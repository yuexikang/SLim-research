import pprint
import numpy as np
from yacs.config import CfgNode as CN
from collections import OrderedDict, defaultdict
from pathlib import Path
from loguru import logger
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
    ExponentialLR,
    CosineAnnealingWarmRestarts,
)
import pytorch_lightning as pl
from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.profilers import PassThroughProfiler
import torch.distributed as dist

from src.slim import SLiM
from src.loss import SLiM_Loss
from src.utils.misc import CudaTimer
from src.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from utils.plotting import make_matching_figures
from utils.comm import all_gather, gather
from utils.misc import flattenList, print_params_summary

test_timer_list = [
    "feat_extract_time",
    "upsample_time",
    "coarse_match_time",
    "fine_match_time",
    "refine_time",
]


def get_mean_time_across_ranks(local_time):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        all_times = [torch.zeros_like(local_time) for _ in range(world_size)]
        dist.all_gather(all_times, local_time)
        return torch.mean(torch.stack(all_times))
    return local_time


class PL_SLiM(pl.LightningModule):
    def __init__(
        self,
        config: CN,
        pretrained_ckpt: str = None,
        profiler: Profiler = None,
        dump_dir: str = None,
    ):
        super().__init__()

        self.config = config
        self.pretrained_ckpt = pretrained_ckpt
        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir

        self.num_devices = None
        self.true_batch_size = (
            config.BATCH_SIZE * config.TRAINER.ACCUMULATE_GRAD_BATCHES
        )
        self.amp = config.AMP

        # Model
        self.slim = SLiM(config=config["MODEL"])
        self.loss = SLiM_Loss(config=config["LOSS"])
        self.slim.max_coarse_matches *= config.BATCH_SIZE
        self.slim.max_fine_matches *= config.BATCH_SIZE

        # Read pretrained checkpoint if exists
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            self.slim.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        self.coarse_scale = self.config.MODEL.COARSE_SCALE  # Coarse scale
        self.lr = self.config.TRAINER.TRUE_LR

        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        self.validation_outputs = []
        self.test_outputs = []
        self.train_forward_times = []
        self.val_forward_times = []
        self.val_parts_times = {}
        self.test_forward_times = []
        self.test_parts_times = {}

    def configure_optimizers(self):
        # optimizer
        if self.config.TRAINER.OPTIMIZER == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.config.TRAINER.ADAM_DECAY,
            )
        elif self.config.TRAINER.OPTIMIZER == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.config.TRAINER.ADAMW_DECAY,
            )
        elif self.config.TRAINER.OPTIMIZER == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.config.TRAINER.SGD_MOMENTUM,
            )
        else:
            # Default: AdamW
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.config.TRAINER.ADAMW_DECAY,
            )

        # learning rate scheduler
        scheduler = {}
        if self.config.TRAINER.SCHEDULER == "MultiStepLR":
            scheduler.update(
                {
                    "interval": "epoch",
                    "scheduler": MultiStepLR(
                        optimizer=optimizer,
                        milestones=self.config.TRAINER.MSLR_MILESTONES,
                        gamma=self.config.TRAINER.MSLR_GAMMA,
                    ),
                }
            )
        elif self.config.TRAINER.SCHEDULER == "CosineAnnealing":
            T_max = int(
                self.config.TRAINER.COSA_TMAX
                * len(self.trainer.datamodule.train_dataloader())
                / self.num_devices
                / self.true_batch_size
            )
            scheduler.update(
                {
                    "interval": "step",
                    "scheduler": CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=T_max,
                        eta_min=self.config.TRAINER.COSA_ETA_MIN,
                    ),
                }
            )
        elif self.config.TRAINER.SCHEDULER == "ExponentialLR":
            scheduler.update(
                {
                    "interval": "epoch",
                    "scheduler": ExponentialLR(
                        optimizer=optimizer, gamma=self.config.TRAINER.ELR_GAMMA
                    ),
                }
            )
        elif self.config.TRAINER.SCHEDULER == "CosineAnnealingWarmRestarts":
            T_0 = int(
                self.config.TRAINER.COSAWR_T0
                * len(self.trainer.datamodule.train_dataloader())
                / self.num_devices
                / self.true_batch_size
            )
            scheduler.update(
                {
                    "interval": "step",
                    "scheduler": CosineAnnealingWarmRestarts(
                        optimizer=optimizer,
                        T_0=T_0,
                        T_mult=self.config.TRAINER.COSAWR_TMULT,
                        eta_min=self.config.TRAINER.COSAWR_ETAMIN,
                    ),
                }
            )
        else:
            # Default: MultiStepLR
            scheduler.update(
                {
                    "interval": "epoch",
                    "scheduler": MultiStepLR(
                        optimizer=optimizer,
                        milestones=self.config.TRAINER.MSLR_MILESTONES,
                        gamma=self.config.TRAINER.MSLR_GAMMA,
                    ),
                }
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if (
            self.trainer.global_step * self.num_devices * self.true_batch_size
            < warmup_step
        ):
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                self.lr = base_lr + (
                    self.trainer.global_step
                    * self.num_devices
                    * self.true_batch_size
                    / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = self.lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch, training: bool = True):
        if training:
            with self.profiler.profile("Compute coarse supervision"):
                compute_supervision_coarse(
                    batch, coarse_scale=self.coarse_scale, config=self.config
                )

        with self.profiler.profile("Main forward"):
            with CudaTimer() as timer:
                self.slim(batch, training=training)

        if training:
            with self.profiler.profile("Compute fine supervision"):
                compute_supervision_fine(batch, config=self.config)

            with self.profiler.profile("Compute losses"):
                self.loss(batch)

        # Timer
        train_time = torch.tensor(timer.elapsed_time, device=self.device)
        mean_train_time = get_mean_time_across_ranks(train_time)
        batch.update({"forward_time": mean_train_time.item()})
        if training:
            self.train_forward_times.append(mean_train_time.item())
        else:
            self.val_forward_times.append(mean_train_time.item())
        torch.cuda.empty_cache()

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            # compute epi_errs for each match
            compute_symmetrical_epipolar_errors(batch)
            # compute R_errs, t_errs, pose_errs for each pair
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    batch["epi_errs"][batch["b_idx_it"] == b].cpu().numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    @staticmethod
    def _is_remote_batch(batch):
        dataset_name = batch.get("dataset_name", "")
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0] if dataset_name else ""
        return str(dataset_name).lower() == "remotesensing"

    @staticmethod
    def _get_pair_names(batch):
        pair_names = batch.get("pair_names")
        if pair_names is None:
            bs = batch["image0"].size(0)
            return [(str(i), str(i)) for i in range(bs)]
        return list(zip(*pair_names))

    @staticmethod
    def _get_batch_value(values, idx, default=""):
        if isinstance(values, torch.Tensor):
            if values.ndim == 0:
                return values.detach().cpu().item()
            if idx < values.shape[0]:
                return values[idx].detach().cpu().item()
            return default
        if isinstance(values, (list, tuple)):
            return values[idx] if idx < len(values) else default
        if values is None:
            return default
        return values

    def _get_remote_identifier(self, batch, pair_name, batch_idx):
        remote_id = self._get_batch_value(batch.get("remote_id"), batch_idx, "")
        aug_variant = self._get_batch_value(batch.get("remote_aug_variant"), batch_idx, "")
        pair_id = self._get_batch_value(batch.get("pair_id"), batch_idx, "")
        return "#".join(
            [
                str(pair_name[0]) if len(pair_name) > 0 else "",
                str(pair_name[1]) if len(pair_name) > 1 else "",
                f"remote_id={remote_id}",
                f"pair_id={pair_id}",
                f"aug={aug_variant}",
            ]
        )

    @staticmethod
    def _warp_points_by_homography(points, H):
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=1)
        warped_h = points_h @ H.T
        denom = warped_h[:, 2:3].clamp(min=1e-8)
        return warped_h[:, :2] / denom

    def _compute_remote_homography_metrics(self, batch):
        with self.profiler.profile("Compute remote homography metrics"):
            bs = batch["image0"].size(0)
            pair_names = self._get_pair_names(batch)
            b_ids = batch.get(
                "b_idx_it",
                torch.empty(0, device=batch["image0"].device, dtype=torch.long),
            )
            pts0 = batch.get("fine_coord_0")
            pts1 = batch.get("fine_coord_1")
            H_0to1 = batch["H_0to1"].to(device=batch["image0"].device)

            reproj_errors = []
            num_matches = []
            identifiers = []
            aug_variants = []
            for b in range(bs):
                identifiers.append(self._get_remote_identifier(batch, pair_names[b], b))
                aug_variants.append(
                    str(self._get_batch_value(batch.get("remote_aug_variant"), b, "unknown"))
                )
                if pts0 is None or pts1 is None or b_ids.numel() == 0:
                    reproj_errors.append(np.empty(0, dtype=np.float32))
                    num_matches.append(0)
                    continue

                mask = b_ids == b
                b_pts0 = pts0[mask]
                b_pts1 = pts1[mask]
                num_matches.append(int(b_pts0.shape[0]))
                if b_pts0.numel() == 0:
                    reproj_errors.append(np.empty(0, dtype=np.float32))
                    continue

                H = H_0to1[b].to(dtype=b_pts0.dtype)
                warped_pts0 = self._warp_points_by_homography(b_pts0, H)
                errors = torch.linalg.norm(warped_pts0 - b_pts1, dim=1)
                reproj_errors.append(errors.detach().cpu().numpy())

            return {
                "identifiers": identifiers,
                "reproj_errors": reproj_errors,
                "num_matches": num_matches,
                "aug_variants": aug_variants,
            }

    @staticmethod
    def _aggregate_remote_homography_metrics(metrics):
        unique_ids = OrderedDict(
            (identifier, idx) for idx, identifier in enumerate(metrics["identifiers"])
        )
        unique_indices = list(unique_ids.values())
        errors_per_pair = [
            np.asarray(metrics["reproj_errors"][idx], dtype=np.float32)
            for idx in unique_indices
        ]
        num_matches = [int(metrics["num_matches"][idx]) for idx in unique_indices]
        aug_variants = metrics.get("aug_variants")
        aug_variants = (
            [str(aug_variants[idx]) for idx in unique_indices]
            if aug_variants is not None
            else ["unknown"] * len(unique_indices)
        )
        all_errors = (
            np.concatenate(errors_per_pair)
            if any(len(errors) > 0 for errors in errors_per_pair)
            else np.empty(0, dtype=np.float32)
        )

        val_metrics = {
            "remote_num_pairs": len(unique_indices),
            "remote_num_matches": int(sum(num_matches)),
            "remote_mean_matches": float(np.mean(num_matches)) if num_matches else 0.0,
            "remote_mean_error": float(np.mean(all_errors)) if len(all_errors) else 0.0,
            "remote_median_error": float(np.median(all_errors)) if len(all_errors) else 0.0,
        }
        for thr in [1, 3, 5, 10]:
            val_metrics[f"remote_inlier@{thr}"] = (
                float(np.mean(all_errors < thr)) if len(all_errors) else 0.0
            )
            pair_precisions = [
                float(np.mean(errors < thr)) if len(errors) else 0.0
                for errors in errors_per_pair
            ]
            val_metrics[f"remote_pair_inlier@{thr}"] = (
                float(np.mean(pair_precisions)) if pair_precisions else 0.0
            )
        for variant in sorted(set(aug_variants)):
            variant_indices = [i for i, v in enumerate(aug_variants) if v == variant]
            variant_errors = [
                errors_per_pair[i] for i in variant_indices if len(errors_per_pair[i]) > 0
            ]
            variant_all_errors = (
                np.concatenate(variant_errors)
                if variant_errors
                else np.empty(0, dtype=np.float32)
            )
            key = str(variant).replace("/", "_")
            val_metrics[f"remote_{key}_num_pairs"] = len(variant_indices)
            val_metrics[f"remote_{key}_inlier@5"] = (
                float(np.mean(variant_all_errors < 5)) if len(variant_all_errors) else 0.0
            )
            val_metrics[f"remote_{key}_median_error"] = (
                float(np.median(variant_all_errors)) if len(variant_all_errors) else 0.0
            )
        return val_metrics

    def _collect_remote_batch_debug_info(self, batch, batch_idx):
        if not self._is_remote_batch(batch):
            return None

        bs = int(batch["image0"].shape[0])
        remote_ids = batch.get("remote_id", [""] * bs)
        aug_variants = batch.get("remote_aug_variant", [""] * bs)
        remote_modes = batch.get("remote_mode", [""] * bs)
        pair_types = batch.get("remote_pair_type", [""] * bs)
        pair_names = self._get_pair_names(batch)

        def get_value(values, idx, default=""):
            if isinstance(values, torch.Tensor):
                return default
            if not isinstance(values, (list, tuple)):
                return values
            return values[idx] if idx < len(values) else default

        items = []
        for b in range(bs):
            names = pair_names[b] if b < len(pair_names) else ("", "")
            items.append(
                {
                    "batch_pos": b,
                    "remote_id": str(get_value(remote_ids, b)),
                    "mode": str(get_value(remote_modes, b)),
                    "pair_type": str(get_value(pair_types, b)),
                    "aug_variant": str(get_value(aug_variants, b)),
                    "image0": str(names[0]) if len(names) > 0 else "",
                    "image1": str(names[1]) if len(names) > 1 else "",
                }
            )

        return {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "batch_idx": int(batch_idx),
            "global_rank": int(getattr(self, "global_rank", 0)),
            "local_rank": int(getattr(self, "local_rank", 0)),
            "items": items,
        }

    def training_step(self, batch, batch_idx):
        remote_debug_info = self._collect_remote_batch_debug_info(batch, batch_idx)
        try:
            self._trainval_inference(batch, training=True)
        except Exception:
            if remote_debug_info is not None:
                logger.error(
                    "Remote batch failed before/during training_step:\n{}",
                    pprint.pformat(remote_debug_info, width=160),
                )
            raise
        log_batch_size = batch["image0"].shape[0]
        log_device = batch["image0"].device
        self.log(
            "train/loss",
            batch["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=log_batch_size,
        )
        for k, v in batch["loss_scalars"].items():
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=log_device, dtype=batch["loss"].dtype)
            else:
                v = v.to(device=log_device, dtype=batch["loss"].dtype)
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=log_batch_size,
            )
        num_matches = torch.as_tensor(
            batch["num_matches"], device=log_device, dtype=batch["loss"].dtype
        )
        self.log(
            "train/num_matches",
            num_matches,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=log_batch_size,
        )
        forward_time = torch.as_tensor(
            batch["forward_time"], device=log_device, dtype=batch["loss"].dtype
        )
        self.log(
            "train/forward_time",
            forward_time,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=log_batch_size,
        )
        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(
                    f"train/{k}",
                    v,
                    self.global_step * self.num_devices * self.true_batch_size,
                )

            # num matches
            self.logger.experiment.add_scalar(
                "train/num_matches",
                batch["num_matches"],
                self.global_step * self.num_devices * self.true_batch_size,
            )

            # time
            self.logger.experiment.add_scalar(
                "train/forward_time",
                batch["forward_time"],
                self.global_step * self.num_devices * self.true_batch_size,
            )
            keys = test_timer_list
            for k in keys:
                self.logger.experiment.add_scalar(
                    f"train/{k}",
                    batch[k],
                    self.global_step * self.num_devices * self.true_batch_size,
                )

        return {"loss": batch["loss"]}

    def on_train_epoch_start(self):
        if self.trainer.current_epoch == 0:
            # Print model summary
            if self.trainer.global_rank == 0:
                print(self.slim)
                print_params_summary(self.slim, recursive=False)
        self.slim.train()
        self.slim.initial_forward()

    def on_train_epoch_end(self):
        # time
        if self.trainer.global_rank == 0 and len(self.train_forward_times) > 0:
            avg_forward_time = sum(self.train_forward_times) / len(
                self.train_forward_times
            )
            self.logger.experiment.add_scalar(
                "train/avg_forward_time", avg_forward_time, self.trainer.current_epoch
            )
        self.train_forward_times.clear()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self._trainval_inference(batch, training=False)

        # All timers
        keys = test_timer_list
        if len(self.val_parts_times.keys()) == 0:
            self.val_parts_times = {k: [] for k in keys}
        for key in keys:
            if key in batch.keys():
                self.val_parts_times[key].append(batch[key])

        if self._is_remote_batch(batch):
            self.validation_outputs.append(
                {"remote_metrics": self._compute_remote_homography_metrics(batch)}
            )
            return

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        self.validation_outputs.append(
            {
                **ret_dict,
                "figures": figures,
            }
        )

    def on_validation_epoch_start(self):
        self.slim.eval()
        self.slim.initial_forward()

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            torch.cuda.empty_cache()
            return
        if self.trainer.sanity_checking:
            self.validation_outputs.clear()
            torch.cuda.empty_cache()
            return

        if "remote_metrics" in self.validation_outputs[0]:
            cur_epoch = self.trainer.current_epoch
            _metrics = [o["remote_metrics"] for o in self.validation_outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            val_metrics_4tb = self._aggregate_remote_homography_metrics(metrics)

            for k, v in val_metrics_4tb.items():
                self.log(k, v, prog_bar=(k == "remote_inlier@5"), sync_dist=True)

            if self.trainer.global_rank == 0:
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"remote_val/{k}", v, global_step=cur_epoch
                    )
                if len(self.val_forward_times) > 0:
                    avg_forward_time = self.calculate_trimmed_mean(
                        self.val_forward_times
                    )
                    self.logger.experiment.add_scalar(
                        "remote_val/avg_forward_time",
                        avg_forward_time,
                        self.trainer.current_epoch,
                    )
                    self.val_forward_times.clear()
                keys = test_timer_list
                if len(self.val_parts_times.keys()) and len(
                    self.val_parts_times[keys[0]]
                ):
                    for k in keys:
                        avg_time_for_this_part = self.calculate_trimmed_mean(
                            self.val_parts_times[k]
                        )
                        self.logger.experiment.add_scalar(
                            f"remote_val/{k}",
                            avg_time_for_this_part,
                            self.trainer.current_epoch,
                        )
                        self.val_parts_times[k].clear()

            self.validation_outputs.clear()
            torch.cuda.empty_cache()
            return

        # handle multiple validation sets
        multi_outputs = (
            [self.validation_outputs]
            if not isinstance(self.validation_outputs[0], (list, tuple))
            else self.validation_outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if self.trainer.sanity_checking:
                self.validation_outputs.clear()
                torch.cuda.empty_cache()
                return

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            for thr in [5, 10, 20]:
                metric_name = f"auc@{thr}"
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])
                self.log(
                    metric_name,
                    val_metrics_4tb[metric_name],
                    prog_bar=True,
                    sync_dist=True,
                )

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                                fig,
                                cur_epoch,
                                close=True,
                            )
                # time
                if len(self.val_forward_times) > 0:
                    avg_forward_time = self.calculate_trimmed_mean(
                        self.val_forward_times
                    )
                    self.logger.experiment.add_scalar(
                        "test/avg_forward_time",
                        avg_forward_time,
                        self.trainer.current_epoch,
                    )
                    self.val_forward_times.clear()
                keys = test_timer_list
                if len(self.val_parts_times.keys()) and len(
                    self.val_parts_times[keys[0]]
                ):
                    for k in keys:
                        avg_time_for_this_part = self.calculate_trimmed_mean(
                            self.val_parts_times[k]
                        )
                        self.logger.experiment.add_scalar(
                            f"test/{k}",
                            avg_time_for_this_part,
                            self.trainer.current_epoch,
                        )
                        self.val_parts_times[k].clear()

        self.validation_outputs.clear()
        torch.cuda.empty_cache()

    def test_step(self, batch: dict, batch_idx):
        # inference
        with torch.inference_mode(True):
            with torch.cuda.amp.autocast(enabled=self.amp):
                with CudaTimer() as timer:
                    self.slim(batch)

        # All timers
        keys = test_timer_list
        if len(self.test_parts_times.keys()) == 0:
            self.test_parts_times = {k: [] for k in keys}
        for key in keys:
            if key in batch.keys():
                self.test_parts_times[key].append(batch[key])

        # Timer
        local_time = torch.tensor(timer.elapsed_time, device=self.device)
        mean_time = get_mean_time_across_ranks(local_time)
        batch.update({"forward_time": mean_time.item()})
        self.test_forward_times.append(mean_time.item())
        torch.cuda.empty_cache()

        # metrics
        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # dump results
        keys_to_save = {"fine_coord_0", "fine_coord_1", "epi_errs"}
        pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].shape[0]
        dumps = []

        for b_id in range(bs):
            item = {}
            mask = batch["b_idx_it"] == b_id
            item["pair_names"] = pair_names[b_id]
            item["identifier"] = "#".join(rel_pair_names[b_id])
            for key in keys_to_save:
                item[key] = batch[key][mask].cpu().numpy()
            for key in ["R_errs", "t_errs", "inliers"]:
                item[key] = batch[key][b_id]
            dumps.append(item)
        ret_dict["dumps"] = dumps

        self.test_outputs.append(ret_dict)

    def on_test_epoch_start(self):
        self.slim.eval()
        self.slim.initial_forward()

    def on_test_epoch_end(self):
        with torch.inference_mode(False):
            # metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in self.test_outputs]
            metrics = {
                k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }

            # [{key: [{...}, *#bs]}, *#batch]
            if self.dump_dir is not None:
                Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
                _dumps = flattenList(
                    [o["dumps"] for o in self.test_outputs]
                )  # [{...}, #bs*#batch]
                dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
                logger.info(
                    f"Prediction and evaluation results will be saved to: {self.dump_dir}"
                )
        if self.trainer.global_rank == 0:
            # time
            if len(self.test_forward_times) > 0:
                avg_forward_time = self.calculate_trimmed_mean(self.test_forward_times)
                print(f"Avg forward time: {avg_forward_time * 1000: 8.4}ms")
                self.test_forward_times.clear()
            keys = test_timer_list
            if len(self.test_parts_times.keys()) and len(
                self.test_parts_times[keys[0]]
            ):
                for k in keys:
                    avg_time_for_this_part = self.calculate_trimmed_mean(
                        self.test_parts_times[k]
                    )
                    print(
                        f"Avg forward time for {k}: {avg_time_for_this_part * 1000: 8.4}ms"
                    )
                    self.test_parts_times[k].clear()
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "slim_pred_eval", dumps)
        self.test_outputs.clear()

    @staticmethod
    def calculate_trimmed_mean(_list):
        n = len(_list)
        trim_count = int(0.1 * n)
        sorted_list = sorted(_list)
        trimmed_list = (
            sorted_list[trim_count:-trim_count] if trim_count > 0 else sorted_list
        )
        avg = sum(trimmed_list) / len(trimmed_list)
        return avg
