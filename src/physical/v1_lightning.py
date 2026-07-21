import math
import time
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .matching import PPMatchingLoss, coarse_homography_correspondences
from .metrics import descriptor_statistics
from .models import count_trainable_parameters
from .v1_losses import orientation_equivariance_loss
from .v1_models import build_physical_v1_encoder


class PhysicalV1Module(pl.LightningModule):
    def __init__(
        self,
        model_name="physical_v1_core",
        similarity_mode="chunked",
        learning_rate=1e-4,
        physical_learning_rate=1e-5,
        weight_decay=0.01,
        max_epochs=20,
        coarse_scale=8,
        chunk_size=256,
        orientation_weight=0.1,
        branch_weight=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = str(model_name)
        self.similarity_mode = str(similarity_mode)
        self.learning_rate = float(learning_rate)
        self.physical_learning_rate = float(physical_learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs_config = int(max_epochs)
        self.coarse_scale = int(coarse_scale)
        self.orientation_weight = float(orientation_weight)
        self.branch_weight = float(branch_weight)
        self.encoder = build_physical_v1_encoder(self.model_name)
        self.fused_loss_fn = PPMatchingLoss(
            gamma=2.0,
            positive_percent=0.9,
            temperature=0.05,
            chunk_size=chunk_size,
        )
        self.branch_loss_fn = PPMatchingLoss(
            gamma=2.0,
            positive_percent=0.25,
            temperature=0.07,
            chunk_size=chunk_size,
        )
        self.branch_loss_fn.log_temperature.requires_grad_(False)
        self.parameter_count = count_trainable_parameters(self.encoder)
        self.validation_accumulator = defaultdict(float)
        self.validation_log_prefix = "val"
        self.train_epoch_started = None
        self.train_samples = 0

    @property
    def active_branches(self):
        if self.encoder.response_mode == "energy_only":
            return ("stable",)
        return ("edge", "contour", "stable")

    def forward(self, image):
        return self.encoder(image)

    def configure_optimizers(self):
        physical = self.encoder.physical_filter_parameters()
        physical_ids = {id(parameter) for parameter in physical}
        temperature = [self.fused_loss_fn.log_temperature]
        temperature_ids = {id(parameter) for parameter in temperature}
        remaining = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
            and id(parameter) not in physical_ids
            and id(parameter) not in temperature_ids
        ]
        parameter_groups = []
        if physical:
            parameter_groups.append(
                {
                    "params": physical,
                    "lr": self.physical_learning_rate,
                    "weight_decay": 0.0,
                    "name": "physical_filters",
                }
            )
        parameter_groups.extend(
            [
                {
                    "params": temperature,
                    "lr": self.learning_rate,
                    "weight_decay": 0.0,
                    "name": "temperature",
                },
                {
                    "params": remaining,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "name": "encoder",
                },
            ]
        )
        optimizer = AdamW(parameter_groups)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs_config,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _forward_pair(self, batch):
        output0 = self.encoder(batch["image0"])
        output1 = self.encoder(batch["image1"])
        correspondences = coarse_homography_correspondences(batch, self.coarse_scale)
        return output0, output1, correspondences

    @staticmethod
    def _selected_indices(correspondences, percent, deterministic=False):
        count = correspondences[0].numel()
        selected_count = max(1, int(count * float(percent)))
        if deterministic:
            return torch.arange(selected_count, device=correspondences[0].device)
        return torch.randperm(count, device=correspondences[0].device)[:selected_count]

    def _losses(self, batch, output0, output1, correspondences, deterministic=False):
        fused_selected = self._selected_indices(
            correspondences, self.fused_loss_fn.positive_percent, deterministic
        )
        fused = self.fused_loss_fn(
            output0["fused"],
            output1["fused"],
            correspondences,
            mode=self.similarity_mode,
            selected=fused_selected,
        )
        orientation = orientation_equivariance_loss(
            output0,
            output1,
            batch,
            correspondences,
            coarse_scale=self.coarse_scale,
        )
        branch_selected = self._selected_indices(
            correspondences, self.branch_loss_fn.positive_percent, deterministic
        )
        branch_losses = {
            name: self.branch_loss_fn(
                output0[name],
                output1[name],
                correspondences,
                mode="chunked",
                selected=branch_selected,
            )
            for name in self.active_branches
        }
        branch = torch.stack(list(branch_losses.values())).mean()
        total = fused + self.orientation_weight * orientation + self.branch_weight * branch
        return total, fused, orientation, branch, branch_losses

    def training_step(self, batch, batch_idx):
        output0, output1, correspondences = self._forward_pair(batch)
        total, fused, orientation, branch, branch_losses = self._losses(
            batch, output0, output1, correspondences
        )
        batch_size = int(batch["image0"].shape[0])
        self.train_samples += batch_size
        values = {
            "train/loss": total,
            "train/loss_pp_fused": fused,
            "train/loss_orientation": orientation,
            "train/loss_branch": branch,
        }
        values.update(
            {f"train/loss_branch_{name}": value for name, value in branch_losses.items()}
        )
        for name, value in values.items():
            self.log(
                name,
                value,
                on_step=name == "train/loss",
                on_epoch=True,
                prog_bar=name == "train/loss",
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log(
            "train/temperature",
            self.fused_loss_fn.temperature.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return total

    def on_train_epoch_start(self):
        self.train_epoch_started = time.perf_counter()
        self.train_samples = 0
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.set_epoch(self.current_epoch)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_end(self):
        elapsed = max(time.perf_counter() - self.train_epoch_started, 1e-6)
        global_throughput = self.train_samples * self.trainer.world_size / elapsed
        self.log("train/samples_per_second", global_throughput, sync_dist=True)
        if self.device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(self.device) / 1024**3
            self.log("train/peak_allocated_gib", allocated, sync_dist=True)
            self.log("train/peak_reserved_gib", reserved, sync_dist=True)
        self.log("model/trainable_parameters", float(self.parameter_count), sync_dist=True)

    def on_validation_epoch_start(self):
        self.validation_accumulator.clear()
        data_module = self.trainer.datamodule
        self.validation_log_prefix = (
            "full_val"
            if data_module is not None
            and getattr(data_module, "full_validation_enabled", False)
            else "val"
        )

    def _accumulate_descriptor_stats(self, prefix, descriptor0, descriptor1, correspondences, batch):
        stats = descriptor_statistics(
            descriptor0,
            descriptor1,
            correspondences,
            self.fused_loss_fn.temperature.detach(),
            variants=batch.get("remote_aug_variant"),
        )
        for variant, values in stats.items():
            for name, value in values.items():
                self.validation_accumulator[f"{prefix}/{variant}/{name}"] += value

    def _accumulate_diagnostics(self, output0, output1, batch_size):
        for output in (output0, output1):
            for gate_name in ("scale_weights", "expert_weights"):
                gate = output[gate_name].float()
                entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=1)
                normalized_entropy = entropy / math.log(gate.shape[1])
                self.validation_accumulator[f"diag/{gate_name}/entropy_sum"] += float(
                    normalized_entropy.mean()
                ) * batch_size
                for index in range(gate.shape[1]):
                    values = gate[:, index]
                    self.validation_accumulator[f"diag/{gate_name}/{index}_mean_sum"] += float(
                        values.mean()
                    ) * batch_size
                    self.validation_accumulator[f"diag/{gate_name}/{index}_std_sum"] += float(
                        values.std(unbiased=False)
                    ) * batch_size
            confidence = output["confidence"].float().flatten()
            self.validation_accumulator["diag/confidence/mean_sum"] += float(
                confidence.mean()
            ) * batch_size
            for quantile in (0.25, 0.5, 0.75):
                self.validation_accumulator[
                    f"diag/confidence/q{int(quantile * 100)}_sum"
                ] += float(torch.quantile(confidence, quantile)) * batch_size
            self.validation_accumulator["diag/confidence/low_sum"] += float(
                (confidence < 0.25).float().mean()
            ) * batch_size
            self.validation_accumulator["diag/count"] += batch_size

    def validation_step(self, batch, batch_idx):
        output0, output1, correspondences = self._forward_pair(batch)
        total, fused, orientation, branch, branch_losses = self._losses(
            batch, output0, output1, correspondences, deterministic=True
        )
        batch_size = int(batch["image0"].shape[0])
        for name, value in {
            "total": total,
            "fused": fused,
            "orientation": orientation,
            "branch": branch,
        }.items():
            self.validation_accumulator[f"loss/{name}_sum"] += float(value.detach()) * batch_size
        self.validation_accumulator["loss/count"] += batch_size
        for name, value in branch_losses.items():
            self.validation_accumulator[f"loss/branch_{name}_sum"] += float(
                value.detach()
            ) * batch_size
        self._accumulate_descriptor_stats(
            "fused", output0["fused"], output1["fused"], correspondences, batch
        )
        for name in self.active_branches:
            self._accumulate_descriptor_stats(
                f"branch/{name}", output0[name], output1[name], correspondences, batch
            )
        self._accumulate_diagnostics(output0, output1, batch_size)

    def _distributed_sum(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.cpu())

    def on_validation_epoch_end(self):
        log_prefix = self.validation_log_prefix
        reduced = {
            key: self._distributed_sum(value)
            for key, value in self.validation_accumulator.items()
        }
        loss_count = max(reduced.get("loss/count", 0.0), 1.0)
        for name in ("total", "fused", "orientation", "branch"):
            self.log(
                f"{log_prefix}/loss_{name}",
                reduced.get(f"loss/{name}_sum", 0.0) / loss_count,
                sync_dist=True,
            )
        for branch_name in self.active_branches:
            self.log(
                f"{log_prefix}/loss_branch_{branch_name}",
                reduced.get(f"loss/branch_{branch_name}_sum", 0.0) / loss_count,
                sync_dist=True,
            )

        variants = ("all", "translation", "scale", "yaw", "pitch", "roll")
        for prefix in ("fused", *[f"branch/{name}" for name in self.active_branches]):
            for variant in variants:
                base = f"{prefix}/{variant}/"
                count = reduced.get(base + "count", 0.0)
                if count <= 0:
                    continue
                values = {
                    "r0": reduced[base + "correct0"] / count,
                    "r1": reduced[base + "correct1"] / count,
                    "positive_similarity": reduced[base + "positive"] / count,
                    "hard_negative_similarity": reduced[base + "hard_negative"] / count,
                    "margin": reduced[base + "margin"] / count,
                    "entropy": reduced[base + "entropy"] / count,
                    "normalized_entropy": reduced[base + "normalized_entropy"] / count,
                }
                output_prefix = (
                    log_prefix if prefix == "fused" else f"{log_prefix}/{prefix}"
                )
                for name, value in values.items():
                    show = prefix == "fused" and variant == "all" and name in {"r0", "r1"}
                    self.log(
                        f"{output_prefix}/{variant}_{name}",
                        value,
                        prog_bar=show,
                        sync_dist=True,
                    )

        diagnostic_count = max(reduced.get("diag/count", 0.0), 1.0)
        for gate_name in ("scale_weights", "expert_weights"):
            self.log(
                f"{log_prefix}/{gate_name}_entropy",
                reduced.get(f"diag/{gate_name}/entropy_sum", 0.0) / diagnostic_count,
                sync_dist=True,
            )
            for index in range(3):
                for statistic in ("mean", "std"):
                    self.log(
                        f"{log_prefix}/{gate_name}_{index}_{statistic}",
                        reduced.get(
                            f"diag/{gate_name}/{index}_{statistic}_sum", 0.0
                        )
                        / diagnostic_count,
                        sync_dist=True,
                    )
        for statistic in ("mean", "q25", "q50", "q75", "low"):
            self.log(
                f"{log_prefix}/confidence_{statistic}",
                reduced.get(f"diag/confidence/{statistic}_sum", 0.0)
                / diagnostic_count,
                sync_dist=True,
            )


__all__ = ["PhysicalV1Module"]
