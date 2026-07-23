import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .matching import PPMatchingLoss, coarse_homography_correspondences
from .metrics import descriptor_statistics
from .models import count_trainable_parameters
from .v1_losses import orientation_equivariance_loss
from .v2_losses import recovery_and_preservation_loss
from .v2_models import FrozenSLiMCoarseExtractor, build_physical_v2_encoder
from .v2_visualization import write_latest_feature_maps


class PhysicalV2Module(pl.LightningModule):
    IMPLEMENTATION_VERSION = "2.1.4"

    def __init__(
        self,
        model_name="physical_v2_core",
        slim_checkpoint="ckpt/megadepth_19epochs.ckpt",
        learning_rate=1e-4,
        physical_learning_rate=1e-5,
        weight_decay=0.01,
        max_epochs=20,
        coarse_scale=8,
        chunk_size=256,
        gradient_log_interval=200,
        polar_chunk_size=1024,
        feature_visualization_interval=20,
        feature_visualization_dir=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = str(model_name)
        self.learning_rate = float(learning_rate)
        self.physical_learning_rate = float(physical_learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs_config = int(max_epochs)
        self.coarse_scale = int(coarse_scale)
        self.chunk_size = int(chunk_size)
        self.gradient_log_interval = int(gradient_log_interval)
        self.feature_visualization_interval = int(feature_visualization_interval)
        self.feature_visualization_dir = (
            Path(feature_visualization_dir) if feature_visualization_dir else None
        )
        self.encoder = build_physical_v2_encoder(
            self.model_name, polar_chunk_size=polar_chunk_size
        )
        self.frozen_slim = FrozenSLiMCoarseExtractor(slim_checkpoint)
        self.physical_loss = PPMatchingLoss(
            gamma=2.0,
            positive_percent=1.0,
            temperature=0.05,
            chunk_size=chunk_size,
            stable_log=True,
        )
        self.unary_loss = PPMatchingLoss(
            gamma=2.0,
            positive_percent=0.25,
            temperature=0.07,
            chunk_size=chunk_size,
            stable_log=True,
        )
        self.unary_loss.log_temperature.requires_grad_(False)
        self.parameter_count = count_trainable_parameters(self.encoder) + 1
        self.validation_accumulator = defaultdict(float)
        self.validation_log_prefix = "val"
        self.train_epoch_started = None
        self.train_samples = 0
        self._gradient_contract_checked = False
        self._active_batch_context = {}
        self._last_visualized_step = None
        self._gabor_sanitized_gradients = {}
        self._gabor_gradient_warning_emitted = False
        for name, parameter in self.encoder.gabor.named_parameters():
            if parameter.requires_grad:
                parameter.register_hook(
                    lambda gradient, parameter_name=f"gabor.{name}": self._sanitize_gabor_gradient(
                        parameter_name, gradient
                    )
                )

    def _sanitize_gabor_gradient(self, name, gradient):
        finite = torch.isfinite(gradient)
        invalid_count = int((~finite).sum().item())
        if invalid_count:
            self._gabor_sanitized_gradients[name] = invalid_count
            if not self._gabor_gradient_warning_emitted:
                warnings.warn(
                    "Non-finite Gabor scalar gradients were replaced with zero; "
                    "see train/gabor_sanitized_grad_elements."
                )
                self._gabor_gradient_warning_emitted = True
            gradient = torch.where(finite, gradient, torch.zeros_like(gradient))
        return gradient

    def train(self, mode=True):
        super().train(mode)
        self.frozen_slim.eval()
        return self

    def on_save_checkpoint(self, checkpoint):
        state = checkpoint.get("state_dict", {})
        for key in [name for name in state if name.startswith("frozen_slim.")]:
            del state[key]
        checkpoint["frozen_slim_checkpoint"] = self.hparams.slim_checkpoint

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.setdefault("state_dict", {})
        current = self.state_dict()
        for key, value in current.items():
            if key.startswith("frozen_slim.") and key not in state:
                state[key] = value

    def configure_optimizers(self):
        physical = self.encoder.physical_filter_parameters()
        physical_ids = {id(parameter) for parameter in physical}
        temperature = [self.physical_loss.log_temperature]
        temperature_ids = {id(parameter) for parameter in temperature}
        remaining = [
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad and id(parameter) not in physical_ids
        ]
        groups = [
            {
                "params": physical,
                "lr": self.physical_learning_rate,
                "weight_decay": 0.0,
                "name": "physical_filters",
            },
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
                "name": "physical_v2",
            },
        ]
        optimizer = AdamW(groups)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.max_epochs_config, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, image0, image1):
        base0, base1 = self.frozen_slim(image0, image1)
        output0, output1 = self.encoder.forward_pair(image0, image1)
        output0["enhanced"] = base0 + output0["delta"]
        output1["enhanced"] = base1 + output1["delta"]
        return output0, output1

    def _forward_pair(self, batch):
        base0, base1 = self.frozen_slim(batch["image0"], batch["image1"])
        output0, output1 = self.encoder.forward_pair(batch["image0"], batch["image1"])
        output0["enhanced"] = base0 + output0["delta"]
        output1["enhanced"] = base1 + output1["delta"]
        correspondences = coarse_homography_correspondences(batch, self.coarse_scale)
        return base0, base1, output0, output1, correspondences

    @staticmethod
    def _selected_indices(correspondences, percent, deterministic=False):
        count = correspondences[0].numel()
        selected_count = max(1, int(count * float(percent)))
        if deterministic:
            return torch.arange(selected_count, device=correspondences[0].device)
        return torch.randperm(count, device=correspondences[0].device)[:selected_count]

    def _losses(
        self,
        batch,
        base0,
        base1,
        output0,
        output1,
        correspondences,
        deterministic=False,
    ):
        recover, keep, recovery_diag = recovery_and_preservation_loss(
            base0,
            base1,
            output0["enhanced"],
            output1["enhanced"],
            correspondences,
            self.frozen_slim.temperature,
            chunk_size=self.chunk_size,
            recovery_weighting=self.encoder.recovery_weighting,
        )
        physical = self.physical_loss(
            output0["physical"],
            output1["physical"],
            correspondences,
            mode="chunked",
        )
        selected = self._selected_indices(correspondences, 0.25, deterministic)
        unary_losses = [
            self.unary_loss(
                descriptor0,
                descriptor1,
                correspondences,
                mode="chunked",
                selected=selected,
            )
            for descriptor0, descriptor1 in zip(output0["unary"], output1["unary"])
        ]
        unary = torch.stack(unary_losses).mean()
        orientation = orientation_equivariance_loss(
            {
                "orientation": output0["orientation"],
                "confidence": output0["reliability"],
            },
            {
                "orientation": output1["orientation"],
                "confidence": output1["reliability"],
            },
            batch,
            correspondences,
            coarse_scale=self.coarse_scale,
        )
        if self.current_epoch < 3:
            physical_weight, unary_weight = 1.0, 0.3
        else:
            physical_weight, unary_weight = 0.5, 0.2
        total = (
            recover
            + 0.25 * keep
            + physical_weight * physical
            + unary_weight * unary
            + 0.1 * orientation
        )
        components = {
            "recover": recover,
            "keep": keep,
            "physical": physical,
            "unary": unary,
            "orientation": orientation,
        }
        return total, components, recovery_diag

    def _log_gradient_norms(self, components):
        if self.gradient_log_interval <= 0 or self.global_step % self.gradient_log_interval:
            return
        parameters = [
            parameter for parameter in self.encoder.parameters() if parameter.requires_grad
        ]
        for name, loss in components.items():
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared = sum(
                gradient.detach().float().square().sum()
                for gradient in gradients
                if gradient is not None
            )
            if not torch.is_tensor(squared):
                squared = loss.new_zeros(())
            self.log(
                f"train/grad_norm_{name}",
                squared.sqrt(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

    def _maybe_visualize_features(
        self, batch, batch_idx, base0, base1, output0, output1, total, components
    ):
        if (
            self.feature_visualization_dir is None
            or self.feature_visualization_interval <= 0
            or not self.trainer.is_global_zero
        ):
            return
        step = int(self.global_step)
        if self._last_visualized_step == step:
            return
        if self._last_visualized_step is not None and step % self.feature_visualization_interval:
            return
        wavelength, sigma, gamma = self.encoder.gabor.constrained_parameters()
        gabor_parameters = {
            "wavelength": wavelength.detach().float().cpu().tolist(),
            "sigma": sigma.detach().float().cpu().tolist(),
            "gamma": gamma.detach().float().cpu().tolist(),
        }
        try:
            write_latest_feature_maps(
                self.feature_visualization_dir,
                batch,
                base0,
                base1,
                output0,
                output1,
                epoch=self.current_epoch,
                global_step=step,
                batch_idx=batch_idx,
                losses={"total": total, **components},
                gabor_parameters=gabor_parameters,
            )
            self._last_visualized_step = step
        except Exception as exc:
            error_path = self.feature_visualization_dir / "visualization_error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(f"step={step}\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
            warnings.warn(f"Physical V2 feature visualization failed: {exc}")

    def training_step(self, batch, batch_idx):
        self._gabor_sanitized_gradients = {}
        self._active_batch_context = {
            "batch_idx": int(batch_idx),
            "remote_ids": self._batch_values(batch, "remote_id"),
            "variants": self._batch_values(batch, "remote_aug_variant"),
        }
        base0, base1, output0, output1, correspondences = self._forward_pair(batch)
        total, components, diagnostics = self._losses(
            batch, base0, base1, output0, output1, correspondences
        )
        self._check_finite_training_state(
            batch, batch_idx, total, components, output0, output1
        )
        self._maybe_visualize_features(
            batch, batch_idx, base0, base1, output0, output1, total, components
        )
        self._log_gradient_norms(components)
        batch_size = int(batch["image0"].shape[0])
        self.train_samples += batch_size
        self.log(
            "train/loss",
            total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        for name, value in components.items():
            self.log(
                f"train/loss_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        for name, value in diagnostics.items():
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log(
            "train/physical_temperature",
            self.physical_loss.temperature.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return total

    def on_train_epoch_start(self):
        self.frozen_slim.eval()
        self.train_epoch_started = time.perf_counter()
        self.train_samples = 0
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.set_epoch(self.current_epoch)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_end(self):
        elapsed = max(time.perf_counter() - self.train_epoch_started, 1e-6)
        throughput = self.train_samples * self.trainer.world_size / elapsed
        self.log("train/samples_per_second", throughput, sync_dist=True)
        self.log("model/trainable_parameters", float(self.parameter_count), sync_dist=True)
        if self.device.type == "cuda":
            self.log(
                "train/peak_allocated_gib",
                torch.cuda.max_memory_allocated(self.device) / 1024**3,
                sync_dist=True,
            )
            self.log(
                "train/peak_reserved_gib",
                torch.cuda.max_memory_reserved(self.device) / 1024**3,
                sync_dist=True,
            )

    def on_after_backward(self):
        missing = [
            name
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        nonfinite = [
            name
            for name, parameter in self.encoder.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        frozen_with_grad = [
            name
            for name, parameter in self.frozen_slim.named_parameters()
            if parameter.grad is not None
        ]
        if missing:
            self._fail_nonfinite("missing_gradient", {"parameters": missing})
        if nonfinite:
            self._fail_nonfinite("nonfinite_gradient", {"parameters": nonfinite})
        if frozen_with_grad:
            self._fail_nonfinite(
                "frozen_slim_gradient", {"parameters": frozen_with_grad}
            )
        self.log(
            "train/gabor_sanitized_grad_elements",
            float(sum(self._gabor_sanitized_gradients.values())),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self._gradient_contract_checked = True

    def on_before_zero_grad(self, optimizer):
        nonfinite = [
            name
            for name, parameter in self.encoder.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if nonfinite:
            self._fail_nonfinite(
                "nonfinite_parameter_after_optimizer_step",
                {"parameters": nonfinite},
            )

    @staticmethod
    def _batch_values(batch, key):
        value = batch.get(key, [])
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    def _diagnostic_path(self):
        loggers = getattr(self.trainer, "loggers", [])
        if loggers and getattr(loggers[0], "log_dir", None):
            return Path(loggers[0].log_dir) / "paper_logs" / "nonfinite_failure.json"
        return Path(self.trainer.default_root_dir) / "nonfinite_failure.json"

    def _fail_nonfinite(self, stage, details):
        payload = {
            "stage": stage,
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            **self._active_batch_context,
            **details,
        }
        if self.trainer.is_global_zero:
            output = self._diagnostic_path()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        raise FloatingPointError(
            f"Physical V2 encountered {stage} at epoch={self.current_epoch}, "
            f"global_step={self.global_step}: {details}"
        )

    def _check_finite_training_state(
        self, batch, batch_idx, total, components, output0, output1
    ):
        nonfinite_losses = [
            name for name, value in {"total": total, **components}.items()
            if not torch.isfinite(value).all()
        ]
        if not nonfinite_losses:
            return
        nonfinite_outputs = []
        for side, output in (("image0", output0), ("image1", output1)):
            for name in (
                "physical",
                "delta",
                "enhanced",
                "orientation",
                "reliability",
                "scale_weights",
            ):
                if not torch.isfinite(output[name]).all():
                    nonfinite_outputs.append(f"{side}/{name}")
            for index, unary in enumerate(output["unary"]):
                if not torch.isfinite(unary).all():
                    nonfinite_outputs.append(f"{side}/unary_{index}")
        self._fail_nonfinite(
            "nonfinite_forward_or_loss",
            {
                "batch_idx": int(batch_idx),
                "losses": nonfinite_losses,
                "outputs": nonfinite_outputs,
                "remote_ids": self._batch_values(batch, "remote_id"),
                "variants": self._batch_values(batch, "remote_aug_variant"),
            },
        )

    def on_validation_epoch_start(self):
        self.frozen_slim.eval()
        self.validation_accumulator.clear()
        data_module = self.trainer.datamodule
        self.validation_log_prefix = (
            "full_val"
            if data_module is not None
            and getattr(data_module, "full_validation_enabled", False)
            else "val"
        )

    def validation_step(self, batch, batch_idx):
        base0, base1, output0, output1, correspondences = self._forward_pair(batch)
        total, components, diagnostics = self._losses(
            batch,
            base0,
            base1,
            output0,
            output1,
            correspondences,
            deterministic=True,
        )
        batch_size = int(batch["image0"].shape[0])
        self.validation_accumulator["loss/total"] += float(total.detach()) * batch_size
        self.validation_accumulator["loss/count"] += batch_size
        for name, value in components.items():
            self.validation_accumulator[f"loss/{name}"] += float(value.detach()) * batch_size
        for name, value in diagnostics.items():
            self.validation_accumulator[f"diag/{name}"] += float(value.detach()) * batch_size
        stats = descriptor_statistics(
            output0["enhanced"],
            output1["enhanced"],
            correspondences,
            self.frozen_slim.temperature,
            variants=batch.get("remote_aug_variant"),
            chunk_size=self.chunk_size,
            scale_by_sqrt_dim=True,
        )
        for variant, values in stats.items():
            for name, value in values.items():
                self.validation_accumulator[f"enhanced/{variant}/{name}"] += value
        for output in (output0, output1):
            weights = output["scale_weights"].float()
            entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
            self.validation_accumulator["diag/scale_entropy"] += float(
                entropy.mean() / math.log(3.0)
            ) * batch_size
            self.validation_accumulator["diag/odd_fraction"] += float(
                output["oe_selector"].float().mean()
            ) * batch_size
            self.validation_accumulator["diag/routing_count"] += batch_size

    def _distributed_sum(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.cpu())

    def on_validation_epoch_end(self):
        values = {
            key: self._distributed_sum(value)
            for key, value in self.validation_accumulator.items()
        }
        prefix = self.validation_log_prefix
        loss_count = max(values.get("loss/count", 0.0), 1.0)
        for name in ("total", "recover", "keep", "physical", "unary", "orientation"):
            self.log(
                f"{prefix}/loss_{name}",
                values.get(f"loss/{name}", 0.0) / loss_count,
                sync_dist=True,
            )
        for name in (
            "base_positive_confidence",
            "enhanced_positive_confidence",
            "recovery_weight",
        ):
            self.log(
                f"{prefix}/{name}",
                values.get(f"diag/{name}", 0.0) / loss_count,
                sync_dist=True,
            )
        routing_count = max(values.get("diag/routing_count", 0.0), 1.0)
        self.log(
            f"{prefix}/scale_weight_entropy",
            values.get("diag/scale_entropy", 0.0) / routing_count,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/odd_selection_fraction",
            values.get("diag/odd_fraction", 0.0) / routing_count,
            sync_dist=True,
        )
        for variant in ("all", "translation", "scale", "yaw", "pitch", "roll"):
            base = f"enhanced/{variant}/"
            count = values.get(base + "count", 0.0)
            if count <= 0:
                continue
            for name, source in (
                ("r0", "correct0"),
                ("r1", "correct1"),
                ("positive_similarity", "positive"),
                ("hard_negative_similarity", "hard_negative"),
                ("margin", "margin"),
                ("entropy", "entropy"),
                ("normalized_entropy", "normalized_entropy"),
            ):
                self.log(
                    f"{prefix}/enhanced_{variant}_{name}",
                    values[base + source] / count,
                    prog_bar=variant == "all" and name in {"r0", "r1"},
                    sync_dist=True,
                )
                if variant == "all" and name == "r0":
                    self.log(
                        f"{prefix}/enhanced_r0",
                        values[base + source] / count,
                        sync_dist=True,
                    )


__all__ = ["PhysicalV2Module"]
