"""Lightning module shared by the Pointwise-P5 and RectConv-P5 controls."""

from __future__ import annotations

import time
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .matching import PPMatchingLoss, coarse_homography_correspondences
from .metrics import descriptor_statistics
from .v3_p5_models import IMPLEMENTATION_VERSION, build_physical_v3_p5_encoder


class PhysicalV3P5Module(pl.LightningModule):
    def __init__(
        self,
        model_name,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=3,
        coarse_scale=8,
        chunk_size=256,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["implementation_version"] = IMPLEMENTATION_VERSION
        self.encoder = build_physical_v3_p5_encoder(
            model_name,
            coarse_scale=coarse_scale,
        )
        self.loss_fn = PPMatchingLoss(chunk_size=chunk_size, stable_log=True)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs_config = int(max_epochs)
        self.coarse_scale = int(coarse_scale)
        self.validation_accumulator = defaultdict(float)
        self.train_started = None
        self.train_samples = 0

    def forward(self, image):
        return self.encoder(image)

    def configure_optimizers(self):
        optimizer = AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs_config,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def descriptors_and_correspondences(self, batch):
        descriptor0 = self.encoder(batch["image0"])
        descriptor1 = self.encoder(batch["image1"])
        correspondences = coarse_homography_correspondences(
            batch,
            self.coarse_scale,
        )
        return descriptor0, descriptor1, correspondences

    def training_step(self, batch, batch_idx):
        descriptor0, descriptor1, correspondences = (
            self.descriptors_and_correspondences(batch)
        )
        loss = self.loss_fn(
            descriptor0,
            descriptor1,
            correspondences,
            mode="chunked",
        )
        batch_size = int(batch["image0"].shape[0])
        self.train_samples += batch_size
        self.log(
            "train/loss_pp",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_start(self):
        self.train_started = time.perf_counter()
        self.train_samples = 0
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.set_epoch(self.current_epoch)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_end(self):
        elapsed = max(time.perf_counter() - self.train_started, 1e-6)
        self.log("train/samples_per_second", self.train_samples / elapsed)
        self.log(
            "model/trainable_parameters",
            float(self.encoder.trainable_parameters),
        )
        if self.device.type == "cuda":
            self.log(
                "train/peak_memory_gib",
                torch.cuda.max_memory_allocated(self.device) / 1024**3,
            )

    def on_validation_epoch_start(self):
        self.validation_accumulator.clear()

    def validation_step(self, batch, batch_idx):
        descriptor0, descriptor1, correspondences = (
            self.descriptors_and_correspondences(batch)
        )
        positive_count = correspondences[0].numel()
        selected_count = max(1, int(positive_count * self.loss_fn.positive_percent))
        selected = torch.arange(selected_count, device=self.device)
        loss = self.loss_fn(
            descriptor0,
            descriptor1,
            correspondences,
            mode="chunked",
            selected=selected,
        )
        statistics = descriptor_statistics(
            descriptor0,
            descriptor1,
            correspondences,
            self.loss_fn.temperature.detach(),
            variants=batch.get("remote_aug_variant"),
        )
        batch_size = int(batch["image0"].shape[0])
        self.validation_accumulator["loss_sum"] += float(loss.detach()) * batch_size
        self.validation_accumulator["loss_count"] += batch_size
        for variant, values in statistics.items():
            for name, value in values.items():
                self.validation_accumulator[f"stats/{variant}/{name}"] += value

    def distributed_sum(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.cpu())

    def on_validation_epoch_end(self):
        reduced = {
            key: self.distributed_sum(value)
            for key, value in self.validation_accumulator.items()
        }
        loss = reduced.get("loss_sum", 0.0) / max(
            reduced.get("loss_count", 0.0),
            1.0,
        )
        self.log("val/loss_pp", loss)
        for variant in ("all", "translation", "scale", "yaw", "pitch", "roll"):
            prefix = f"stats/{variant}/"
            count = reduced.get(prefix + "count", 0.0)
            if count <= 0:
                continue
            values = {
                "r0": reduced[prefix + "correct0"] / count,
                "r1": reduced[prefix + "correct1"] / count,
                "positive_similarity": reduced[prefix + "positive"] / count,
                "hard_negative_similarity": reduced[prefix + "hard_negative"]
                / count,
                "margin": reduced[prefix + "margin"] / count,
            }
            for name, value in values.items():
                self.log(f"val/{variant}_{name}", value)
            if variant == "all":
                self.log("val_r0", values["r0"], prog_bar=True)


__all__ = ["PhysicalV3P5Module"]
