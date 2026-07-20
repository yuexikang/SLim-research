import time
from collections import defaultdict

import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .matching import PPMatchingLoss, coarse_homography_correspondences
from .metrics import descriptor_statistics
from .models import build_physical_v0_encoder, count_trainable_parameters


class PhysicalV0Module(pl.LightningModule):
    def __init__(
        self,
        model_name="physical_full",
        similarity_mode="full",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=20,
        coarse_scale=8,
        chunk_size=256,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.similarity_mode = similarity_mode
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs_config = int(max_epochs)
        self.coarse_scale = int(coarse_scale)
        self.encoder = build_physical_v0_encoder(model_name)
        self.loss_fn = PPMatchingLoss(chunk_size=chunk_size)
        self.parameter_count = count_trainable_parameters(self.encoder)
        self.validation_accumulator = defaultdict(float)
        self.train_epoch_started = None
        self.train_samples = 0

    def forward(self, image):
        return self.encoder(image)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs_config,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _descriptors_and_correspondences(self, batch):
        descriptor0 = self.encoder(batch["image0"])
        descriptor1 = self.encoder(batch["image1"])
        correspondences = coarse_homography_correspondences(batch, self.coarse_scale)
        return descriptor0, descriptor1, correspondences

    def training_step(self, batch, batch_idx):
        descriptor0, descriptor1, correspondences = self._descriptors_and_correspondences(batch)
        loss = self.loss_fn(
            descriptor0,
            descriptor1,
            correspondences,
            mode=self.similarity_mode,
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
            sync_dist=True,
        )
        self.log(
            "train/temperature",
            self.loss_fn.temperature.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        self.train_epoch_started = time.perf_counter()
        self.train_samples = 0
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.set_epoch(self.current_epoch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_end(self):
        elapsed = max(time.perf_counter() - self.train_epoch_started, 1e-6)
        global_throughput = self.train_samples * self.trainer.world_size / elapsed
        self.log("train/samples_per_second", global_throughput, sync_dist=True)
        if self.device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(self.device) / 1024**3
            self.log("train/peak_memory_gib", peak_gib, sync_dist=True)
        self.log("model/trainable_parameters", float(self.parameter_count), sync_dist=True)

    def on_validation_epoch_start(self):
        self.validation_accumulator.clear()

    def validation_step(self, batch, batch_idx):
        descriptor0, descriptor1, correspondences = self._descriptors_and_correspondences(batch)
        positive_count = correspondences[0].numel()
        selected_count = max(1, int(positive_count * self.loss_fn.positive_percent))
        selected = torch.arange(selected_count, device=self.device)
        loss = self.loss_fn(
            descriptor0,
            descriptor1,
            correspondences,
            mode=self.similarity_mode,
            selected=selected,
        )
        stats = descriptor_statistics(
            descriptor0,
            descriptor1,
            correspondences,
            self.loss_fn.temperature.detach(),
            variants=batch.get("remote_aug_variant"),
        )
        batch_size = int(batch["image0"].shape[0])
        self.validation_accumulator["loss_sum"] += float(loss.detach()) * batch_size
        self.validation_accumulator["loss_count"] += batch_size
        for variant, values in stats.items():
            for name, value in values.items():
                self.validation_accumulator[f"stats/{variant}/{name}"] += value

    def _distributed_sum(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.cpu())

    def on_validation_epoch_end(self):
        reduced = {
            key: self._distributed_sum(value)
            for key, value in self.validation_accumulator.items()
        }
        loss = reduced.get("loss_sum", 0.0) / max(reduced.get("loss_count", 0.0), 1.0)
        self.log("val/loss_pp", loss, sync_dist=True)
        variants = ["all", "translation", "scale", "yaw", "pitch", "roll"]
        for variant in variants:
            prefix = f"stats/{variant}/"
            count = reduced.get(prefix + "count", 0.0)
            if count <= 0:
                continue
            values = {
                "r0": reduced[prefix + "correct0"] / count,
                "r1": reduced[prefix + "correct1"] / count,
                "positive_similarity": reduced[prefix + "positive"] / count,
                "hard_negative_similarity": reduced[prefix + "hard_negative"] / count,
                "margin": reduced[prefix + "margin"] / count,
                "entropy": reduced[prefix + "entropy"] / count,
                "normalized_entropy": reduced[prefix + "normalized_entropy"] / count,
            }
            for name, value in values.items():
                self.log(f"val/{variant}_{name}", value, sync_dist=True)
            if variant == "all":
                self.log("val_r1", values["r1"], prog_bar=True, sync_dist=True)
