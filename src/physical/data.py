import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.datasets.remote_sensing import RemoteSensingHomographyDataset


def stratified_manifest_indices(rows, ratio, seed, max_rows=0):
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"train_data_ratio must be in (0, 1], got {ratio}.")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.get("dataset", "unknown"), row.get("subset", "unknown"))].append(index)

    target = int(math.floor(len(rows) * ratio + 0.5))
    if max_rows and int(max_rows) > 0:
        target = min(target, int(max_rows))
    target = max(1, min(target, len(rows)))

    exact = {key: len(indices) * target / len(rows) for key, indices in groups.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1

    rng = np.random.default_rng(int(seed))
    selected = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(indices)
        selected.extend(indices[: quotas[key]].tolist())
    return sorted(selected)


def manifest_indices_from_selected_rows(rows, selected_rows):
    available = defaultdict(list)
    for index, row in enumerate(rows):
        signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
        available[signature].append(index)
    selected_indices = []
    missing = []
    for row in selected_rows:
        signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
        candidates = available.get(signature)
        if not candidates:
            missing.append(row.get("id", signature[:120]))
            continue
        selected_indices.append(candidates.pop(0))
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise ValueError(
            f"{len(missing)} selected rows are absent from the training manifest. "
            f"First missing rows: {preview}"
        )
    return sorted(selected_indices)


class PhysicalV0DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_manifest,
        val_manifest,
        experiment_dir,
        image_size=512,
        batch_size=4,
        val_batch_size=1,
        num_workers=6,
        train_data_ratio=0.3,
        max_train_rows=0,
        max_val_rows=0,
        homography_difficulty=0.3,
        seed=66,
        selected_train_rows=None,
        train_one_variant_per_row=False,
        val_one_variant_per_row=False,
        rotation_limit_degrees=35.0,
        minimum_region_sampler=False,
        photometric_augmentation=None,
        valid_crop_rectification=False,
    ):
        super().__init__()
        self.train_manifest = Path(train_manifest)
        self.val_manifest = Path(val_manifest)
        self.experiment_dir = Path(experiment_dir)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.train_data_ratio = float(train_data_ratio)
        self.max_train_rows = int(max_train_rows)
        self.max_val_rows = int(max_val_rows)
        self.homography_difficulty = float(homography_difficulty)
        self.seed = int(seed)
        self.selected_train_rows = (
            Path(selected_train_rows) if selected_train_rows is not None else None
        )
        self.train_one_variant_per_row = bool(train_one_variant_per_row)
        self.val_one_variant_per_row = bool(val_one_variant_per_row)
        self.rotation_limit_degrees = float(rotation_limit_degrees)
        self.minimum_region_sampler = bool(minimum_region_sampler)
        self.photometric_augmentation = photometric_augmentation
        self.valid_crop_rectification = bool(valid_crop_rectification)
        self.variants = list(RemoteSensingHomographyDataset.DEFAULT_AUG_VARIANTS)
        self.train_dataset = None
        self.val_dataset = None
        self.full_val_dataset = None
        self.full_validation_enabled = False
        self.selected_indices = None

    def setup(self, stage=None):
        if stage not in (None, "fit") or self.train_dataset is not None:
            return
        rows = RemoteSensingHomographyDataset.load_manifest_rows(
            self.train_manifest, split="train"
        )
        if self.selected_train_rows is not None:
            selected_rows = RemoteSensingHomographyDataset.load_manifest_rows(
                self.selected_train_rows, split="all"
            )
            self.selected_indices = manifest_indices_from_selected_rows(rows, selected_rows)
            if self.max_train_rows > 0:
                self.selected_indices = self.selected_indices[: self.max_train_rows]
        else:
            self.selected_indices = stratified_manifest_indices(
                rows,
                ratio=self.train_data_ratio,
                seed=self.seed,
                max_rows=self.max_train_rows,
            )
        self.train_dataset = RemoteSensingHomographyDataset(
            manifest_path=self.train_manifest,
            image_size=self.image_size,
            mode="train",
            row_indices=self.selected_indices,
            homography_difficulty=self.homography_difficulty,
            left_identity=True,
            aug_variants=self.variants,
            seed=self.seed,
            deterministic_train=True,
            one_variant_per_row=self.train_one_variant_per_row,
            rotation_limit_degrees=self.rotation_limit_degrees,
            minimum_region_sampler=self.minimum_region_sampler,
            photometric_augmentation=self.photometric_augmentation,
            valid_crop_rectification=self.valid_crop_rectification,
        )
        self.val_dataset = RemoteSensingHomographyDataset(
            manifest_path=self.val_manifest,
            image_size=self.image_size,
            mode="val",
            max_samples=self.max_val_rows,
            homography_difficulty=self.homography_difficulty,
            left_identity=True,
            aug_variants=self.variants,
            seed=self.seed,
            one_variant_per_row=self.val_one_variant_per_row,
            rotation_limit_degrees=self.rotation_limit_degrees,
            minimum_region_sampler=self.minimum_region_sampler,
            photometric_augmentation=None,
            valid_crop_rectification=self.valid_crop_rectification,
        )
        if self.val_one_variant_per_row:
            self.full_val_dataset = RemoteSensingHomographyDataset(
                manifest_path=self.val_manifest,
                image_size=self.image_size,
                mode="val",
                max_samples=self.max_val_rows,
                homography_difficulty=self.homography_difficulty,
                left_identity=True,
                aug_variants=self.variants,
                seed=self.seed,
                rotation_limit_degrees=self.rotation_limit_degrees,
                minimum_region_sampler=self.minimum_region_sampler,
                photometric_augmentation=None,
                valid_crop_rectification=self.valid_crop_rectification,
            )
        else:
            self.full_val_dataset = self.val_dataset
        self._save_selected_rows(rows)

    def _save_selected_rows(self, rows):
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        output = self.experiment_dir / "selected_train_rows.jsonl"
        content = "".join(
            json.dumps(rows[index], ensure_ascii=False, sort_keys=True) + "\n"
            for index in self.selected_indices
        )
        if output.exists() and output.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Selected training subset differs from the existing run record: {output}"
            )
        output.write_text(content, encoding="utf-8")

    def set_epoch(self, epoch):
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)

    def enable_full_validation(self):
        if self.full_val_dataset is None:
            raise RuntimeError("Data module must be set up before full validation.")
        self.full_validation_enabled = True

    def disable_full_validation(self):
        self.full_validation_enabled = False

    @property
    def active_val_dataset(self):
        if self.full_validation_enabled:
            return self.full_val_dataset
        return self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.active_val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
