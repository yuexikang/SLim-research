import json
import math
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LGPhotometricAugmentation:
    """Deterministic per-sample low-light and degradation augmentation."""

    MAX_ATTEMPTS = 4
    MIN_MEAN_INTENSITY = 18.0
    MIN_RELATIVE_MEAN = 0.4
    MAX_NEAR_BLACK_FRACTION = 0.45
    NEAR_BLACK_THRESHOLD = 5

    def __init__(self):
        self.transform = A.Compose(
            [
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
                A.OneOf(
                    [
                        A.Blur(blur_limit=(3, 9)),
                        A.MotionBlur(blur_limit=(3, 25)),
                        A.ISONoise(),
                        A.ImageCompression(),
                    ],
                    p=0.1,
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.1, blur_limit=(3, 25)),
                A.RandomBrightnessContrast(
                    p=0.5,
                    brightness_limit=(-0.4, 0.0),
                    contrast_limit=(-0.3, 0.0),
                ),
                A.CLAHE(p=0.2),
            ],
            p=0.95,
            seed=0,
        )

    def __call__(self, image, seed):
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for attempt in range(self.MAX_ATTEMPTS):
            attempt_seed = (int(seed) + attempt * 104729) % (2**32)
            self.transform.set_random_seed(attempt_seed)
            augmented = self.transform(image=rgb)["image"]
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            if self._has_usable_brightness(image, augmented):
                return augmented
        return image.copy()

    @classmethod
    def _has_usable_brightness(cls, source, augmented):
        source_mean = float(np.mean(source, dtype=np.float64))
        augmented_mean = float(np.mean(augmented, dtype=np.float64))
        required_mean = max(
            cls.MIN_MEAN_INTENSITY,
            cls.MIN_RELATIVE_MEAN * source_mean,
        )
        near_black_fraction = float(
            np.mean(augmented <= cls.NEAR_BLACK_THRESHOLD)
        )
        return (
            augmented_mean >= required_mean
            and near_black_fraction <= cls.MAX_NEAR_BLACK_FRACTION
        )


class RemoteSensingHomographyDataset(Dataset):
    """Manifest-driven remote-sensing dataset with homography supervision.

    Supported manifest rows:
    - aligned_pairs: image0/image1 are already aligned in their source coordinates.
    - gt_pairs: image0/image1 are paired with a 3x3 homography or 2x3 affine file.
    - single_synth: image is used twice with independent synthetic homographies.

    The returned H_0to1 maps model input pixels from image0 to image1.
    """

    DEFAULT_AUG_VARIANTS = ("translation", "scale", "yaw", "pitch", "roll")

    def __init__(
        self,
        manifest_path,
        image_size=256,
        mode="train",
        max_samples=None,
        homography_difficulty=0.35,
        left_identity=True,
        aug_variants=None,
        manifest_split=None,
        row_indices=None,
        seed=0,
        deterministic_train=False,
        one_variant_per_row=False,
        rotation_limit_degrees=35.0,
        minimum_region_sampler=False,
        photometric_augmentation=None,
        valid_crop_rectification=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = int(image_size)
        self.mode = mode
        self.homography_difficulty = float(homography_difficulty)
        self.left_identity = bool(left_identity)
        self.aug_variants = self._parse_aug_variants(aug_variants)
        self.seed = int(seed)
        self.deterministic_train = bool(deterministic_train)
        self.one_variant_per_row = bool(one_variant_per_row)
        self.rotation_limit_degrees = float(rotation_limit_degrees)
        self.minimum_region_sampler = bool(minimum_region_sampler)
        if not 0.0 <= self.homography_difficulty < 1.0:
            raise ValueError("homography_difficulty must be in [0, 1).")
        if self.rotation_limit_degrees < 0.0:
            raise ValueError("rotation_limit_degrees must be non-negative.")
        self.photometric_augmentation = self._parse_photometric_augmentation(
            photometric_augmentation
        )
        self.valid_crop_rectification = bool(valid_crop_rectification)
        if self.one_variant_per_row and self.mode not in {"train", "val"}:
            raise ValueError(
                "one_variant_per_row is only supported for training or validation datasets."
            )
        self.epoch = 0

        split_filter = mode if manifest_split is None else manifest_split
        rows = self.load_manifest_rows(self.manifest_path, split=split_filter)
        if row_indices is not None:
            rows = [rows[int(i)] for i in row_indices]
        elif max_samples is not None and int(max_samples) > 0:
            rows = rows[: int(max_samples)]
        if not rows:
            raise ValueError(f"No rows for split='{split_filter}' in {self.manifest_path}")
        self.rows = rows

    @staticmethod
    def _parse_photometric_augmentation(profile):
        if profile is None or str(profile).strip().lower() in {"", "none"}:
            return None
        if str(profile).strip().lower() == "lg":
            return LGPhotometricAugmentation()
        raise ValueError(f"Unknown photometric augmentation profile: {profile}")

    def __len__(self):
        if self.one_variant_per_row:
            return len(self.rows)
        return len(self.rows) * len(self.aug_variants)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        if self.one_variant_per_row:
            row_idx = idx
            epoch_offset = self.epoch * 1000003 if self.mode == "train" else 0
            variant_seed = (
                self.seed + epoch_offset + row_idx * 9973 + 7919
            ) % (2**32)
            variant_rng = np.random.default_rng(variant_seed)
            variant_idx = int(variant_rng.integers(0, len(self.aug_variants)))
        else:
            row_idx = idx // len(self.aug_variants)
            variant_idx = idx % len(self.aug_variants)
        aug_variant = self.aug_variants[variant_idx]
        row = self.rows[row_idx]
        rng_idx = idx
        if self.one_variant_per_row and self.mode == "val":
            rng_idx = row_idx * len(self.aug_variants) + variant_idx
        rng = self._rng(rng_idx)
        source_quad0 = None
        source_quad1 = None

        if row["mode"] == "aligned_pairs":
            base0, _ = self._read_square_gray_with_transform(row["image0"])
            base1, _ = self._read_square_gray_with_transform(row["image1"])
            if base0.shape != base1.shape:
                base1 = cv2.resize(base1, (base0.shape[1], base0.shape[0]), interpolation=cv2.INTER_AREA)
            H_base_0to1 = np.eye(3, dtype=np.float32)
            H0 = np.eye(3, dtype=np.float32) if self.left_identity else self._sample_base_to_view_h(rng, "mixed")
            H1 = self._sample_base_to_view_h(rng, aug_variant)
            image0 = cv2.warpPerspective(base0, H0, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
            image1 = cv2.warpPerspective(base1, H1, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
            pair_name = (row["image0"], row["image1"])
            pair_type = row.get("pair_type", "unknown")
        elif row["mode"] == "gt_pairs":
            base0, T0 = self._read_square_gray_with_transform(row["image0"])
            base1, T1 = self._read_square_gray_with_transform(row["image1"])
            H_orig_0to1 = self._read_gt_matrix(row)
            H_base_0to1 = T1 @ H_orig_0to1 @ np.linalg.inv(T0)
            H0 = np.eye(3, dtype=np.float32) if self.left_identity else self._sample_base_to_view_h(rng, "mixed")
            H1 = self._sample_base_to_view_h(rng, aug_variant)
            image0 = cv2.warpPerspective(base0, H0, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
            image1 = cv2.warpPerspective(base1, H1, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
            pair_name = (row["image0"], row["image1"])
            pair_type = row.get("pair_type", "unknown")
        elif row["mode"] == "single_synth":
            if self.valid_crop_rectification:
                base = self._read_original_gray(row["image"])
                image0, image1, H0, H1, source_quad0, source_quad1 = (
                    self._rectify_valid_source_quads(base, rng, aug_variant)
                )
            else:
                base, _ = self._read_square_gray_with_transform(row["image"])
                H0 = (
                    np.eye(3, dtype=np.float32)
                    if self.left_identity
                    else self._sample_base_to_view_h(rng, "mixed")
                )
                H1 = self._sample_base_to_view_h(rng, aug_variant)
                image0 = cv2.warpPerspective(
                    base, H0, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR
                )
                image1 = cv2.warpPerspective(
                    base, H1, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR
                )
            H_base_0to1 = np.eye(3, dtype=np.float32)
            pair_name = (row["image"], row["image"])
            pair_type = "single_synth"
        else:
            raise ValueError(f"Unsupported remote manifest mode: {row['mode']}")

        H_0to1 = H1 @ H_base_0to1 @ np.linalg.inv(H0)
        H_1to0 = np.linalg.inv(H_0to1)
        if self.mode == "train" and self.photometric_augmentation is not None:
            image1 = self.photometric_augmentation(
                image1, seed=self._photometric_seed(rng_idx)
            )

        sample = {
            "image0": self._to_tensor(image0),
            "image1": self._to_tensor(image1),
            "H_0to1": torch.from_numpy(H_0to1.astype(np.float32)),
            "H_1to0": torch.from_numpy(H_1to0.astype(np.float32)),
            "scale0": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "scale1": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "dataset_name": "RemoteSensing",
            "scene_id": row.get("dataset", "remote"),
            "pair_id": idx,
            "pair_names": pair_name,
            "remote_mode": row["mode"],
            "remote_pair_type": pair_type,
            "remote_aug_variant": aug_variant,
            "remote_photometric_profile": (
                "lg" if self.photometric_augmentation is not None else "none"
            ),
            "remote_id": row["id"],
        }
        if source_quad0 is not None:
            sample["remote_source_quad0"] = torch.from_numpy(source_quad0.copy())
            sample["remote_source_quad1"] = torch.from_numpy(source_quad1.copy())
        return sample

    @classmethod
    def _parse_aug_variants(cls, aug_variants):
        if aug_variants is None:
            return list(cls.DEFAULT_AUG_VARIANTS)
        if isinstance(aug_variants, str):
            variants = [v.strip() for v in aug_variants.split(",") if v.strip()]
        else:
            variants = [str(v).strip() for v in aug_variants if str(v).strip()]
        if not variants:
            return list(cls.DEFAULT_AUG_VARIANTS)

        valid = set(cls.DEFAULT_AUG_VARIANTS) | {"mixed"}
        unknown = [v for v in variants if v not in valid]
        if unknown:
            raise ValueError(
                f"Unknown remote homography variants: {unknown}. Valid variants: {sorted(valid)}"
            )
        return variants

    @staticmethod
    def load_manifest_rows(manifest_path, split="all"):
        rows = []
        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_split = row.get("split")
                if (
                    split == "all"
                    or row_split == split
                    or row_split is None
                    or (split == "val" and row_split == "test")
                ):
                    rows.append(row)
        return rows

    def _rng(self, idx):
        if self.mode == "train":
            if self.deterministic_train:
                seed = (self.seed + self.epoch * 1000003 + idx * 9973) % (2**32)
                return np.random.default_rng(seed)
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx * 9973)

    def _photometric_seed(self, idx):
        epoch_offset = self.epoch * 1000003 if self.mode == "train" else 0
        return (self.seed + epoch_offset + int(idx) * 9973 + 424243) % (2**32)

    def _read_square_gray_with_transform(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA if max(image.shape[:2]) > self.image_size else cv2.INTER_LINEAR,
        )
        T_orig_to_square = np.array(
            [
                [self.image_size / float(w), 0.0, 0.0],
                [0.0, self.image_size / float(h), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return image, T_orig_to_square

    @staticmethod
    def _read_original_gray(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return image

    def _rectify_valid_source_quads(self, image, rng, variant):
        height, width = image.shape[:2]
        quad0, quad1 = self._sample_valid_source_quads(
            rng, width=width, height=height, variant=variant
        )
        destination = self._corners()
        H0 = cv2.getPerspectiveTransform(quad0, destination)
        H1 = cv2.getPerspectiveTransform(quad1, destination)
        warp_kwargs = {
            "dsize": (self.image_size, self.image_size),
            "flags": cv2.INTER_LINEAR,
            "borderMode": cv2.BORDER_REFLECT_101,
        }
        image0 = cv2.warpPerspective(image, H0, **warp_kwargs)
        image1 = cv2.warpPerspective(image, H1, **warp_kwargs)
        return (
            image0,
            image1,
            H0.astype(np.float32),
            H1.astype(np.float32),
            quad0,
            quad1,
        )

    def _sample_valid_source_quads(self, rng, width, height, variant):
        minimum_ratio = self.minimum_region_ratio
        border = 2.0
        max_base_ratio = max(minimum_ratio, min(0.72, 1.0 - 0.2 * minimum_ratio))
        for _ in range(256):
            ratio = rng.uniform(minimum_ratio, max_base_ratio)
            half_width = 0.5 * ratio * (width - 1)
            half_height = 0.5 * ratio * (height - 1)
            if half_width + border >= 0.5 * width or half_height + border >= 0.5 * height:
                continue
            center = np.array(
                [
                    rng.uniform(half_width + border, width - 1 - half_width - border),
                    rng.uniform(half_height + border, height - 1 - half_height - border),
                ],
                dtype=np.float32,
            )
            local = np.array(
                [
                    [-half_width, -half_height],
                    [half_width, -half_height],
                    [half_width, half_height],
                    [-half_width, half_height],
                ],
                dtype=np.float32,
            )
            quad0 = local + center
            quad1 = self._transform_source_quad(rng, local, center, variant)
            if self._source_quad_is_valid(quad1, width, height, border):
                return quad0.astype(np.float32), quad1.astype(np.float32)
        raise RuntimeError(
            f"Unable to sample an in-bounds '{variant}' quadrilateral after 256 attempts."
        )

    def _transform_source_quad(self, rng, local, center, variant):
        transformed = local.copy()
        if variant == "translation":
            maximum = self.homography_difficulty * min(abs(local[0, 0]), abs(local[0, 1]))
            transformed += rng.uniform(-maximum, maximum, size=2).astype(np.float32)
        elif variant == "scale":
            scale = math.exp(
                rng.uniform(
                    math.log(self.minimum_region_ratio),
                    -math.log(self.minimum_region_ratio),
                )
            )
            transformed *= scale
        elif variant == "roll":
            angle = math.radians(
                rng.uniform(-self.rotation_limit_degrees, self.rotation_limit_degrees)
            )
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ],
                dtype=np.float32,
            )
            transformed = transformed @ rotation.T
        elif variant in {"yaw", "pitch"}:
            amount = rng.uniform(0.08, 0.5 * self.homography_difficulty)
            sign = -1.0 if rng.random() < 0.5 else 1.0
            if variant == "yaw":
                side = (1, 2) if sign > 0 else (0, 3)
                transformed[side[0], 1] += amount * (2.0 * abs(local[0, 1]))
                transformed[side[1], 1] -= amount * (2.0 * abs(local[0, 1]))
                transformed[list(side), 0] -= sign * amount * abs(local[0, 0]) * 0.35
            else:
                side = (0, 1) if sign > 0 else (3, 2)
                transformed[side[0], 0] += amount * (2.0 * abs(local[0, 0]))
                transformed[side[1], 0] -= amount * (2.0 * abs(local[0, 0]))
                transformed[list(side), 1] += sign * amount * abs(local[0, 1]) * 0.35
        else:
            raise ValueError(f"Unsupported remote homography variant: {variant}")
        return transformed + center

    def _source_quad_is_valid(self, quad, width, height, border):
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
            return False
        if (
            quad[:, 0].min() < border
            or quad[:, 0].max() > width - 1 - border
            or quad[:, 1].min() < border
            or quad[:, 1].max() > height - 1 - border
        ):
            return False
        minimum_area = (
            self.minimum_region_ratio**2 * float(width - 1) * float(height - 1)
        )
        return abs(cv2.contourArea(quad.astype(np.float32))) >= minimum_area

    @staticmethod
    def _read_gt_matrix(row):
        gt_path = Path(row["gt"])
        if gt_path.suffix.lower() == ".npy":
            H = np.load(gt_path)
        else:
            H = np.loadtxt(gt_path)
        H = np.asarray(H, dtype=np.float32)
        if H.shape == (2, 3):
            H = np.vstack([H, np.array([0, 0, 1], dtype=np.float32)])
        if H.shape != (3, 3):
            raise ValueError(f"GT matrix must be 3x3 or 2x3, got {H.shape}: {gt_path}")
        if row.get("gt_direction", "0to1") in {"1to0", "b_to_a"}:
            H = np.linalg.inv(H)
        if abs(float(H[2, 2])) > 1e-8:
            H = H / H[2, 2]
        return H.astype(np.float32)

    def _sample_base_to_view_h(self, rng, variant="mixed"):
        s = self.image_size
        if self.homography_difficulty <= 0:
            return np.eye(3, dtype=np.float32)

        if variant == "translation":
            return self._sample_translation_h(rng)
        if variant == "scale":
            return self._sample_scale_h(rng)
        if variant == "yaw":
            return self._sample_yaw_h(rng)
        if variant == "pitch":
            return self._sample_pitch_h(rng)
        if variant == "roll":
            return self._sample_roll_h(rng)
        if variant != "mixed":
            raise ValueError(f"Unsupported remote homography variant: {variant}")

        if self.minimum_region_sampler:
            max_jitter = max(1.0, 0.5 * (s - self.minimum_region_size))
        else:
            max_jitter = max(1.0, s * 0.18 * self.homography_difficulty)
        corners = np.array(
            [[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]],
            dtype=np.float32,
        )
        src = corners.copy()
        src[0] += rng.uniform(0, max_jitter, 2)
        src[1] += np.array([-rng.uniform(0, max_jitter), rng.uniform(0, max_jitter)], dtype=np.float32)
        src[2] -= rng.uniform(0, max_jitter, 2)
        src[3] += np.array([rng.uniform(0, max_jitter), -rng.uniform(0, max_jitter)], dtype=np.float32)

        rotation_limit = self.rotation_limit_degrees if self.minimum_region_sampler else 12.0
        angle = (
            rng.uniform(-rotation_limit, rotation_limit)
            * (1.0 if self.minimum_region_sampler else self.homography_difficulty)
            * math.pi
            / 180.0
        )
        scale = 1.0 + rng.uniform(-0.08, 0.08) * self.homography_difficulty
        center = np.array([(s - 1) * 0.5, (s - 1) * 0.5], dtype=np.float32)
        rot = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ],
            dtype=np.float32,
        ) * scale
        src = (src - center) @ rot.T + center
        src = np.clip(src, 0, s - 1).astype(np.float32)

        dst = corners
        H = cv2.getPerspectiveTransform(src, dst)
        return H.astype(np.float32)

    def _sample_translation_h(self, rng):
        if self.minimum_region_sampler:
            max_shift = self.image_size - self.minimum_region_size
        else:
            max_shift = self.image_size * 0.22 * self.homography_difficulty
        tx, ty = rng.uniform(-max_shift, max_shift, 2)
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _sample_scale_h(self, rng):
        if self.minimum_region_sampler:
            minimum_ratio = self.minimum_region_ratio
            scale = math.exp(rng.uniform(math.log(minimum_ratio), -math.log(minimum_ratio)))
        else:
            scale = 1.0 + rng.uniform(-0.35, 0.35) * self.homography_difficulty
        return self._centered_affine_h(np.array([[scale, 0.0], [0.0, scale]], dtype=np.float32))

    def _sample_roll_h(self, rng):
        angle = (
            rng.uniform(-self.rotation_limit_degrees, self.rotation_limit_degrees)
            * (1.0 if self.minimum_region_sampler else self.homography_difficulty)
            * math.pi
            / 180.0
        )
        rot = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float32,
        )
        return self._centered_affine_h(rot)

    def _sample_yaw_h(self, rng):
        corners = self._corners()
        amount = self._sample_perspective_amount(rng)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        dst = corners.copy()
        if sign > 0:
            dst[1, 0] -= amount * 0.35
            dst[2, 0] -= amount * 0.35
            dst[1, 1] += amount
            dst[2, 1] -= amount
        else:
            dst[0, 0] += amount * 0.35
            dst[3, 0] += amount * 0.35
            dst[0, 1] += amount
            dst[3, 1] -= amount
        return self._quad_to_h(corners, self._clip_quad(dst))

    def _sample_pitch_h(self, rng):
        corners = self._corners()
        amount = self._sample_perspective_amount(rng)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        dst = corners.copy()
        if sign > 0:
            dst[0, 0] += amount
            dst[1, 0] -= amount
            dst[0, 1] += amount * 0.35
            dst[1, 1] += amount * 0.35
        else:
            dst[3, 0] += amount
            dst[2, 0] -= amount
            dst[3, 1] -= amount * 0.35
            dst[2, 1] -= amount * 0.35
        return self._quad_to_h(corners, self._clip_quad(dst))

    @property
    def minimum_region_ratio(self):
        return max(1e-3, 1.0 - self.homography_difficulty)

    @property
    def minimum_region_size(self):
        return self.image_size * self.minimum_region_ratio

    def _sample_perspective_amount(self, rng):
        if self.minimum_region_sampler:
            maximum = 0.5 * (self.image_size - self.minimum_region_size)
            minimum = min(self.image_size * 0.08, maximum)
            return rng.uniform(minimum, maximum)
        return (
            self.image_size
            * rng.uniform(0.08, 0.28)
            * self.homography_difficulty
        )

    def _centered_affine_h(self, linear):
        center = np.array([(self.image_size - 1) * 0.5, (self.image_size - 1) * 0.5], dtype=np.float32)
        H = np.eye(3, dtype=np.float32)
        H[:2, :2] = linear
        H[:2, 2] = center - linear @ center
        return H

    def _corners(self):
        s = self.image_size
        return np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32)

    def _clip_quad(self, quad):
        return np.clip(quad, 0, self.image_size - 1).astype(np.float32)

    @staticmethod
    def _quad_to_h(src, dst):
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        return H.astype(np.float32)

    @staticmethod
    def _to_tensor(image):
        return torch.from_numpy(image).float()[None] / 255.0
