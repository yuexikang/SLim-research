"""Code-traceable PolarP descriptor and fixed no-training matcher for V3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PolarPConfig:
    patch_size: int = 72
    spatial_bins: int = 12
    orientation_bins: int = 12
    rotation_invariant: bool = True
    multiple_orientations: bool = True
    orientation_peak_ratio: float = 0.8
    max_keypoints: int = 1000
    keypoint_quality: float = 0.01
    keypoint_min_distance: float = 4.0
    keypoint_block_size: int = 7
    descriptor_normalization: str = "none"


def detect_fixed_keypoints(
    structural_image: np.ndarray,
    config: PolarPConfig,
) -> np.ndarray:
    finite = np.nan_to_num(structural_image, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        return np.empty((0, 2), dtype=np.float32)
    display = np.clip((finite - low) / (high - low), 0, 1)
    display = np.round(display * 255).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(
        display,
        maxCorners=config.max_keypoints,
        qualityLevel=config.keypoint_quality,
        minDistance=config.keypoint_min_distance,
        blockSize=config.keypoint_block_size,
        useHarrisDetector=False,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    points = points[:, 0].astype(np.float32)
    radius = config.patch_size // 2
    height, width = display.shape
    valid = (
        (points[:, 0] >= radius)
        & (points[:, 0] < width - radius)
        & (points[:, 1] >= radius)
        & (points[:, 1] < height - radius)
    )
    return points[valid]


def _polar_layout(config: PolarPConfig) -> Dict[str, np.ndarray]:
    radius = config.patch_size // 2
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coordinate, coordinate)
    circle = (xx**2 + yy**2) < (radius + 1) ** 2
    radius0_square = radius**2 / (2 * config.spatial_bins + 1)
    radius1_square = radius0_square * (config.spatial_bins + 1)
    squared = xx**2 + yy**2
    radial = np.full(squared.shape, -1, dtype=np.int16)
    radial[circle & (squared <= radius0_square)] = 0
    radial[circle & (squared > radius0_square) & (squared <= radius1_square)] = 1
    radial[circle & (squared > radius1_square)] = 2
    theta = np.arctan2(yy, xx) + math.pi
    return {
        "radial": radial,
        "theta": theta,
        "circle": circle,
        "radius0_square": np.array(radius0_square),
        "radius1_square": np.array(radius1_square),
    }


def _orientation_histogram(
    magnitude_patch: np.ndarray,
    orientation_patch: np.ndarray,
    circle: np.ndarray,
    bins: int,
) -> np.ndarray:
    orientation_bin = np.floor(orientation_patch * bins / math.pi).astype(np.int64)
    orientation_bin = np.mod(orientation_bin, bins)
    valid_bins = orientation_bin[circle]
    valid_weights = magnitude_patch[circle]
    return np.bincount(valid_bins, weights=valid_weights, minlength=bins).astype(
        np.float64
    )


def _base_directions(
    magnitude_patch: np.ndarray,
    orientation_patch: np.ndarray,
    circle: np.ndarray,
    config: PolarPConfig,
) -> List[float]:
    if not config.rotation_invariant:
        return [0.0]
    histogram = _orientation_histogram(
        magnitude_patch,
        orientation_patch,
        circle,
        config.orientation_bins,
    )
    maximum = float(histogram.max(initial=0.0))
    if maximum <= 0:
        return [0.0]
    threshold = config.orientation_peak_ratio * maximum
    candidates: List[Tuple[float, float]] = []
    for index in range(config.orientation_bins):
        left = histogram[(index - 1) % config.orientation_bins]
        center = histogram[index]
        right = histogram[(index + 1) % config.orientation_bins]
        if center > left and center > right and center > threshold:
            denominator = left + right - 2.0 * center
            offset = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
            interpolated = (index + 1 + offset - 1) % config.orientation_bins + 1
            angle = interpolated * math.pi / config.orientation_bins
            candidates.append((float(center), float(angle)))
    if not candidates:
        index = int(np.argmax(histogram))
        candidates = [(float(histogram[index]), (index + 1) * math.pi / config.orientation_bins)]
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not config.multiple_orientations:
        return [candidates[0][1]]
    return [angle for _, angle in candidates]


def _normalize_descriptor(descriptor: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return descriptor
    if mode == "l2":
        return descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)
    if mode == "root":
        total = max(float(np.abs(descriptor).sum()), 1e-12)
        return np.sqrt(np.clip(descriptor / total, 0, None))
    raise ValueError(f"Unknown descriptor normalization: {mode}")


def _descriptor_for_orientation(
    magnitude_patch: np.ndarray,
    orientation_patch: np.ndarray,
    layout: Dict[str, np.ndarray],
    base_direction: float,
    config: PolarPConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    radial = layout["radial"]
    theta = layout["theta"]
    spatial_bin = np.mod(
        np.floor((theta - base_direction) * config.spatial_bins / (2 * math.pi)),
        config.spatial_bins,
    ).astype(np.int64)
    orientation_bin = np.mod(
        np.floor(
            (orientation_patch - base_direction)
            * config.orientation_bins
            / math.pi
        ),
        config.orientation_bins,
    ).astype(np.int64)

    center = np.zeros(config.orientation_bins, dtype=np.float64)
    outer = np.zeros(
        (config.orientation_bins, config.spatial_bins, 3), dtype=np.float64
    )
    center_mask = radial == 0
    center += np.bincount(
        orientation_bin[center_mask],
        weights=magnitude_patch[center_mask],
        minlength=config.orientation_bins,
    )
    for ring in (1, 2):
        mask = radial == ring
        combined = (
            orientation_bin[mask] * config.spatial_bins + spatial_bin[mask]
        )
        histogram = np.bincount(
            combined,
            weights=magnitude_patch[mask],
            minlength=config.orientation_bins * config.spatial_bins,
        )
        outer[:, :, ring - 1] = histogram.reshape(
            config.orientation_bins, config.spatial_bins
        )

    next_sector = np.roll(np.arange(config.spatial_bins), -1)
    weight1 = math.sqrt(
        float(layout["radius0_square"]) / float(layout["radius1_square"])
    )
    weight2 = math.sqrt(float(layout["radius0_square"])) / (config.patch_size // 2)
    outer[:, :, 2] = (
        (outer[:, :, 0] + outer[:, next_sector, 0]) * weight1
        + (outer[:, :, 1] + outer[:, next_sector, 1]) * weight2
    ) / 2.0 + center[:, None] / 6.0
    all_feature = outer[:, :, 2].mean(axis=1)
    skip = np.concatenate(
        [
            (
                outer[:, 0::2, 0] * weight1
                + outer[:, 1::2, 1] * weight2
            ).mean(axis=1),
            (
                outer[:, 0::2, 1] * weight2
                + outer[:, 1::2, 0] * weight1
            ).mean(axis=1),
        ]
    ) / 2.0

    psd_swapped = False
    if config.rotation_invariant:
        half = config.spatial_bins // 2
        first = outer[:, :half, :]
        second = outer[:, half:, :]
        comparison_count = int(
            (np.var(first, axis=0, ddof=1) - np.var(second, axis=0, ddof=1) >= 0).sum()
        )
        if comparison_count > 3 * config.spatial_bins / 4:
            outer = np.concatenate([second, first], axis=1)
            psd_swapped = True

    descriptor = np.concatenate(
        [
            center,
            outer.reshape(-1, order="F"),
            skip,
            all_feature,
        ]
    ).astype(np.float32)
    descriptor = _normalize_descriptor(descriptor, config.descriptor_normalization)
    diagnostics = {
        "base_direction": float(base_direction),
        "psd_swapped": float(psd_swapped),
    }
    return descriptor.astype(np.float32), diagnostics


def describe_keypoints(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    keypoints: np.ndarray,
    config: PolarPConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    config = config or PolarPConfig()
    layout = _polar_layout(config)
    radius = config.patch_size // 2
    descriptors = []
    output_points = []
    diagnostics: List[Dict[str, float]] = []
    for x_float, y_float in keypoints:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        magnitude_patch = magnitude[y - radius : y + radius + 1, x - radius : x + radius + 1]
        orientation_patch = orientation[y - radius : y + radius + 1, x - radius : x + radius + 1]
        if magnitude_patch.shape != layout["circle"].shape:
            continue
        directions = _base_directions(
            magnitude_patch,
            orientation_patch,
            layout["circle"],
            config,
        )
        for base_direction in directions:
            descriptor, diagnostic = _descriptor_for_orientation(
                magnitude_patch,
                orientation_patch,
                layout,
                base_direction,
                config,
            )
            descriptors.append(descriptor)
            output_points.append([x_float, y_float])
            diagnostics.append(diagnostic)
    if not descriptors:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 480), dtype=np.float32),
            [],
        )
    return (
        np.asarray(output_points, dtype=np.float32),
        np.stack(descriptors).astype(np.float32),
        diagnostics,
    )


def match_code_exact_descriptors(
    descriptor0: np.ndarray,
    descriptor1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbor matching with one retained source for each target."""
    if not len(descriptor0) or not len(descriptor1):
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, np.empty((0,), dtype=np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = matcher.match(descriptor0.astype(np.float32), descriptor1.astype(np.float32))
    best_by_target = {}
    for match in raw:
        previous = best_by_target.get(match.trainIdx)
        if previous is None or match.distance < previous.distance:
            best_by_target[match.trainIdx] = match
    matches = sorted(best_by_target.values(), key=lambda match: match.queryIdx)
    source = np.asarray([match.queryIdx for match in matches], dtype=np.int64)
    target = np.asarray([match.trainIdx for match in matches], dtype=np.int64)
    distance = np.asarray([match.distance for match in matches], dtype=np.float32)
    return source, target, distance
