"""Fixed diagnostics used by the Physical V3 no-training ablation suite."""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def select_gpu_with_most_free_memory() -> Tuple[int, Dict[str, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    candidates = []
    for line in output.strip().splitlines():
        index, free, total, utilization = [
            int(value.strip()) for value in line.split(",")
        ]
        candidates.append(
            {
                "index": index,
                "memory_free_mib": free,
                "memory_total_mib": total,
                "utilization_percent": utilization,
            }
        )
    if not candidates:
        raise RuntimeError("nvidia-smi returned no visible GPUs")
    selected = max(
        candidates,
        key=lambda row: (row["memory_free_mib"], -row["utilization_percent"]),
    )
    return int(selected["index"]), selected


def physical_branch_arrays(
    state: Dict[str, np.ndarray],
    branch: str,
) -> Tuple[np.ndarray, np.ndarray]:
    odd = np.maximum(state["magnitude_odd"], 0.0)
    even = np.maximum(state["magnitude_even"], 0.0)
    orientation_odd = np.mod(state["orientation_odd"], math.pi)
    orientation_even = np.mod(state["orientation_even"], math.pi)

    if branch == "pvalid":
        return np.maximum(state["weight"], 0.0), np.mod(
            state["orientation"], math.pi
        )
    if branch == "odd":
        return np.power(odd, 0.25), orientation_odd
    if branch == "even":
        return np.power(even, 0.25), orientation_even
    if branch == "hard_magnitude":
        choose_odd = odd >= even
        magnitude = np.where(choose_odd, odd, even)
        orientation = np.where(choose_odd, orientation_odd, orientation_even)
        return np.power(magnitude, 0.25), orientation
    if branch == "soft_magnitude":
        x = odd * np.cos(2.0 * orientation_odd)
        x += even * np.cos(2.0 * orientation_even)
        y = odd * np.sin(2.0 * orientation_odd)
        y += even * np.sin(2.0 * orientation_even)
        orientation = np.mod(0.5 * np.arctan2(y, x), math.pi)
        return np.power(odd + even, 0.25), orientation
    raise ValueError(f"Unknown physical branch: {branch}")


def project_points(points: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    projected = homogeneous @ matrix.T
    valid = np.abs(projected[:, 2]) > 1e-8
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    output[valid] = projected[valid, :2] / projected[valid, 2:3]
    valid &= np.isfinite(output).all(axis=1)
    return output, valid


def nearest_point_errors(
    projected: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if not len(projected) or not len(targets):
        return np.full((len(projected),), np.inf, dtype=np.float64)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.match(
        projected.astype(np.float32),
        targets.astype(np.float32),
    )
    errors = np.full((len(projected),), np.inf, dtype=np.float64)
    for match in matches:
        errors[match.queryIdx] = match.distance
    return errors


def _raw_matches(
    descriptors0: np.ndarray,
    descriptors1: np.ndarray,
):
    if not len(descriptors0) or not len(descriptors1):
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return matcher.match(
        descriptors0.astype(np.float32),
        descriptors1.astype(np.float32),
    )


def unique_target_matches(
    descriptors0: np.ndarray,
    descriptors1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    raw = _raw_matches(descriptors0, descriptors1)
    best_by_target = {}
    target_counts = Counter()
    for match in raw:
        target_counts[match.trainIdx] += 1
        previous = best_by_target.get(match.trainIdx)
        if previous is None or match.distance < previous.distance:
            best_by_target[match.trainIdx] = match
    retained = sorted(best_by_target.values(), key=lambda match: match.queryIdx)
    source = np.asarray([match.queryIdx for match in retained], dtype=np.int64)
    target = np.asarray([match.trainIdx for match in retained], dtype=np.int64)
    distance = np.asarray([match.distance for match in retained], dtype=np.float32)
    diagnostics = {
        "raw_matches": float(len(raw)),
        "unique_target_ratio": (
            float(len(target_counts) / len(raw)) if raw else 0.0
        ),
        "max_target_fan_in": (
            float(max(target_counts.values())) if target_counts else 0.0
        ),
    }
    return source, target, distance, diagnostics


def descriptor_oracle_diagnostics(
    points0: np.ndarray,
    descriptors0: np.ndarray,
    points1: np.ndarray,
    descriptors1: np.ndarray,
    matrix: np.ndarray,
    correct_threshold: float,
    top_k: int = 10,
) -> Dict[str, float]:
    empty = {
        "oracle_eligible": 0.0,
        "oracle_r1": 0.0,
        "oracle_r5": 0.0,
        "oracle_r10": 0.0,
        "oracle_mrr10": 0.0,
        "positive_distance": 0.0,
        "hard_negative_distance_top10": 0.0,
        "distance_margin_top10": 0.0,
    }
    if not len(points0) or not len(points1):
        return empty

    projected, projection_valid = project_points(points0, matrix)
    squared = (
        projected[:, None, 0] - points1[None, :, 0]
    ) ** 2 + (
        projected[:, None, 1] - points1[None, :, 1]
    ) ** 2
    spatial_positive = squared <= correct_threshold**2
    spatial_positive[~projection_valid] = False
    eligible = spatial_positive.any(axis=1)
    eligible_indices = np.flatnonzero(eligible)
    if not len(eligible_indices):
        return empty

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    maximum_k = min(top_k, len(descriptors1))
    nearest = matcher.knnMatch(
        descriptors0.astype(np.float32),
        descriptors1.astype(np.float32),
        k=maximum_k,
    )
    ranks = []
    reciprocal_ranks = []
    positive_distances = []
    negative_distances = []
    for query_index in eligible_indices:
        candidates = nearest[query_index]
        rank = top_k + 1
        first_negative = None
        for candidate_rank, match in enumerate(candidates, start=1):
            if spatial_positive[query_index, match.trainIdx]:
                rank = min(rank, candidate_rank)
            elif first_negative is None:
                first_negative = float(match.distance)

        positive_indices = np.flatnonzero(spatial_positive[query_index])
        differences = descriptors1[positive_indices] - descriptors0[query_index]
        positive_distance = float(
            np.sqrt(np.square(differences, dtype=np.float64).sum(axis=1)).min()
        )
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank if rank <= top_k else 0.0)
        positive_distances.append(positive_distance)
        if first_negative is not None:
            negative_distances.append(first_negative)

    ranks_array = np.asarray(ranks)
    positive_mean = float(np.mean(positive_distances))
    negative_mean = (
        float(np.mean(negative_distances)) if negative_distances else 0.0
    )
    return {
        "oracle_eligible": float(len(eligible_indices)),
        "oracle_r1": float(np.mean(ranks_array <= 1)),
        "oracle_r5": float(np.mean(ranks_array <= 5)),
        "oracle_r10": float(np.mean(ranks_array <= 10)),
        "oracle_mrr10": float(np.mean(reciprocal_ranks)),
        "positive_distance": positive_mean,
        "hard_negative_distance_top10": negative_mean,
        "distance_margin_top10": negative_mean - positive_mean,
    }


def sample_bilinear(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0,), dtype=np.float32)
    map_x = points[:, 0].astype(np.float32)[None]
    map_y = points[:, 1].astype(np.float32)[None]
    sampled = cv2.remap(
        field.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    return sampled[0]


def safe_correlation(values0: Iterable[float], values1: Iterable[float]) -> float:
    values0 = np.asarray(list(values0), dtype=np.float64)
    values1 = np.asarray(list(values1), dtype=np.float64)
    valid = np.isfinite(values0) & np.isfinite(values1)
    values0 = values0[valid]
    values1 = values1[valid]
    if len(values0) < 2 or values0.std() < 1e-12 or values1.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(values0, values1)[0, 1])


def physical_state_diagnostics(
    state0: Dict[str, np.ndarray],
    state1: Dict[str, np.ndarray],
    points0: np.ndarray,
    matrix_model: np.ndarray,
) -> Dict[str, float]:
    projected, valid = project_points(points0, matrix_model)
    height, width = state1["orientation"].shape
    valid &= projected[:, 0] >= 0
    valid &= projected[:, 0] <= width - 1
    valid &= projected[:, 1] >= 0
    valid &= projected[:, 1] <= height - 1
    source = points0[valid]
    target = projected[valid]
    if not len(source):
        return {
            "physical_samples": 0.0,
            "orientation_median_error_deg": 90.0,
            "orientation_mean_error_deg": 90.0,
            "odd_correlation": 0.0,
            "even_correlation": 0.0,
            "roe_correlation": 0.0,
            "vimo_correlation": 0.0,
            "vimo_agreement": 0.0,
            "vimo_iou": 0.0,
        }

    orientation0 = sample_bilinear(state0["orientation"], source)
    orientation1 = sample_bilinear(state1["orientation"], target)
    delta = np.arctan2(
        np.sin(2.0 * (orientation0 - orientation1)),
        np.cos(2.0 * (orientation0 - orientation1)),
    )
    angular_error = np.abs(delta) * 90.0 / math.pi

    odd0 = np.log1p(sample_bilinear(state0["magnitude_odd"], source))
    odd1 = np.log1p(sample_bilinear(state1["magnitude_odd"], target))
    even0 = np.log1p(sample_bilinear(state0["magnitude_even"], source))
    even1 = np.log1p(sample_bilinear(state1["magnitude_even"], target))
    roe0 = np.abs(odd0 - even0) / (odd0 + even0 + 1e-6)
    roe1 = np.abs(odd1 - even1) / (odd1 + even1 + 1e-6)
    vimo0 = sample_bilinear(state0["weight"], source)
    vimo1 = sample_bilinear(state1["weight"], target)
    vimo_binary0 = vimo0 >= 0.5
    vimo_binary1 = vimo1 >= 0.5
    union = np.logical_or(vimo_binary0, vimo_binary1).sum()
    return {
        "physical_samples": float(len(source)),
        "orientation_median_error_deg": float(np.nanmedian(angular_error)),
        "orientation_mean_error_deg": float(np.nanmean(angular_error)),
        "odd_correlation": safe_correlation(odd0, odd1),
        "even_correlation": safe_correlation(even0, even1),
        "roe_correlation": safe_correlation(roe0, roe1),
        "vimo_correlation": safe_correlation(vimo0, vimo1),
        "vimo_agreement": float(np.mean(vimo_binary0 == vimo_binary1)),
        "vimo_iou": (
            float(np.logical_and(vimo_binary0, vimo_binary1).sum() / union)
            if union
            else 1.0
        ),
    }
