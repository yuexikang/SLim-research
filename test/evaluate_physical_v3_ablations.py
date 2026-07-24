# 作用：执行Physical V3.0.2无需训练的分层消融，同时诊断关键点重复率、
# 描述子GT检索、匹配hubness、HIMO方向及Odd/Even物理状态稳定性。

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physical.v3_ablation import (
    descriptor_oracle_diagnostics,
    nearest_point_errors,
    physical_branch_arrays,
    physical_state_diagnostics,
    project_points,
    select_gpu_with_most_free_memory,
    unique_target_matches,
)
from src.physical.v3_himo import (
    HIMO_IMPLEMENTATION_VERSION,
    HIMO_SOURCE_COMMIT,
    FrozenHIMO,
    HIMOConfig,
)
from src.physical.v3_polarp import (
    PolarPConfig,
    describe_keypoints,
    detect_fixed_keypoints,
)


@dataclass(frozen=True)
class Variant:
    name: str
    normalization: str
    multiple_orientations: bool = True
    rotation_invariant: bool = True
    half_turn_mode: str = "psd"
    branch: str = "pvalid"


VARIANTS = (
    Variant("code_exact_raw_multi_psd", "none"),
    Variant("l2_multi_psd", "l2"),
    Variant("root_multi_psd", "root"),
    Variant("l2_single_psd", "l2", multiple_orientations=False),
    Variant("l2_multi_no_psd", "l2", half_turn_mode="none"),
    Variant("l2_multi_symmetric_halfturn", "l2", half_turn_mode="symmetric"),
    Variant("l2_multi_psd_odd", "l2", branch="odd"),
    Variant("l2_multi_psd_even", "l2", branch="even"),
    Variant("l2_multi_psd_hard_magnitude", "l2", branch="hard_magnitude"),
    Variant("l2_multi_psd_soft_magnitude", "l2", branch="soft_magnitude"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Physical V3.0.2 no-training ablation suite."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument(
        "--manifest_split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' chooses the visible GPU with the most free memory.",
    )
    parser.add_argument("--pairs_per_modality", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_keypoints", type=int, default=1000)
    parser.add_argument("--keypoint_quality", type=float, default=0.01)
    parser.add_argument("--keypoint_min_distance", type=float, default=4.0)
    parser.add_argument("--patch_size", type=int, default=72)
    parser.add_argument("--spatial_bins", type=int, default=12)
    parser.add_argument("--orientation_bins", type=int, default=12)
    parser.add_argument("--feature_cache_size", type=int, default=4)
    parser.add_argument("--correct_thr", type=float, default=5.0)
    parser.add_argument("--success_ncm", type=int, default=20)
    parser.add_argument("--failed_rmse", type=float, default=10.0)
    parser.add_argument("--max_cooccurrence_levels", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=[variant.name for variant in VARIANTS],
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one pair per modality and at most 100 anchors.",
    )
    return parser.parse_args()


def modality_pair(record: Dict) -> str:
    return f"{record.get('modality0', 'unknown')}-{record.get('modality1', 'unknown')}"


def read_manifest_stratified(
    path: Path,
    split: str,
    per_modality: int,
    seed: int,
) -> List[Dict]:
    groups = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and row.get("split") != split:
                continue
            groups[modality_pair(row)].append(row)
    if not groups:
        raise ValueError(f"No rows selected from {path} for split={split}")

    rng = np.random.default_rng(seed)
    selected = []
    for group_name in sorted(groups):
        rows = groups[group_name]
        count = len(rows) if per_modality <= 0 else min(per_modality, len(rows))
        indices = np.sort(rng.choice(len(rows), count, replace=False))
        selected.extend(rows[index] for index in indices)
    return selected


def read_matrix(record: Dict) -> np.ndarray:
    path = Path(record.get("gt") or record.get("matrix_path") or "")
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth matrix: {path}")
    matrix = np.load(path) if path.suffix.lower() == ".npy" else np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix in {path}, got {matrix.shape}")
    direction = str(record.get("gt_direction", "0to1")).lower()
    if direction in {"1to0", "b_to_a", "1_to_0"}:
        matrix = np.linalg.inv(matrix)
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def model_coordinate_matrix(
    matrix: np.ndarray,
    size0: Tuple[int, int],
    size1: Tuple[int, int],
    image_size: int,
) -> np.ndarray:
    width0, height0 = size0
    width1, height1 = size1
    model_to_original0 = np.diag(
        [width0 / image_size, height0 / image_size, 1.0]
    )
    original_to_model1 = np.diag(
        [image_size / width1, image_size / height1, 1.0]
    )
    result = original_to_model1 @ matrix @ model_to_original0
    return result / result[2, 2] if abs(result[2, 2]) > 1e-12 else result


def reprojection_errors(
    points0: np.ndarray,
    points1: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    projected, valid = project_points(points0, matrix)
    errors = np.linalg.norm(projected - points1, axis=1)
    errors[~valid | ~np.isfinite(errors)] = np.inf
    return errors


class FeatureCache:
    def __init__(
        self,
        extractor: FrozenHIMO,
        variants: Iterable[Variant],
        args,
        device: torch.device,
    ):
        self.extractor = extractor
        self.variants = tuple(variants)
        self.args = args
        self.device = device
        self.capacity = max(0, args.feature_cache_size)
        self.entries: OrderedDict[str, Dict] = OrderedDict()

    def get(self, path: str) -> Tuple[Dict, bool]:
        if path in self.entries:
            entry = self.entries.pop(path)
            self.entries[path] = entry
            return entry, True
        entry = self._extract(path)
        if self.capacity:
            self.entries[path] = entry
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return entry, False

    def _extract(self, path: str) -> Dict:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        original_size = (image.shape[1], image.shape[0])
        interpolation = (
            cv2.INTER_AREA
            if max(image.shape) > self.args.image_size
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            image,
            (self.args.image_size, self.args.image_size),
            interpolation=interpolation,
        )
        tensor = torch.from_numpy(resized.astype(np.float32))[None, None].to(
            self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        output = self.extractor(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        extraction_ms = (time.perf_counter() - started) * 1000.0
        names = (
            "magnitude_odd",
            "magnitude_even",
            "orientation_odd",
            "orientation_even",
            "orientation",
            "weight",
        )
        state = {
            name: output[name][0, 0].detach().cpu().numpy() for name in names
        }
        structural = np.maximum(
            state["magnitude_odd"],
            state["magnitude_even"],
        ) ** 0.25
        detector_config = self._polar_config(self.variants[0])
        anchors = detect_fixed_keypoints(structural, detector_config)
        features = {
            variant.name: self._describe(state, anchors, variant)
            for variant in self.variants
        }
        return {
            "state": state,
            "anchors": anchors,
            "features": features,
            "original_size": original_size,
            "extraction_ms": extraction_ms,
        }

    def _polar_config(self, variant: Variant) -> PolarPConfig:
        return PolarPConfig(
            patch_size=self.args.patch_size,
            spatial_bins=self.args.spatial_bins,
            orientation_bins=self.args.orientation_bins,
            rotation_invariant=variant.rotation_invariant,
            multiple_orientations=variant.multiple_orientations,
            max_keypoints=self.args.max_keypoints,
            keypoint_quality=self.args.keypoint_quality,
            keypoint_min_distance=self.args.keypoint_min_distance,
            descriptor_normalization=variant.normalization,
            half_turn_mode=variant.half_turn_mode,
        )

    def _describe(
        self,
        state: Dict[str, np.ndarray],
        anchors: np.ndarray,
        variant: Variant,
    ) -> Dict:
        magnitude, orientation = physical_branch_arrays(state, variant.branch)
        points, descriptors, diagnostics = describe_keypoints(
            magnitude,
            orientation,
            anchors,
            self._polar_config(variant),
        )
        if descriptors.shape[1:] != (480,):
            raise RuntimeError(
                f"{variant.name} must produce 480-D descriptors, got "
                f"{descriptors.shape}"
            )
        if not np.isfinite(descriptors).all():
            raise FloatingPointError(f"{variant.name} produced NaN/Inf")
        return {
            "points": points,
            "descriptors": descriptors,
            "diagnostics": diagnostics,
        }


def scale_points(
    points: np.ndarray,
    original_size: Tuple[int, int],
    image_size: int,
) -> np.ndarray:
    width, height = original_size
    return points.astype(np.float64) * np.array(
        [width / image_size, height / image_size],
        dtype=np.float64,
    )


def detector_diagnostics(
    entry0: Dict,
    entry1: Dict,
    matrix: np.ndarray,
    image_size: int,
) -> Dict[str, float]:
    points0 = scale_points(entry0["anchors"], entry0["original_size"], image_size)
    points1 = scale_points(entry1["anchors"], entry1["original_size"], image_size)
    projected, valid = project_points(points0, matrix)
    width1, height1 = entry1["original_size"]
    valid &= projected[:, 0] >= 0
    valid &= projected[:, 0] < width1
    valid &= projected[:, 1] >= 0
    valid &= projected[:, 1] < height1
    errors = nearest_point_errors(projected[valid], points1)
    return {
        "detector_projected_valid": float(valid.sum()),
        "detector_repeatability_r3": (
            float(np.mean(errors <= 3.0)) if len(errors) else 0.0
        ),
        "detector_repeatability_r5": (
            float(np.mean(errors <= 5.0)) if len(errors) else 0.0
        ),
    }


def evaluate_variant(
    variant: Variant,
    entry0: Dict,
    entry1: Dict,
    matrix: np.ndarray,
    args,
) -> Dict[str, float]:
    feature0 = entry0["features"][variant.name]
    feature1 = entry1["features"][variant.name]
    points0 = scale_points(
        feature0["points"], entry0["original_size"], args.image_size
    )
    points1 = scale_points(
        feature1["points"], entry1["original_size"], args.image_size
    )
    source, target, distances, hubness = unique_target_matches(
        feature0["descriptors"],
        feature1["descriptors"],
    )
    errors = reprojection_errors(points0[source], points1[target], matrix)
    correct = errors <= args.correct_thr
    ncm = int(correct.sum())
    matches = int(len(errors))
    success = ncm >= args.success_ncm
    rmse = (
        float(np.sqrt(np.mean(errors[correct] ** 2)))
        if success and ncm
        else float(args.failed_rmse)
    )
    oracle = descriptor_oracle_diagnostics(
        points0,
        feature0["descriptors"],
        points1,
        feature1["descriptors"],
        matrix,
        args.correct_thr,
    )
    return {
        "keypoints0": float(len(feature0["descriptors"])),
        "keypoints1": float(len(feature1["descriptors"])),
        "matches": float(matches),
        "NCM": float(ncm),
        "Pre": float(ncm / matches) if matches else 0.0,
        "SR": float(success),
        "RMSE": rmse,
        "mean_descriptor_distance": (
            float(distances.mean()) if len(distances) else 0.0
        ),
        **hubness,
        **oracle,
    }


def weighted_mean(
    rows: Iterable[Dict],
    value: str,
    weight: str,
) -> float:
    rows = list(rows)
    denominator = sum(float(row[weight]) for row in rows)
    if denominator <= 0:
        return 0.0
    return float(
        sum(float(row[value]) * float(row[weight]) for row in rows) / denominator
    )


def aggregate_variant(rows: Iterable[Dict]) -> Dict[str, float]:
    rows = list(rows)
    pair_mean_fields = (
        "NCM",
        "Pre",
        "SR",
        "RMSE",
        "matches",
        "keypoints0",
        "keypoints1",
        "mean_descriptor_distance",
        "unique_target_ratio",
        "max_target_fan_in",
    )
    result = {"num_pairs": len(rows)}
    for field in pair_mean_fields:
        result[field] = float(np.mean([row[field] for row in rows]))
    for field in (
        "oracle_r1",
        "oracle_r5",
        "oracle_r10",
        "oracle_mrr10",
        "positive_distance",
        "hard_negative_distance_top10",
        "distance_margin_top10",
    ):
        result[field] = weighted_mean(rows, field, "oracle_eligible")
    result["oracle_eligible"] = float(
        sum(row["oracle_eligible"] for row in rows)
    )
    return result


def aggregate_physical(rows: Iterable[Dict]) -> Dict[str, float]:
    rows = list(rows)
    result = {"num_pairs": len(rows)}
    for field in (
        "detector_repeatability_r3",
        "detector_repeatability_r5",
    ):
        result[field] = weighted_mean(rows, field, "detector_projected_valid")
    result["detector_projected_valid"] = float(
        sum(row["detector_projected_valid"] for row in rows)
    )
    for field in (
        "orientation_median_error_deg",
        "orientation_mean_error_deg",
        "odd_correlation",
        "even_correlation",
        "roe_correlation",
        "vimo_correlation",
        "vimo_agreement",
        "vimo_iou",
    ):
        result[field] = float(np.mean([row[field] for row in rows]))
    result["physical_samples"] = float(sum(row["physical_samples"] for row in rows))
    return result


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.smoke:
        args.pairs_per_modality = 1
        args.max_keypoints = min(args.max_keypoints, 100)
        args.feature_cache_size = min(args.feature_cache_size, 2)

    gpu_metadata = None
    if args.device == "auto":
        gpu_index, gpu_metadata = select_gpu_with_most_free_memory()
        args.device = f"cuda:{gpu_index}"
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)

    selected_names = set(args.variants or [variant.name for variant in VARIANTS])
    variants = tuple(
        variant for variant in VARIANTS if variant.name in selected_names
    )
    rows_manifest = read_manifest_stratified(
        args.manifest_path,
        args.manifest_split,
        args.pairs_per_modality,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "selected_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    extractor = FrozenHIMO(
        HIMOConfig(max_cooccurrence_levels=args.max_cooccurrence_levels)
    ).to(device).eval()
    if sum(parameter.numel() for parameter in extractor.parameters()):
        raise RuntimeError("Frozen HIMO unexpectedly contains parameters")
    cache = FeatureCache(extractor, variants, args, device)

    pair_rows = []
    physical_rows = []
    started = time.perf_counter()
    for pair_index, record in enumerate(
        tqdm(rows_manifest, desc="Physical V3.0.2 no-training ablations")
    ):
        pair_started = time.perf_counter()
        entry0, cache_hit0 = cache.get(record["image0"])
        entry1, cache_hit1 = cache.get(record["image1"])
        matrix = read_matrix(record)
        group = modality_pair(record)

        detector = detector_diagnostics(entry0, entry1, matrix, args.image_size)
        matrix_model = model_coordinate_matrix(
            matrix,
            entry0["original_size"],
            entry1["original_size"],
            args.image_size,
        )
        physical = physical_state_diagnostics(
            entry0["state"],
            entry1["state"],
            entry0["anchors"],
            matrix_model,
        )
        physical_rows.append(
            {
                "pair_index": pair_index,
                "id": record.get("id", ""),
                "modality_pair": group,
                **detector,
                **physical,
            }
        )

        for variant in variants:
            metrics = evaluate_variant(variant, entry0, entry1, matrix, args)
            pair_rows.append(
                {
                    "pair_index": pair_index,
                    "id": record.get("id", ""),
                    "modality_pair": group,
                    "variant": variant.name,
                    **metrics,
                    "cache_hit0": int(cache_hit0),
                    "cache_hit1": int(cache_hit1),
                    "pair_runtime_ms": (
                        time.perf_counter() - pair_started
                    )
                    * 1000.0,
                }
            )

    variant_groups = defaultdict(list)
    variant_modality_groups = defaultdict(list)
    for row in pair_rows:
        variant_groups[row["variant"]].append(row)
        variant_modality_groups[(row["variant"], row["modality_pair"])].append(
            row
        )
    physical_groups = defaultdict(list)
    for row in physical_rows:
        physical_groups[row["modality_pair"]].append(row)

    variant_summary = {
        name: aggregate_variant(rows)
        for name, rows in sorted(variant_groups.items())
    }
    variant_modality_summary = {
        name: {
            modality: aggregate_variant(
                variant_modality_groups[(name, modality)]
            )
            for modality in sorted(
                {
                    key[1]
                    for key in variant_modality_groups
                    if key[0] == name
                }
            )
        }
        for name in sorted(variant_groups)
    }
    physical_summary = aggregate_physical(physical_rows)
    physical_modality_summary = {
        name: aggregate_physical(rows)
        for name, rows in sorted(physical_groups.items())
    }

    summary = {
        "version": f"v{HIMO_IMPLEMENTATION_VERSION}",
        "evaluation": "physical_v3_no_training_ablation_suite",
        "manifest_path": str(args.manifest_path),
        "manifest_split": args.manifest_split,
        "seed": args.seed,
        "pairs_per_modality": args.pairs_per_modality,
        "num_pairs": len(rows_manifest),
        "device": str(device),
        "selected_gpu": gpu_metadata,
        "image_size": args.image_size,
        "max_keypoints": args.max_keypoints,
        "correct_threshold": args.correct_thr,
        "himo_source_commit": HIMO_SOURCE_COMMIT,
        "variants": [asdict(variant) for variant in variants],
        "variant_summary": variant_summary,
        "variant_by_modality": variant_modality_summary,
        "physical_summary": physical_summary,
        "physical_by_modality": physical_modality_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "pair_variant_metrics.csv", pair_rows)
    write_csv(args.output_dir / "physical_pair_metrics.csv", physical_rows)
    write_csv(
        args.output_dir / "variant_summary.csv",
        [
            {"variant": name, **metrics}
            for name, metrics in variant_summary.items()
        ],
    )
    modality_rows = []
    for variant, groups in variant_modality_summary.items():
        for group, metrics in groups.items():
            modality_rows.append(
                {"variant": variant, "modality_pair": group, **metrics}
            )
    write_csv(args.output_dir / "variant_modality_summary.csv", modality_rows)
    write_csv(
        args.output_dir / "physical_modality_summary.csv",
        [
            {"modality_pair": name, **metrics}
            for name, metrics in physical_modality_summary.items()
        ],
    )
    (args.output_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
