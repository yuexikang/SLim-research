# 作用：无需训练，使用冻结 HIMO 与公开代码定义的 480 维 PolarP
# 描述子测评真值影像对。
# V3.0.1范围：单尺度可审计 HIMO 核心、固定 Shi-Tomasi anchor、
# PolarP和最近邻唯一匹配；不使用SLiM、神经网络或RANSAC。

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    match_code_exact_descriptors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the training-free Physical V3 HIMO+PolarP baseline."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument(
        "--manifest_split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_keypoints", type=int, default=1000)
    parser.add_argument("--keypoint_quality", type=float, default=0.01)
    parser.add_argument("--keypoint_min_distance", type=float, default=4.0)
    parser.add_argument("--patch_size", type=int, default=72)
    parser.add_argument("--spatial_bins", type=int, default=12)
    parser.add_argument("--orientation_bins", type=int, default=12)
    parser.add_argument(
        "--descriptor_normalization",
        choices=["none", "l2", "root"],
        default="none",
        help="'none' is the code-exact raw descriptor; other choices are V3 ablations.",
    )
    parser.add_argument(
        "--cooccurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max_cooccurrence_levels",
        type=int,
        default=8192,
        help=(
            "Safety limit for the source-faithful dynamic CoF matrix. "
            "This does not quantize gray levels."
        ),
    )
    parser.add_argument(
        "--rotation_invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--multiple_orientations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep all official base-direction peaks; disable only for the "
            "dense single-direction ablation."
        ),
    )
    parser.add_argument("--feature_cache_size", type=int, default=16)
    parser.add_argument("--correct_thr", type=float, default=5.0)
    parser.add_argument("--success_ncm", type=int, default=20)
    parser.add_argument("--failed_rmse", type=float, default=10.0)
    parser.add_argument("--num_vis_pairs", type=int, default=10)
    parser.add_argument("--max_vis_matches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=66)
    return parser.parse_args()


def read_manifest(path: Path, split: str, maximum: int) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and row.get("split") != split:
                continue
            rows.append(row)
            if maximum > 0 and len(rows) >= maximum:
                break
    if not rows:
        raise ValueError(f"No rows selected from {path} for split={split}")
    return rows


def read_matrix(record: Dict) -> np.ndarray:
    path = Path(record.get("gt") or record.get("matrix_path") or "")
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth matrix: {path}")
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        matrix = np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix in {path}, got {matrix.shape}")
    direction = str(record.get("gt_direction", "0to1")).lower()
    if direction in {"1to0", "b_to_a", "1_to_0"}:
        matrix = np.linalg.inv(matrix)
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def original_size(path: str) -> Tuple[int, int]:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.shape[1], image.shape[0]


def reprojection_errors(
    points0: np.ndarray,
    points1: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    if not len(points0):
        return np.empty((0,), dtype=np.float64)
    homogeneous = np.concatenate(
        [points0, np.ones((len(points0), 1), dtype=np.float64)], axis=1
    )
    projected = homogeneous @ matrix.T
    denominator = projected[:, 2]
    valid = np.abs(denominator) > 1e-8
    warped = np.full((len(points0), 2), np.nan, dtype=np.float64)
    warped[valid] = projected[valid, :2] / denominator[valid, None]
    errors = np.linalg.norm(warped - points1, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "pair"


def draw_matches(
    record: Dict,
    points0: np.ndarray,
    points1: np.ndarray,
    correct: np.ndarray,
    path: Path,
    max_matches: int,
):
    image0 = Image.open(record["image0"]).convert("RGB")
    image1 = Image.open(record["image1"]).convert("RGB")
    canvas = Image.new(
        "RGB",
        (image0.width + image1.width, max(image0.height, image1.height)),
        "white",
    )
    canvas.paste(image0, (0, 0))
    canvas.paste(image1, (image0.width, 0))
    draw = ImageDraw.Draw(canvas)
    indices = np.arange(len(points0))
    if len(indices) > max_matches:
        indices = np.linspace(0, len(indices) - 1, max_matches).round().astype(int)
    for index in indices:
        color = (40, 190, 80) if correct[index] else (220, 65, 60)
        x0, y0 = points0[index]
        x1, y1 = points1[index]
        draw.line(
            (float(x0), float(y0), float(x1) + image0.width, float(y1)),
            fill=color,
            width=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


class FeatureCache:
    def __init__(
        self,
        extractor: FrozenHIMO,
        polar_config: PolarPConfig,
        device: torch.device,
        image_size: int,
        capacity: int,
    ):
        self.extractor = extractor
        self.polar_config = polar_config
        self.device = device
        self.image_size = image_size
        self.capacity = max(0, capacity)
        self.entries: OrderedDict[str, Dict] = OrderedDict()

    def get(self, path: str) -> Tuple[Dict, bool]:
        if path in self.entries:
            value = self.entries.pop(path)
            self.entries[path] = value
            return value, True
        value = self._extract(path)
        if self.capacity > 0:
            self.entries[path] = value
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return value, False

    def _extract(self, path: str) -> Dict:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        resized = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=(
                cv2.INTER_AREA
                if max(image.shape) > self.image_size
                else cv2.INTER_LINEAR
            ),
        )
        tensor = torch.from_numpy(resized.astype(np.float32))[None, None].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        state = self.extractor(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        extraction_ms = (time.perf_counter() - started) * 1000.0

        magnitude_odd = state["magnitude_odd"][0, 0].cpu().numpy()
        magnitude_even = state["magnitude_even"][0, 0].cpu().numpy()
        structural = np.maximum(magnitude_odd, magnitude_even) ** 0.25
        orientation = state["orientation"][0, 0].cpu().numpy()
        weight = state["weight"][0, 0].cpu().numpy()
        keypoints = detect_fixed_keypoints(structural, self.polar_config)
        points, descriptors, diagnostics = describe_keypoints(
            weight,
            orientation,
            keypoints,
            self.polar_config,
        )
        if descriptors.shape[1:] != (480,):
            raise RuntimeError(f"PolarP descriptor must be 480-D, got {descriptors.shape}")
        return {
            "points": points,
            "descriptors": descriptors,
            "diagnostics": diagnostics,
            "extraction_ms": extraction_ms,
            "num_detected": int(len(keypoints)),
        }


def aggregate(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    if not rows:
        return {
            "num_pairs": 0,
            "NCM": 0.0,
            "Pre": 0.0,
            "SR": 0.0,
            "RMSE": 0.0,
            "mean_matches": 0.0,
            "mean_runtime_ms": 0.0,
            "mean_keypoints0": 0.0,
            "mean_keypoints1": 0.0,
        }
    return {
        "num_pairs": len(rows),
        "NCM": float(np.mean([row["NCM"] for row in rows])),
        "Pre": float(np.mean([row["Pre"] for row in rows])),
        "SR": float(np.mean([row["SR"] for row in rows])),
        "RMSE": float(np.mean([row["RMSE"] for row in rows])),
        "mean_matches": float(np.mean([row["matches"] for row in rows])),
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in rows])),
        "mean_keypoints0": float(np.mean([row["keypoints0"] for row in rows])),
        "mean_keypoints1": float(np.mean([row["keypoints1"] for row in rows])),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(device)
    rows_manifest = read_manifest(args.manifest_path, args.manifest_split, args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    himo_config = HIMOConfig(
        cooccurrence=args.cooccurrence,
        patch_size=args.patch_size,
        spatial_bins=args.spatial_bins,
        max_cooccurrence_levels=args.max_cooccurrence_levels,
    )
    polar_config = PolarPConfig(
        patch_size=args.patch_size,
        spatial_bins=args.spatial_bins,
        orientation_bins=args.orientation_bins,
        rotation_invariant=args.rotation_invariant,
        multiple_orientations=args.multiple_orientations,
        max_keypoints=args.max_keypoints,
        keypoint_quality=args.keypoint_quality,
        keypoint_min_distance=args.keypoint_min_distance,
        descriptor_normalization=args.descriptor_normalization,
    )
    extractor = FrozenHIMO(himo_config).to(device).eval()
    if sum(parameter.numel() for parameter in extractor.parameters()) != 0:
        raise RuntimeError("Frozen HIMO unexpectedly contains trainable parameters.")
    cache = FeatureCache(
        extractor,
        polar_config,
        device,
        args.image_size,
        args.feature_cache_size,
    )

    rng = np.random.default_rng(args.seed)
    visual_indices = set(
        rng.choice(
            len(rows_manifest),
            min(args.num_vis_pairs, len(rows_manifest)),
            replace=False,
        ).tolist()
    )
    result_rows = []
    for index, record in enumerate(
        tqdm(
            rows_manifest,
            desc=(
                f"V{HIMO_IMPLEMENTATION_VERSION} "
                "HIMO+PolarP no-training baseline"
            ),
        )
    ):
        pair_started = time.perf_counter()
        feature0, cache_hit0 = cache.get(record["image0"])
        feature1, cache_hit1 = cache.get(record["image1"])
        source_indices, target_indices, distances = match_code_exact_descriptors(
            feature0["descriptors"],
            feature1["descriptors"],
        )
        points0 = feature0["points"][source_indices].astype(np.float64)
        points1 = feature1["points"][target_indices].astype(np.float64)
        width0, height0 = original_size(record["image0"])
        width1, height1 = original_size(record["image1"])
        points0 *= np.array([width0 / args.image_size, height0 / args.image_size])
        points1 *= np.array([width1 / args.image_size, height1 / args.image_size])
        matrix = read_matrix(record)
        errors = reprojection_errors(points0, points1, matrix)
        correct = errors <= args.correct_thr
        ncm = int(correct.sum())
        matches = int(len(errors))
        success = ncm >= args.success_ncm
        rmse = (
            float(np.sqrt(np.mean(errors[correct] ** 2)))
            if success and ncm
            else float(args.failed_rmse)
        )
        runtime_ms = (time.perf_counter() - pair_started) * 1000.0
        modality_pair = (
            f"{record.get('modality0', 'unknown')}-"
            f"{record.get('modality1', 'unknown')}"
        )
        row = {
            "index": index,
            "id": record.get("id", ""),
            "collection": record.get("collection", ""),
            "subset": record.get("subset", ""),
            "modality_pair": modality_pair,
            "image0": Path(record["image0"]).name,
            "image1": Path(record["image1"]).name,
            "keypoints0": int(len(feature0["descriptors"])),
            "keypoints1": int(len(feature1["descriptors"])),
            "matches": matches,
            "NCM": ncm,
            "Pre": float(ncm / matches) if matches else 0.0,
            "SR": int(success),
            "RMSE": rmse,
            "mean_descriptor_distance": float(distances.mean()) if len(distances) else 0.0,
            "runtime_ms": runtime_ms,
            "cache_hit0": int(cache_hit0),
            "cache_hit1": int(cache_hit1),
        }
        result_rows.append(row)
        if index in visual_indices:
            draw_matches(
                record,
                points0,
                points1,
                correct,
                args.output_dir
                / "visualizations"
                / f"{index:04d}_{safe_name(record.get('id', str(index)))}.jpg",
                args.max_vis_matches,
            )

    groups = defaultdict(list)
    for row in result_rows:
        groups[row["modality_pair"]].append(row)
    overall = aggregate(result_rows)
    by_modality = {name: aggregate(group) for name, group in sorted(groups.items())}
    summary = {
        "version": f"v{HIMO_IMPLEMENTATION_VERSION}",
        "evaluation": "physical_v3_himo_polarp_no_training",
        "implementation_scope": {
            "included": [
                "inspectable single-scale HIMO core",
                "fixed Shi-Tomasi anchors on HIMO structural magnitude",
                "code-traceable 480-D PolarP",
                "nearest-neighbor matching with unique targets",
            ],
            "excluded": [
                "opaque official Deal_Extreme.p and Preproscessing.p",
                "DoFS keypoint enhancement",
                "CDMS multiscale matching",
                "RANSAC filtering",
                "all learned modules",
            ],
            "himo_source_commit": HIMO_SOURCE_COMMIT,
        },
        "protocol": {
            "correct": f"original-coordinate reprojection error <= {args.correct_thr}px",
            "success": f"NCM >= {args.success_ncm}",
            "failed_rmse": args.failed_rmse,
            "filtering": "ground-truth label only; no RANSAC",
            "descriptor_normalization": args.descriptor_normalization,
        },
        "manifest_path": str(args.manifest_path),
        "manifest_split": args.manifest_split,
        "image_size": args.image_size,
        "config": {
            "cooccurrence": args.cooccurrence,
            "max_cooccurrence_levels": args.max_cooccurrence_levels,
            "rotation_invariant": args.rotation_invariant,
            "multiple_orientations": args.multiple_orientations,
            "patch_size": args.patch_size,
            "spatial_bins": args.spatial_bins,
            "orientation_bins": args.orientation_bins,
            "max_keypoints": args.max_keypoints,
            "keypoint_quality": args.keypoint_quality,
            "keypoint_min_distance": args.keypoint_min_distance,
        },
        "overall": overall,
        "by_modality_pair": by_modality,
    }

    fields = list(result_rows[0].keys())
    with (args.output_dir / "pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    with (args.output_dir / "modality_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields_modality = ["modality_pair", *overall.keys()]
        writer = csv.DictWriter(handle, fieldnames=fields_modality)
        writer.writeheader()
        for name, values in by_modality.items():
            writer.writerow({"modality_pair": name, **values})
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
