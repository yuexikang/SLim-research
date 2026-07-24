# 作用：在固定真值影像对上测试训练后的Pointwise-P5或RectConv-P5，
# 使用64x64 dense cosine mutual-nearest-neighbor输出统一匹配指标。

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physical.metrics import nearest_neighbors
from src.physical.v3_ablation import project_points, select_gpu_with_most_free_memory
from src.physical.v3_p5_lightning import PhysicalV3P5Module
from src.physical.v3_p5_models import IMPLEMENTATION_VERSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_rows", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--correct_thr", type=float, default=5.0)
    parser.add_argument("--success_ncm", type=int, default=20)
    parser.add_argument("--failed_rmse", type=float, default=10.0)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--feature_cache_size", type=int, default=16)
    return parser.parse_args()


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_matrix(record):
    path = Path(record.get("gt") or record.get("matrix_path") or "")
    matrix = np.load(path) if path.suffix.lower() == ".npy" else np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    direction = str(record.get("gt_direction", "0to1")).lower()
    if direction in {"1to0", "b_to_a", "1_to_0"}:
        matrix = np.linalg.inv(matrix)
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


class DescriptorCache:
    def __init__(self, encoder, device, image_size, capacity):
        self.encoder = encoder
        self.device = device
        self.image_size = int(image_size)
        self.capacity = int(capacity)
        self.entries = OrderedDict()

    def get(self, path):
        if path in self.entries:
            result = self.entries.pop(path)
            self.entries[path] = result
            return result
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        original_size = (image.shape[1], image.shape[0])
        interpolation = (
            cv2.INTER_AREA
            if max(image.shape) > self.image_size
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=interpolation,
        )
        tensor = torch.from_numpy(resized.astype(np.float32))[None, None].to(
            self.device
        )
        descriptor = self.encoder(tensor)
        result = (descriptor, original_size)
        if self.capacity > 0:
            self.entries[path] = result
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return result


def descriptor_points(indices, width, coarse_scale, original_size, image_size):
    x = indices.remainder(width).double() + 0.5
    y = torch.div(indices, width, rounding_mode="floor").double() + 0.5
    points = torch.stack([x, y], dim=1).cpu().numpy() * coarse_scale
    original_width, original_height = original_size
    return points * np.array(
        [original_width / image_size, original_height / image_size]
    )


def aggregate(rows):
    return {
        "num_pairs": len(rows),
        "NCM": float(np.mean([row["NCM"] for row in rows])),
        "Pre": float(np.mean([row["Pre"] for row in rows])),
        "SR": float(np.mean([row["SR"] for row in rows])),
        "RMSE": float(np.mean([row["RMSE"] for row in rows])),
        "mean_matches": float(np.mean([row["matches"] for row in rows])),
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in rows])),
    }


def main():
    args = parse_args()
    snapshot = None
    if args.device == "auto":
        gpu, snapshot = select_gpu_with_most_free_memory()
        args.device = f"cuda:{gpu}"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module = PhysicalV3P5Module.load_from_checkpoint(
        args.ckpt_path,
        map_location=device,
    )
    encoder = module.encoder.to(device).eval()
    rows = read_rows(args.selected_rows)
    cache = DescriptorCache(
        encoder,
        device,
        args.image_size,
        args.feature_cache_size,
    )
    result_rows = []
    with torch.inference_mode():
        for index, record in enumerate(tqdm(rows, desc=module.hparams.model_name)):
            started = time.perf_counter()
            descriptor0, size0 = cache.get(record["image0"])
            descriptor1, size1 = cache.get(record["image1"])
            source, target, _ = nearest_neighbors(
                descriptor0,
                descriptor1,
                chunk_size=args.chunk_size,
            )
            points0 = descriptor_points(
                source,
                descriptor0.shape[-1],
                module.coarse_scale,
                size0,
                args.image_size,
            )
            points1 = descriptor_points(
                target,
                descriptor1.shape[-1],
                module.coarse_scale,
                size1,
                args.image_size,
            )
            projected, valid = project_points(points0, read_matrix(record))
            errors = np.linalg.norm(projected - points1, axis=1)
            errors[~valid | ~np.isfinite(errors)] = np.inf
            correct = errors <= args.correct_thr
            ncm = int(correct.sum())
            matches = int(len(errors))
            success = ncm >= args.success_ncm
            rmse = (
                float(np.sqrt(np.mean(errors[correct] ** 2)))
                if success and ncm
                else float(args.failed_rmse)
            )
            result_rows.append(
                {
                    "index": index,
                    "id": record.get("id", ""),
                    "modality_pair": (
                        f"{record.get('modality0', 'unknown')}-"
                        f"{record.get('modality1', 'unknown')}"
                    ),
                    "matches": matches,
                    "NCM": ncm,
                    "Pre": float(ncm / matches) if matches else 0.0,
                    "SR": int(success),
                    "RMSE": rmse,
                    "runtime_ms": (time.perf_counter() - started) * 1000.0,
                }
            )

    groups = defaultdict(list)
    for row in result_rows:
        groups[row["modality_pair"]].append(row)
    summary = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "evaluation": "physical_v3_p5_dense_mutual_nn",
        "model": module.hparams.model_name,
        "checkpoint": str(args.ckpt_path),
        "selected_rows": str(args.selected_rows),
        "device": str(device),
        "selected_gpu": snapshot,
        "image_size": args.image_size,
        "trainable_parameters": encoder.trainable_parameters,
        "overall": aggregate(result_rows),
        "by_modality_pair": {
            name: aggregate(group) for name, group in sorted(groups.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output_dir / "pair_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
