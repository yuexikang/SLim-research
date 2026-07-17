# 作用：用 SLiM 权重测评多模态真值影像对，并按 PRISMatch 论文协议输出 NCM、Pre、SR、RMSE。
# 协议：原始图像坐标下误差不超过 5 像素为正确匹配；NCM 不少于 20 为成功；失败对 RMSE 固定为 10；不做 RANSAC 过滤。

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from default_config import get_config
from infer_pairs import load_model, safe_name
from src.datasets.remote_sensing import RemoteSensingHomographyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLiM with the PRISMatch paper metric protocol.")
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config_name", choices=["outdoor_test", "indoor_test"], default="outdoor_test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--thr", type=float, default=None, help="Override coarse and fine thresholds.")
    parser.add_argument("--refine_iters", type=int, default=4)
    parser.add_argument("--correct_thr", type=float, default=5.0)
    parser.add_argument("--success_ncm", type=int, default=20)
    parser.add_argument("--failed_rmse", type=float, default=10.0)
    parser.add_argument("--num_vis_pairs", type=int, default=10)
    parser.add_argument("--max_vis_matches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--skip_warmup", action="store_true")
    return parser.parse_args()


def make_batch(item, device):
    return {
        "image0": item["image0"][None].to(device),
        "image1": item["image1"][None].to(device),
        "H_0to1": item["H_0to1"][None].to(device),
        "scale0": item["scale0"][None].to(device),
        "scale1": item["scale1"][None].to(device),
        "pair_names": item["pair_names"],
        "use_full_coarse_matching": True,
    }


def original_sizes(record):
    size0 = cv2.imread(record["image0"], cv2.IMREAD_GRAYSCALE)
    size1 = cv2.imread(record["image1"], cv2.IMREAD_GRAYSCALE)
    if size0 is None or size1 is None:
        raise FileNotFoundError(f"Could not read {record['image0']} or {record['image1']}")
    return (size0.shape[1], size0.shape[0]), (size1.shape[1], size1.shape[0])


def reprojection_errors(points0, points1, matrix):
    if not len(points0):
        return np.empty((0,), dtype=np.float64)
    homogeneous = np.concatenate([points0, np.ones((len(points0), 1))], axis=1)
    projected = homogeneous @ matrix.astype(np.float64).T
    denom = projected[:, 2]
    valid = np.abs(denom) > 1e-8
    warped = np.full((len(points0), 2), np.nan, dtype=np.float64)
    warped[valid] = projected[valid, :2] / denom[valid, None]
    errors = np.linalg.norm(warped - points1, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def draw_matches(record, points0, points1, correct, path, max_matches):
    image0 = Image.open(record["image0"]).convert("RGB")
    image1 = Image.open(record["image1"]).convert("RGB")
    canvas = Image.new("RGB", (image0.width + image1.width, max(image0.height, image1.height)), "white")
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
        draw.line((float(x0), float(y0), float(x1) + image0.width, float(y1)), fill=color, width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize(rows):
    def aggregate(group):
        if not group:
            return {"num_pairs": 0, "NCM": 0.0, "Pre": 0.0, "SR": 0.0, "RMSE": 0.0, "mean_matches": 0.0, "mean_runtime_ms": 0.0}
        return {
            "num_pairs": len(group),
            "NCM": float(np.mean([row["NCM"] for row in group])),
            "Pre": float(np.mean([row["Pre"] for row in group])),
            "SR": float(np.mean([row["SR"] for row in group])),
            "RMSE": float(np.mean([row["RMSE"] for row in group])),
            "mean_matches": float(np.mean([row["matches"] for row in group])),
            "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in group])),
        }

    groups = defaultdict(list)
    for row in rows:
        groups[row["modality_pair"]].append(row)
    return aggregate(rows), {name: aggregate(group) for name, group in sorted(groups.items())}


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("SLiM inference requires a CUDA device.")
    torch.cuda.set_device(torch.device(args.device))

    config = get_config(args.config_name)
    config.MODEL.REFINE_ITERS = config.REFINE_ITERS = args.refine_iters
    if args.thr is not None:
        config.MODEL.COARSE_THRES = config.COARSE_THRES = args.thr
        config.MODEL.FINE_THRES = config.FINE_THRES = args.thr
    if args.image_size % int(config.MODEL.COARSE_SCALE):
        raise ValueError(f"image_size must be divisible by {config.MODEL.COARSE_SCALE}")

    dataset = RemoteSensingHomographyDataset(
        manifest_path=args.manifest_path,
        image_size=args.image_size,
        mode="val",
        homography_difficulty=0.0,
        left_identity=True,
        aug_variants=["mixed"],
        manifest_split="test",
        seed=args.seed,
    )
    device = torch.device(args.device)
    model = load_model(config, args.ckpt_path, device, skip_warmup=args.skip_warmup)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_indices = set(np.random.default_rng(args.seed).choice(len(dataset), min(args.num_vis_pairs, len(dataset)), replace=False))
    rows = []

    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="Evaluating PRISMatch protocol"):
            item = dataset[index]
            record = dataset.rows[index]
            batch = make_batch(item, device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(batch, training=False)
            torch.cuda.synchronize(device)
            runtime_ms = (time.perf_counter() - started) * 1000.0

            points0 = batch["fine_coord_0"].detach().cpu().numpy().astype(np.float64)
            points1 = batch["fine_coord_1"].detach().cpu().numpy().astype(np.float64)
            (width0, height0), (width1, height1) = original_sizes(record)
            points0 *= np.array([width0 / args.image_size, height0 / args.image_size])
            points1 *= np.array([width1 / args.image_size, height1 / args.image_size])
            matrix = dataset._read_gt_matrix(record)
            errors = reprojection_errors(points0, points1, matrix)
            correct = errors <= args.correct_thr
            ncm = int(correct.sum())
            matches = int(len(points0))
            success = ncm >= args.success_ncm
            rmse = float(np.sqrt(np.mean(errors[correct] ** 2))) if success and ncm else float(args.failed_rmse)
            modality_pair = f"{record.get('modality0', 'unknown')}-{record.get('modality1', 'unknown')}"
            row = {
                "index": index,
                "id": record.get("id", ""),
                "collection": record.get("collection", ""),
                "subset": record.get("subset", ""),
                "modality_pair": modality_pair,
                "image0": Path(record["image0"]).name,
                "image1": Path(record["image1"]).name,
                "matches": matches,
                "NCM": ncm,
                "Pre": float(ncm / matches) if matches else 0.0,
                "SR": int(success),
                "RMSE": rmse,
                "runtime_ms": runtime_ms,
            }
            rows.append(row)
            if index in vis_indices:
                draw_matches(record, points0, points1, correct, args.output_dir / "visualizations" / f"{index:04d}_{safe_name(record['id'])}.jpg", args.max_vis_matches)
            del batch
            torch.cuda.empty_cache()

    overall, by_modality = summarize(rows)
    summary = {
        "protocol": {
            "NCM": f"mean correct matches per pair, reprojection error <= {args.correct_thr}px in original image coordinates",
            "Pre": "mean of per-pair NCM / produced matches",
            "SR": f"fraction of pairs with NCM >= {args.success_ncm}",
            "RMSE": f"correct-match RMSE for successful pairs; {args.failed_rmse} for failed pairs",
            "filtering": "ground-truth label only; no RANSAC",
        },
        "manifest_path": str(args.manifest_path),
        "ckpt_path": str(args.ckpt_path),
        "image_size": args.image_size,
        "overall": overall,
        "by_modality_pair": by_modality,
    }
    fields = ["index", "id", "collection", "subset", "modality_pair", "image0", "image1", "matches", "NCM", "Pre", "SR", "RMSE", "runtime_ms"]
    with (args.output_dir / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "modality_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modality_pair", *overall.keys()])
        writer.writeheader()
        for name, values in by_modality.items():
            writer.writerow({"modality_pair": name, **values})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
