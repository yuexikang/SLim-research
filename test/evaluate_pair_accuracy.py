# 作用：输入一对影像 A/B 和对应的 3x3 真实配准矩阵 .npy，调用 SLiM 模型匹配两图，并用重投影误差统计匹配精度。
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from default_config import get_config
from infer_pairs import (
    IMAGE_EXTS,
    draw_matches,
    load_model,
    make_batch,
    safe_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SLiM matches on one image pair with a ground-truth matrix."
    )
    parser.add_argument("--image0_path", type=Path, required=True, help="Path to image A.")
    parser.add_argument("--image1_path", type=Path, required=True, help="Path to image B.")
    parser.add_argument(
        "--matrix_path",
        type=Path,
        default=None,
        help="Path to the corresponding 3x3 .npy matrix. Defaults to image0 stem without _A + .npy.",
    )
    parser.add_argument(
        "--matrix_direction",
        type=str,
        default="auto",
        choices=["auto", "a_to_b", "b_to_a"],
        help="Direction of the .npy matrix. auto selects the direction with lower median reprojection error.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/test_accuracy"),
        help="Where to save npz, visualizations, and metrics json.",
    )
    parser.add_argument(
        "--pair_name",
        type=str,
        default=None,
        help="Output name. Defaults to image stem names.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("ckpt/megadepth_19epochs.ckpt"),
        help="Path to a SLiM checkpoint.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_test",
        choices=["outdoor_test", "indoor_test"],
        help="Use outdoor_test for general/outdoor imagery.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--thr", type=float, default=None, help="Override match threshold.")
    parser.add_argument("--refine_iters", type=int, default=4, help="Refinement iterations.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Resize longer image side before padding. Defaults to min(config IMAGE_SIZE, 960).",
    )
    parser.add_argument(
        "--inlier_threshold",
        type=float,
        default=5.0,
        help="Pixel threshold used for the main accuracy/inlier ratio.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,3,5,10",
        help="Comma-separated pixel thresholds for additional inlier ratios.",
    )
    parser.add_argument(
        "--max_vis_matches",
        type=int,
        default=300,
        help="Maximum matches drawn in the all-match visualization.",
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip model.initial_forward(); useful if Triton warmup fails on a GPU.",
    )
    return parser.parse_args()


def default_matrix_path(image0_path: Path):
    stem = image0_path.stem
    if stem.endswith("_A"):
        stem = stem[:-2]
    return image0_path.with_name(f"{stem}.npy")


def validate_inputs(image0_path: Path, image1_path: Path, matrix_path: Path):
    for path in [image0_path, image1_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {path}")
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")


def project_points(points_xy: np.ndarray, matrix: np.ndarray):
    points_h = np.concatenate(
        [points_xy.astype(np.float64), np.ones((len(points_xy), 1), dtype=np.float64)],
        axis=1,
    )
    projected_h = points_h @ matrix.T
    denom = projected_h[:, 2:3]
    valid = np.abs(denom[:, 0]) > 1e-8
    projected = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    projected[valid] = projected_h[valid, :2] / denom[valid]
    return projected, valid


def reprojection_errors(pts0: np.ndarray, pts1: np.ndarray, matrix: np.ndarray):
    projected, valid = project_points(pts0, matrix)
    errors = np.linalg.norm(projected - pts1, axis=1)
    valid = valid & np.isfinite(errors)
    return errors[valid], projected[valid], valid


def select_matrix_direction(pts0, pts1, matrix, matrix_direction):
    errors_ab, projected_ab, valid_ab = reprojection_errors(pts0, pts1, matrix)
    inv_matrix = np.linalg.inv(matrix)
    errors_ba, projected_ba, valid_ba = reprojection_errors(pts0, pts1, inv_matrix)
    diagnostics = {
        "a_to_b_median_error_px": float(np.median(errors_ab)) if len(errors_ab) else None,
        "inverse_median_error_px": float(np.median(errors_ba)) if len(errors_ba) else None,
        "a_to_b_mean_error_px": float(np.mean(errors_ab)) if len(errors_ab) else None,
        "inverse_mean_error_px": float(np.mean(errors_ba)) if len(errors_ba) else None,
    }

    if matrix_direction == "a_to_b":
        return matrix, "a_to_b", errors_ab, projected_ab, valid_ab, diagnostics

    if matrix_direction == "b_to_a":
        return (
            inv_matrix,
            "b_to_a_used_as_inverse",
            errors_ba,
            projected_ba,
            valid_ba,
            diagnostics,
        )

    median_ab = float(np.median(errors_ab)) if len(errors_ab) else float("inf")
    median_ba = float(np.median(errors_ba)) if len(errors_ba) else float("inf")
    if median_ab <= median_ba:
        return matrix, "a_to_b", errors_ab, projected_ab, valid_ab, diagnostics
    return inv_matrix, "b_to_a_used_as_inverse", errors_ba, projected_ba, valid_ba, diagnostics


def summarize_errors(errors: np.ndarray, thresholds, inlier_threshold: float):
    if len(errors) == 0:
        return {
            "valid_matches": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "rmse_px": None,
            "main_inlier_threshold_px": inlier_threshold,
            "main_accuracy": 0.0,
            "inlier_ratios": {str(t): 0.0 for t in thresholds},
        }

    inlier_ratios = {str(t): float(np.mean(errors <= t)) for t in thresholds}
    return {
        "valid_matches": int(len(errors)),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors**2))),
        "main_inlier_threshold_px": float(inlier_threshold),
        "main_accuracy": float(np.mean(errors <= inlier_threshold)),
        "inlier_ratios": inlier_ratios,
    }


def draw_inlier_matches(path0, path1, pts0, pts1, inlier_mask, out_path):
    img0 = Image.open(path0).convert("RGB")
    img1 = Image.open(path1).convert("RGB")
    w0, h0 = img0.size
    w1, h1 = img1.size
    canvas = Image.new("RGB", (w0 + w1, max(h0, h1)), "white")
    canvas.paste(img0, (0, 0))
    canvas.paste(img1, (w0, 0))
    draw = ImageDraw.Draw(canvas)

    indices = np.flatnonzero(inlier_mask)
    if len(indices) > 500:
        indices = np.linspace(0, len(indices) - 1, 500).round().astype(np.int64)
        indices = np.flatnonzero(inlier_mask)[indices]

    for idx in indices:
        x0, y0 = float(pts0[idx, 0]), float(pts0[idx, 1])
        x1, y1 = float(pts1[idx, 0]) + w0, float(pts1[idx, 1])
        draw.line((x0, y0, x1, y1), fill=(40, 220, 90), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    matrix_path = args.matrix_path or default_matrix_path(args.image0_path)
    validate_inputs(args.image0_path, args.image1_path, matrix_path)

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("SLiM inference in this repo uses CUDA timers; use a CUDA device.")
    torch.cuda.set_device(torch.device(args.device))

    config = get_config(args.config_name)
    config.MODEL.REFINE_ITERS = config.REFINE_ITERS = args.refine_iters
    if args.thr is not None:
        config.MODEL.COARSE_THRES = config.COARSE_THRES = args.thr
        config.MODEL.FINE_THRES = config.FINE_THRES = args.thr

    image_size = args.image_size or min(config.IMAGE_SIZE, 960)
    coarse_scale = int(config.MODEL.COARSE_SCALE)
    if image_size % coarse_scale != 0:
        raise ValueError(f"--image_size must be divisible by {coarse_scale}.")

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    pair_name = safe_name(args.pair_name or f"{args.image0_path.stem}__{args.image1_path.stem}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(config, args.ckpt_path, device, skip_warmup=args.skip_warmup)
    batch = make_batch(args.image0_path, args.image1_path, image_size, coarse_scale, device)

    with torch.inference_mode():
        model(batch, training=False)

    pts0 = batch["fine_coord_0"].detach().cpu().numpy()
    pts1 = batch["fine_coord_1"].detach().cpu().numpy()
    matrix = np.load(matrix_path).astype(np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got shape {matrix.shape}: {matrix_path}")

    used_matrix, used_direction, errors, projected, valid_mask, direction_diagnostics = select_matrix_direction(
        pts0, pts1, matrix, args.matrix_direction
    )
    metrics = summarize_errors(errors, thresholds, args.inlier_threshold)
    metrics.update(
        {
            "pair_name": pair_name,
            "image0": str(args.image0_path),
            "image1": str(args.image1_path),
            "matrix_path": str(matrix_path),
            "matrix_direction": used_direction,
            "direction_diagnostics": direction_diagnostics,
            "total_matches": int(len(pts0)),
        }
    )

    valid_indices = np.flatnonzero(valid_mask)
    full_inlier_mask = np.zeros(len(pts0), dtype=bool)
    full_inlier_mask[valid_indices] = errors <= args.inlier_threshold

    npz_path = args.output_dir / f"{pair_name}_matches_eval.npz"
    all_vis_path = args.output_dir / f"{pair_name}_matches_all.jpg"
    inlier_vis_path = args.output_dir / f"{pair_name}_matches_inliers.jpg"
    metrics_path = args.output_dir / f"{pair_name}_metrics.json"

    np.savez_compressed(
        npz_path,
        image0=str(args.image0_path),
        image1=str(args.image1_path),
        matrix_path=str(matrix_path),
        matrix=matrix,
        used_matrix=used_matrix,
        fine_coord_0=pts0,
        fine_coord_1=pts1,
        valid_indices=valid_indices,
        projected_coord_1=projected,
        reprojection_errors=errors,
        inlier_mask=full_inlier_mask,
    )
    draw_matches(args.image0_path, args.image1_path, pts0, pts1, all_vis_path, args.max_vis_matches)
    draw_inlier_matches(args.image0_path, args.image1_path, pts0, pts1, full_inlier_mask, inlier_vis_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nEvaluation results:")
    print(f"pair: {pair_name}")
    print(f"matches: {metrics['total_matches']} total, {metrics['valid_matches']} valid")
    print(f"matrix direction: {metrics['matrix_direction']}")
    print("direction diagnostics:", metrics["direction_diagnostics"])
    if metrics["mean_error_px"] is None:
        print("no valid reprojection errors")
        return
    print(f"mean error: {metrics['mean_error_px']:.4f}px")
    print(f"median error: {metrics['median_error_px']:.4f}px")
    print(f"rmse: {metrics['rmse_px']:.4f}px")
    print(
        f"accuracy@{metrics['main_inlier_threshold_px']:.1f}px: "
        f"{metrics['main_accuracy'] * 100:.2f}%"
    )
    print("inlier ratios:", metrics["inlier_ratios"])
    print(f"npz: {npz_path}")
    print(f"all matches: {all_vis_path}")
    print(f"inliers: {inlier_vis_path}")
    print(f"metrics: {metrics_path}")

    del batch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
