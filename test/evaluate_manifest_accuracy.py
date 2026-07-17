# 作用：指定 SLiM 权重和遥感验证/测试集 jsonl 索引，批量评估匹配精度，并随机保存若干对匹配可视化结果。

# MPLCONFIGDIR=/tmp/matplotlib /root/miniconda3/envs/slim/bin/python test/evaluate_manifest_accuracy.py \
#   --manifest_path data/remote_archive/manifests/test_GoogleEarth.jsonl \
#   --manifest_split test \
#   --ckpt_path "logs/tb_logs/googleearth_pairs/lr5e-5_bs6_512_epoch40_resume/checkpoints/best-epoch=32-remote_inlier@5=0.992-remote_median_error=0.33.ckpt" \
#   --output_dir outputs/eval_googleearth_best \
#   --device cuda:1 \
#   --image_size 512 \
#   --num_vis_pairs 10

# 如果你想测试“五种随机扰动下的鲁棒性”,再加
# --eval_aug_variants default

# MPLCONFIGDIR=/tmp/matplotlib /root/miniconda3/envs/slim/bin/python test/evaluate_manifest_accuracy.py \
#   --manifest_path data/remote_archive/manifests/val_optical_single_images.jsonl \
#   --manifest_split val \
#   --ckpt_path "logs/tb_logs/opt_single/lr5e-5_bs6_512_epoch30/checkpoints/last.ckpt" \
#   --output_dir outputs/eval_opt_single_val \
#   --device cuda:0 \
#   --image_size 512 \
#   --eval_aug_variants default \
#   --homography_difficulty 0.3 \
#   --num_vis_pairs 10


import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from default_config import get_config
from infer_pairs import load_model, safe_name
from src.datasets.remote_sensing import RemoteSensingHomographyDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a SLiM checkpoint on a remote-sensing manifest."
    )
    parser.add_argument("--manifest_path", type=Path, required=True, help="Remote-sensing jsonl manifest.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to a SLiM checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/manifest_accuracy"),
        help="Directory for metrics, per-pair csv, and sampled visualizations.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_test",
        choices=["outdoor_test", "indoor_test"],
        help="Test config used to build the SLiM model.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, for example cuda:0.")
    parser.add_argument("--thr", type=float, default=None, help="Override coarse and fine matching thresholds.")
    parser.add_argument("--refine_iters", type=int, default=4, help="Number of refinement iterations.")
    parser.add_argument("--image_size", type=int, default=512, help="Square image size used by the remote dataset.")
    parser.add_argument(
        "--coarse_matcher",
        type=str,
        default="full",
        choices=["full", "norm_clue"],
        help="Use full coarse dual-softmax matching for paper evaluation, or the faster norm-clue candidate matcher.",
    )
    parser.add_argument(
        "--manifest_split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
        help="Which split to read from the manifest. mode=val also accepts split=test in the dataset.",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Limit base manifest rows; 0 means no limit.")
    parser.add_argument(
        "--eval_aug_variants",
        type=str,
        default="none",
        help=(
            "Evaluation perturbations. Use 'none' for fixed manifest pairs, "
            "'default' for translation,scale,yaw,pitch,roll, or comma-separated variants."
        ),
    )
    parser.add_argument(
        "--homography_difficulty",
        type=float,
        default=0.25,
        help="Difficulty for optional online synthetic perturbation evaluation.",
    )
    parser.add_argument(
        "--left_identity",
        action="store_true",
        default=True,
        help="Keep image0 unwarped for optional perturbation evaluation.",
    )
    parser.add_argument("--seed", type=int, default=20260627, help="Seed for deterministic val perturbations and sampled visualizations.")
    parser.add_argument("--thresholds", type=str, default="1,3,5,10", help="Comma-separated pixel thresholds.")
    parser.add_argument("--main_threshold", type=float, default=5.0, help="Main inlier threshold in pixels.")
    parser.add_argument("--ncm_threshold", type=float, default=5.0, help="Pixel threshold for Number of Correct Matches.")
    parser.add_argument("--sr_min_ncm", type=int, default=20, help="A pair is successful when NCM is no less than this value.")
    parser.add_argument("--failed_rmse", type=float, default=10.0, help="RMSE assigned to failed image pairs.")
    parser.add_argument("--auc_threshold", type=float, default=5.0, help="Upper threshold tau for corner-error AUC.")
    parser.add_argument("--ransac_reproj_threshold", type=float, default=5.0, help="RANSAC reprojection threshold for estimating H_hat.")
    parser.add_argument("--num_vis_pairs", type=int, default=10, help="Number of random pairs to visualize.")
    parser.add_argument("--max_vis_matches", type=int, default=300, help="Maximum matches drawn in each visualization.")
    parser.add_argument("--skip_warmup", action="store_true", help="Skip model.initial_forward().")
    return parser.parse_args()


def parse_thresholds(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_eval_aug_variants(value):
    value = (value or "none").strip()
    if value.lower() in {"none", "off", "false", "0"}:
        return ["mixed"], 0.0
    if value.lower() in {"default", "remote"}:
        return list(RemoteSensingHomographyDataset.DEFAULT_AUG_VARIANTS), None
    return [v.strip() for v in value.split(",") if v.strip()], None


def tensor_to_pil(image_tensor):
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def warp_points(points_xy, H):
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)
    points_h = np.concatenate(
        [points_xy.astype(np.float64), np.ones((len(points_xy), 1), dtype=np.float64)],
        axis=1,
    )
    warped_h = points_h @ H.astype(np.float64).T
    denom = warped_h[:, 2]
    valid = np.abs(denom) > 1e-8
    warped = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    warped[valid] = warped_h[valid, :2] / denom[valid, None]
    return warped, valid


def compute_pair_metrics(pts0, pts1, H_0to1, thresholds, main_threshold):
    projected, valid = warp_points(pts0, H_0to1)
    errors = np.linalg.norm(projected - pts1, axis=1)
    valid = valid & np.isfinite(errors)
    errors = errors[valid]

    if len(errors) == 0:
        return {
            "total_matches": int(len(pts0)),
            "valid_matches": 0,
            "mean_error": np.nan,
            "median_error": np.nan,
            "rmse": np.nan,
            "main_inlier": 0.0,
            **{f"inlier@{format_threshold(t)}": 0.0 for t in thresholds},
        }, errors, valid

    metrics = {
        "total_matches": int(len(pts0)),
        "valid_matches": int(len(errors)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "main_inlier": float(np.mean(errors <= main_threshold)),
    }
    for threshold in thresholds:
        metrics[f"inlier@{format_threshold(threshold)}"] = float(np.mean(errors <= threshold))
    return metrics, errors, valid


def estimate_homography(pts0, pts1, ransac_reproj_threshold):
    if len(pts0) < 4 or len(pts1) < 4:
        return None, 0
    H_hat, inlier_mask = cv2.findHomography(
        pts0.astype(np.float32),
        pts1.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold),
    )
    if H_hat is None or not np.isfinite(H_hat).all():
        return None, 0
    if abs(float(H_hat[2, 2])) > 1e-8:
        H_hat = H_hat / H_hat[2, 2]
    ransac_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return H_hat.astype(np.float64), ransac_inliers


def corner_error(H_hat, H_gt, image_size):
    if H_hat is None:
        return np.inf
    s = float(image_size - 1)
    corners = np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=np.float64)
    pred, pred_valid = warp_points(corners, H_hat)
    gt, gt_valid = warp_points(corners, H_gt)
    valid = pred_valid & gt_valid & np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if not valid.all():
        return np.inf
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def corner_auc(ace_values, tau):
    values = np.asarray(ace_values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    finite_under_tau = np.sort(values[np.isfinite(values) & (values <= tau)])
    if finite_under_tau.size == 0:
        return 0.0
    recalls = np.arange(1, len(finite_under_tau) + 1, dtype=np.float64) / float(len(values))
    x = np.concatenate([[0.0], finite_under_tau, [tau]])
    y = np.concatenate([[0.0], recalls, [recalls[-1]]])
    return float(np.trapz(y, x) / tau)


def format_threshold(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def aggregate_pair_metrics(pair_rows, thresholds, main_threshold, ncm_threshold, sr_min_ncm, failed_rmse, auc_threshold):
    all_errors = []
    correct_errors = []
    pair_inliers = {threshold: [] for threshold in thresholds}
    total_matches = 0
    valid_matches = 0
    total_correct = 0
    successful_pairs = 0
    rmse_for_all_pairs = []
    all_ace = []
    successful_ace = []
    for row in pair_rows:
        total_matches += int(row["total_matches"])
        valid_matches += int(row["valid_matches"])
        total_correct += int(row["ncm"])
        if bool(row["success"]):
            successful_pairs += 1
            successful_ace.append(float(row["ace"]))
        all_ace.append(float(row["ace"]))
        rmse_for_all_pairs.append(float(row["paper_rmse"]))
        if row["_errors"].size:
            all_errors.append(row["_errors"])
            correct = row["_errors"][row["_errors"] <= ncm_threshold]
            if correct.size:
                correct_errors.append(correct)
        for threshold in thresholds:
            pair_inliers[threshold].append(row[f"inlier@{format_threshold(threshold)}"])

    errors = np.concatenate(all_errors) if all_errors else np.empty((0,), dtype=np.float64)
    ncm_errors = np.concatenate(correct_errors) if correct_errors else np.empty((0,), dtype=np.float64)
    summary = {
        "num_pairs": len(pair_rows),
        "total_matches": int(total_matches),
        "valid_matches": int(valid_matches),
        "ncm_threshold": float(ncm_threshold),
        "NCM": int(total_correct),
        "Precision": float(total_correct / total_matches) if total_matches else 0.0,
        "sr_min_ncm": int(sr_min_ncm),
        "SR": float(successful_pairs / len(pair_rows)) if pair_rows else 0.0,
        "successful_pairs": int(successful_pairs),
        "mean_matches": float(total_matches / len(pair_rows)) if pair_rows else 0.0,
        "mean_valid_matches": float(valid_matches / len(pair_rows)) if pair_rows else 0.0,
        "mean_error": float(np.mean(errors)) if len(errors) else None,
        "median_error": float(np.median(errors)) if len(errors) else None,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "RMSE_correct_matches": float(np.sqrt(np.mean(ncm_errors**2))) if len(ncm_errors) else float(failed_rmse),
        "RMSE_pair_protocol": float(np.mean(rmse_for_all_pairs)) if rmse_for_all_pairs else float(failed_rmse),
        "failed_rmse": float(failed_rmse),
        "MACE": float(np.mean(successful_ace)) if successful_ace else None,
        "AUC_threshold": float(auc_threshold),
        "AUC": corner_auc(all_ace, auc_threshold),
        "main_threshold": float(main_threshold),
        "main_inlier": float(np.mean(errors <= main_threshold)) if len(errors) else 0.0,
    }
    for threshold in thresholds:
        key = format_threshold(threshold)
        summary[f"inlier@{key}"] = float(np.mean(errors <= threshold)) if len(errors) else 0.0
        summary[f"pair_inlier@{key}"] = float(np.mean(pair_inliers[threshold])) if pair_inliers[threshold] else 0.0
    return summary


def draw_eval_matches(image0, image1, pts0, pts1, errors, valid_mask, out_path, threshold, max_matches):
    img0 = tensor_to_pil(image0)
    img1 = tensor_to_pil(image1)
    w0, h0 = img0.size
    canvas = Image.new("RGB", (w0 + img1.size[0], max(h0, img1.size[1])), "white")
    canvas.paste(img0, (0, 0))
    canvas.paste(img1, (w0, 0))
    draw = ImageDraw.Draw(canvas)

    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) > max_matches:
        pick = np.linspace(0, len(valid_indices) - 1, max_matches).round().astype(np.int64)
        valid_indices = valid_indices[pick]

    valid_errors = errors
    full_errors = np.full(len(pts0), np.nan, dtype=np.float64)
    full_errors[np.flatnonzero(valid_mask)] = valid_errors
    for idx in valid_indices:
        err = full_errors[idx]
        color = (40, 220, 90) if err <= threshold else (230, 70, 60)
        x0, y0 = float(pts0[idx, 0]), float(pts0[idx, 1])
        x1, y1 = float(pts1[idx, 0]) + w0, float(pts1[idx, 1])
        draw.line((x0, y0, x1, y1), fill=color, width=1)
        draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), outline=color, width=1)
        draw.ellipse((x1 - 2, y1 - 2, x1 + 2, y1 + 2), outline=color, width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_dataset(args, aug_variants, difficulty):
    mode = "val" if args.manifest_split in {"val", "test"} else args.manifest_split
    manifest_split = args.manifest_split
    return RemoteSensingHomographyDataset(
        manifest_path=args.manifest_path,
        image_size=args.image_size,
        mode=mode,
        max_samples=args.max_samples,
        homography_difficulty=difficulty,
        left_identity=args.left_identity,
        aug_variants=aug_variants,
        manifest_split=manifest_split,
        seed=args.seed,
    )


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


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("SLiM inference in this repo uses CUDA timers; use a CUDA device.")
    torch.cuda.set_device(torch.device(args.device))

    config = get_config(args.config_name)
    config.MODEL.REFINE_ITERS = config.REFINE_ITERS = args.refine_iters
    if args.thr is not None:
        config.MODEL.COARSE_THRES = config.COARSE_THRES = args.thr
        config.MODEL.FINE_THRES = config.FINE_THRES = args.thr

    coarse_scale = int(config.MODEL.COARSE_SCALE)
    if args.image_size % coarse_scale != 0:
        raise ValueError(f"--image_size must be divisible by coarse scale {coarse_scale}.")

    aug_variants, forced_difficulty = parse_eval_aug_variants(args.eval_aug_variants)
    difficulty = args.homography_difficulty if forced_difficulty is None else forced_difficulty
    thresholds = parse_thresholds(args.thresholds)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args, aug_variants, difficulty)
    model = load_model(config, args.ckpt_path, device, skip_warmup=args.skip_warmup)

    rng = np.random.default_rng(args.seed)
    n_vis = min(args.num_vis_pairs, len(dataset))
    vis_indices = set(rng.choice(len(dataset), size=n_vis, replace=False).tolist()) if n_vis else set()

    pair_rows = []
    with torch.inference_mode():
        for idx in tqdm(range(len(dataset)), desc="Evaluating manifest"):
            item = dataset[idx]
            batch = make_batch(item, device)
            batch["use_full_coarse_matching"] = args.coarse_matcher == "full"
            model(batch, training=False)

            pts0 = batch["fine_coord_0"].detach().cpu().numpy()
            pts1 = batch["fine_coord_1"].detach().cpu().numpy()
            H = item["H_0to1"].detach().cpu().numpy()
            pair_metrics, errors, valid_mask = compute_pair_metrics(
                pts0, pts1, H, thresholds, args.main_threshold
            )
            ncm = int(np.sum(errors <= args.ncm_threshold))
            pair_precision = float(ncm / len(pts0)) if len(pts0) else 0.0
            success = ncm >= args.sr_min_ncm
            correct_errors = errors[errors <= args.ncm_threshold]
            paper_rmse = (
                float(np.sqrt(np.mean(correct_errors**2)))
                if success and correct_errors.size
                else float(args.failed_rmse)
            )
            H_hat, ransac_inliers = estimate_homography(
                pts0, pts1, args.ransac_reproj_threshold
            )
            ace = corner_error(H_hat, H, args.image_size) if success else np.inf

            row = {
                "index": idx,
                "remote_id": item.get("remote_id", ""),
                "remote_mode": item.get("remote_mode", ""),
                "remote_pair_type": item.get("remote_pair_type", ""),
                "remote_aug_variant": item.get("remote_aug_variant", ""),
                "ncm": ncm,
                "precision": pair_precision,
                "success": bool(success),
                "paper_rmse": paper_rmse,
                "ace": float(ace),
                "ransac_inliers": ransac_inliers,
                **pair_metrics,
                "_errors": errors,
            }
            pair_rows.append(row)

            if idx in vis_indices:
                name = safe_name(
                    f"{idx:06d}_{row['remote_id']}_{row['remote_aug_variant']}"
                )
                draw_eval_matches(
                    item["image0"],
                    item["image1"],
                    pts0,
                    pts1,
                    errors,
                    valid_mask,
                    args.output_dir / "visualizations" / f"{name}.jpg",
                    args.main_threshold,
                    args.max_vis_matches,
                )

            del batch
            torch.cuda.empty_cache()

    summary = aggregate_pair_metrics(
        pair_rows,
        thresholds,
        args.main_threshold,
        args.ncm_threshold,
        args.sr_min_ncm,
        args.failed_rmse,
        args.auc_threshold,
    )
    summary.update(
        {
            "manifest_path": str(args.manifest_path),
            "ckpt_path": str(args.ckpt_path),
            "config_name": args.config_name,
            "manifest_split": args.manifest_split,
            "image_size": args.image_size,
            "coarse_matcher": args.coarse_matcher,
            "eval_aug_variants": aug_variants,
            "homography_difficulty": difficulty,
            "ransac_reproj_threshold": args.ransac_reproj_threshold,
            "num_visualizations": n_vis,
        }
    )

    pair_csv = args.output_dir / "pair_metrics.csv"
    csv_fields = [
        "index",
        "remote_id",
        "remote_mode",
        "remote_pair_type",
        "remote_aug_variant",
        "ncm",
        "precision",
        "success",
        "paper_rmse",
        "ace",
        "ransac_inliers",
        "total_matches",
        "valid_matches",
        "mean_error",
        "median_error",
        "rmse",
        "main_inlier",
        *[f"inlier@{format_threshold(t)}" for t in thresholds],
    ]
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({k: row.get(k, "") for k in csv_fields})

    summary_path = args.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nManifest evaluation summary:")
    print(f"manifest: {args.manifest_path}")
    print(f"pairs evaluated: {summary['num_pairs']}")
    print(f"matches: {summary['total_matches']} total, {summary['valid_matches']} valid")
    print(f"NCM@{args.ncm_threshold:g}: {summary['NCM']}")
    print(f"Precision: {summary['Precision'] * 100:.2f}%")
    print(f"SR(NCM>={args.sr_min_ncm}): {summary['SR'] * 100:.2f}%")
    print(f"RMSE(correct matches): {summary['RMSE_correct_matches']:.4f}px")
    print(f"RMSE(pair protocol, failed={args.failed_rmse:g}): {summary['RMSE_pair_protocol']:.4f}px")
    if summary["MACE"] is not None:
        print(f"MACE(successful pairs): {summary['MACE']:.4f}px")
    else:
        print("MACE(successful pairs): None")
    print(f"AUC@{args.auc_threshold:g}: {summary['AUC'] * 100:.2f}%")
    print(f"mean matches per pair: {summary['mean_matches']:.2f}")
    if summary["mean_error"] is not None:
        print(f"mean error: {summary['mean_error']:.4f}px")
        print(f"median error: {summary['median_error']:.4f}px")
        print(f"rmse: {summary['rmse']:.4f}px")
    print(f"main inlier@{args.main_threshold:g}: {summary['main_inlier'] * 100:.2f}%")
    for threshold in thresholds:
        key = format_threshold(threshold)
        print(
            f"inlier@{key}: {summary[f'inlier@{key}'] * 100:.2f}% "
            f"(pair avg {summary[f'pair_inlier@{key}'] * 100:.2f}%)"
        )
    print(f"summary: {summary_path}")
    print(f"pair csv: {pair_csv}")
    print(f"visualizations: {args.output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
