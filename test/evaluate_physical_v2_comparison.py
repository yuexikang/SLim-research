# 作用：公平比较官方 SLiM 与 SLiM+Physical V2 在 coarse 和完整匹配链路上的表现。
# 四条路线共享输入、SLiM backbone、匹配阈值和坐标协议；enhanced 仅比 base 增加训练后的 V2 delta。

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
from src.physical.models import count_trainable_parameters
from src.physical.v2_models import PhysicalEncoderV2, build_physical_v2_encoder


ROUTES = ("base_coarse", "enhanced_coarse", "base_full", "enhanced_full")
METRICS = ("matches", "NCM", "Pre", "SR", "RMSE", "runtime_ms")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare official SLiM and Physical V2 with matched protocols."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument(
        "--manifest_split", choices=["train", "val", "test", "all"], default="test"
    )
    parser.add_argument("--physical_ckpt_path", type=Path, required=True)
    parser.add_argument(
        "--slim_ckpt_path",
        type=Path,
        default=Path("ckpt/megadepth_19epochs.ckpt"),
    )
    parser.add_argument(
        "--model", choices=list(PhysicalEncoderV2.MODEL_CONFIGS), default=None
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--correct_thr", type=float, default=5.0)
    parser.add_argument("--success_ncm", type=int, default=20)
    parser.add_argument("--failed_rmse", type=float, default=10.0)
    parser.add_argument("--num_vis_pairs", type=int, default=10)
    parser.add_argument("--max_vis_matches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--skip_warmup", action="store_true")
    parser.add_argument(
        "--physical_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run only the Physical V2 encoder under BF16 autocast.",
    )
    return parser.parse_args()


def load_v2_encoder(checkpoint_path, device, requested_model=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    model_name = requested_model or hparams.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint does not record model_name; provide --model.")
    encoder = build_physical_v2_encoder(
        model_name,
        polar_chunk_size=int(hparams.get("polar_chunk_size", 1024)),
    )
    state = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder.* parameters found in {checkpoint_path}.")
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder.to(device).eval(), str(model_name), hparams, checkpoint


def make_batch(item, device):
    return {
        "image0": item["image0"][None].to(device, non_blocking=True),
        "image1": item["image1"][None].to(device, non_blocking=True),
        "scale0": item["scale0"][None].to(device, non_blocking=True),
        "scale1": item["scale1"][None].to(device, non_blocking=True),
        "pair_names": item["pair_names"],
        "use_full_coarse_matching": True,
    }


def synchronized_time(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


def prepare_route_data(batch, coarse0, coarse1, fine0=None, fine1=None):
    data = {
        "image0": batch["image0"],
        "image1": batch["image1"],
        "scale0": batch["scale0"],
        "scale1": batch["scale1"],
        "pair_names": batch["pair_names"],
        "use_full_coarse_matching": True,
        "batch_size": batch["image0"].shape[0],
        "hw0_i": batch["image0"].shape[2:],
        "hw1_i": batch["image1"].shape[2:],
        "feat0_c": coarse0,
        "feat1_c": coarse1,
        "hw0_c": coarse0.shape[2:],
        "hw1_c": coarse1.shape[2:],
        "coarse_scale": batch["image0"].shape[-1] / coarse0.shape[-1],
    }
    if fine0 is not None and fine1 is not None:
        data.update(
            {
                "feat0_f": fine0,
                "feat1_f": fine1,
                "hw0_f": fine0.shape[2:],
                "hw1_f": fine1.shape[2:],
                "fine_scale": batch["image0"].shape[-1] / fine0.shape[-1],
            }
        )
    return data


@torch.no_grad()
def run_coarse_route(slim, batch, coarse0, coarse1):
    data = prepare_route_data(batch, coarse0, coarse1)
    slim._coarse_correlation(data, coarse0, coarse1)
    slim._get_coarse_coord_test(data)
    scale = float(data["coarse_scale"])
    width0 = int(data["hw0_c"][1])
    width1 = int(data["hw1_c"][1])
    index0 = data["i_idx_c"]
    index1 = data["j_idx_c"]
    points0 = torch.stack(
        [(index0 % width0) + 0.5, (index0 // width0) + 0.5], dim=1
    ).float() * scale
    points1 = torch.stack(
        [(index1 % width1) + 0.5, (index1 // width1) + 0.5], dim=1
    ).float() * scale
    return points0, points1


@torch.no_grad()
def run_full_route(slim, batch, coarse0, coarse1, fine0, fine1):
    data = prepare_route_data(batch, coarse0, coarse1, fine0, fine1)
    slim._coarse_correlation(data, coarse0, coarse1)
    slim._get_coarse_coord_test(data)
    slim._fine_correlation(data)
    slim._get_fine_coord_test(data)
    slim._refinement(data)
    return data["fine_coord_0"], data["fine_coord_1"]


def original_sizes(record):
    image0 = cv2.imread(record["image0"], cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(record["image1"], cv2.IMREAD_GRAYSCALE)
    if image0 is None or image1 is None:
        raise FileNotFoundError(
            f"Could not read {record['image0']} or {record['image1']}"
        )
    return (image0.shape[1], image0.shape[0]), (image1.shape[1], image1.shape[0])


def restore_original_coordinates(points0, points1, record, image_size):
    (width0, height0), (width1, height1) = original_sizes(record)
    points0 = points0.detach().float().cpu().numpy().astype(np.float64)
    points1 = points1.detach().float().cpu().numpy().astype(np.float64)
    points0 *= np.array([width0 / image_size, height0 / image_size])
    points1 *= np.array([width1 / image_size, height1 / image_size])
    return points0, points1


def reprojection_errors(points0, points1, matrix):
    if not len(points0):
        return np.empty((0,), dtype=np.float64)
    homogeneous = np.concatenate(
        [points0, np.ones((len(points0), 1), dtype=np.float64)], axis=1
    )
    projected = homogeneous @ matrix.astype(np.float64).T
    denominator = projected[:, 2]
    valid = np.abs(denominator) > 1e-8
    warped = np.full((len(points0), 2), np.nan, dtype=np.float64)
    warped[valid] = projected[valid, :2] / denominator[valid, None]
    errors = np.linalg.norm(warped - points1, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def evaluate_points(
    points0,
    points1,
    matrix,
    runtime_ms,
    correct_thr,
    success_ncm,
    failed_rmse,
):
    errors = reprojection_errors(points0, points1, matrix)
    correct = errors <= correct_thr
    ncm = int(correct.sum())
    matches = int(len(errors))
    success = ncm >= success_ncm
    rmse = (
        float(np.sqrt(np.mean(errors[correct] ** 2)))
        if success and ncm
        else float(failed_rmse)
    )
    return {
        "matches": matches,
        "NCM": ncm,
        "Pre": ncm / matches if matches else 0.0,
        "SR": int(success),
        "RMSE": rmse,
        "runtime_ms": float(runtime_ms),
    }, correct


def draw_matches(record, points0, points1, correct, output_path, max_matches):
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def aggregate(rows, route):
    if not rows:
        return {
            "num_pairs": 0,
            **{name: 0.0 for name in METRICS},
        }
    return {
        "num_pairs": len(rows),
        **{
            name: float(np.mean([row[route][name] for row in rows]))
            for name in METRICS
        },
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["modality_pair"]].append(row)
    return {
        route: {
            "overall": aggregate(rows, route),
            "by_modality_pair": {
                name: aggregate(group, route)
                for name, group in sorted(groups.items())
            },
        }
        for route in ROUTES
    }


def comparison_delta(base, enhanced):
    output = {}
    for name in METRICS:
        output[name] = float(enhanced[name] - base[name])
    output["relative_NCM"] = (
        float(enhanced["NCM"] / base["NCM"]) if base["NCM"] else None
    )
    output["relative_Pre"] = (
        float(enhanced["Pre"] / base["Pre"]) if base["Pre"] else None
    )
    return output


def flatten_row(row):
    flat = {
        key: row[key]
        for key in (
            "index",
            "id",
            "collection",
            "subset",
            "modality_pair",
            "image0",
            "image1",
        )
    }
    for route in ROUTES:
        for metric in METRICS:
            flat[f"{route}_{metric}"] = row[route][metric]
    return flat


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Physical V2 comparison requires a CUDA device.")
    if args.image_size % 8:
        raise ValueError("image_size must be divisible by the coarse scale 8.")
    if not args.physical_ckpt_path.is_file():
        raise FileNotFoundError(args.physical_ckpt_path)
    if not args.slim_ckpt_path.is_file():
        raise FileNotFoundError(args.slim_ckpt_path)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = get_config("outdoor_test")
    slim = load_model(
        config,
        args.slim_ckpt_path,
        device,
        skip_warmup=args.skip_warmup,
    )
    encoder, model_name, hparams, checkpoint = load_v2_encoder(
        args.physical_ckpt_path, device, requested_model=args.model
    )
    dataset = RemoteSensingHomographyDataset(
        manifest_path=args.manifest_path,
        image_size=args.image_size,
        mode="val",
        max_samples=args.max_samples,
        homography_difficulty=0.0,
        left_identity=True,
        aug_variants=["mixed"],
        manifest_split=args.manifest_split,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_count = min(args.num_vis_pairs, len(dataset))
    vis_indices = set(
        np.random.default_rng(args.seed).choice(
            len(dataset), vis_count, replace=False
        )
    )

    rows = []
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="Physical V2 fair comparison"):
            item = dataset[index]
            record = dataset.rows[index]
            batch = make_batch(item, device)

            started = synchronized_time(device)
            pyramid0, pyramid1 = slim.feature_backbone(
                batch["image0"], batch["image1"]
            )
            backbone_ms = (synchronized_time(device) - started) * 1000.0
            base0 = pyramid0[slim.coarse_scale_idx]
            base1 = pyramid1[slim.coarse_scale_idx]

            started = synchronized_time(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.physical_amp
            ):
                output0, output1 = encoder.forward_pair(
                    batch["image0"], batch["image1"]
                )
            physical_ms = (synchronized_time(device) - started) * 1000.0
            enhanced0 = base0 + output0["delta"].float()
            enhanced1 = base1 + output1["delta"].float()

            started = synchronized_time(device)
            fine0, fine1 = slim.fine_upsample(
                pyramid0[slim.fine_scale_idx], pyramid1[slim.fine_scale_idx]
            )
            fine_upsample_ms = (synchronized_time(device) - started) * 1000.0

            route_points = {}
            route_runtime = {}
            for route, feature0, feature1, full in (
                ("base_coarse", base0, base1, False),
                ("enhanced_coarse", enhanced0, enhanced1, False),
                ("base_full", base0, base1, True),
                ("enhanced_full", enhanced0, enhanced1, True),
            ):
                started = synchronized_time(device)
                if full:
                    route_points[route] = run_full_route(
                        slim, batch, feature0, feature1, fine0, fine1
                    )
                else:
                    route_points[route] = run_coarse_route(
                        slim, batch, feature0, feature1
                    )
                matching_ms = (synchronized_time(device) - started) * 1000.0
                route_runtime[route] = backbone_ms + matching_ms
                if route.startswith("enhanced"):
                    route_runtime[route] += physical_ms
                if full:
                    route_runtime[route] += fine_upsample_ms

            matrix = dataset._read_gt_matrix(record)
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
            }
            for route in ROUTES:
                points0, points1 = restore_original_coordinates(
                    *route_points[route], record, args.image_size
                )
                metrics, correct = evaluate_points(
                    points0,
                    points1,
                    matrix,
                    route_runtime[route],
                    args.correct_thr,
                    args.success_ncm,
                    args.failed_rmse,
                )
                row[route] = metrics
                if index in vis_indices:
                    draw_matches(
                        record,
                        points0,
                        points1,
                        correct,
                        args.output_dir
                        / "visualizations"
                        / route
                        / f"{index:04d}_{safe_name(record.get('id', str(index)))}.jpg",
                        args.max_vis_matches,
                    )
            rows.append(row)

    routes = summarize(rows)
    coarse_delta = comparison_delta(
        routes["base_coarse"]["overall"],
        routes["enhanced_coarse"]["overall"],
    )
    full_delta = comparison_delta(
        routes["base_full"]["overall"],
        routes["enhanced_full"]["overall"],
    )
    summary = {
        "evaluation": "physical_v2_fair_comparison",
        "protocol": {
            "shared": (
                "same resized inputs, official SLiM backbone, dual-softmax coarse "
                "selection, thresholds and original-coordinate GT"
            ),
            "coarse": "coarse token centers after official SLiM coarse selection",
            "full": "official SLiM coarse, fine matching and recurrent refinement",
            "enhanced": "base SLiM coarse + Physical V2 adapter delta",
            "correct": (
                f"original-coordinate reprojection error <= {args.correct_thr}px"
            ),
            "success": f"NCM >= {args.success_ncm}",
            "failed_rmse": args.failed_rmse,
            "filtering": "ground-truth label only; no RANSAC",
        },
        "model": model_name,
        "physical_checkpoint": str(args.physical_ckpt_path),
        "physical_checkpoint_epoch": checkpoint.get("epoch"),
        "official_slim_checkpoint": str(args.slim_ckpt_path),
        "manifest_path": str(args.manifest_path),
        "image_size": args.image_size,
        "physical_amp": args.physical_amp,
        "trainable_physical_parameters": count_trainable_parameters(encoder),
        "routes": routes,
        "enhanced_minus_base": {
            "coarse": coarse_delta,
            "full": full_delta,
        },
        "checkpoint_hparams": hparams,
    }

    flat_rows = [flatten_row(row) for row in rows]
    fields = list(flat_rows[0]) if flat_rows else ["index"]
    with (args.output_dir / "pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    with (args.output_dir / "route_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["route", "modality_pair", "num_pairs", *METRICS]
        )
        writer.writeheader()
        for route in ROUTES:
            writer.writerow(
                {
                    "route": route,
                    "modality_pair": "overall",
                    **routes[route]["overall"],
                }
            )
            for modality, values in routes[route]["by_modality_pair"].items():
                writer.writerow(
                    {
                        "route": route,
                        "modality_pair": modality,
                        **values,
                    }
                )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
