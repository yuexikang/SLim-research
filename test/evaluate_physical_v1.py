# 作用：统一测评 Physical Encoder V1 的合成几何能力、多模态论文指标，以及四个描述分支与官方 SLiM 的互补性。

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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from default_config import get_config
from src.datasets.remote_sensing import RemoteSensingHomographyDataset
from src.physical.matching import coarse_homography_correspondences
from src.physical.metrics import descriptor_statistics, nearest_neighbors
from src.physical.models import count_trainable_parameters
from src.physical.v1_models import PhysicalEncoderV1, build_physical_v1_encoder
from src.slim import SLiM


BRANCHES = ("fused", "edge", "contour", "stable")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a standalone Physical Encoder V1 checkpoint."
    )
    parser.add_argument("--evaluation", choices=["synthetic", "multimodal"], required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument(
        "--manifest_split", choices=["train", "val", "test", "all"], default=None
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument(
        "--model", choices=list(PhysicalEncoderV1.MODEL_CONFIGS), default=None
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--homography_difficulty", type=float, default=0.3)
    parser.add_argument("--correct_thr", type=float, default=5.0)
    parser.add_argument("--success_ncm", type=int, default=20)
    parser.add_argument("--failed_rmse", type=float, default=10.0)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--slim_ckpt_path", type=Path, default=Path("ckpt/megadepth_19epochs.ckpt")
    )
    parser.add_argument("--skip_slim_crr", action="store_true")
    parser.add_argument("--retention_reference_summary", type=Path, default=None)
    return parser.parse_args()


def load_physical_v1_checkpoint(path, device, requested_model=None):
    checkpoint = torch.load(path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    model_name = requested_model or hparams.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint does not record model_name; provide --model explicitly.")
    encoder = build_physical_v1_encoder(model_name)
    state = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        encoder_state = state
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder.to(device).eval(), model_name, hparams


def load_official_slim(path, device):
    config = get_config("outdoor_test")
    model = SLiM(config.MODEL)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), config


def make_batch(item, device):
    return {
        "image0": item["image0"][None].to(device),
        "image1": item["image1"][None].to(device),
        "H_0to1": item["H_0to1"][None].to(device),
        "scale0": item["scale0"][None].to(device),
        "scale1": item["scale1"][None].to(device),
        "remote_aug_variant": [item["remote_aug_variant"]],
    }


def merge_statistics(accumulator, prefix, stats):
    for variant, values in stats.items():
        for key, value in values.items():
            accumulator[f"{prefix}/{variant}/{key}"] += float(value)


def finalize_statistics(accumulator, prefix):
    output = {}
    for variant in ("all", "translation", "scale", "yaw", "pitch", "roll"):
        base = f"{prefix}/{variant}/"
        count = accumulator.get(base + "count", 0.0)
        if count <= 0:
            continue
        output[variant] = {
            "num_correspondences": int(count),
            "R@0": accumulator[base + "correct0"] / count,
            "R@1": accumulator[base + "correct1"] / count,
            "positive_similarity": accumulator[base + "positive"] / count,
            "hard_negative_similarity": accumulator[base + "hard_negative"] / count,
            "mean_margin": accumulator[base + "margin"] / count,
            "entropy": accumulator[base + "entropy"] / count,
            "normalized_entropy": accumulator[base + "normalized_entropy"] / count,
        }
    return output


def gate_diagnostics(outputs):
    result = {}
    for gate_name in ("scale_weights", "expert_weights"):
        gate = outputs[gate_name].float()
        entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=1)
        result[gate_name] = {
            "mean": gate.mean(dim=(0, 2, 3)).cpu().tolist(),
            "std": gate.std(dim=(0, 2, 3), unbiased=False).cpu().tolist(),
            "normalized_entropy": float((entropy / np.log(gate.shape[1])).mean()),
        }
    confidence = outputs["confidence"].float().flatten()
    result["confidence"] = {
        "mean": float(confidence.mean()),
        "q25": float(torch.quantile(confidence, 0.25)),
        "q50": float(torch.quantile(confidence, 0.50)),
        "q75": float(torch.quantile(confidence, 0.75)),
        "fraction_below_0.25": float((confidence < 0.25).float().mean()),
    }
    return result


def evaluate_synthetic(args, encoder, model_name, hparams, device):
    split = args.manifest_split or "test"
    dataset = RemoteSensingHomographyDataset(
        manifest_path=args.manifest_path,
        image_size=args.image_size,
        mode="val",
        max_samples=args.max_samples,
        homography_difficulty=args.homography_difficulty,
        left_identity=True,
        aug_variants=list(RemoteSensingHomographyDataset.DEFAULT_AUG_VARIANTS),
        manifest_split=split,
        seed=args.seed,
    )
    accumulator = defaultdict(float)
    diagnostic_rows = []
    runtimes = []
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="Synthetic Physical V1 evaluation"):
            batch = make_batch(dataset[index], device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output0 = encoder(batch["image0"])
            output1 = encoder(batch["image1"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtimes.append((time.perf_counter() - started) * 1000.0)
            correspondences = coarse_homography_correspondences(batch, 8)
            for branch in BRANCHES:
                stats = descriptor_statistics(
                    output0[branch],
                    output1[branch],
                    correspondences,
                    temperature=torch.tensor(0.05, device=device),
                    variants=batch["remote_aug_variant"],
                    chunk_size=args.chunk_size,
                )
                merge_statistics(accumulator, branch, stats)
            diagnostic_rows.extend([gate_diagnostics(output0), gate_diagnostics(output1)])

    diagnostics = {}
    if diagnostic_rows:
        for gate_name in ("scale_weights", "expert_weights"):
            diagnostics[gate_name] = {
                key: np.mean([row[gate_name][key] for row in diagnostic_rows], axis=0).tolist()
                if key in {"mean", "std"}
                else float(np.mean([row[gate_name][key] for row in diagnostic_rows]))
                for key in ("mean", "std", "normalized_entropy")
            }
        diagnostics["confidence"] = {
            key: float(np.mean([row["confidence"][key] for row in diagnostic_rows]))
            for key in diagnostic_rows[0]["confidence"]
        }
    summary = {
        "evaluation": "synthetic",
        "model": model_name,
        "checkpoint": str(args.ckpt_path),
        "manifest": str(args.manifest_path),
        "base_rows": len(dataset.rows),
        "generated_pairs": len(dataset),
        "image_size": args.image_size,
        "homography_difficulty": args.homography_difficulty,
        "trainable_parameters": count_trainable_parameters(encoder),
        "mean_encoder_pair_runtime_ms": float(np.mean(runtimes)) if runtimes else 0.0,
        "metrics": {branch: finalize_statistics(accumulator, branch) for branch in BRANCHES},
        "diagnostics": diagnostics,
        "checkpoint_hparams": hparams,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def reprojection_errors(points0, points1, homography):
    if len(points0) == 0:
        return np.empty((0,), dtype=np.float64)
    homogeneous = np.concatenate([points0, np.ones((len(points0), 1))], axis=1)
    warped = homogeneous @ homography.astype(np.float64).T
    denominator = warped[:, 2]
    valid = np.abs(denominator) > 1e-8
    projected = np.full((len(points0), 2), np.nan, dtype=np.float64)
    projected[valid] = warped[valid, :2] / denominator[valid, None]
    errors = np.linalg.norm(projected - points1, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors


@torch.no_grad()
def row_predictions(feature0, feature1, chunk_size=256):
    flat0 = feature0.flatten(2).transpose(1, 2)[0]
    flat1 = feature1.flatten(2).transpose(1, 2)[0]
    predictions = []
    for start in range(0, flat0.shape[0], chunk_size):
        similarity = flat0[start : start + chunk_size] @ flat1.transpose(0, 1)
        predictions.append(similarity.argmax(dim=1))
    return torch.cat(predictions)


def update_crr(accumulator, prefix, physical_prediction, slim_prediction, correspondences, width):
    _, source, target = correspondences
    if source.numel() == 0:
        return
    for radius, label in ((0, "r0"), (1, "r1")):
        gt_x, gt_y = target % width, target // width

        def correct(prediction):
            selected = prediction[source]
            x, y = selected % width, selected // width
            return torch.maximum((x - gt_x).abs(), (y - gt_y).abs()) <= radius

        physical_correct = correct(physical_prediction)
        slim_correct = correct(slim_prediction)
        values = {
            "both_correct": physical_correct & slim_correct,
            "physical_only": physical_correct & ~slim_correct,
            "slim_only": ~physical_correct & slim_correct,
            "both_wrong": ~physical_correct & ~slim_correct,
        }
        accumulator[f"{prefix}/{label}/total"] += int(source.numel())
        for name, mask in values.items():
            accumulator[f"{prefix}/{label}/{name}"] += int(mask.sum())


def finalize_crr(accumulator, prefixes):
    result = {}
    for prefix in prefixes:
        result[prefix] = {}
        for label in ("r0", "r1"):
            base = f"{prefix}/{label}/"
            total = accumulator.get(base + "total", 0)
            if total == 0:
                continue
            values = {
                name: int(accumulator.get(base + name, 0))
                for name in ("both_correct", "physical_only", "slim_only", "both_wrong")
            }
            slim_wrong = values["physical_only"] + values["both_wrong"]
            values["CRR"] = values["physical_only"] / slim_wrong if slim_wrong else 0.0
            values["num_correspondences"] = int(total)
            result[prefix][label] = values
    return result


def aggregate_protocol(rows):
    if not rows:
        return {
            "num_pairs": 0,
            "NCM": 0.0,
            "Pre": 0.0,
            "SR": 0.0,
            "RMSE": 0.0,
            "mean_matches": 0.0,
            "mean_runtime_ms": 0.0,
        }
    return {
        "num_pairs": len(rows),
        "NCM": float(np.mean([row["NCM"] for row in rows])),
        "Pre": float(np.mean([row["Pre"] for row in rows])),
        "SR": float(np.mean([row["SR"] for row in rows])),
        "RMSE": float(np.mean([row["RMSE"] for row in rows])),
        "mean_matches": float(np.mean([row["matches"] for row in rows])),
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in rows])),
    }


def reference_optical_precision(path):
    if path is None:
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary.get("by_modality_pair", {}).get("optical-optical", {}).get("Pre")


def evaluate_multimodal(args, encoder, model_name, hparams, device):
    split = args.manifest_split or "test"
    dataset = RemoteSensingHomographyDataset(
        manifest_path=args.manifest_path,
        image_size=args.image_size,
        mode="val",
        max_samples=args.max_samples,
        homography_difficulty=0.0,
        left_identity=True,
        aug_variants=["mixed"],
        manifest_split=split,
        seed=args.seed,
    )
    slim = slim_config = None
    if not args.skip_slim_crr:
        slim, slim_config = load_official_slim(args.slim_ckpt_path, device)

    rows = []
    crr = defaultdict(int)
    crr_prefixes = defaultdict(set)
    diagnostic_rows = []
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="Multimodal Physical V1 evaluation"):
            item = dataset[index]
            record = dataset.rows[index]
            batch = make_batch(item, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output0 = encoder(batch["image0"])
            output1 = encoder(batch["image1"])
            source, target, _ = nearest_neighbors(
                output0["fused"], output1["fused"], chunk_size=args.chunk_size
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_ms = (time.perf_counter() - started) * 1000.0

            points0 = torch.stack(
                [
                    (source % output0["fused"].shape[-1]) + 0.5,
                    (source // output0["fused"].shape[-1]) + 0.5,
                ],
                dim=1,
            ).float() * 8.0
            points1 = torch.stack(
                [
                    (target % output1["fused"].shape[-1]) + 0.5,
                    (target // output1["fused"].shape[-1]) + 0.5,
                ],
                dim=1,
            ).float() * 8.0
            image0 = cv2.imread(record["image0"], cv2.IMREAD_GRAYSCALE)
            image1 = cv2.imread(record["image1"], cv2.IMREAD_GRAYSCALE)
            if image0 is None or image1 is None:
                raise FileNotFoundError(
                    f"Could not read {record['image0']} or {record['image1']}"
                )
            points0 = points0.cpu().numpy() * np.array(
                [image0.shape[1] / args.image_size, image0.shape[0] / args.image_size]
            )
            points1 = points1.cpu().numpy() * np.array(
                [image1.shape[1] / args.image_size, image1.shape[0] / args.image_size]
            )
            errors = reprojection_errors(points0, points1, dataset._read_gt_matrix(record))
            correct = errors <= args.correct_thr
            ncm = int(correct.sum())
            matches = int(len(errors))
            success = ncm >= args.success_ncm
            rmse = (
                float(np.sqrt(np.mean(errors[correct] ** 2)))
                if success and ncm
                else float(args.failed_rmse)
            )
            modality = (
                f"{record.get('modality0', 'unknown')}-{record.get('modality1', 'unknown')}"
            )
            rows.append(
                {
                    "index": index,
                    "id": record.get("id", ""),
                    "modality_pair": modality,
                    "matches": matches,
                    "NCM": ncm,
                    "Pre": ncm / matches if matches else 0.0,
                    "SR": int(success),
                    "RMSE": rmse,
                    "runtime_ms": runtime_ms,
                }
            )
            diagnostic_rows.extend([gate_diagnostics(output0), gate_diagnostics(output1)])

            if slim is not None:
                features0, features1 = slim.feature_backbone(
                    batch["image0"], batch["image1"]
                )
                coarse0 = features0[slim_config.MODEL.COARSE_SCALE_IDX]
                coarse1 = features1[slim_config.MODEL.COARSE_SCALE_IDX]
                slim_prediction = row_predictions(coarse0, coarse1, args.chunk_size)
                correspondences = coarse_homography_correspondences(batch, 8)
                for branch in BRANCHES:
                    physical_prediction = row_predictions(
                        output0[branch], output1[branch], args.chunk_size
                    )
                    for group in ("overall", modality):
                        prefix = f"{branch}/{group}"
                        update_crr(
                            crr,
                            prefix,
                            physical_prediction,
                            slim_prediction,
                            correspondences,
                            output1[branch].shape[-1],
                        )
                        crr_prefixes[branch].add(prefix)

    groups = defaultdict(list)
    for row in rows:
        groups[row["modality_pair"]].append(row)
    overall = aggregate_protocol(rows)
    by_modality = {
        name: aggregate_protocol(group) for name, group in sorted(groups.items())
    }
    optical_precision = by_modality.get("optical-optical", {}).get("Pre")
    if optical_precision is None:
        optical_precision = reference_optical_precision(args.retention_reference_summary)
    retention = {
        name: values["Pre"] / optical_precision
        for name, values in by_modality.items()
        if optical_precision and name != "optical-optical"
    }
    complementarity = None
    if slim is not None:
        complementarity = {
            branch: {
                prefix.split("/", 1)[1]: values
                for prefix, values in finalize_crr(
                    crr, sorted(crr_prefixes[branch])
                ).items()
            }
            for branch in BRANCHES
        }
    summary = {
        "evaluation": "multimodal",
        "protocol": {
            "matching": "fused coarse cosine mutual-nearest-neighbor; no RANSAC",
            "correct": f"original-coordinate reprojection error <= {args.correct_thr}px",
            "success": f"NCM >= {args.success_ncm}",
            "failed_rmse": args.failed_rmse,
        },
        "model": model_name,
        "checkpoint": str(args.ckpt_path),
        "manifest": str(args.manifest_path),
        "image_size": args.image_size,
        "trainable_parameters": count_trainable_parameters(encoder),
        "overall": overall,
        "by_modality_pair": by_modality,
        "optical_reference_precision": optical_precision,
        "cross_modal_retention": retention,
        "slim_coarse_complementarity": complementarity,
        "mean_diagnostics": {
            "scale_weights": np.mean(
                [row["scale_weights"]["mean"] for row in diagnostic_rows], axis=0
            ).tolist()
            if diagnostic_rows
            else [],
            "expert_weights": np.mean(
                [row["expert_weights"]["mean"] for row in diagnostic_rows], axis=0
            ).tolist()
            if diagnostic_rows
            else [],
            "confidence": float(
                np.mean([row["confidence"]["mean"] for row in diagnostic_rows])
            )
            if diagnostic_rows
            else 0.0,
        },
        "checkpoint_hparams": hparams,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()) if rows else ["index"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Physical V1 evaluation requires CUDA.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encoder, model_name, hparams = load_physical_v1_checkpoint(
        args.ckpt_path, device, requested_model=args.model
    )
    if args.evaluation == "synthetic":
        evaluate_synthetic(args, encoder, model_name, hparams, device)
    else:
        evaluate_multimodal(args, encoder, model_name, hparams, device)


if __name__ == "__main__":
    main()
