"""Build separate GT manifests for the two official SwinMatcher test collections."""

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/disk1/Data/datasets/SwinMatcher/SwinMatcher_test_datasets"),
    )
    parser.add_argument(
        "--output_dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/remote_archive/manifests"),
            Path("/home/disk1/Data/remote_archive/manifests"),
        ],
    )
    return parser.parse_args()


def normalize_modality(token):
    token = token.lower()
    aliases = {
        "rgb": "optical",
        "opt": "optical",
        "ir": "infrared",
        "day": "optical_day",
        "night": "optical_night",
    }
    return aliases.get(token, token)


def parse_modalities(subset):
    text = subset.replace("-", "_").replace(" ", "_")
    parts = [part for part in text.split("_") if part and not part.isdigit()]
    if parts and parts[0].lower().startswith("scene"):
        parts = parts[1:]
    if len(parts) < 2:
        return "unknown", "unknown"
    return normalize_modality(parts[0]), normalize_modality(parts[1])


def build_collection_records(collection_dir):
    records = []
    skipped = 0
    collection = collection_dir.name
    for subset_dir in sorted(path for path in collection_dir.iterdir() if path.is_dir()):
        modality0, modality1 = parse_modalities(subset_dir.name)
        pair_type = "optical_optical" if modality0 == modality1 == "optical" else "multimodal"
        for pair_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
            images = sorted(
                path for path in pair_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            )
            if len(images) < 3:
                skipped += 1
                continue
            image0 = images[0]
            # The source layout contains 00.PNG, an auxiliary 00_.PNG, then targets 01..10.
            for image1 in images[2:]:
                gt = pair_dir / f"{image1.stem}.npy"
                if not gt.is_file():
                    skipped += 1
                    continue
                records.append(
                    {
                        "dataset": "SwinMatcher",
                        "id": f"swinmatcher/{collection}/{subset_dir.name}/{pair_dir.name}/{image1.stem}",
                        "split": "test",
                        "mode": "gt_pairs",
                        "pair_type": pair_type,
                        "image0": str(image0),
                        "image1": str(image1),
                        "gt": str(gt),
                        "gt_format": "npy",
                        "gt_direction": "0to1",
                        "modality0": modality0,
                        "modality1": modality1,
                        "collection": collection,
                        "scene": collection,
                        "subset": subset_dir.name,
                    }
                )
    return records, skipped


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    args = parse_args()
    collections = {
        "proposed": args.data_root / "proposed_test_dataset",
        "expanded_MRSI": args.data_root / "expanded_MRSI_dataset",
    }
    all_records = []
    for name, collection_dir in collections.items():
        if not collection_dir.is_dir():
            raise FileNotFoundError(collection_dir)
        records, skipped = build_collection_records(collection_dir)
        all_records.extend(records)
        summary = {
            "collection": collection_dir.name,
            "records": len(records),
            "skipped": skipped,
            "by_modality_pair": dict(
                sorted(Counter(f"{row['modality0']}-{row['modality1']}" for row in records).items())
            ),
        }
        for output_dir in args.output_dirs:
            write_jsonl(output_dir / f"test_SwinMatcher_{name}_gt.jsonl", records)
            (output_dir / f"test_SwinMatcher_{name}_gt_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(json.dumps(summary, ensure_ascii=False))

    for output_dir in args.output_dirs:
        write_jsonl(output_dir / "test_SwinMatcher_all_gt.jsonl", all_records)


if __name__ == "__main__":
    main()
