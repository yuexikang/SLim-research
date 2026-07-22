#!/usr/bin/env python3
"""Build leakage-free GoogleEarth single-image train/validation manifests.

The source manifest contains paired views. Pair IDs are stratified by subset,
split deterministically, and only then expanded to single-image records so two
views from the same pair can never cross the train/validation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("data/remote_archive/manifests/train_GoogleEarth.jsonl")
DEFAULT_TRAIN = Path("data/remote_archive/manifests/train_GoogleEarth_single.jsonl")
DEFAULT_VAL = Path("data/remote_archive/manifests/val_GoogleEarth_single.jsonl")
DEFAULT_SUMMARY = Path(
    "data/remote_archive/manifests/GoogleEarth_single_split_summary.json"
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()


def allocate_validation_counts(
    subset_sizes: dict[str, int], val_ratio: float
) -> dict[str, int]:
    target = round(sum(subset_sizes.values()) * val_ratio)
    exact = {subset: size * val_ratio for subset, size in subset_sizes.items()}
    counts = {subset: math.floor(value) for subset, value in exact.items()}
    remainder = target - sum(counts.values())
    order = sorted(
        subset_sizes,
        key=lambda subset: (-(exact[subset] - counts[subset]), subset),
    )
    for subset in order[:remainder]:
        counts[subset] += 1
    return counts


def split_pair_ids(rows: list[dict], seed: int, val_ratio: float) -> set[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("subset", "")].append(row)
    val_counts = allocate_validation_counts(
        {subset: len(group) for subset, group in groups.items()}, val_ratio
    )
    validation_ids: set[str] = set()
    for subset, group in groups.items():
        ranked = sorted(group, key=lambda row: stable_rank(seed, row["id"]))
        validation_ids.update(row["id"] for row in ranked[: val_counts[subset]])
    return validation_ids


def single_view_id(image: str) -> str:
    return hashlib.sha1(image.encode("utf-8")).hexdigest()[:12]


def expand_pair(row: dict, split: str) -> list[dict]:
    records = []
    for view in (0, 1):
        image = row[f"image{view}"]
        records.append(
            {
                "id": f"single/optical/{single_view_id(image)}",
                "dataset": "GoogleEarth",
                "subset": row.get("subset", ""),
                "mode": "single_synth",
                "split": split,
                "image": image,
                "modality": "optical",
                "source_modality": row[f"modality{view}"],
                "source_pair_id": row["id"],
                "source_pair_type": row.get("pair_type", "optical_optical"),
                "source_view": view,
                "gt": None,
                "notes": "GoogleEarth single image for online synthetic homography training",
            }
        )
    return records


def validate_source(rows: list[dict]) -> None:
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest contains duplicate pair IDs.")
    for row in rows:
        if row.get("dataset") != "GoogleEarth":
            raise ValueError(f"Unexpected dataset in {row.get('id')}: {row.get('dataset')}")
        if row.get("pair_type") != "optical_optical":
            raise ValueError(f"Unexpected pair type in {row.get('id')}")
        if not row.get("image0") or not row.get("image1"):
            raise ValueError(f"Missing paired image path in {row.get('id')}")


def duplicate_count(rows: list[dict], key: str) -> int:
    values = [row[key] for row in rows]
    return len(values) - len(set(values))


def distribution(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = stream.read(1)
            if not byte:
                return None
            if byte != b"\xff":
                continue
            while byte == b"\xff":
                byte = stream.read(1)
            marker = byte[0]
            if marker in sof_markers:
                length = struct.unpack(">H", stream.read(2))[0]
                payload = stream.read(length - 2)
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            raw_length = stream.read(2)
            if len(raw_length) != 2:
                return None
            length = struct.unpack(">H", raw_length)[0]
            stream.seek(length - 2, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train_output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val_output", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--summary_output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rows = read_jsonl(args.source)
    validate_source(rows)
    validation_ids = split_pair_ids(rows, args.seed, args.val_ratio)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    train_pair_ids: set[str] = set()
    val_pair_ids: set[str] = set()
    for row in rows:
        if row["id"] in validation_ids:
            val_rows.extend(expand_pair(row, "val"))
            val_pair_ids.add(row["id"])
        else:
            train_rows.extend(expand_pair(row, "train"))
            train_pair_ids.add(row["id"])

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.val_output, val_rows)
    train_images = {row["image"] for row in train_rows}
    val_images = {row["image"] for row in val_rows}
    all_rows = train_rows + val_rows
    missing_paths = [row["image"] for row in all_rows if not Path(row["image"]).is_file()]
    dimensions = Counter()
    unreadable_headers = []
    if not missing_paths:
        for image in sorted({row["image"] for row in all_rows}):
            size = jpeg_dimensions(Path(image))
            if size is None:
                unreadable_headers.append(image)
            else:
                dimensions[f"{size[0]}x{size[1]}"] += 1
    summary = {
        "version": "Physical Encoder V2.1.2",
        "source_manifest": str(args.source),
        "source_sha256": sha256(args.source),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source_pair_count": len(rows),
        "train_pair_count": len(train_pair_ids),
        "val_pair_count": len(val_pair_ids),
        "train_single_count": len(train_rows),
        "val_single_count": len(val_rows),
        "train_subset_distribution": distribution(train_rows, "subset"),
        "val_subset_distribution": distribution(val_rows, "subset"),
        "train_source_modality_distribution": distribution(
            train_rows, "source_modality"
        ),
        "val_source_modality_distribution": distribution(val_rows, "source_modality"),
        "duplicate_train_ids": duplicate_count(train_rows, "id"),
        "duplicate_val_ids": duplicate_count(val_rows, "id"),
        "duplicate_train_images": duplicate_count(train_rows, "image"),
        "duplicate_val_images": duplicate_count(val_rows, "image"),
        "pair_overlap": len(train_pair_ids & val_pair_ids),
        "image_overlap": len(train_images & val_images),
        "missing_path_count": len(missing_paths),
        "image_dimensions": dict(sorted(dimensions.items())),
        "minimum_image_side": min(
            (min(map(int, size.split("x"))) for size in dimensions), default=None
        ),
        "unreadable_image_header_count": len(unreadable_headers),
        "train_manifest": str(args.train_output),
        "train_sha256": sha256(args.train_output),
        "val_manifest": str(args.val_output),
        "val_sha256": sha256(args.val_output),
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if missing_paths:
        raise FileNotFoundError(f"Found {len(missing_paths)} missing image paths.")
    if unreadable_headers:
        raise ValueError(f"Found {len(unreadable_headers)} unreadable JPEG headers.")


if __name__ == "__main__":
    main()
