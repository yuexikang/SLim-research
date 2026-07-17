#!/usr/bin/env python3
# 作用：把现有遥感数据集整理成统一 JSONL manifest；只索引原始文件路径，不移动或复制数据。

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def split_by_hash(key: str, val_ratio: float = 0.1) -> str:
    # Stable split without depending on Python's randomized hash.
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return "val" if bucket < int(val_ratio * 1000) else "train"


def numeric_id_from_stem(stem: str, prefix: str) -> str | None:
    m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", stem)
    return m.group(1) if m else None


def build_3mos(root: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {
        "paired_by_sensor": {},
        "single_synth_by_group": {},
        "unmatched_optical": {},
        "unmatched_sar": {},
    }

    for sensor_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sensor = sensor_dir.name
        opt_root = sensor_dir / "opt"
        sar_root = sensor_dir / "sar"

        # SEN keeps regions as another layer; pair only when a region has both opt and sar.
        if sensor == "SEN":
            region_dirs = sorted(p for p in sensor_dir.iterdir() if p.is_dir())
            for region_dir in region_dirs:
                group = f"{sensor}/{region_dir.name}"
                records.extend(build_3mos_group(region_dir, "3MOS", group, stats))
            continue

        if opt_root.exists() or sar_root.exists():
            records.extend(build_3mos_group(sensor_dir, "3MOS", sensor, stats))

    return records, stats


def build_3mos_group(root: Path, dataset: str, group: str, stats: dict) -> list[dict]:
    records: list[dict] = []
    opt_root = root / "opt"
    sar_root = root / "sar"
    opt_by_id: dict[str, Path] = {}
    sar_by_id: dict[str, Path] = {}

    if opt_root.exists():
        for p in image_files(opt_root):
            image_id = numeric_id_from_stem(p.stem, "opt")
            if image_id is not None:
                opt_by_id[image_id] = p
    if sar_root.exists():
        for p in image_files(sar_root):
            image_id = numeric_id_from_stem(p.stem, "sar")
            if image_id is not None:
                sar_by_id[image_id] = p

    common = sorted(set(opt_by_id) & set(sar_by_id), key=lambda x: int(x))
    for image_id in common:
        split = split_by_hash(f"{group}:{image_id}")
        records.append(
            {
                "id": f"3mos/{group}/{image_id}",
                "dataset": dataset,
                "subset": group,
                "mode": "aligned_pairs",
                "split": split,
                "image0": str(opt_by_id[image_id]),
                "image1": str(sar_by_id[image_id]),
                "modality0": "optical",
                "modality1": "sar",
                "pair_type": "multimodal",
                "aligned": True,
                "gt": None,
                "notes": "paired by matching opt_<id> and sar_<id> filenames",
            }
        )

    opt_only = sorted(set(opt_by_id) - set(sar_by_id), key=lambda x: int(x))
    sar_only = sorted(set(sar_by_id) - set(opt_by_id), key=lambda x: int(x))
    stats["paired_by_sensor"][group] = len(common)
    if opt_only:
        stats["unmatched_optical"][group] = len(opt_only)
    if sar_only:
        stats["unmatched_sar"][group] = len(sar_only)

    for image_id in opt_only:
        split = split_by_hash(f"{group}:opt:{image_id}")
        records.append(single_record("3MOS", group, image_id, opt_by_id[image_id], "optical", split))
    for image_id in sar_only:
        split = split_by_hash(f"{group}:sar:{image_id}")
        records.append(single_record("3MOS", group, image_id, sar_by_id[image_id], "sar", split))
    if opt_only or sar_only:
        stats["single_synth_by_group"][group] = len(opt_only) + len(sar_only)

    return records


def single_record(dataset: str, subset: str, image_id: str, image: Path, modality: str, split: str) -> dict:
    return {
        "id": f"{dataset.lower()}/{subset}/{modality}/{image_id}",
        "dataset": dataset,
        "subset": subset,
        "mode": "single_synth",
        "split": split,
        "image": str(image),
        "modality": modality,
        "gt": None,
        "notes": "single image source for online synthetic homography training",
    }


def build_googleearth(root: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {"paired": {}, "missing_current": {}, "missing_past": {}}

    for split_name, split_dir in [("train", "Train"), ("val", "Val")]:
        past_root = root / "training_data" / "past" / split_dir
        current_root = root / "training_data" / "current" / split_dir
        rows, missing0, missing1 = pair_same_names(
            past_root,
            current_root,
            dataset="GoogleEarth",
            subset=f"training_data/{split_dir}",
            split=split_name,
            modality0="optical_past",
            modality1="optical_current",
        )
        records.extend(rows)
        stats["paired"][split_name] = len(rows)
        if missing0:
            stats["missing_past"][split_name] = missing0
        if missing1:
            stats["missing_current"][split_name] = missing1

    source_root = root / "evaluation_data" / "source"
    target_root = root / "evaluation_data" / "target"
    source = {p.stem.replace("source_", ""): p for p in image_files(source_root)}
    target = {p.stem.replace("target_", ""): p for p in image_files(target_root)}
    common = sorted(set(source) & set(target))
    for image_id in common:
        records.append(
            {
                "id": f"googleearth/evaluation/{image_id}",
                "dataset": "GoogleEarth",
                "subset": "evaluation_data",
                "mode": "aligned_pairs",
                "split": "test",
                "image0": str(source[image_id]),
                "image1": str(target[image_id]),
                "modality0": "optical_source",
                "modality1": "optical_target",
                "pair_type": "optical_optical",
                "aligned": True,
                "gt": None,
                "notes": "paired by source_<id> and target_<id> filenames",
            }
        )
    stats["paired"]["test"] = len(common)
    return records, stats


def pair_same_names(
    root0: Path,
    root1: Path,
    dataset: str,
    subset: str,
    split: str,
    modality0: str,
    modality1: str,
) -> tuple[list[dict], int, int]:
    files0 = {p.name: p for p in image_files(root0)}
    files1 = {p.name: p for p in image_files(root1)}
    common = sorted(set(files0) & set(files1))
    rows = [
        {
            "id": f"{dataset.lower()}/{subset}/{Path(name).stem}",
            "dataset": dataset,
            "subset": subset,
        "mode": "aligned_pairs",
            "split": split,
            "image0": str(files0[name]),
            "image1": str(files1[name]),
            "modality0": modality0,
            "modality1": modality1,
            "pair_type": infer_pair_type(modality0, modality1),
            "aligned": True,
            "gt": None,
            "notes": "paired by identical filename",
        }
        for name in common
    ]
    return rows, len(set(files0) - set(files1)), len(set(files1) - set(files0))


def build_jl1flight(root: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {"paired": {}, "missing_b_t0": {}, "extra_b_without_a": {}}
    for split, rel in [("train", "train/affine_pairs_train"), ("test", "test/affine_pairs_test")]:
        folder = root / rel
        a_files: dict[tuple[str, str], Path] = {}
        b0_files: dict[tuple[str, str], Path] = {}
        all_b_keys: set[tuple[str, str]] = set()

        for p in image_files(folder):
            ma = re.fullmatch(r"a_(\d+)_(\d+)", p.stem)
            mb = re.fullmatch(r"b_(\d+)_(\d+)_t(\d+)", p.stem)
            if ma:
                a_files[(ma.group(1), ma.group(2))] = p
            elif mb:
                key = (mb.group(1), mb.group(2))
                all_b_keys.add(key)
                if mb.group(3) == "0":
                    b0_files[key] = p

        common = sorted(set(a_files) & set(b0_files), key=lambda xy: (int(xy[0]), int(xy[1])))
        for x, y in common:
            records.append(
                {
                    "id": f"jl1flight/{split}/{x}_{y}",
                    "dataset": "jl1flight",
                    "subset": rel,
                    "mode": "aligned_pairs",
                    "split": split,
                    "image0": str(a_files[(x, y)]),
                    "image1": str(b0_files[(x, y)]),
                    "modality0": "optical_a",
                    "modality1": "optical_b_t0",
                    "pair_type": "optical_optical",
                    "aligned": True,
                    "gt": None,
                    "notes": "user-defined pairing: a_<x>_<y> with first B image b_<x>_<y>_t0; other B_t* files are not used",
                    "group_id": f"{x}_{y}",
                }
            )

        stats["paired"][split] = len(common)
        missing_b0 = set(a_files) - set(b0_files)
        extra_b = all_b_keys - set(a_files)
        if missing_b0:
            stats["missing_b_t0"][split] = len(missing_b0)
        if extra_b:
            stats["extra_b_without_a"][split] = len(extra_b)

    return records, stats


def modality_family(modality: str) -> str:
    if modality.startswith("optical"):
        return "optical"
    return modality.split("_", 1)[0]


def infer_pair_type(modality0: str, modality1: str) -> str:
    fam0 = modality_family(modality0)
    fam1 = modality_family(modality1)
    if fam0 == "optical" and fam1 == "optical":
        return "optical_optical"
    return "multimodal"


def write_readme(path: Path, summary: dict) -> None:
    text = f"""# Remote Dataset Archive

这个目录是遥感匹配数据的统一归档索引。原始数据没有被移动、复制或改名；JSONL 里的路径直接指向 `/home/disk1/Data/datasets/...`。

## 字段约定

- 每个 `.jsonl` 保持纯净：要么只放 `optical_optical` 成对，要么只放 `multimodal` 成对，要么只放 `single_synth` 单张图。
- `mode`: 训练数据模式，当前包含 `aligned_pairs`、`gt_pairs` 和 `single_synth`。
- `pair_type`: 成对样本类型，当前包含 `optical_optical` 和 `multimodal`；单张图没有该字段。
- `image0` / `image1`: 成对样本的两张图。
- `image`: 单图自监督样本。
- `modality0` / `modality1` / `modality`: 模态标注，例如 `optical`、`sar`。
- `gt`: 当前三个数据集都按无真值归档，所以为 `null`。
- `gt_pairs` 额外需要 `gt` 和 `gt_direction`；`gt` 可为 `.npy` 或文本矩阵，支持 3x3 homography 或 2x3 affine。
- `split`: `train`、`val` 或 `test`。

## 数据集规则

- `3MOS.jsonl`: 只放 `opt_<id>` 与 `sar_<id>` 配成的多模态成对样本；无法配对的单模态图单独写入 `3MOS_single_images.jsonl`。无官方 split 的部分使用稳定 90/10 规则切成 train/val。
- `GoogleEarth`: `training_data/past/<split>` 与 `training_data/current/<split>` 按相同文件名配对；`evaluation_data/source` 与 `target` 按编号配对为 test。
- `jl1flight`: 按你的说明作为无真值成对数据，使用 `a_<x>_<y>.png` 和第一张 `b_<x>_<y>_t0.png` 配对，其它 `b_*_t1...t9` 暂不纳入训练 manifest。

## 统计

```json
{json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    path.write_text(text, encoding="utf-8")


def pure_suffix(row: dict) -> str:
    if row["mode"] == "single_synth":
        return "single_images"
    pair_type = row.get("pair_type")
    mode_prefix = "gt_" if row["mode"] == "gt_pairs" else ""
    if pair_type == "optical_optical":
        return f"{mode_prefix}optical_optical_pairs"
    if pair_type == "multimodal":
        return f"{mode_prefix}multimodal_pairs"
    raise ValueError(f"Unsupported row kind: mode={row.get('mode')} pair_type={pair_type}")


def write_dataset_manifests(manifest_dir: Path, dataset_name: str, records: list[dict]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[pure_suffix(row)].append(row)

    for suffix, rows in sorted(groups.items()):
        if dataset_name == "3MOS" and suffix == "multimodal_pairs":
            # Keep the user's expected 3MOS.jsonl name, but make it pure.
            name = "3MOS.jsonl"
        elif suffix == "optical_optical_pairs" and dataset_name in {"GoogleEarth", "jl1flight"}:
            # These datasets currently only contribute optical-optical pairs.
            name = f"{dataset_name}.jsonl"
        else:
            name = f"{dataset_name}_{suffix}.jsonl"
        path = manifest_dir / name
        write_jsonl(path, rows)
        outputs[suffix] = str(path)
    return outputs


def write_pure_group_manifests(manifest_dir: Path, records: list[dict]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(pure_suffix(row), "all")].append(row)
        groups[(pure_suffix(row), row["split"])].append(row)

    for (suffix, split), rows in sorted(groups.items()):
        name = f"{suffix}.jsonl" if split == "all" else f"{split}_{suffix}.jsonl"
        path = manifest_dir / name
        write_jsonl(path, rows)
        outputs[f"{split}_{suffix}" if split != "all" else suffix] = str(path)
    return outputs


def single_view_id(path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return digest


def build_single_view_records(records: list[dict]) -> list[dict]:
    """Create pure single-image manifests from every available view.

    These derived records are useful for single-image synthetic homography
    training while keeping pair manifests pure. Records are deduplicated by
    image path.
    """
    by_path: dict[str, dict] = {}
    for row in records:
        if row["mode"] == "single_synth":
            image = row["image"]
            modality = row["modality"]
            candidate = {
                "id": f"single/{modality_family(modality)}/{single_view_id(image)}",
                "dataset": row["dataset"],
                "subset": row.get("subset", ""),
                "mode": "single_synth",
                "split": row["split"],
                "image": image,
                "modality": modality_family(modality),
                "source_modality": modality,
                "gt": None,
                "notes": "single image source from explicit single_synth manifest",
            }
            by_path.setdefault(image, candidate)
            continue

        for view_idx in (0, 1):
            image = row[f"image{view_idx}"]
            modality = row[f"modality{view_idx}"]
            family = modality_family(modality)
            candidate = {
                "id": f"single/{family}/{single_view_id(image)}",
                "dataset": row["dataset"],
                "subset": row.get("subset", ""),
                "mode": "single_synth",
                "split": row["split"],
                "image": image,
                "modality": family,
                "source_modality": modality,
                "source_pair_id": row["id"],
                "source_pair_type": row.get("pair_type"),
                "source_view": view_idx,
                "gt": None,
                "notes": "single image source derived from a pure paired manifest",
            }
            by_path.setdefault(image, candidate)
    return sorted(by_path.values(), key=lambda r: (r["modality"], r["dataset"], r["split"], r["image"]))


def write_single_view_manifests(manifest_dir: Path, records: list[dict]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        modality = modality_family(row["modality"])
        groups[(modality, "all")].append(row)
        groups[(modality, row["split"])].append(row)

    for (modality, split), rows in sorted(groups.items()):
        name = (
            f"{modality}_single_images.jsonl"
            if split == "all"
            else f"{split}_{modality}_single_images.jsonl"
        )
        path = manifest_dir / name
        write_jsonl(path, rows)
        outputs[f"{split}_{modality}" if split != "all" else modality] = str(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=Path, default=Path("/home/disk1/Data/datasets"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/remote_archive"))
    args = parser.parse_args()

    out_dir = args.out_dir
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale mixed manifests from earlier archive versions. Keeping them
    # around is risky because they violate the "one file, one sample kind" rule.
    for stale in manifest_dir.glob("*.jsonl"):
        stale.unlink()

    builders = {
        "3MOS": build_3mos,
        "GoogleEarth": build_googleearth,
        "jl1flight": build_jl1flight,
    }
    all_records: list[dict] = []
    summary: dict = {"datasets_root": str(args.datasets_root), "datasets": {}, "outputs": {}}

    for name, builder in builders.items():
        root = args.datasets_root / name
        records, stats = builder(root)
        all_records.extend(records)
        dataset_outputs = write_dataset_manifests(manifest_dir, name, records)
        summary["datasets"][name] = {
            "root": str(root),
            "records": len(records),
            "by_mode": dict(Counter(r["mode"] for r in records)),
            "by_pair_type": dict(Counter(r.get("pair_type", "single") for r in records)),
            "by_split": dict(Counter(r["split"] for r in records)),
            "details": stats,
        }
        summary["outputs"][name] = dataset_outputs

    summary["outputs"]["pure_groups"] = write_pure_group_manifests(manifest_dir, all_records)
    single_view_records = build_single_view_records(all_records)
    summary["outputs"]["single_views_by_modality"] = write_single_view_manifests(
        manifest_dir, single_view_records
    )
    summary["single_view_records"] = len(single_view_records)
    summary["single_view_by_modality"] = dict(Counter(r["modality"] for r in single_view_records))

    summary["total_records"] = len(all_records)
    summary["total_by_mode"] = dict(Counter(r["mode"] for r in all_records))
    summary["total_by_pair_type"] = dict(Counter(r.get("pair_type", "single") for r in all_records))
    summary["total_by_split"] = dict(Counter(r["split"] for r in all_records))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
