#!/usr/bin/env python3
# 作用：校验统一遥感 JSONL manifest 的字段完整性和文件路径可用性。

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    errors: list[str] = []
    counts = Counter()
    by_dataset = Counter()
    by_split = Counter()
    by_mode = Counter()
    by_pair_type = Counter()

    with args.manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid json: {exc}")
                continue

            counts["rows"] += 1
            by_dataset[row.get("dataset", "<missing>")] += 1
            by_split[row.get("split", "<missing>")] += 1
            by_mode[row.get("mode", "<missing>")] += 1
            if "pair_type" in row:
                by_pair_type[row["pair_type"]] += 1

            required = ["id", "dataset", "mode", "split"]
            missing = [k for k in required if k not in row]
            if missing:
                errors.append(f"line {line_no}: missing required keys {missing}")
                continue

            mode = row["mode"]
            if mode in {"aligned_pairs", "gt_pairs"}:
                required = ["image0", "image1", "modality0", "modality1", "pair_type"]
                if mode == "gt_pairs":
                    required += ["gt", "gt_direction"]
                missing = [k for k in required if k not in row]
                if missing:
                    errors.append(f"line {line_no}: {mode} missing {missing}")
                    continue
                if row["pair_type"] not in {"optical_optical", "multimodal"}:
                    errors.append(f"line {line_no}: invalid pair_type: {row['pair_type']}")
                for key in ("image0", "image1"):
                    if not Path(row[key]).is_file():
                        errors.append(f"line {line_no}: {key} does not exist: {row[key]}")
                if mode == "gt_pairs" and not Path(row["gt"]).is_file():
                    errors.append(f"line {line_no}: gt does not exist: {row['gt']}")
            elif mode == "single_synth":
                required = ["image", "modality"]
                missing = [k for k in required if k not in row]
                if missing:
                    errors.append(f"line {line_no}: single_synth missing {missing}")
                    continue
                if not Path(row["image"]).is_file():
                    errors.append(f"line {line_no}: image does not exist: {row['image']}")
            else:
                errors.append(f"line {line_no}: unsupported mode: {mode}")

            if len(errors) >= args.max_errors:
                break

    if len(by_mode) > 1:
        errors.append(f"manifest is not pure by mode: {dict(by_mode)}")
    if by_pair_type and len(by_pair_type) > 1:
        errors.append(f"manifest is not pure by pair_type: {dict(by_pair_type)}")
    if by_pair_type and by_mode.get("single_synth", 0):
        errors.append("manifest mixes paired samples and single_synth samples")

    print("rows:", counts["rows"])
    print("by_dataset:", dict(sorted(by_dataset.items())))
    print("by_split:", dict(sorted(by_split.items())))
    print("by_mode:", dict(sorted(by_mode.items())))
    if by_pair_type:
        print("by_pair_type:", dict(sorted(by_pair_type.items())))

    if errors:
        print("errors:")
        for err in errors:
            print("-", err)
        raise SystemExit(1)
    print("manifest ok")


if __name__ == "__main__":
    main()
