#!/usr/bin/env python3
"""重新生成 jl1flight 的遥感配准索引。

这个脚本读取 jl1flight 官方/预处理好的 index/*.npz：
- image_paths: 每张 patch 图像的相对路径
- affine_matrices: 每张 patch 到公共坐标系的 2x3 仿射矩阵
- pair_infos: 可训练/测试的成对图像下标

输出统一的 SLiM RemoteSensing manifest，文件名带 _gt：
- mode=gt_pairs
- image0/image1 为真实成对影像
- gt 为 image0 像素坐标到 image1 像素坐标的 3x3 单应矩阵 .txt

gt txt 规范：
- 只写 3x3 矩阵数字
- 3 行，每行 3 个浮点数，空格分隔
- 不写注释、不写路径、不写额外字段
- txt 放在原 jl1flight 数据目录的 affine_pairs_train / affine_pairs_test 下

注意：不能假设 b_<x>_<y>_t0 和 a_<x>_<y> 对齐。真值应由 index 里的
affine_matrices 计算，而不是写成 aligned_pairs。
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build clean jl1flight gt-pair manifests.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/disk1/Data/datasets/jl1flight"),
        help="jl1flight dataset root.",
    )
    parser.add_argument(
        "--archive_root",
        type=Path,
        default=Path("data/remote_archive"),
        help="Remote archive root under the SLiM repo.",
    )
    return parser.parse_args()


def to_homogeneous(matrix_2x3):
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError(f"Expected a 2x3 affine matrix, got {matrix.shape}")
    return np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float64)])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_split(dataset_root, archive_root, split_name, npz_path, image_root, gt_dir):
    data = np.load(npz_path, allow_pickle=True)
    image_paths = data["image_paths"]
    affine_matrices = data["affine_matrices"]
    pair_infos = data["pair_infos"]

    rows = []
    for pair_idx, (idx0, idx1) in enumerate(pair_infos):
        rel0 = str(image_paths[int(idx0)])
        rel1 = str(image_paths[int(idx1)])
        image0 = image_root / rel0
        image1 = image_root / rel1
        if not image0.is_file():
            raise FileNotFoundError(image0)
        if not image1.is_file():
            raise FileNotFoundError(image1)

        # jl1flight stores the affine transform associated with each image.
        # For a pair (a, b_t), a is identity and b_t's matrix maps a -> b_t.
        # For the general case, image0 -> image1 is M1 @ inv(M0).
        H0 = to_homogeneous(affine_matrices[int(idx0)])
        H1 = to_homogeneous(affine_matrices[int(idx1)])
        H_0to1 = H1 @ np.linalg.inv(H0)
        H_0to1 = H_0to1 / H_0to1[2, 2]

        gt_path = image1.with_name(f"{image1.stem}_H_0to1.txt")
        np.savetxt(gt_path, H_0to1.astype(np.float64), fmt="%.10f")

        stem0 = Path(rel0).stem
        stem1 = Path(rel1).stem
        group_id = stem0[2:] if stem0.startswith("a_") else stem0
        rows.append(
            {
                "dataset": "jl1flight",
                "id": f"jl1flight/{split_name}/{pair_idx:06d}",
                "split": split_name,
                "subset": rel0.split("/")[0],
                "mode": "gt_pairs",
                "pair_type": "optical_optical",
                "modality0": "optical_a",
                "modality1": "optical_b",
                "group_id": group_id,
                "image0": str(image0),
                "image1": str(image1),
                "gt": str(gt_path),
                "gt_direction": "0to1",
                "source_index": str(npz_path),
                "source_image0_index": int(idx0),
                "source_image1_index": int(idx1),
                "notes": (
                    "gt is image0->image1, computed as "
                    "M_image1 @ inv(M_image0) from jl1flight index affine_matrices"
                ),
            }
        )

    return rows


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    archive_root = args.archive_root

    train_rows = build_split(
        dataset_root=dataset_root,
        archive_root=archive_root,
        split_name="train",
        npz_path=dataset_root / "index/scene_info/affine_pairs_train.npz",
        image_root=dataset_root / "train",
        gt_dir=None,
    )
    test_rows = build_split(
        dataset_root=dataset_root,
        archive_root=archive_root,
        split_name="test",
        npz_path=dataset_root / "index/scene_info_val/affine_pairs_test.npz",
        image_root=dataset_root / "test",
        gt_dir=None,
    )

    manifest_dir = archive_root / "manifests"
    write_jsonl(manifest_dir / "train_jl1flight_gt.jsonl", train_rows)
    write_jsonl(manifest_dir / "test_jl1flight_gt.jsonl", test_rows)
    write_jsonl(manifest_dir / "jl1flight_gt.jsonl", train_rows + test_rows)

    print(f"train rows: {len(train_rows)}")
    print(f"test rows: {len(test_rows)}")
    print(f"combined rows: {len(train_rows) + len(test_rows)}")
    print(f"manifest dir: {manifest_dir}")
    print("gt txt location: original jl1flight affine_pairs_train / affine_pairs_test folders")


if __name__ == "__main__":
    main()
