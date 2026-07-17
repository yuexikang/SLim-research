# Remote Dataset Archive

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
{
  "datasets": {
    "3MOS": {
      "by_mode": {
        "aligned_pairs": 38871,
        "single_synth": 16212
      },
      "by_pair_type": {
        "multimodal": 38871,
        "single": 16212
      },
      "by_split": {
        "train": 49535,
        "val": 5548
      },
      "details": {
        "paired_by_sensor": {
          "ALOS": 2650,
          "GF3": 14208,
          "RCM": 3745,
          "Radarsat": 7827,
          "SEN/SEN1_region1": 8941,
          "SEN/SEN1_region6": 1500
        },
        "single_synth_by_group": {
          "GF3": 1,
          "SEN/SEN1_region1": 16211
        },
        "unmatched_optical": {},
        "unmatched_sar": {
          "GF3": 1,
          "SEN/SEN1_region1": 16211
        }
      },
      "records": 55083,
      "root": "/home/disk1/Data/datasets/3MOS"
    },
    "GoogleEarth": {
      "by_mode": {
        "aligned_pairs": 10125
      },
      "by_pair_type": {
        "optical_optical": 10125
      },
      "by_split": {
        "test": 500,
        "train": 9000,
        "val": 625
      },
      "details": {
        "missing_current": {},
        "missing_past": {},
        "paired": {
          "test": 500,
          "train": 9000,
          "val": 625
        }
      },
      "records": 10125,
      "root": "/home/disk1/Data/datasets/GoogleEarth"
    },
    "jl1flight": {
      "by_mode": {
        "aligned_pairs": 1100
      },
      "by_pair_type": {
        "optical_optical": 1100
      },
      "by_split": {
        "test": 351,
        "train": 749
      },
      "details": {
        "extra_b_without_a": {},
        "missing_b_t0": {},
        "paired": {
          "test": 351,
          "train": 749
        }
      },
      "records": 1100,
      "root": "/home/disk1/Data/datasets/jl1flight"
    }
  },
  "datasets_root": "/home/disk1/Data/datasets",
  "outputs": {
    "3MOS": {
      "multimodal_pairs": "data/remote_archive/manifests/3MOS.jsonl",
      "single_images": "data/remote_archive/manifests/3MOS_single_images.jsonl"
    },
    "GoogleEarth": {
      "optical_optical_pairs": "data/remote_archive/manifests/GoogleEarth.jsonl"
    },
    "jl1flight": {
      "optical_optical_pairs": "data/remote_archive/manifests/jl1flight.jsonl"
    },
    "pure_groups": {
      "multimodal_pairs": "data/remote_archive/manifests/multimodal_pairs.jsonl",
      "optical_optical_pairs": "data/remote_archive/manifests/optical_optical_pairs.jsonl",
      "single_images": "data/remote_archive/manifests/single_images.jsonl",
      "test_optical_optical_pairs": "data/remote_archive/manifests/test_optical_optical_pairs.jsonl",
      "train_multimodal_pairs": "data/remote_archive/manifests/train_multimodal_pairs.jsonl",
      "train_optical_optical_pairs": "data/remote_archive/manifests/train_optical_optical_pairs.jsonl",
      "train_single_images": "data/remote_archive/manifests/train_single_images.jsonl",
      "val_multimodal_pairs": "data/remote_archive/manifests/val_multimodal_pairs.jsonl",
      "val_optical_optical_pairs": "data/remote_archive/manifests/val_optical_optical_pairs.jsonl",
      "val_single_images": "data/remote_archive/manifests/val_single_images.jsonl"
    },
    "single_views_by_modality": {
      "optical": "data/remote_archive/manifests/optical_single_images.jsonl",
      "sar": "data/remote_archive/manifests/sar_single_images.jsonl",
      "test_optical": "data/remote_archive/manifests/test_optical_single_images.jsonl",
      "train_optical": "data/remote_archive/manifests/train_optical_single_images.jsonl",
      "train_sar": "data/remote_archive/manifests/train_sar_single_images.jsonl",
      "val_optical": "data/remote_archive/manifests/val_optical_single_images.jsonl",
      "val_sar": "data/remote_archive/manifests/val_sar_single_images.jsonl"
    }
  },
  "single_view_by_modality": {
    "optical": 61321,
    "sar": 55083
  },
  "single_view_records": 116404,
  "total_by_mode": {
    "aligned_pairs": 50096,
    "single_synth": 16212
  },
  "total_by_pair_type": {
    "multimodal": 38871,
    "optical_optical": 11225,
    "single": 16212
  },
  "total_by_split": {
    "test": 851,
    "train": 59284,
    "val": 6173
  },
  "total_records": 66308
}
```
