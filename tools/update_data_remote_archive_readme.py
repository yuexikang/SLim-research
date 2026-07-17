from pathlib import Path


README = Path("/home/disk1/Data/remote_archive/README.md")

CONTENT = """# Remote Dataset Archive

这个目录是遥感匹配数据的统一归档索引。原始影像不移动、不复制、不改名；JSONL 里的路径直接指向 `/home/disk1/Data/datasets/...`。

## Manifest 纯净原则

- 一个 `.jsonl` 只放一种数据形态：`optical_optical` 成对、`multimodal` 成对、或 `single_synth` 单张图。
- 有真实几何关系的数据集，manifest 文件名必须带 `_gt`，例如 `train_jl1flight_gt.jsonl`。
- 无真实几何关系、只能假设已对齐的数据，才使用 `aligned_pairs`。
- 单张图自监督数据使用 `single_synth`，训练/验证时在线生成随机 homography，不在本地保存扰动图。

## 字段约定

- `mode`: 训练数据模式，当前包含 `aligned_pairs`、`gt_pairs` 和 `single_synth`。
- `pair_type`: 成对样本类型，例如 `optical_optical`、`multimodal`。
- `image0` / `image1`: 成对样本的两张图。
- `image`: 单图自监督样本。
- `modality0` / `modality1` / `modality`: 模态标注，例如 `optical`、`sar`。
- `gt`: 只在 `gt_pairs` 中使用，指向真实 3x3 矩阵 txt。
- `gt_direction`: 真值方向；统一写 `0to1`，表示 `gt` 把 `image0` 像素坐标映射到 `image1` 像素坐标。
- `split`: `train`、`val` 或 `test`。

## 有真值影像对整理模板

以后所有“有真实变换关系”的遥感影像对，都统一整理成：

```text
image0: /path/to/A.png
image1: /path/to/B.png
gt:     /path/to/B_H_0to1.txt
```

`gt` txt 只写 3x3 矩阵数字，不写注释、不写路径、不写额外字段：

```text
h00 h01 h02
h10 h11 h12
h20 h21 h22
```

含义必须固定为：

```text
[x1, y1, 1]^T ~ H_0to1 @ [x0, y0, 1]^T
```

对应 manifest 行模板：

```json
{
  "dataset": "dataset_name",
  "id": "dataset_name/train/000000",
  "split": "train",
  "mode": "gt_pairs",
  "pair_type": "optical_optical",
  "image0": "/path/to/A.png",
  "image1": "/path/to/B.png",
  "gt": "/path/to/B_H_0to1.txt",
  "gt_direction": "0to1",
  "modality0": "optical",
  "modality1": "optical"
}
```

推荐把 txt 放在原数据集对应影像目录中，manifest 只记录绝对路径即可。这样数据集本体自洽，SLiM 项目里只保留索引。

## 当前数据集规则

- `3MOS.jsonl`: 只放 `opt_<id>` 与 `sar_<id>` 配成的多模态成对样本；无法配对的单模态图单独写入 `3MOS_single_images.jsonl`。
- `GoogleEarth`: `past/current` 按相同文件名配对。注意这类数据没有逐像素真实 homography，只能作为 `aligned_pairs` 或单图合成训练使用；有建筑视角差时不适合用 H 指标硬评估屋顶角点。
- `jl1flight`: 使用原始 `index/*.npz` 中的 `pair_infos` 和 `affine_matrices` 生成 `gt_pairs`。不能假设 `b_t0` 与 `a` 对齐。真值矩阵按 `H_A_to_B = M_B @ inv(M_A)` 计算，txt 放回原数据集目录。

## jl1flight 当前索引

```text
data/remote_archive/manifests/train_jl1flight_gt.jsonl  7490 pairs
data/remote_archive/manifests/test_jl1flight_gt.jsonl   1755 pairs
data/remote_archive/manifests/jl1flight_gt.jsonl        9245 pairs
```

每条记录都是 `mode=gt_pairs`，`gt` 指向：

```text
/home/disk1/Data/datasets/jl1flight/train/affine_pairs_train/*_H_0to1.txt
/home/disk1/Data/datasets/jl1flight/test/affine_pairs_test/*_H_0to1.txt
```

## 生成脚本

jl1flight 的索引和 txt 真值由 SLiM 项目中的脚本生成：

```bash
cd /home/disk1/SLiM
/root/miniconda3/envs/slim/bin/python data/remote_archive/scripts/build_jl1flight_manifest.py
```

这个脚本只读取 `/home/disk1/Data/datasets/jl1flight/index/*.npz`，不会移动原始影像。
"""


README.write_text(CONTENT, encoding="utf-8")
print(f"updated {README}")
