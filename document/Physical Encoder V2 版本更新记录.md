# Physical Encoder V2 版本更新记录

> 当前实现：V2.1.2。基础结构规范见[Physical Encoder V2](./Physical%20Encoder%20V2.md)。本文件只记录实现修订，不重复完整模型设计。

## V2.1.2

### 更新摘要

- 正式训练数据改为 GoogleEarth 单图，不再混入低分辨率 3MOS。
- 从成对索引先按 pair 划分训练/验证，再展开为单图，消除同一影像对跨集合泄漏。
- 每张基础影像每个 epoch 仍只在线生成一种随机扰动。
- 模型结构、loss 和 checkpoint 参数拓扑不变。

### 具体更新

#### 数据索引

源索引：

```text
data/remote_archive/manifests/train_GoogleEarth.jsonl
```

使用 seed 66，在 `subset` 内稳定排序，并按 9:1 划分 pair ID。划分完成后，每对影像展开为两条 `single_synth` 记录：

| 索引 | pair 数 | 单图数 |
|---|---:|---:|
| `train_GoogleEarth_single.jsonl` | 8,201 | 16,402 |
| `val_GoogleEarth_single.jsonl` | 911 | 1,822 |

检查结果：18,224 张源图均为 `1080x1080`；重复 ID 为 0、重复路径为 0、训练/验证 pair 重叠为 0、影像路径重叠为 0、缺失路径为 0。完整 SHA256 和 subset 分布记录在：

```text
data/remote_archive/manifests/GoogleEarth_single_split_summary.json
```

生成命令：

```bash
/root/miniconda3/envs/slim/bin/python tools/build_googleearth_single_manifests.py
```

通用索引 `train_optical_single_images.jsonl` 中的 34,957 条 3MOS 记录已删除；该文件现在保留 18,000 条 GoogleEarth 和 1,498 条 jl1flight。V2.1.2正式训练直接使用新的 GoogleEarth 专用索引，不依赖这个通用索引。

#### 在线扰动

训练使用 `--train_one_variant_per_row`。每张基础影像在每个 epoch 只从以下五种扰动中选择一种：

```text
translation / scale / yaw / pitch / roll
```

选择由 `seed + epoch + row_index` 决定，不写入磁盘；恢复训练时可复现。训练期验证使用 `--val_one_variant_per_row`，每张验证图固定一种扰动。训练结束后仍可由入口对 best checkpoint 执行五种扰动完整验证。

#### 代码与兼容性

- `train_physical_v2.py` 默认训练/验证索引改为 GoogleEarth 单图索引。
- 实现版本、checkpoint hparams、可视化 metadata 和 W&B tag 更新为 `2.1.2`。
- V2网络参数名和尺寸不变，V2.1/V2.1.1 checkpoint 可加载。
- 为公平评估新数据配方，GoogleEarth 正式实验应从冻结的官方 SLiM 和随机初始化的 V2头重新开始，不恢复旧混合数据实验。

## V2.1.1

### 更新摘要

- 增加训练中关键物理特征图可视化。
- 每 20 train step 覆盖同一组文件，不保存历史图片。
- 提高输入图和低分辨率特征图的展示清晰度。

### 具体更新

输出目录：

```text
<experiment>/paper_logs/latest_feature_maps/
```

固定输出 `inputs.png`、`odd_even_features.png`、`physical_gates.png`、`descriptor_features.png` 和 `latest_step.json`。下一次更新先写临时文件再原子替换，因此目录始终只表示最新 step。

主要内容包括：输入影像、Odd/Even响应、可靠性、尺度权重、Hard O/E选择、physical descriptor、delta、base SLiM和enhanced feature。输入图按模型实际的 `512x512` 张量显示；`64x64`特征图采用最近邻放大，避免插值造成额外模糊。

该版本只增加可观测性，不新增参数，不改变训练目标。

## V2.1

### 更新摘要

- 修复长时间 BF16 训练中的 Pair Linear Attention 非有限值。
- 修复弱纹理区域 `atan2(0,0)` 的未定义反向梯度。
- 增加前向、loss 和梯度三级非有限值诊断。

### 具体更新

Pair Linear Attention 的 Q/K/V、ELU映射、`K^T V`、归一化、输出投影、LayerNorm和MLP固定使用 FP32，返回主路径前再转回原 dtype。

无符号方向角不再直接计算 `0.5*atan2(y,x)`。当 `x^2+y^2 < 1e-6` 时，安全输入替换为常量 `(1,0)`，无效位置输出零角度且不向原方向向量传播梯度：

$$
\phi=\frac{1}{2}\operatorname{atan2}(y_s,x_s).
$$

异常诊断依次检查模型输出、各项 loss 和反向梯度；异常时报告阶段、epoch、global step、batch内样本 ID、扰动类型和首个非有限参数。此修订解决了 Gabor 参数在约 global step 927 出现非有限梯度的问题。

## V2

### 更新摘要

- 建立冻结 SLiM coarse feature 上的独立 Physical Encoder V2 残差分支。
- 只研究 `64x64` coarse correspondence discovery，不训练 Fine 和 Refinement。

### 具体更新

核心路径为三尺度抗混叠金字塔、共享 LDN、三频八方向参数化 Gabor、HIMO Odd/Even、MASW、两轮 Linear Pair Transformer、Hard O/E、可靠性尺度融合、双轴 Polar Descriptor 和 zero-init `96->192` Adapter：

$$
F_i^{enhanced}=F_i^{SLiM}+\Delta F_i.
$$

SLiM始终冻结并保持 `eval()`；训练只更新 Physical V2参数。完整结构、loss和实验边界以基础规范文档为准。
