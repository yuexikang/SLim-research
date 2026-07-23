# Physical Encoder V2 版本更新记录

> 当前实现：V2.1.4。基础结构规范见[Physical Encoder V2](./Physical%20Encoder%20V2.md)。本文件只记录实现修订，不重复完整模型设计。

## V2.1.4

### 更新摘要

- 合成几何改为在原始分辨率影像内部采样两组有效四边形，再分别透视展开成完整的`512x512` patch。
- 不再把整张正方形图旋转或透视后用黑色填充越界区域，避免补边成为物理头的强伪边缘。
- View 0和View 1均来自同一原图的有效区域，输出监督Homography同步按两个rectification矩阵计算。
- LG增强增加低亮度质量保护，防止森林等暗场景被连续负亮度扰动压成近全黑图像。
- V2网络、loss、GoogleEarth索引、LG光度增强和每图每epoch一种扰动均保持不变。

### 具体更新

#### 有效区域采样与Rectification

在原图坐标中先采样凸四边形`Q0`，再按本epoch选中的`translation / scale / yaw / pitch / roll`得到`Q1`。两者必须同时满足：顶点位于原图内部、保持凸性、距边界至少2像素，且面积不小于难度`d=0.7`所规定的最小区域面积：

$$
A(Q_i) \ge (1-d)^2WH = 0.09WH.
$$

令`R0`和`R1`分别把`Q0/Q1`映射到完整的`512x512`目标矩形，则：

$$
I_0=\operatorname{warp}(I,R_0),\qquad
I_1=\operatorname{warp}(I,R_1),\qquad
H_{0\rightarrow1}=R_1R_0^{-1}.
$$

因此两张输入都由原图有效像素构成，不依赖黑色padding；源图只在最终采样时插值到512，不先压缩为512再变形。边界模式使用`BORDER_REFLECT_101`作为数值保险，但有效四边形约束使正常采样不会读取原图外像素。

#### 低亮度质量保护

LG光度增强仍保留V2.1.3的概率与参数，但每次增强结果必须同时满足：

- 平均灰度不低于`18/255`；
- 平均灰度不低于增强前View B的40%；
- 灰度不超过5的近黑像素占比不高于45%。

不满足时使用由原样本seed确定性派生的新seed重新采样，最多尝试4次。四次都不合格时回退到增强前View B。该保护只拦截光度增强造成的亮度塌缩，不改变原始暗场景、不修改几何Homography，恢复训练时仍可复现。

#### 物理特征图解释修订

可视化首列名称由`physical`改为`physical channel activity`。该图是96维物理描述子在通道维的标准差，并且每个面板独立按分位数着色，不是描述子L2范数；黄色只表示当前面板内相对活跃。黑色补边曾经通过LDN和Gabor产生强边界响应，因此可能在物理活动图中显示为高值，而冻结SLiM的特征范数在无纹理补边中通常较低。

#### 代码与兼容性

- `train_physical_v2.py`默认启用`--valid_crop_rectification`，可用`--no-valid_crop_rectification`仅作旧实验复现。
- 训练和快速/完整验证共用相同有效区域几何协议；验证仍不使用LG光度增强。
- 新增LG低亮度质量门槛、确定性重试及原图回退，避免无有效纹理的近全黑样本进入训练。
- 新增五类几何的边界、无黑色补边、确定性和Homography方向回归测试。
- 实现版本、checkpoint hparams、特征图metadata、W&B tag和正式训练脚本更新为`2.1.4`。
- 模型参数拓扑不变，旧V2 checkpoint可以完整恢复；当前V2.1.4正式实验按实验要求从V2.1.3的`last.ckpt`继续训练。

## V2.1.3

### 更新摘要

- Homography difficulty由0.3提高到0.7，最小有效区域边长为原图的30%。
- roll旋转范围改为`[-45°, 45°]`，不再重复乘difficulty。
- 新增总体概率0.95的`lg`在线光度增强，只作用于训练对的第二张图。
- GoogleEarth索引、每图每epoch一种几何扰动、模型结构和loss保持不变。

### 具体更新

#### 几何采样

设输入尺寸为`W x H`、难度为`d=0.7`：

$$
W_{min}=W(1-d)=0.3W,\qquad H_{min}=H(1-d)=0.3H.
$$

translation的最大位移由最小重叠边长约束；scale在`0.3`与`1/0.3`之间按log尺度采样；yaw和pitch的梯形短边不小于0.3倍原边长；roll独立使用`±45°`。五种几何类型仍由`seed + epoch + row_index`每图每epoch选择一种。

#### LG光度增强

整套Compose执行概率为`0.95`，内部配置如下：

| 变换 | 概率 | 参数 |
|---|---:|---|
| RandomGamma | 0.1 | `gamma_limit=(15,65)` |
| HueSaturationValue | 0.1 | `val_shift_limit=(-100,-40)` |
| Blur/MotionBlur/ISONoise/ImageCompression四选一 | 0.1 | Blur `3-9`，MotionBlur `3-25` |
| Blur | 0.1 | `blur_limit=(3,9)` |
| MotionBlur | 0.1 | `blur_limit=(3,25)` |
| RandomBrightnessContrast | 0.5 | brightness `[-0.4,0]`，contrast `[-0.3,0]` |
| CLAHE | 0.2 | Albumentations默认参数 |

输入灰度图先复制为RGB执行增强，再转回灰度。训练中的`image0`保持干净参照，`image1`执行`lg`；光度随机种子与epoch和row绑定，可重复运行。常规验证与最终完整验证都不启用随机光度增强。

#### 代码与兼容性

- V2入口默认参数更新为`homography_difficulty=0.7`、`rotation_limit_degrees=45`、`minimum_region_sampler=true`和`photometric_augmentation=lg`。
- 参数化Gabor的核生成、响应卷积和幅值链统一使用FP32，防止强几何与低亮度增强下BF16反向溢出；V1默认路径不变。
- 若个别强增强样本仍使三个Gabor标量参数产生NaN/Inf梯度，仅把对应非有限元素置零，有限梯度和其他模块不变；W&B通过`train/gabor_sanitized_grad_elements`记录触发数量，防止异常值进入AdamW状态。
- 实现版本、checkpoint hparams、特征图metadata和W&B tag更新为`2.1.3`。
- 模型参数拓扑不变，旧V2 checkpoint仍可加载；V2.1.3 GoogleEarth实验从头训练。

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
