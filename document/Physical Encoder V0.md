# Physical Encoder V0

## 1. 实验目标

Physical Encoder V0 是一个独立的、面向稠密粗匹配的轻量物理结构编码器。第一阶段只回答一个问题：

> 固定的方向结构先验，加上轻量可学习后端，能否从单模态光学 Homography 监督中学到具有几何稳定性、跨模态保留性，并且与 SLiM 粗特征互补的描述空间？

V0 刻意不包含以下模块：

- SLiM Backbone、Fine Matching 和 Refinement；
- Repeatability Head；
- 单独的 Orientation Head 或 Orientation Loss；
- CleanDIFT；
- PP、PF、FP、FF 四路融合；
- RANSAC 或其他后处理。

输入和输出为：

$$
I\in\mathbb{R}^{B\times1\times512\times512},\qquad
P\in\mathbb{R}^{B\times128\times64\times64}.
$$

输出的每个空间位置是一个 128 维 L2 归一化描述子。完整实现位于 `src/physical/`，独立训练入口为 `train_physical_v0.py`。

---

## 2. 设计与实现对照

当前 `physical_full` 基本按照本文档的核心设计实现。

| 设计项 | 当前实现 | 状态 |
| --- | --- | --- |
| 灰度输入，输出 1/8 分辨率 128 维描述子 | `[B,1,512,512] -> [B,128,64,64]` | 已实现 |
| 三尺度输入 | `1, 1/2, 1/4` | 已实现 |
| 固定 8 方向 quadrature bank | 固定 9x9 Gabor odd/even，共 16 个核 | 已实现 |
| 相位极性不敏感能量 | `sqrt(even^2 + odd^2 + eps)` | 已实现 |
| Soft Major Orientation | doubled-angle soft orientation | 已实现 |
| 方向通道 canonicalization | 连续循环线性采样 | 已实现 |
| 三尺度共享 Structure Encoder | 同一个模块重复作用于三个尺度 | 已实现 |
| 学习式 1/8 下采样 | 三条路径分别含 3、2、1 个 stride-2 block | 已实现 |
| Soft Cross-Scale Fusion | `Conv1x1(192->3) + softmax` | 已实现 |
| 128 维描述子头 | DWConv、PWConv、LayerNorm2d、L2 normalize | 已实现 |
| 只训练 `L_PP` | partial dual-softmax focal loss | 已实现 |
| Full 与 Chunked 数学等价 | loss 和梯度测试通过 | 已实现 |
| 参数量匹配 Tiny CNN | 42,992 vs. Physical 44,899，差 4.25% | 已实现 |
| NoCanon、SingleScale 消融 | 可构建、已做前向测试 | 已实现，尚未正式训练 |
| 直接复用 SLiM loss 类 | 实际采用独立重写的数学等价 `PPMatchingLoss` | 实现方式不同 |
| Gabor / Log-Gabor-like 可选 | 当前固定为标准 Gabor | 收敛为单一实现 |
| 分文件 quadrature/orientation/encoder | 当前集中在 `src/physical/models.py` | 仅代码组织不同 |
| 接入 SLiM 形成统一匹配结果 | 当前只支持并行测评和互补统计 | 尚未实现 |

因此，当前模型在网络结构和实验约束上遵循 V0 设计；差异主要是代码组织以及 loss 复用方式，不改变实验定义。

---

## 3. Physical-Full 总体结构

```text
Gray Image I [B,1,512,512]
│
├── scale 1     [B,1,512,512]
├── scale 1/2   [B,1,256,256]
└── scale 1/4   [B,1,128,128]
     │
     │ 每个尺度共享以下模块
     ▼
Fixed Gabor Quadrature Bank
16 responses = 8 even + 8 odd
     │
     ▼
Orientation Energy [B,8,Hs,Ws]
     │
     ▼
Soft Major Orientation
     │
     ▼
Orientation Channel Canonicalization [B,8,Hs,Ws]
     │
     ▼
Shared Structure Encoder [B,64,Hs,Ws]
     │
     ├── scale 1:   3 x DownsampleBlock -> [B,64,64,64]
     ├── scale 1/2: 2 x DownsampleBlock -> [B,64,64,64]
     └── scale 1/4: 1 x DownsampleBlock -> [B,64,64,64]
     │
     ▼
Concat [B,192,64,64]
     │
     ▼
Conv1x1(192->3) + Softmax
     │
     ▼
Weighted Scale Sum [B,64,64,64]
     │
     ▼
Descriptor Head
DWConv3x3(64) -> Conv1x1(64->128) -> LayerNorm2d -> L2 Normalize
     │
     ▼
P [B,128,64,64]
```

两张训练影像使用同一个共享权重编码器：

```text
I0 -> Physical Encoder -> P0 [B,128,64,64]
I1 -> Physical Encoder -> P1 [B,128,64,64]
                         │
                         ▼
              similarity [B,4096,4096]
                         │
                         ▼
                        L_PP
```

模型没有两套分支权重；`I0` 和 `I1` 都调用 `self.encoder`。

---

## 4. 模型具体结构

### 4.1 三尺度图像金字塔

当前使用双线性插值和 antialias：

| 分支 | 输入尺寸 | 后续下采样次数 | 对齐后尺寸 |
| --- | ---: | ---: | ---: |
| scale 1 | 512x512 | 3 | 64x64 |
| scale 1/2 | 256x256 | 2 | 64x64 |
| scale 1/4 | 128x128 | 1 | 64x64 |

三尺度共同覆盖较细边缘、中等结构和较大区域结构。三条路径最终位于同一个 1/8 粗网格。

### 4.2 固定 Gabor Quadrature Bank

使用 8 个无符号方向：

$$
\theta_k=\frac{k\pi}{8},\qquad k=0,\ldots,7.
$$

每个方向包含 even 和 odd 两个相差 90 度相位的 Gabor 核。当前超参数为：

| 参数 | 数值 |
| --- | ---: |
| orientations | 8 |
| kernel size | 9 |
| sigma | 2.5 |
| wavelength | 4.0 |
| gamma | 0.5 |
| energy epsilon | `1e-6` |

滤波核先去均值，再按 L2 范数归一化。方向能量为：

$$
A_{s,k}=\sqrt{(I_s*K_k^e)^2+(I_s*K_k^o)^2+\epsilon}.
$$

这一步消除了 odd/even 响应符号，使亮暗翻转、边缘极性反转时仍可能保留相近的结构能量。

这些核使用 `register_buffer` 保存，不是 `Parameter`，因此：

- 会进入 checkpoint；
- 会随模型移动到 GPU；
- 不参与反向更新；
- 不计入 44,899 个可训练参数。

### 4.3 Soft Major Orientation

对每个位置的 8 个方向能量计算 softmax，当前方向温度固定为 `0.1`：

$$
\pi_k(x)=\operatorname{softmax}_k\left(\frac{A_k(x)}{\tau_o}\right).
$$

使用 doubled-angle 表示无符号方向：

$$
v_x=\sum_k\pi_k\cos(2\theta_k),\qquad
v_y=\sum_k\pi_k\sin(2\theta_k),
$$

$$
\theta(x)=\frac{1}{2}\operatorname{atan2}(v_y,v_x).
$$

使用 `2theta` 是因为结构方向满足 `theta` 与 `theta + pi` 等价。AMP 下角度估计强制使用 FP32，减少小方向差异被 BF16 舍入的问题。

### 4.4 方向通道 Canonicalization

局部主方向被转换为连续通道位移：

$$
\delta(x)=\frac{8\theta(x)}{\pi}.
$$

随后沿 8 个方向通道进行循环线性采样：

$$
\widetilde A_k(x)=\operatorname{CircularSample}(A(x),k+\delta(x)).
$$

整数位移对应方向通道循环移动；非整数位移由相邻两个方向通道线性插值。该模块没有可学习参数。

### 4.5 Shared Structure Encoder

三个尺度共享下面同一个网络：

| 层 | 配置 | 输出通道 | 参数量 |
| --- | --- | ---: | ---: |
| 1 | Conv 1x1, 8 -> 32, no bias | 32 | 256 |
| 2 | DWConv 3x3, groups=32, no bias | 32 | 288 |
| 3 | GELU | 32 | 0 |
| 4 | Conv 1x1, 32 -> 64, no bias | 64 | 2,048 |
| 5 | DWConv 3x3, groups=64, no bias | 64 | 576 |
| 6 | Conv 1x1, 64 -> 64, no bias | 64 | 4,096 |
| 合计 |  |  | **7,264** |

共享参数确保三个尺度映射到同一个 64 维结构空间，而不是形成三个不相容的描述空间。

### 4.6 Physical Downsample Block

每个 block 为：

```text
DWConv 3x3, 64->64, stride=2, groups=64, no bias  576 params
GELU                                                  0 params
Conv 1x1, 64->64, no bias                         4,096 params
总计                                                4,672 params
```

三条路径的参数不共享：

| 路径 | Block 数 | 参数量 |
| --- | ---: | ---: |
| scale 1 | 3 | 14,016 |
| scale 1/2 | 2 | 9,344 |
| scale 1/4 | 1 | 4,672 |
| 合计 | 6 | **28,032** |

这里共享的是前面的 Structure Encoder，不是三条下采样路径。

### 4.7 Cross-Scale Fusion

三个对齐特征拼接为：

$$
Z=[Z_1;Z_2;Z_3]\in\mathbb{R}^{B\times192\times64\times64}.
$$

逐位置预测三个尺度权重：

$$
[w_1,w_2,w_3]=\operatorname{softmax}(\operatorname{Conv}_{1\times1}(Z)).
$$

融合结果为：

$$
Z_{phy}=w_1Z_1+w_2Z_2+w_3Z_3.
$$

`Conv1x1(192->3)` 包含 bias，共 `192*3+3=579` 个参数。

### 4.8 Descriptor Head

| 层 | 配置 | 输出形状 | 参数量 |
| --- | --- | --- | ---: |
| 1 | DWConv 3x3, 64 channels | `[B,64,64,64]` | 576 |
| 2 | Conv 1x1, 64 -> 128 | `[B,128,64,64]` | 8,192 |
| 3 | LayerNorm2d(128), affine | `[B,128,64,64]` | 256 |
| 4 | channel-wise L2 normalize | `[B,128,64,64]` | 0 |
| 合计 |  |  | **9,024** |

最终每个描述子满足：

$$
\|P_{:,x,y}\|_2=1.
$$

### 4.9 参数量汇总

| 模块 | 可训练参数 |
| --- | ---: |
| Fixed Gabor Bank | 0 |
| Soft Orientation + Canonicalization | 0 |
| Shared Structure Encoder | 7,264 |
| 三条 Downsample Paths | 28,032 |
| Scale Fusion | 579 |
| Descriptor Head | 9,024 |
| **Physical-Full 总计** | **44,899** |

训练模块还包含一个可学习的 `log_temperature`，因此 Lightning 训练系统显示 44,900 个可训练参数；模型编码器本身是 44,899 个。

---

## 5. 训练监督

### 5.1 数据生成

训练使用单张光学影像 manifest。每张基础影像在线生成五种 Homography 样本：

```text
translation
scale
yaw
pitch
roll
```

变换不保存到本地。训练随机数由 `seed + epoch + sample index` 确定，使不同模型和恢复训练能够复现同一扰动。

当前 Physical-Full 正式实验实际使用：

| 项目 | 数值 |
| --- | --- |
| train manifest | `train_optical_single_images.jsonl` |
| train ratio | 1.0 |
| selected base rows | 54,455 |
| generated pairs / epoch | 272,275 |
| validation pairs | 25,820 |
| seed | 66 |
| image size | 512 |
| homography difficulty | 0.3 |

原始实验计划要求 Physical-Full 与 Tiny-CNN 使用相同训练子集。当前 Physical-Full 使用 100%，而已启动的 Tiny-CNN 使用 30%，二者不能作为严格公平对照；正式论文比较必须统一比例。

### 5.2 Coarse GT Correspondence

在 1/8 网格上取 token 中心，通过真实 Homography 从 `image0` 投影到 `image1`。投影结果四舍五入到目标 token，并通过逆 Homography 做 round-trip 检查，只保留：

- 投影后仍在图像范围内；
- 正向和反向离散对应一致。

最终得到 GT 三元组：

$$
(b,i,j),
$$

其中 `i` 是源图 token，`j` 是目标图 token。

### 5.3 PP Matching Loss

将两个描述图展平：

$$
P_0,P_1\in\mathbb{R}^{B\times4096\times128}.
$$

相似度矩阵为：

$$
S=\frac{P_0P_1^T}{\tau}\in\mathbb{R}^{B\times4096\times4096}.
$$

`tau` 通过 `log_temperature` 学习，初值为 `0.05`，有效范围限制在 `[0.001,1.0]`。

训练随机选择 90% 有效正样本。对 GT `(i,j)`，分别计算行 softmax 和列 softmax：

$$
p_{ij}=\operatorname{softmax}_{row}(S)_{ij}\cdot
       \operatorname{softmax}_{col}(S)_{ij}.
$$

使用 `gamma=2` 的 focal loss：

$$
L_{PP}=-\frac{1}{N}\sum_{(i,j)}(1-p_{ij})^2\log(p_{ij}).
$$

V0 总损失只有：

$$
L=L_{PP}.
$$

`full` 模式直接构造完整矩阵；`chunked` 模式按正样本分块并使用 gradient checkpoint。二者不做负样本近似，已有 CPU 数值测试验证 loss 和描述子梯度等价。

### 5.4 优化设置

默认训练配置：

| 项目 | 数值 |
| --- | --- |
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `0.01` |
| scheduler | CosineAnnealingLR |
| eta min | `1e-6` |
| precision | BF16 mixed |
| gradient clipping | 1.0 |
| max epochs | 20 |

优化器接收 `PhysicalV0Module.parameters()`，因此 Structure Encoder、下采样路径、Scale Fusion、Descriptor Head 和 loss temperature 都会更新。固定 Gabor 与方向常量不会更新。

---

## 6. 验证指标

### 6.1 R@0 和 R@1

对每个有效 GT 源 token `i`，使用全部目标 token 做最近邻预测：

$$
\hat j=\arg\max_j P_0(i)^TP_1(j).
$$

将预测和 GT 转换为二维粗网格坐标，定义切比雪夫距离：

$$
d=\max(|x_{pred}-x_{gt}|,|y_{pred}-y_{gt}|).
$$

$$
R@0=\frac{\sum[d=0]}{N},\qquad
R@1=\frac{\sum[d\leq1]}{N}.
$$

- `R@0` 要求预测与 GT 位于完全相同的粗 token；
- `R@1` 允许落在 GT 周围 3x3 token 邻域；
- 粗尺度为 8，因此 `R@1` 允许横纵方向各偏差约 8 个输入像素；
- 这是所有有效 token 的 micro-average，不是先对图像对求均值。

验证分别记录总体以及五种扰动的 R@0、R@1。

### 6.2 描述子判别性

同时记录：

$$
s^+=S(i,j^*),
$$

$$
s^-=\max_{j\notin\mathcal{N}_{1}(j^*)}S(i,j),
$$

$$
margin=s^+-s^-.
$$

Hard negative 排除 GT 周围 3x3 token 邻域。还记录匹配分布 entropy 和 normalized entropy，用于判断分布是否清晰，而不是只看 loss。

---

## 7. 模型变体与 Tiny-CNN 对照

构建入口支持四个模型：

| 名称 | 定义 | 参数量 |
| --- | --- | ---: |
| `physical_full` | Canonicalization + 三尺度融合 | 44,899 |
| `physical_no_canon` | 关闭 Canonicalization，保留三尺度 | 44,899 |
| `physical_single_scale` | 保留 Canonicalization，只使用原尺度 | 30,304 |
| `tiny_cnn` | 无物理先验的参数量匹配 CNN | 42,992 |

Tiny-CNN 结构为：

```text
Input [B,1,512,512]
Conv3x3 s2, 1->16 + LayerNorm2d + GELU       -> [B,16,256,256]
DSConvBlock 16->16, residual                  -> [B,16,256,256]
DSConvBlock 16->32, s2                        -> [B,32,128,128]
DSConvBlock 32->32, residual                  -> [B,32,128,128]
DSConvBlock 32->64, s2                        -> [B,64,64,64]
6 x DSConvBlock 64->64, residual              -> [B,64,64,64]
Descriptor Head                               -> [B,128,64,64]
```

Tiny-CNN 与 Physical-Full 参数量差异为：

$$
\frac{|44899-42992|}{44899}=4.25\%.
$$

只有在数据、扰动、batch size、优化器、学习率、epoch 和 seed 全部一致时，二者结果才能用于判断物理归纳偏置是否有效。

---

## 8. 独立多模态评测与 SLiM 互补性

### 8.1 Physical standalone 协议

真实多模态测试采用：

- Physical 粗描述子的 cosine mutual-nearest-neighbor；
- 不使用 RANSAC；
- 匹配点恢复到原始影像坐标；
- 重投影误差不超过 5 px 视为正确；
- `NCM >= 20` 视为图像对成功；
- 失败图像对 RMSE 记为 10。

输出 NCM、Precision、SR、RMSE、平均匹配数、运行时间和按模态分组结果。

Cross-Modal Retention 定义为：

$$
Retention_m=\frac{Pre_m}{Pre_{optical-optical}}.
$$

Retention 只表示相对于模型自身 optical-optical 精度的保留比例；如果 optical-optical 基线本身很低，高 Retention 不代表绝对性能好。

### 8.2 与 SLiM 的互补统计

对同一个 GT token，分别判断 Physical 粗特征和 SLiM 官方粗特征是否预测正确：

| Physical | SLiM | 统计名称 |
| --- | --- | --- |
| 正确 | 正确 | `both_correct` |
| 正确 | 错误 | `physical_only` |
| 错误 | 正确 | `slim_only` |
| 错误 | 错误 | `both_wrong` |

Complementary Recovery Rate 定义为：

$$
CRR=\frac{physical\_only}{physical\_only+both\_wrong}.
$$

它表示 SLiM 粗特征预测错误的位置中，有多少被 Physical 粗特征预测正确。

当前代码只做两个编码器的并行诊断，不会产生真正融合后的统一匹配结果。使用 GT 选择两者正确结果得到的 oracle union 只是理论上限，不能作为可实现融合性能。

---

## 9. 当前 Physical-Full 实验结果

### 9.1 合成光学验证

当前最佳 checkpoint 为 epoch 12：

```text
logs/tb_logs/physical_v0/physical_full_ratio100_gpu1_bs8_seed66_resume/
checkpoints/best-12-0.9969.ckpt
```

| 指标 | Epoch 0 | Epoch 12 |
| --- | ---: | ---: |
| train L_PP | 0.7512 | 0.3832 |
| val L_PP | 0.5711 | 0.3939 |
| R@0 | 88.23% | 90.42% |
| R@1 | 98.69% | 99.69% |

这证明 V0 能学习同源光学影像在合成 Homography 下的粗 token 对应，但不能直接证明真实多模态能力。

### 9.2 SwinMatcher Proposed，400 对

Physical standalone：

| NCM | Precision | SR | RMSE |
| ---: | ---: | ---: | ---: |
| 35.12 | 4.34% | 42.25% | 7.11 px |

与 SLiM 粗特征互补：

| 指标 | R@0 | R@1 |
| --- | ---: | ---: |
| CRR | 1.56% | 3.19% |
| SLiM coarse accuracy | 10.75% | 22.67% |
| Oracle union upper bound | 12.15% | 25.14% |

Optical-SAR 的互补性最高，`CRR@1=7.76%`；Optical-Map 接近完全失败。

### 9.3 Expanded MRSI，600 对

Physical standalone：

| NCM | Precision | SR | RMSE |
| ---: | ---: | ---: | ---: |
| 27.77 | 3.21% | 29.17% | 8.01 px |

与 SLiM 粗特征互补：

| 指标 | R@0 | R@1 |
| --- | ---: | ---: |
| CRR | 1.24% | 2.73% |
| SLiM coarse accuracy | 11.21% | 23.21% |
| Oracle union upper bound | 12.31% | 25.31% |

Optical-Infrared 的互补性最高，`CRR@1=6.78%`；Optical-Map 再次失败。

### 9.4 当前可以得出的结论

1. V0 在合成光学 Homography 验证集上有效并已收敛。
2. 合成验证 R@0/R@1 没有迁移到真实多模态数据，存在明显域差距。
3. Physical-Full 目前不能作为独立 matcher 替代 SLiM。
4. 两个真实数据集都观察到小幅但稳定的非零互补性。
5. 互补性具有数据域和模态依赖：Proposed 的 Optical-SAR 较强，Expanded MRSI 的 Optical-Infrared 较强。
6. Optical-Map 在两个数据集都失败，是当前结构最稳定的弱项。
7. `slim_only` 数量远大于 `physical_only`，无条件相加容易损坏 SLiM 正确结果。
8. Tiny-CNN 尚未形成同数据量公平基线，因此还不能把现象归因于物理先验本身。

---

## 10. 当前局限与下一步判据

### 10.1 当前局限

- 训练只看单张光学影像及合成 Homography，没有真实跨模态监督；
- 方向能量主要表达边缘和纹理结构，难以跨越地图符号化表达；
- 只输出 1/8 粗特征，没有亚像素 refinement；
- standalone 使用 mutual nearest-neighbor，没有置信度学习和几何过滤；
- CRR 使用 GT 做诊断，不等于融合后可以自动恢复这些位置；
- Physical-Full 与当前 Tiny-CNN 的训练数据比例不一致。

### 10.2 是否进入 SLiM 融合

在实现融合前，应至少完成：

1. Physical-Full 与 Tiny-CNN 的同数据、同 seed、同训练预算对照；
2. `physical_no_canon` 的 Roll/Yaw/Pitch 消融；
3. `physical_single_scale` 的尺度消融；
4. 在独立验证集上检验可实现的 confidence/score fusion，而不是 oracle union；
5. 确认融合不会降低 SLiM-only 正确样本。

如果继续做最小集成，建议先采用受控残差形式：

$$
F'=F+\alpha W_P(P),
$$

其中 `P` 从 128 维投影到 SLiM coarse 的 192 维，`alpha` 使用较小可学习门控。该集成目前尚未实现。

---

## 11. 代码位置

| 功能 | 文件 |
| --- | --- |
| Physical-Full、消融、Tiny-CNN | `src/physical/models.py` |
| Homography coarse GT、PP loss | `src/physical/matching.py` |
| R@0/R@1、margin、entropy | `src/physical/metrics.py` |
| 分层抽样与 DataLoader | `src/physical/data.py` |
| Lightning 训练与验证 | `src/physical/lightning.py` |
| 独立训练入口 | `train_physical_v0.py` |
| 合成与多模态测评 | `test/evaluate_physical_v0.py` |
| CPU 单元测试 | `tests/test_physical_v0.py` |

V0 的最小研究问题保持不变：

> Can a physically structured dense encoder trained only on optical homography pairs produce more modality-stable and complementary correspondences than a parameter-matched generic CNN?
