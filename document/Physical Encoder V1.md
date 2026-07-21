# Physical Encoder V1

> 状态：V1-Core 已实现并通过 CPU 单元测试、GPU BF16 smoke test、checkpoint 恢复测试及合成/多模态测评入口检查。本文档同时作为结构设计和实验规范。

## 1. 研究目标

Physical Encoder V0 已经证明固定方向能量能够学习合成 Homography 粗匹配，但真实多模态实验暴露出三个问题：

1. V0 只对齐方向通道，没有对齐局部空间邻域；
2. odd/even 在前端立即合并为幅值能量，物理中间量丢失过早；
3. 62% 参数位于三套独立下采样路径，后端容易退化成 Gabor 预处理后的普通 CNN。

V1 的目标不是加深网络，而是：

$$
\boxed{
\text{少自由 CNN 容量}
+\text{显式方向与空间约束}
+\text{更晚丢失物理中间量}
}
$$

V1 仍然是独立的稠密粗描述编码器。本阶段明确不包含：

- SLiM Backbone、Fine Matching 和 Refinement；
- CleanDIFT、Semantic Student 或其他语义教师；
- Repeatability Head；
- PP、PF、FP、FF 四路融合；
- RANSAC 或其他几何后处理。

---

## 2. 输入、输出与前向接口

输入保持为灰度图：

$$
I\in\mathbb{R}^{B\times1\times512\times512}.
$$

主输出为：

$$
P_{fused}\in\mathbb{R}^{B\times128\times64\times64}.
$$

V1 不提前丢弃三个物理专家，完整前向接口返回具名字典：

```python
{
    "fused":         Tensor[B, 128, 64, 64],
    "edge":          Tensor[B,  32, 64, 64],
    "contour":       Tensor[B,  32, 64, 64],
    "stable":        Tensor[B,  32, 64, 64],
    "orientation":   Tensor[B,   2, 64, 64],
    "confidence":    Tensor[B,   1, 64, 64],
    "scale_weights": Tensor[B,   3, 64, 64],
    "expert_weights":Tensor[B,   3, 64, 64],
}
```

其中：

- `fused` 是 standalone 训练与测评默认使用的描述子；
- `edge` 保留 odd response 主导的边缘专家；
- `contour` 保留 even response 主导的线与轮廓专家；
- `stable` 保留幅值能量和跨频相位一致性专家；
- `orientation` 保存 `[cos(2theta), sin(2theta)]`；
- `confidence` 保存方向集中度 `rho`；
- `scale_weights` 和 `expert_weights` 用于解释门控行为。

四种描述子输出均按通道 L2 归一化。

---

## 3. V1-Core 总体结构

```text
Gray Image [B,1,512,512]
│
├── scale 1     [B,1,512,512]
├── scale 1/2   [B,1,256,256]
└── scale 1/4   [B,1,128,128]
     │
     │ 每个尺度共享以下物理参数与可学习模块
     ▼
Local Divisive Normalization
     │
     ▼
3 Frequencies x 8 Orientations x Odd/Even
Parameterized Steerable Gabor Quadrature Bank
     │
     ├── Signed Odd Responses
     ├── Signed Even Responses
     ├── Magnitude Energy
     └── Multi-frequency Phase Agreement
     │
     ▼
Soft Major Orientation + Confidence rho
     │
     ▼
Confidence-gated Orientation-channel Canonicalization
     │
     ├── Edge Pointwise Encoder      -> 32 channels
     ├── Contour Pointwise Encoder   -> 32 channels
     └── Stability Pointwise Encoder -> 32 channels
     │
     ▼
5x5 Dense Orientation-Aligned Neighborhood at native scale
     │
     ▼
Fixed BlurPool + Shared Pointwise Refinement
     │
     ├── 512 -> 256 -> 128 -> 64
     ├── 256 -> 128 -> 64
     └── 128 -> 64
     │
     ▼
Explicit Cross-scale Agreement and Difference
     │
     ▼
Scale Gate
     │
     ├── P_edge    [B,32,64,64]
     ├── P_contour [B,32,64,64]
     └── P_stable  [B,32,64,64]
     │
     ▼
Expert Gate
     │
     ▼
Conv1x1 32->128 + LayerNorm2d + L2 Normalize
     │
     ▼
P_fused [B,128,64,64]
```

V1 的第一个空间混合操作是方向对齐的 OAN。OAN 之前只允许物理滤波、方向通道采样和 1x1 pointwise 编码，不使用普通 3x3 CNN。

---

## 4. 多尺度 Local Divisive Normalization

先构建 antialiased 图像金字塔：

| 图像尺度 | 空间尺寸 | OAN 后 BlurPool 次数 |
| --- | ---: | ---: |
| `s=1` | 512x512 | 3 |
| `s=1/2` | 256x256 | 2 |
| `s=1/4` | 128x128 | 1 |

每个尺度独立执行固定 Local Divisive Normalization。使用 9x9、`sigma=2` 的归一化 Gaussian 核：

$$
\mu_s=G_\sigma*I_s,
$$

$$
\sigma_s=\sqrt{G_\sigma*(I_s-\mu_s)^2+10^{-4}},
$$

$$
I_s^n=\operatorname{clip}\left(
\frac{I_s-\mu_s}{\sigma_s},-5,5
\right).
$$

LDN 没有可训练参数，作用是削弱全局亮度、局部幅值和辐射对比度变化。它不能被 BatchNorm 替代，因为目标是逐图像、逐位置的确定性物理归一化。

---

## 5. 参数化多频 Steerable Gabor Quadrature Bank

### 5.1 频率与方向

每个图像尺度都计算三个频率和八个无符号方向：

$$
\lambda_f^0\in\{3,6,12\},\qquad f=0,1,2,
$$

$$
\theta_k=\frac{k\pi}{8},\qquad k=0,\ldots,7.
$$

三个尺度和三个频率采用笛卡尔积，共九个尺度频率组合。固定支持尺寸为：

| 基础波长 | Kernel size |
| ---: | ---: |
| 3 | 9 |
| 6 | 15 |
| 12 | 25 |

### 5.2 受约束物理参数

每个频率只学习三个标量，不学习自由卷积核：

$$
\lambda_f=\lambda_f^0\exp\left(0.25\tanh\Delta\lambda_f\right),
$$

$$
\sigma_f=0.56\lambda_f\exp\left(0.20\tanh\Delta\sigma_f\right),
$$

$$
\gamma_f=0.3+0.5\operatorname{sigmoid}(g_f).
$$

初始化满足 `lambda=lambda0`、`sigma=0.56*lambda`、`gamma=0.5`。九个物理标量在三个图像尺度和八个方向之间共享。

对坐标进行方向旋转：

$$
x_\theta=x\cos\theta_k+y\sin\theta_k,
$$

$$
y_\theta=-x\sin\theta_k+y\cos\theta_k.
$$

解析 odd/even quadrature 核为：

$$
K^e_{f,k}(x,y)=
\exp\left[-\frac{x_\theta^2+\gamma_f^2y_\theta^2}{2\sigma_f^2}\right]
\cos\left(\frac{2\pi x_\theta}{\lambda_f}\right),
$$

$$
K^o_{f,k}(x,y)=
\exp\left[-\frac{x_\theta^2+\gamma_f^2y_\theta^2}{2\sigma_f^2}\right]
\sin\left(\frac{2\pi x_\theta}{\lambda_f}\right).
$$

每个核在 FP32 中生成，随后去均值并按 L2 范数归一化，再转换到卷积输入 dtype。不同方向由同一解析参数生成，不能独立漂移成普通卷积核。

### 5.3 响应

对每个尺度得到：

$$
E_{s,f,k}=I_s^n*K^e_{f,k},
$$

$$
O_{s,f,k}=I_s^n*K^o_{f,k},
$$

$$
A_{s,f,k}=\sqrt{E_{s,f,k}^2+O_{s,f,k}^2+10^{-6}}.
$$

响应张量均为 `[B,3,8,Hs,Ws]`。`O` 和 `E` 保留符号，`A` 提供相位和极性不敏感的结构能量。

---

## 6. Multi-frequency Phase Agreement

将每个频率的 quadrature 响应视为复响应：

$$
R_{s,f,k}=E_{s,f,k}+iO_{s,f,k}.
$$

跨频相位一致性定义为：

$$
PC_{s,k}=
\frac{\left|\sum_f R_{s,f,k}\right|}
{\sum_f A_{s,f,k}+10^{-6}}.
$$

根据三角不等式，理论上 `PC` 位于 `[0,1]`；实现中最终 clamp 到 `[0,1]` 消除数值误差。

- `PC` 接近 1：多个频率在该方向形成一致结构；
- `PC` 接近 0：响应可能来自单频纹理、噪声或相位冲突。

Phase Agreement 不替代幅值能量，而是作为 Stability Expert 和方向估计的显式输入。

---

## 7. Soft Major Orientation 与置信度

使用 phase-weighted energy 聚合三个频率：

$$
Q_{s,k}=PC_{s,k}\sum_f A_{s,f,k}.
$$

为降低残余幅值尺度影响，先按方向均值归一化：

$$
\bar Q_{s,k}=\frac{Q_{s,k}}
{\frac{1}{8}\sum_j Q_{s,j}+10^{-6}}.
$$

方向概率和 doubled-angle 主方向为：

$$
\pi_{s,k}=\operatorname{softmax}_k\left(\frac{\bar Q_{s,k}}{0.1}\right),
$$

$$
v_x=\sum_k\pi_{s,k}\cos(2\theta_k),\qquad
v_y=\sum_k\pi_{s,k}\sin(2\theta_k),
$$

$$
\theta_s=\frac{1}{2}\operatorname{atan2}(v_y,v_x).
$$

方向置信度为：

$$
\rho_s=\sqrt{v_x^2+v_y^2}\in[0,1].
$$

`rho` 是解析方向分布的集中度，不增加一个可以任意输出置信度的 CNN Head。

---

## 8. Confidence-gated 方向通道 Canonicalization

对 `[B,3,8,Hs,Ws]` 的 Odd、Even、Energy 分别沿方向维进行连续循环采样。

$$
\delta_s(x)=\frac{8\theta_s(x)}{\pi},
$$

$$
C_{canon,s,f,k}(x)=
\operatorname{CircularSample}\left(C_{s,f}(x),k+\delta_s(x)\right).
$$

最终结果不是无条件使用 canonical feature，而是：

$$
C'=(1-\rho_s)C+\rho_s C_{canon}.
$$

当方向分布均匀时，`rho` 接近 0，模型保留原始表示；当方向明确时，`rho` 接近 1，模型使用规范化方向通道。

---

## 9. 三个 Pointwise Physical Experts

OAN 之前只允许 1x1 编码，不允许普通空间卷积。

### 9.1 Edge Expert

将 canonicalized odd response 的 signed 和 absolute 形式拼接：

$$
X_o=[O',|O'|]\in\mathbb{R}^{B\times48\times H_s\times W_s}.
$$

```text
Conv1x1 48->32, no bias
LayerNorm2d(32)
GELU
```

### 9.2 Contour Expert

$$
X_e=[E',|E'|]\in\mathbb{R}^{B\times48\times H_s\times W_s}.
$$

使用独立的 `Conv1x1 48->32 + LayerNorm2d + GELU`。

### 9.3 Stability Expert

$$
X_a=[A',PC,\rho]\in\mathbb{R}^{B\times33\times H_s\times W_s}.
$$

使用 `Conv1x1 33->32 + LayerNorm2d + GELU`。

三个 expert encoder 的参数在三个图像尺度之间共享，但 Odd、Even、Stable 之间不共享。

---

## 10. Dense Orientation-Aligned Neighborhood

### 10.1 空间坐标对齐

V0 只对齐方向通道；V1 同时对齐局部空间邻域。基础 5x5 offsets 为：

$$
\mathcal P=\{(-2,-2),\ldots,(2,2)\}.
$$

对 canonical patch offset `p`，在原图特征上采样：

$$
p'_s(x)=R_{\theta_s(x)}p.
$$

`torchvision.ops.deform_conv2d` 使用的实际 offset 为：

$$
\Delta p_s(x)=R_{\theta_s(x)}p-p.
$$

这样，图像旋转 `alpha` 后主方向变为 `theta+alpha`，采样窗口也随之旋转，从而把局部邻域映射到相同 canonical frame。

### 10.2 无符号方向的 180 度歧义

主方向满足 `theta` 与 `theta+pi` 等价，但普通 5x5 kernel 对旋转 180 度的 patch 不一定输出相同结果。V1 对每个 depthwise OAN kernel 强制中心对称：

$$
W_{sym}=\frac{1}{2}\left(W_{raw}+\operatorname{rot180}(W_{raw})\right).
$$

因此：

$$
OAN(C,\theta)=OAN(C,\theta+\pi).
$$

这个约束是 OAN 的必要组成，不是可选正则项。

### 10.3 OAN 模块

每个 expert 使用独立 OAN，参数跨三个图像尺度共享：

```text
ReflectPad2
Depthwise DeformConv 5x5, 32 channels, center-symmetric weights
Conv1x1 32->32, no bias
LayerNorm2d(32)
GELU
```

为了避免显式创建 `[B,C,H,W,25]`，offset field 只生成 `[B,50,H,W]`，由 deformable convolution执行采样。

当前 `torchvision 0.16.1` 的 deformable convolution不支持 BF16。OAN 必须局部关闭 autocast：

```python
with torch.autocast(device_type="cuda", enabled=False):
    aligned = oan(feature.float(), theta.float())
aligned = aligned.to(feature.dtype)
```

最终继续进行方向置信度门控：

$$
Z'=(1-\rho_s)Z+\rho_s OAN(Z,\theta_s).
$$

---

## 11. 固定抗混叠下采样

V1 删除 V0 的六个独立 learnable stride-2 depthwise blocks。使用固定 BlurPool：

$$
b=[1,4,6,4,1],\qquad
B=\frac{b^Tb}{256}.
$$

BlurPool 使用 5x5 depthwise fixed convolution，stride 为 2。每次 BlurPool 后使用同一个跨 expert、跨尺度、跨层级共享的 pointwise refinement：

```text
Conv1x1 32->32, no bias
LayerNorm2d(32)
GELU
```

三条路径分别下采样 3、2、1 次到 `[B,32,64,64]`。由于 BlurPool 是各向同性固定滤波，且 refinement 只有 1x1，它不会重新引入一个未对齐的普通空间卷积后端。

方向不能直接平均角度。每个尺度先下采样：

$$
o_s=[\cos(2\theta_s),\sin(2\theta_s)],
$$

再重新单位化；`rho` 使用同一固定 BlurPool 下采样。

---

## 12. 显式 Cross-scale Stability

每个尺度首先根据三个 expert 构造用于比较的 32 维尺度表示：

$$
U_s=\operatorname{Normalize}\left(
\operatorname{LN}\left(
\operatorname{Conv}_{1\times1}
[Z^o_s;Z^e_s;Z^a_s]
\right)\right).
$$

三个尺度两两计算 cosine agreement：

$$
a_{12}=U_1^TU_2,\qquad
a_{23}=U_2^TU_3,\qquad
a_{13}=U_1^TU_3,
$$

以及逐通道差异：

$$
D_{12}=|U_1-U_2|,\quad
D_{23}=|U_2-U_3|,\quad
D_{13}=|U_1-U_3|.
$$

Scale Gate 的输入通道固定为：

```text
U1, U2, U3                 96 channels
D12, D23, D13              96 channels
a12, a23, a13               3 channels
rho1, rho2, rho3             3 channels
total                      198 channels
```

门控网络为：

```text
Conv1x1 198->64, no bias
LayerNorm2d(64)
GELU
Conv1x1 64->3, with bias
Softmax over 3 scales
```

得到逐位置尺度权重 `w1,w2,w3`。同一组权重分别融合三个 expert：

$$
P_{edge}=\operatorname{Normalize}\left(\sum_s w_sZ^o_s\right),
$$

$$
P_{contour}=\operatorname{Normalize}\left(\sum_s w_sZ^e_s\right),
$$

$$
P_{stable}=\operatorname{Normalize}\left(\sum_s w_sZ^a_s\right).
$$

方向输出使用同一尺度权重对 doubled-angle vector 加权后重新归一化；置信度输出为尺度加权 `rho`。

---

## 13. Late Expert Coupling

Expert Gate 输入：

```text
P_edge, P_contour, P_stable  96 channels
a12, a23, a13                 3 channels
fused rho                      1 channel
total                        100 channels
```

门控网络为：

```text
Conv1x1 100->32, no bias
LayerNorm2d(32)
GELU
Conv1x1 32->3, with bias
Softmax over edge/contour/stable
```

融合为：

$$
Z=g_oP_{edge}+g_eP_{contour}+g_aP_{stable}.
$$

最终描述头不增加普通 3x3 卷积：

```text
Conv1x1 32->128, no bias
LayerNorm2d(128)
L2 normalize
```

---

## 14. 默认参数量估算

参数量按上述固定 32 维 expert 配置估算：

| 模块 | 参数量 |
| --- | ---: |
| 三频物理参数 `lambda/sigma/gamma` | 9 |
| Edge Pointwise Encoder | 1,600 |
| Contour Pointwise Encoder | 1,600 |
| Stability Pointwise Encoder | 1,120 |
| 三个 OAN | 5,664 |
| Shared Downsample Refinement | 1,088 |
| Scale Proxy `96->32` | 3,136 |
| Scale Gate | 12,995 |
| Expert Gate | 3,363 |
| Final Descriptor Head | 4,352 |
| **V1-Core Encoder 预计总计** | **34,927** |

说明：

- 固定 LDN、BlurPool、offset base grid 不计参数；
- OAN 原始 depthwise weight 按 32x25 计数，中心对称化不改变 Parameter 数量；
- 训练模块额外包含一个 fused learnable temperature；
- 参数量低于 V0 的 44,899，但计算量会显著高于 V0，主要来自九组多频滤波和三个原分辨率 OAN。

---

## 15. Coarse GT 与训练损失

V1 复用 V0 的 1/8 Homography GT：对 64x64 token 中心做正向投影、逆向 round-trip 检查，只保留离散双向一致且未越界的 `(b,i,j)`。

总损失固定为：

$$
L=L_{PP}^{fused}+0.1L_{orientation}+0.1L_{branch}.
$$

实现中的建议键名依次为 `L_PP_fused`、`L_orientation` 和 `L_branch`，避免训练日志与公式记号之间产生歧义。

### 15.1 Fused PP Loss

`P_fused` 使用与 V0 相同的 partial dual-softmax focal loss：

- 使用 90% 有效 GT；
- focal gamma 为 2；
- temperature 从 0.05 开始学习；
- full 和 chunked 模式保持数学等价。

### 15.2 Orientation Equivariance Loss

对源点 `x`，Homography 写为：

$$
H(x,y)=\left(\frac{n_x}{d},\frac{n_y}{d}\right).
$$

局部 Jacobian 为：

$$
J_H(x)=
\begin{bmatrix}
\frac{h_{11}d-h_{31}n_x}{d^2} & \frac{h_{12}d-h_{32}n_x}{d^2}\\
\frac{h_{21}d-h_{31}n_y}{d^2} & \frac{h_{22}d-h_{32}n_y}{d^2}
\end{bmatrix}.
$$

源方向向量为：

$$
u_0=[\cos\theta_0,\sin\theta_0]^T.
$$

投影后的理论方向为：

$$
u_1^*=\frac{J_Hu_0}{\|J_Hu_0\|_2+10^{-6}}.
$$

因为方向无符号，损失使用绝对内积：

$$
L_{orientation}=
\frac{\sum_i \bar\rho_i
\left(1-|u_{1,i}^Tu_{1,i}^*|\right)}
{\sum_i\bar\rho_i+10^{-6}},
$$

其中：

$$
\bar\rho_i=\operatorname{stopgrad}
\left(\min(\rho_{0,i},\rho_{1,j^*})\right).
$$

### 15.3 Branch Auxiliary Loss

三个 32 维 expert 都使用独立的辅助 matching loss，防止 Expert Gate 永久忽略某个物理分支：

$$
L_{branch}=\frac{1}{3}
\left(L_{PP}^{edge}+L_{PP}^{contour}+L_{PP}^{stable}\right).
$$

辅助分支固定采用：

- 25% GT positives；
- chunk size 256；
- 固定 temperature 0.07；
- gamma 2；
- chunked partial dual-softmax，不构造三个完整 4096x4096 矩阵。

---

## 16. 优化器、数据和 checkpoint

### 16.1 参数组

| 参数组 | LR | Weight decay |
| --- | ---: | ---: |
| `lambda/sigma/gamma` | `1e-5` | 0 |
| fused temperature | `1e-4` | 0 |
| 其余 V1 参数 | `1e-4` | `0.01` |

优化器使用 AdamW，scheduler 使用 20 epoch CosineAnnealingLR，最低 LR 为 `1e-6`，gradient clipping 为 1.0。

主训练使用 BF16 mixed precision，只有解析核生成、方向估计和 OAN deformable convolution强制 FP32。

### 16.2 公平训练子集

首轮 V1-Core 不重新随机抽样，直接复用 Tiny-CNN 的 30% 训练记录：

```text
logs/tb_logs/physical_v0/tiny_cnn_ratio30_gpu3_bs8_seed66/
selected_train_rows.jsonl
```

固定配置：

| 项目 | 数值 |
| --- | --- |
| base rows | 16,337 |
| online variants | 每张图每 epoch 从 translation/scale/yaw/pitch/roll 随机选择一种 |
| pairs per epoch | 16,337 |
| seed | 66 |
| epochs | 20 |
| validation | 100% |
| test | 100% |

只有 30% 公平实验达到成功标准后，才训练 100% V1-Core。

### 16.3 Batch 与恢复

有效全局 batch 固定为 8：

| GPU 数 | per-GPU batch | accumulate gradients |
| ---: | ---: | ---: |
| 1 | 1 | 8 |
| 1 | 2 | 4 |
| 2 | 1 | 4 |
| 2 | 2 | 2 |

Smoke test 按 `2 -> 1` 尝试 per-GPU batch，以峰值显存不超过 22 GiB 为安全线。在满足安全线的方案中选择 samples/s 最高者，不能通过改变有效全局 batch 获得速度优势。

checkpoint 每 2 epoch 保存一次，`best` 监控严格指标 `val/all_r0`，mode 为 `max`；R@1 只记录，不用于选择最佳权重。

---

## 17. 验证与多模态测评

### 17.1 合成验证

沿用 V0 指标：

- 总体及五种扰动分别统计 R@0、R@1；
- positive similarity；
- hardest-negative similarity；
- margin；
- normalized entropy；
- 三个尺度权重和三个 expert 权重的均值、标准差及熵；
- `rho` 的均值、分位数和低置信度比例；
- 三个 branch 的辅助 R@0/R@1。

### 17.2 真实多模态

沿用现有两个 manifest：

```text
test_SwinMatcher_proposed_gt.jsonl
test_SwinMatcher_expanded_MRSI_gt.jsonl
```

Standalone 使用 `P_fused` 的 cosine mutual-nearest-neighbor，不使用 RANSAC，输出 NCM、Precision、SR、RMSE 和 runtime。

同时对 `P_fused/P_edge/P_contour/P_stable` 分别计算与官方 SLiM 粗特征的 PP/FF 四象限和 CRR，判断互补性究竟来自哪个专家。

---

## 18. 消融配置

V1 采用同一实现，通过显式配置开关产生六个首轮模型：

| 模型 | 改动 |
| --- | --- |
| V1-Core (`physical_v1_core`) | 完整 V1-Core |
| V1-NoOAN (`physical_v1_no_oan`) | OAN 恒等映射，只保留方向通道 canonicalization |
| V1-EnergyOnly (`physical_v1_energy_only`) | 删除 Odd/Even expert，仅保留 Energy/Phase Stability |
| V1-NoConfidenceGate (`physical_v1_no_confidence_gate`) | canonicalization 和 OAN 始终启用，等价 `rho=1` |
| V1-SimpleScaleFusion (`physical_v1_simple_scale`) | 删除 agreement/difference，改为 V0 式 concat scale attention |
| V1-FixedBank (`physical_v1_fixed_bank`) | 固定 `lambda/sigma/gamma`，不更新物理参数 |

对应未来配置接口固定为：

```python
enable_oan: bool
response_mode: Literal["full", "energy_only"]
confidence_gate: bool
scale_fusion: Literal["stability", "simple"]
bank_mode: Literal["parameterized", "fixed"]
```

首轮执行顺序：

1. V1-Core 与 V1-NoOAN；
2. 若 OAN 没有提升 Roll/Yaw/Pitch R@0 或跨模态 CRR，停止后续 OAN 扩展；
3. V1-EnergyOnly，验证 odd/even/phase 分离；
4. 其余三个消融验证 gate、scale stability 和 bank adaptation。

---

## 19. 单元测试与 Smoke Test

### 19.1 CPU 单元测试

- `lambda/sigma/gamma` 始终位于约束范围；
- 同频八方向由同一参数解析生成；
- odd/even 在离散误差范围内满足 quadrature 关系；
- 每个 Gabor 核零均值且 L2 范数为 1；
- phase agreement 位于 `[0,1]` 且无 NaN/Inf；
- `rho=0` 时 channel canonicalization 和 OAN 完全旁路；
- `rho=1` 时输出等于完整 canonicalization；
- OAN 有效 kernel 满足 `W=rot180(W)`；
- `theta` 与 `theta+pi` 的 OAN 输出数值一致；
- BlurPool 三条路径都输出 64x64；
- 所有返回字段尺寸正确；
- 四种描述子的通道 L2 范数为 1；
- 所有 loss 都能向对应模块反传。

### 19.2 GPU Smoke Test

- 验证 BF16 主路径和 FP32 OAN 可以混合前向、反向；
- per-GPU batch 2 和 1 分别跑 2 train batch、2 val batch；
- 记录 peak allocated/reserved memory、samples/s 和 OAN 耗时占比；
- 确认 DDP 下没有 unused parameters；
- checkpoint 保存、恢复后下一 batch 的 loss 与未中断运行一致；
- 强制测试 deformable-convolution dtype，发现不支持时立即报出明确错误，不静默退化为普通卷积。

---

## 20. 成功判据

当前基线来自已完成的 V0、Tiny-CNN 和官方 SLiM 测评。

### 20.1 通用成功

V1-Core 必须在两个数据集的总体 CRR@1 同时超过 Tiny-CNN：

| 数据集 | Tiny-CNN CRR@1 |
| --- | ---: |
| SwinMatcher Proposed | 4.22% |
| Expanded MRSI | 3.97% |

达到该条件后，V1 才能被定义为通用物理辅助分支候选。

### 20.2 定向成功

如果总体未超过 Tiny，但同时达到：

| 场景 | Physical V0 基线 |
| --- | --- |
| Proposed Optical-SAR | `Pre=10.59%` 且 `CRR@1=7.83%` |
| MRSI Optical-Infrared | `Pre=7.76%` 且 `CRR@1=6.83%` |

则 V1 定位为 SAR/Infrared 门控专家，不作为通用分支。

### 20.3 失败条件

如果既没有超过 Tiny 的总体 CRR@1，也没有同时超过两个定向 V0 基线，则：

- 不进入 SLiM 融合阶段；
- 不训练 100% 数据版本；
- 根据消融判断是 OAN、phase separation 还是 scale stability 无效；
- 保留实验作为 V1 负结果，不继续增加自由 CNN 容量。

---

## 21. 与总体路线的关系

Physical V1-Core 只对应 `总体路线.md` 中的 Physical Warm-up。它验证：

$$
\text{LDN}
+\text{multi-frequency quadrature}
+\text{odd/even/phase separation}
+\text{confidence-gated channel and spatial canonicalization}
+\text{cross-scale stability}.
$$

以下内容必须留到 V1 达标以后：

```text
Repeatability R
CleanDIFT Teacher
Semantic Student
Physical-Queried Semantic Aggregation
SLiM residual/score fusion
PP/PF/FP/FF four-path matching
```

V1 的最小研究问题为：

> Can explicit channel-and-spatial orientation canonicalization, multi-frequency phase structure, and cross-scale stability produce more modality-stable and SLiM-complementary dense correspondences than both Physical V0 and a parameter-matched generic CNN?

---

## 22. 实现映射与 Smoke 结论

代码落点：

| 文件 | 作用 |
| --- | --- |
| `src/physical/v1_models.py` | V1-Core、六种消融、参数化滤波器、canonicalization、OAN 和双层门控 |
| `src/physical/v1_losses.py` | Homography Jacobian 与无符号方向等变损失 |
| `src/physical/v1_lightning.py` | 三项损失、优化器参数组、验证指标和诊断日志 |
| `train_physical_v1.py` | 独立训练、固定有效 batch、checkpoint、恢复与实验元数据 |
| `test/evaluate_physical_v1.py` | 四分支合成指标、多模态论文指标及 SLiM CRR |
| `tests/test_physical_v1.py` | 解析核、OAN、输出接口、反向传播和选样确定性测试 |

GPU 3、RTX 4090、512x512、BF16、chunked PP 的 smoke 结果：

| GPU 数 | per-GPU batch | accumulation | effective batch | peak allocated/GPU | peak reserved/GPU | global samples/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 8 | 8 | 4.75 GiB | 5.98 GiB | 0.98 |
| 1 | 2 | 4 | 8 | 9.34 GiB | 11.83 GiB | 1.45 |
| 2 | 2 | 2 | 8 | 9.34 GiB | 11.83 GiB | 2.79 |

三种配置均低于 22 GiB 安全线且参数、loss、checkpoint 无 NaN/Inf。双卡 DDP 未出现 unused parameter。正式单卡实验默认使用 `batch_size=2`、`accumulate_grad_batches=4`；双卡使用 `batch_size=2`、`accumulate_grad_batches=2`；验证 batch 固定为 1。

训练集不再在单个 epoch 内将每张基础影像展开为五对。扰动类型由 `seed + epoch + row index` 确定，同一实验和恢复训练可复现，不同 epoch 会重新选择。验证与测试仍完整展开五种扰动，保证分类型指标不变。

---

## 23. 实验运行基线与已验证经验

> 本节记录截至 2026-07-21 已经实际运行验证的配置、结果与操作约定。后续继续 V1 实验时，以本节为运行基线，不依赖聊天记录回忆。

### 23.1 当前训练范围

`train_physical_v1.py` 训练的是独立 Physical V1 编码器，不加载 SLiM checkpoint，也不更新 SLiM Backbone、Fine Matching 或 Refinement。优化器包含：

| 参数组 | 内容 | 学习率 | Weight decay |
| --- | --- | ---: | ---: |
| `physical_filters` | 受约束 Gabor `lambda/sigma/gamma` | `1e-5` | `0` |
| `encoder` | Pointwise experts、OAN、尺度门控、专家门控、描述子头 | `1e-4` | `0.01` |
| `temperature` | Fused PP loss 的可学习温度 | `1e-4` | `0` |

Branch PP loss 的温度固定，不参与更新。训练 checkpoint 保存 Physical V1、loss 温度、优化器、调度器和 epoch/global-step 状态，不包含 SLiM 权重。

### 23.2 固定 30% 训练子集

原始训练索引为：

```text
data/remote_archive/manifests/train_optical_single_images.jsonl
54,455 rows
```

正式 V1-Core 与所有消融固定复用以下分层抽样结果：

```text
logs/tb_logs/physical_v0/tiny_cnn_ratio30_gpu3_bs8_seed66/selected_train_rows.jsonl
16,337 rows
```

该文件按 `dataset/subset` 分层并使用 seed 66 生成，是 Tiny-CNN、V1-Core 和 V1 消融公平比较的共同样本集合。它是关键复现实验资产，不能作为普通 smoke 日志删除。

正式实验必须传入 `--selected_train_rows`，并且不能同时使用 `--resample_train_subset`。提供固定文件后，实际样本由文件内容决定；`--train_data_ratio 0.3` 主要保留为实验语义和 W&B 元数据。每个新实验目录会再次保存自己的 `selected_train_rows.jsonl` 副本。

训练使用 `--train_one_variant_per_row`：每张基础影像每个 epoch 只在线生成一种扰动，因此正式训练为 `16,337 pairs/epoch`，不是五倍展开后的 81,685 对。扰动由 `seed + epoch + row index` 确定，恢复训练和不同模型间可复现。

验证索引包含 5,164 张基础影像，验证阶段仍展开五种扰动：

```text
5,164 x 5 = 25,820 validation pairs
```

验证集不受训练比例影响，也不能在正式结果中使用 pilot 的 `--max_val_rows 100`。

### 23.3 Batch、梯度累积与显存结论

`--batch_size` 是每张 GPU 的 micro batch；`--effective_batch_size` 是每次优化器更新累计使用的全局样本数：

$$
N_{accum}=\frac{B_{effective}}{B_{per\ GPU}\times N_{GPU}}.
$$

固定 `effective_batch_size=8`，确保单卡、双卡、V1-Core 和所有消融的优化条件一致：

| 模式 | 物理 GPU | 程序逻辑设备 | per-GPU batch | accumulation | effective batch |
| --- | --- | --- | ---: | ---: | ---: |
| 单卡 | `3` | `"0,"` | 2 | 4 | 8 |
| 双卡 | `0,3` | `"0,1,"` | 2 | 2 | 8 |

双卡 per-GPU batch 4 已实际复现为反向传播 OOM：每卡当时分配 15.38 GiB、reserved 4.40 GiB，反向仍需申请 3.12 GiB，而可用显存只有 3.10 GiB。因此 24 GiB RTX 4090 的正式安全配置固定为 per-GPU batch 2，不再尝试 batch 4。

双卡 batch 2 的正式 pilot 实测每卡 peak allocated 约 9.34 GiB、peak reserved 约 11.84 GiB，吞吐量约 3.25 samples/s。验证 batch 固定为 1。

当使用 `CUDA_VISIBLE_DEVICES=0,3` 时，程序只看见两张卡并重新编号为逻辑 `0,1`，所以必须写 `--device "0,1,"`，不能写 `--device "0,3,"`。单独暴露物理 GPU 3 时应写：

```bash
CUDA_VISIBLE_DEVICES=3 ... --device "0,"
```

### 23.4 1% Pilot 结果

保留的有效 pilot 位于：

```text
logs/tb_logs/physical_v1/v1_core_ratio01_seed66_pilot_bs2
```

配置为 545 张基础训练影像、100 张基础验证影像、双卡 per-GPU batch 2、effective batch 8、3 epoch。其收敛结果为：

| Epoch | Train loss | Val R@0 | Val R@1 |
| ---: | ---: | ---: | ---: |
| 0 | 12.15 | 39.80% | 50.92% |
| 1 | 7.98 | 45.43% | 57.93% |
| 2 | 6.79 | 47.38% | 60.27% |

最佳 checkpoint：

```text
logs/tb_logs/physical_v1/v1_core_ratio01_seed66_pilot_bs2/checkpoints/best-02.ckpt
```

Epoch 2 五种扰动指标：

| 扰动 | R@0 | R@1 |
| --- | ---: | ---: |
| Pitch | 43.24% | 56.38% |
| Roll | 45.21% | 57.44% |
| Yaw | 46.06% | 59.55% |
| Translation | 50.82% | 63.36% |
| Scale | 51.13% | 64.27% |

Pilot 证明实现、DDP、反向传播、checkpoint 和验证链路正常，但不能代替真实多模态结论。需要继续关注：

- hardest-negative similarity 仍高于 positive similarity，Epoch 2 margin 为 `-0.063`；
- Expert Gate 平均权重约为 Edge 9.2%、Contour 76.4%、Stable 14.5%，存在偏向 Contour 的趋势；
- 三尺度平均权重约为 2.7%、44.0%、53.3%，第一个尺度贡献很低；
- 平均方向置信度约 0.908，低置信旁路很少触发。

这些现象必须通过 30% 正式训练和既定消融判断，不能仅凭 pilot 修改结构。

### 23.5 W&B 约定

Physical V1 默认 W&B 模式为 `online`。日志按以下三层组织：

```text
Project: slim_physical_v1
Group:   数据与实验系列，例如 physical_v1_optical_single_ratio30
Run:     模型_数据_比例_尺寸_GPU_batch_seed_epoch_相似度模式
```

正式 V1-Core 名称固定为：

```text
Group: physical_v1_optical_single_ratio30
Run:   physical_v1_core_optical_single_ratio30_img512_1gpu3_bs2_ebs8_seed66_ep20_chunked
```

每个消融必须只替换模型名，并保留其余字段，使 W&B 图表可以直接公平对比。`--wandb_log_model false` 表示 checkpoint 仍保存到本地实验目录，但不重复上传为 W&B Artifact。

### 23.6 正式单卡训练命令

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib \
/root/miniconda3/envs/slim/bin/python train_physical_v1.py \
  --model physical_v1_core \
  --device "0," \
  --selected_train_rows logs/tb_logs/physical_v0/tiny_cnn_ratio30_gpu3_bs8_seed66/selected_train_rows.jsonl \
  --train_data_ratio 0.3 \
  --train_one_variant_per_row \
  --task_name physical_v1_optical_single_ratio30 \
  --run_name physical_v1_core_optical_single_ratio30_img512_1gpu3_bs2_ebs8_seed66_ep20_chunked \
  --batch_size 2 \
  --val_batch_size 1 \
  --effective_batch_size 8 \
  --num_workers 6 \
  --max_epochs 20 \
  --similarity_mode chunked \
  --chunk_size 256 \
  --seed 66 \
  --save_every_n_epochs 2 \
  --use_wandb \
  --wandb_project slim_physical_v1 \
  --wandb_mode online \
  --wandb_log_model false
```

双卡时只修改以下三项，并同步修改 run name：

```bash
CUDA_VISIBLE_DEVICES=0,3
--device "0,1,"
--run_name physical_v1_core_optical_single_ratio30_img512_2gpu03_bs2_ebs8_seed66_ep20_chunked
```

`batch_size=2` 和 `effective_batch_size=8` 保持不变，程序会将 accumulation 从单卡 4 自动改为双卡 2。

### 23.7 Shell 命令注意事项

多行 Bash 命令中的反斜杠必须是该行最后一个字符：

```bash
# 正确
--device "0," \
  --batch_size 2

# 错误：反斜杠后存在空格
--device "0," \ 
  --batch_size 2
```

反斜杠后的空格会导致 `bash: command not found`、`unrecognized arguments`，并使后续参数被当作新的命令。`RequestsDependencyWarning` 是当前 requests 依赖组合的警告，与该类命令解析失败无关。

### 23.8 后续实验顺序

1. 完成 V1-Core 30% 正式训练，以 `val/all_r0` 选择最佳 checkpoint；
2. 在两个 SwinMatcher 多模态 manifest 上运行 standalone 与 SLiM CRR 测评；
3. 固定同一份 16,337-row 训练子集和全部超参数，依次训练 `physical_v1_no_oan`、`physical_v1_energy_only`、`physical_v1_no_confidence_gate`、`physical_v1_simple_scale`、`physical_v1_fixed_bank`；
4. 依据第 20 节成功判据决定是否进入 100% 数据训练和 SLiM 融合阶段。

内部 Edge、Contour、Stable 的单独验证指标只是 V1-Core 的分支诊断，不代表消融模型已经同步训练。每一种消融都必须拥有独立 run、checkpoint 和多模态测评结果。
