
对，现阶段应该把问题压缩成一个非常明确的实验：

> **先证明 Physical Encoder 自己确实学到了不同于普通深度特征的、对几何变化和模态变化更稳定的对应信息。**

在这个问题没有回答之前，不应该加入 CleanDIFT、四路交叉、Repeatability Head 等模块。否则最后即使性能提升，也不知道到底是谁起作用。

我建议把第一版定成下面这样。

---

# 一、Physical Encoder V0：最小实现

输入仍然使用你当前 SLiM 的灰度图：

[
I\in\mathbb R^{B\times1\times512\times512}
]

输出：

[
P\in\mathbb R^{B\times128\times64\times64}.
]

暂时只输出一个物理描述特征 (P)。

**先不要输出 Repeatability (R)，也先不要单独训练 Orientation (O)。**

完整数据流：

```text
Gray Image
   [1×512×512]
        │
        ▼
Multi-scale Image Pyramid
   ┌────┼────┐
   ×1  ×1/2 ×1/4
   │    │    │
   ▼    ▼    ▼
Fixed Quadrature Orientation Bank
   │    │    │
Odd / Even Responses
   │    │    │
Orientation Energy
   │    │    │
Soft Major Orientation
   │    │    │
Orientation-Axis Canonicalization
   │    │    │
Local Structure Encoding
   │    │    │
   └────┼────┘
        │
Cross-Scale Fusion
        │
Descriptor Head
        │
        ▼
 P [128×64×64]
```

这版只回答一个问题：

> 物理结构归纳偏置能不能产生一个有效的 dense matching representation？

---

# 二、第一层：固定 Quadrature Orientation Bank

第一版我反而不建议一开始就把滤波器全部做成可学习的。

否则假设训练有效，你无法判断提升来自：

* HIMO启发的物理结构；
* 还是普通卷积网络自己学出来的。

所以 V0 最好采用：

[
\boxed{\text{Fixed Physical Front-end + Learnable Descriptor Back-end}}
]

---

## 2.1 多尺度输入

生成：

[
I^1,\qquad I^{1/2},\qquad I^{1/4}.
]

例如512输入：

```text
Scale 1    512×512
Scale 2    256×256
Scale 3    128×128
```

---

## 2.2 方向滤波

每个尺度使用：

[
K=8
]

个无符号方向：

[
\theta_k=\frac{k\pi}{8},
\qquad k=0,\ldots,7.
]

每个方向都有一对 quadrature filters：

[
K_k^e,\qquad K_k^o.
]

分别计算：

[
E_{s,k}=I_s*K_k^e,
]

[
O_{s,k}=I_s*K_k^o.
]

然后构造方向能量：

[
A_{s,k}
=======

\sqrt{
E_{s,k}^2+
O_{s,k}^2+
\epsilon
}.
]

这里最重要的是：

[
A_{s,k}
]

对 odd/even 响应的正负不敏感。

这对于跨模态是合理的，因为同一个结构在不同成像模式下可能出现：

```text
bright → dark
edge polarity reversal
ridge → valley
```

但是局部结构方向仍可能保持。

这正是 HIMO 中利用奇偶相位结构的重要出发点。

### 第一版滤波器选择

我建议先用固定的 **Gabor / Log-Gabor-like quadrature bank**。

不要第一版就自己学习 kernel。

后续消融再做：

```text
Fixed
vs
Fixed + Learnable Residual
vs
Fully Learnable Steerable
```

---

# 三、Soft Major Orientation

对于每个尺度：

[
A_s
\in
\mathbb R^{B\times8\times H_s\times W_s}.
]

计算：

[
\pi_{s,k}(x)
============

\operatorname{Softmax}*k
\left(
\frac{A*{s,k}(x)}{\tau_o}
\right).
]

再使用 doubled-angle orientation：

[
v_x=
\sum_k
\pi_k\cos2\theta_k,
]

[
v_y=
\sum_k
\pi_k\sin2\theta_k.
]

于是：

[
\theta(x)
=========

\frac12
\operatorname{atan2}(v_y,v_x).
]

使用 (2\theta) 的原因是：

[
\theta
\equiv
\theta+\pi.
]

例如一条道路方向是：

[
20^\circ
]

还是：

[
200^\circ
]

对于结构匹配而言应该视为同一方向。

---

# 四、Orientation-Axis Canonicalization

这是 V0 中真正值得重点验证的模块。

原始方向特征：

[
A_s(x)
======

[A_{s,0},A_{s,1},\ldots,A_{s,7}].
]

如果图像旋转：

[
45^\circ,
]

这些方向响应大体会在 orientation channel 上移动两个位置。

因此不直接旋转图像 patch，而根据局部主方向：

[
\delta(x)
=========

\frac{K\theta(x)}{\pi}
]

对 orientation channel 做连续循环对齐：

[
\widetilde A_{s,k}(x)
=====================

\operatorname{CircularSample}
\left(
A_s(x),
k+\delta(x)
\right).
]

数据流就是：

```text
原图

方向通道
0  22.5  45  67.5 ...
█   ▂     ▂    ▂

旋转45°
↓
▂   ▂     █    ▂

根据主方向重新对齐
↓
█   ▂     ▂    ▂
```

第一版只 canonicalize：

[
A_{s,k}.
]

不要同时处理几十组复杂特征。

这样最容易验证这个设计本身有没有价值。

---

# 五、每个尺度的轻量 Structure Encoder

Canonicalized feature只有8个方向通道：

[
\widetilde A_s\in\mathbb R^{B\times8\times H_s\times W_s}.
]

每个尺度经过同一个共享的小网络：

```text
Conv 1×1
8 → 32

Depthwise Conv 3×3

GELU

Conv 1×1
32 → 64

Depthwise Conv 3×3

Conv 1×1
64 → 64
```

得到：

[
Z_s\in\mathbb R^{B\times64\times H_s\times W_s}.
]

这里建议三个尺度**共享参数**。

原因是你希望尺度发生变化时：

[
Z_1,\ Z_2,\ Z_3
]

处于同一个描述空间。

---

# 六、对齐到1/8分辨率

全部变成：

[
64\times64.
]

例如：

[
\hat Z_s
========

D_s(Z_s).
]

这里不要简单把512特征一次 average pooling 8倍。

可以：

```text
高分辨率 Z1
→ stride2 block
→ stride2 block
→ stride2 block

Z2
→ stride2
→ stride2

Z3
→ stride2
```

最终全部：

[
[B,64,64,64].
]

---

# 七、最小 Cross-Scale Fusion

V0先不要设计复杂 DoFS 网络。

直接把三个尺度拼起来：

[
Z=
[Z_1;Z_2;Z_3]
\in
\mathbb R^{B\times192\times64\times64}.
]

预测每个位置的三个尺度权重：

[
[w_1,w_2,w_3]
=============

\operatorname{Softmax}
\left(
Conv_{1\times1}(Z)
\right).
]

然后：

[
Z_{phy}
=======

w_1Z_1+w_2Z_2+w_3Z_3.
]

最后：

```text
64-d
 ↓
3×3 Depthwise Conv
 ↓
1×1 Conv
 ↓
128-d
 ↓
LayerNorm2d
 ↓
L2 normalize
```

得到：

[
P\in
\mathbb R^{B\times128\times64\times64}.
]

---

# 八、V0完整数据流

对于一对训练图像：

[
I_0,\qquad I_1
]

分别独立经过**完全共享权重**的 Physical Encoder：

```text
I0                                  I1
│                                   │
Physical Encoder                    Physical Encoder
│                                   │
P0                                  P1
[B,128,64,64]                       [B,128,64,64]
│                                   │
└───────────────┬───────────────────┘
                │
          Flatten + Dot Product
                │
                ▼
       S_PP [B,4096,4096]
                │
          Existing GT Matrix
                │
                ▼
             L_PP
```

你的当前数据集已经提供 single-image synthetic homography：

[
I\rightarrow I_0,I_1,H_{0\rightarrow1},
]

并且现有代码能够根据 Homography 直接构造 coarse-level `conf_matrix_gt`。

所以第一版几乎不需要改数据集。

---

# 九、第一版训练只使用一个 Loss

我建议最小实验阶段就用：

[
\boxed{
L=L_{PP}
}
]

不要一开始加入：

[
L_{ori},
L_{rep},
L_{scale}.
]

因为固定方向滤波器已经具有方向结构，canonicalization也已经显式存在。

第一步就是看：

> 仅靠这种网络结构，加上普通 matching supervision，能不能学出有效 descriptor？

Similarity：

[
S_{PP}
======

\frac{
P_0P_1^\top
}{
\tau
}.
]

直接复用 SLiM 当前 coarse loss即可。当前 coarse loss本身就是利用 GT correspondence 进行双方向 partial softmax 和 focal supervision。

这非常适合第一版。

---

# 十、不要马上接进SLiM，先单独验证

我建议顺序是：

## Experiment 1：Physical Encoder standalone

只训练：

```text
Optical single images
        ↓
Synthetic Homography
        ↓
Physical Encoder
        ↓
PP matching
```

完全不加载 SLiM。

因为这样最容易回答：

> Physical Encoder 自己到底有没有能力？

---

# 十一、怎么判断它是否有效

这里不能只看一个 validation loss。

我建议从四个层面判断。

---

## 11.1 第一层：几何匹配能力

在你已有的 optical synthetic validation set 上测试。

对于每一个 coarse token (i)，已知真实对应：

[
j^*.
]

计算预测：

[
\hat j
======

\arg\max_j S_{PP}(i,j).
]

统计：

### Exact coarse Recall

[
R@0
===

P(\hat j=j^*).
]

### 邻域Recall

例如：

[
R@1
===

P(
|\hat j-j^*|_\infty\le1
).
]

分别统计：

```text
Translation
Scale
Roll
Yaw
Pitch
```

你当前数据已经按这些变换类型生成训练样本。

这里尤其重要的是：

### Roll

因为：

> Orientation Canonicalization 理论上最直接应该提升旋转变化。

如果：

```text
No Canonicalization
和
With Canonicalization
```

在 roll 数据上几乎没有差别，那么这个模块可能没有实际贡献。

---

# 十二、第二层：匹配判别性

不仅看对不对，还看匹配峰是否足够清楚。

对于 GT：

[
j^*
]

计算：

[
s^+
===

S(i,j^*).
]

Hardest negative：

[
s^-
===

\max_{j\notin\mathcal N(j^*)}
S(i,j).
]

Margin：

[
M=s^+-s^-.
]

统计：

[
\operatorname{MeanMargin}.
]

还可以计算 Match Distribution Entropy：

[
H_i
===

*

\sum_j
p_{ij}\log p_{ij}.
]

理想情况：

```text
正确位置相似度高
Hard negative低
分布更加尖锐
```

这比只看 loss 更能判断 descriptor 是否真正具有区分性。

---

# 十三、第三层：零样本跨模态测试

这是最关键的。

Physical Encoder训练时只看：

```text
Optical
+
Homography
```

训练完成后直接测试：

```text
Optical–Optical
Optical–Infrared
Optical–SAR
Optical–Map
Optical–Depth
Day–Night
```

不进行任何微调。

但这里不能只看绝对精度。

因为一个轻量 Physical Encoder 很可能：

```text
绝对匹配能力低于完整SLiM
```

这并不代表它没有价值。

我建议计算：

# Cross-Modal Retention

例如：

[
Retention_{\mathrm{SAR}}
========================

\frac{
Pre_{\mathrm{Optical-SAR}}
}{
Pre_{\mathrm{Optical-Optical}}
}.
]

对于你当前 SLiM官方权重：

[
\frac{0.175}{0.493}
\approx0.355.
]

也就是说 SAR 情况下保留约35.5%的 optical-optical precision。你已有测试结果支持这个基线。

假设 Physical Encoder结果：

```text
Optical-Optical Pre = 0.30
Optical-SAR     Pre = 0.18
```

绝对SAR还是0.18。

但是：

[
Retention=0.60.
]

这说明它：

> 绝对能力不如完整 SLiM，但对模态变化更加稳定。

那么它作为第二个空间就是有价值的。

---

# 十四、第四层：与SLiM是否互补

这是决定后续要不要做四路的**最重要指标**。

对于每个真实 correspondence：

[
i\rightarrow j^*
]

分别判断：

### Physical

[
PP_{\mathrm{correct}}
]

### SLiM

[
FF_{\mathrm{correct}}.
]

统计四种情况：

| Physical | SLiM | 含义                  |
| -------- | ---- | ------------------- |
| ✓        | ✓    | 两者都能匹配              |
| ✓        | ✗    | **Physical可以救SLiM** |
| ✗        | ✓    | SLiM具有额外任务信息        |
| ✗        | ✗    | 两者都失败               |

最值得关注：

[
\boxed{
P(PP=\checkmark\mid FF=\times)
}
]

即：

> SLiM失败的地方，有多少能被Physical Encoder正确识别？

我会把这个指标直接叫：

## Complementary Recovery Rate

[
CRR=
\frac{
N(PP\ correct,\ FF\ wrong)
}{
N(FF\ wrong)
}.
]

这个指标比单独的 PP precision 还重要。

因为你后续四路的理论前提就是：

[
P\text{和}F
]

存在互补。

假设结果：

```text
SLiM错误点10000个

Physical也错     8500
Physical正确     1500
```

那么：

[
CRR=15%.
]

这说明至少有15%的 SLiM失败 correspondence存在潜在可恢复信息。

如果：

[
CRR\approx1%
]

说明：

```text
Physical Space
基本只是SLiM Feature Space的弱版本
```

那四路设计的意义就很弱。

---

# 十五、必须设置一个普通CNN对照

这一点非常重要。

做一个参数量接近的：

```text
Tiny CNN Encoder
```

例如：

```text
Conv
↓
Residual Block
↓
Downsample
↓
Residual Block
↓
Downsample
↓
128-d feature
```

同样：

```text
Optical-only
Homography
same L_PP
same epochs
same optimizer
```

比较：

```text
Tiny CNN
Physical Encoder
```

否则即使 Physical Encoder有效，也无法证明：

> 是物理结构设计带来的。

可能只是：

> 多加了一个独立网络。

---

# 十六、第一阶段我建议只做四个模型

这样实验非常干净：

### A. Tiny CNN

普通学习式descriptor。

### B. Physical-NoCanon

方向奇偶结构 + 多尺度，但没有方向canonicalization。

### C. Physical-SingleScale

只使用单尺度。

### D. Physical-Full

```text
Quadrature
+
Canonicalization
+
Multi-scale Fusion
```

然后比较：

```text
Synthetic geometric
Cross-modal zero-shot
Cross-modal retention
CRR with SLiM
```

这样你能够分别回答：

> 方向结构有没有作用？

> Canonicalization有没有作用？

> 多尺度有没有作用？

> Physical Space和SLiM是不是互补？

---

# 十七、什么时候接入SLiM

只有 standalone 满足下面三个条件之一，我才建议继续。

### 条件1

跨模态 retention 明显优于 Tiny CNN。

### 条件2

在 SAR、Infrared等至少多个模态上：

[
CRR
]

存在稳定的非零优势。

### 条件3

Canonicalization 在 roll/rotation 测试上表现出明确作用。

满足后，做最简单的融合测试。

---

# 十八、最小SLiM集成方式

先不要四路。

直接：

[
P_i=E_P(I_i)
]

SLiM：

[
F_i\in\mathbb R^{192}.
]

映射：

[
\Delta F_i
==========

W_P(P_i)
\in\mathbb R^{192}.
]

然后：

[
F_i'
====

F_i+
\alpha\Delta F_i.
]

第一版可以：

[
\alpha=0.1
]

或者使用可学习标量。

数据流：

```text
Image
 ├────────────→ SLiM Backbone → F
 │
 └────────────→ Physical Encoder → P
                                  ↓
                              Projector
                                  ↓
                                 ΔF
                                  │
                         F' = F + αΔF
                                  │
                         Original SLiM Coarse
```

其余：

```text
Fine
Refinement
Loss
```

全部不改。

这样你能回答第二个问题：

> Physical特征作为辅助信息，能不能直接改善现有matcher？

---

# 十九、我认为现在最合理的代码实施顺序

```text
src/physical/
├── quadrature_bank.py
├── orientation_canonicalization.py
├── physical_encoder.py
└── physical_matcher.py
```

先实现：

```python
P0 = encoder(image0)
P1 = encoder(image1)

sim_matrix = correlation(P0, P1)

data["sim_matrix"] = sim_matrix
loss = coarse_loss(data)
```

复用你现有：

```text
RemoteSensingHomographyDataset
spvs_coarse_homography
SLiM coarse loss
```

因此第一阶段基本不用重新设计训练基础设施。

---

## 我会把当前最小实验目标定成一句话

[
\boxed{
\text{Can a physically structured dense encoder trained only on optical homography pairs produce more modality-stable and complementary correspondences than a generic CNN?}
}
]

只要这个问题得到肯定结果，后面的研究链条自然成立：

```text
Physical Encoder有效
        ↓
与SLiM Feature互补
        ↓
建立Image Space / Feature Space
        ↓
PP / FF / PF / FP
        ↓
再考虑CleanDIFT
```

现在最值得优先实现的是 **Physical-Full 与 Tiny-CNN 两条完全同训练条件的 standalone baseline**。这会最快告诉你整个 Image Space 方向是否值得继续。
