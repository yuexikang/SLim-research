# Physical Encoder V3 设计初稿

> 状态：结构草案。当前可运行实现为 V3.0.0 无训练基线，版本变更见[Physical Encoder V3 版本更新记录](./Physical%20Encoder%20V3%20版本更新记录.md)。本文只确定冻结 HIMO、紧凑 Physical State 和 Dense Polar Physical Encoder，不锁定最终后端网络，也不进入 SLiM 融合实现。

## 1. 出发点

V2 的公平对照表明，直接将可学习物理分支作为残差加入 SLiM coarse feature，能够扩大候选覆盖并提高 SR，但会引入较多低质量匹配，尤其在 MRSI 上明显降低 Precision 和 NCM。

V3 改变物理模块的职责：

1. HIMO 不再参与端到端优化，而是固定的物理状态生成器；
2. 不让普通神经网络覆盖 Odd、Even 和主方向的物理定义；
3. 神经网络只学习如何编码方向对齐后的局部物理邻域；
4. 第一阶段先验证独立物理描述子的价值，不直接设计 SLiM residual Adapter。

总体形式为：

$$
P=H(I),\qquad
T_i=S_{\mathrm{polar}}(P,x_i),\qquad
D_i=E_{\mathrm{polar}}(T_i).
$$

其中 $H$ 完全冻结，$S_{\mathrm{polar}}$ 是由 HIMO 方向控制的确定性区域划分、直方图统计和 PSD 处理，只有 $E_{\mathrm{polar}}$ 及其后端网络可训练。

## 2. 研究边界

V3-Core 首先回答：

> 固定 HIMO 物理状态经过方向对齐的局部极坐标组织后，能否形成具有几何重复性和跨模态稳定性的 dense descriptor？

本阶段包含：

- 冻结 HIMO；
- 5 通道 Physical State；
- 基于原始 PolarP 分区、方向直方图、主方向和 PSD 的 Dense Polar 编码；
- 小型 Polar Encoder；
- 独立 dense descriptor 训练和评估。

本阶段暂不包含：

- 可学习 Gabor 或可学习 HIMO；
- Pair Transformer；
- SLiM residual Adapter；
- fine matching 和 recurrent refinement；
- 与 SLiM、语义特征或其他分支的融合。

## 3. 总体结构

```text
image I [B,1,512,512]
  -> official HIMO intensity normalization + CoF residual filtering
  -> frozen four-scale / six-orientation Log-Gabor + Odd/Even Sobel
  -> frozen Deep-Shallow Odd / Even responses
  -> frozen MASW + hard Odd/Even orientation coupling
  -> Physical State P [B,5,H0,W0], H0=W0=512 or 256
  -> dense code-faithful PolarP statistics (center + two 12-sector rings)
  -> base direction alignment + PSD reversal handling
  -> small shared structured Polar Encoder
  -> high-resolution local descriptor D0 [B,Cd,H0,W0]
  -> anti-aliased bottom-up feature pyramid
  -> top-down FPN + lateral fusion
  -> descriptors at 256 / 128 / 64
```

冻结 HIMO 的输出必须可以脱离训练代码单独检查、保存和可视化。网络不得访问完整 Log-Gabor 响应堆栈，只能接收 IMO、有效性和约定的紧凑 Physical State。

V3 默认采用 `H0=W0=256`。`512` 分辨率 Polar 编码保留为高成本消融，不作为首轮默认，因为其 anchor 数、Polar 区域统计量和激活显存均为 `256` 版本的 4 倍。

## 4. 冻结 HIMO

### 4.1 来源边界与复现目标

本节以 HIMO 论文公式 (1)-(20) 和作者公开 MATLAB 实现为唯一物理定义来源。实现时必须区分三类内容：

| 标记 | 含义 |
|---|---|
| `HIMO-Original` | 论文与公开代码已经定义的计算，不改变公式 |
| `Code-Reproduction` | 论文描述与公开代码存在实现细节差异时，以固定 commit 的公开代码复现 |
| `V3-Extension` | 为 dense 神经网络接口新增的状态重组或后端，不宣称属于原始 HIMO |

冻结模块定义为：

$$
\mathcal{S}_{\mathrm{HIMO}}=H_{\mathrm{official}}(I).
$$

`HIMO-Original` 公开输出是耦合后的 IMO 方向图和权重/有效性图，不是 V3 的 5 通道状态。V3 只从原始算法内部已经存在的 Odd/Even MASW 中间量构造紧凑状态。

HIMO 中不存在可训练 `Parameter`，始终处于 `eval()`，并在 `torch.no_grad()` 与 FP32 下执行。HIMO 状态不接受描述子损失的梯度。不得用可学习 Gabor、LDN、BatchNorm 或普通 CNN 替换本节任何步骤。

### 4.2 强度归一化、CoF 与尺度空间

官方 demo 在进入 `Build_Himo_Pyramid` 前调用 `Deal_Extreme.p` 和 `Preproscessing.p`，但仓库只发布了 P-code，无法从源码审计其公式。V3 不为这两个黑盒步骤臆造等价公式：冻结 HIMO 的公开可复现边界从传入 `Build_Himo_Pyramid` 的单通道图像 $I$ 开始；端到端回归时由 MATLAB 官方入口导出其预处理结果，作为 PyTorch HIMO 的固定输入夹具。

公开 `Build_Himo_Pyramid.m` 首先对非零像素中位数做强度归一化：

$$
I\leftarrow
\frac{255}{2\,\operatorname{median}(I[I\neq0])}I.
$$

随后在每个 octave 之前执行 Co-occurrence Filter：

$$
I_{\mathrm{co}}(p)=
\frac{\sum_{q\in W_p}G_{\sigma_s}(p,q)M(I_p,I_q)I_q}
{\sum_{q\in W_p}G_{\sigma_s}(p,q)M(I_p,I_q)},
$$

$$
M(a,b)=\frac{C(a,b)}{h(a)h(b)}.
$$

其中：

$$
C(a,b)=
\sum_{p,q}
\exp\left(-\frac{d(p,q)^2}{2\sigma_{\mathrm{oc}}^2}\right)
\mathbf{1}[I_p=a]\mathbf{1}[I_q=b].
$$

公开代码固定：

- `sigma_s=5`；
- `sigma_oc=1.6`；
- 残差混合 $I\leftarrow0.75I+0.25I_{\mathrm{co}}$；
- `nOctaves=3`、`nLayers=2`；
- octave 缩放因子 `G_resize=1.2`；
- Gaussian 基础尺度 `G_sigma=1.6`。

公开 `Gaussian_Scaling` 在每个 octave 的 layer 1 不再额外卷积，layer 2 使用由 `Get_Gaussian_Scale` 计算的增量 Gaussian。复现时不得把两个 layer 都重复施加 `sigma=1.6`。

这些是首个 source-faithful 实现的固定默认值。V3 默认从第一 octave 的高分辨率层构造 dense state；完整 `3x2` HIMO pyramid 同时保留为诊断输出和后续消融，不能把 FPN 误称为原始 HIMO 的 CDMS。

### 4.3 Deep-Shallow Odd/Even 提取

#### 4.3.1 Log-Gabor 深层响应

论文使用二维 Log-Gabor：

$$
L_{s,o}(\rho,\theta)=
\exp\left(
-\frac{\log^2(\rho/\rho_s)}{2\log^2\sigma_\rho}
\right)
\exp\left(
-\frac{(\theta-\theta_o)^2}{2\sigma_\theta^2}
\right),
$$

其空间复核为：

$$
L^*_{s,o}=L^{\mathrm{even}}_{s,o}
+iL^{\mathrm{odd}}_{s,o}.
$$

公开代码使用：

- `Ns=4` 个尺度；
- `No=6` 个方向，$\theta_o=o\pi/6$；
- `minWaveLength=3`；
- `mult=1.6`；
- `sigmaOnf=0.75`；
- 第 $s$ 个尺度权重为 $N_s-s+1$。

图像响应写作：

$$
G_{s,o}=I*L^*_{s,o}
=G^{\mathrm{even}}_{s,o}+iG^{\mathrm{odd}}_{s,o}.
$$

Odd 分量直接按方向投影：

$$
G^{\mathrm{odd}}_{x,\mathrm{LG}}
=\sum_s\sum_o
G^{\mathrm{odd}}_{s,o}\cos\theta_o,
\qquad
G^{\mathrm{odd}}_{y,\mathrm{LG}}
=\sum_s\sum_o
G^{\mathrm{odd}}_{s,o}\sin\theta_o.
$$

Even 分量先由同位置 Odd 符号异化：

$$
\operatorname{sgn}^*(x)=
\begin{cases}
1,&x\geq0,\\
-1,&x<0,
\end{cases}
$$

$$
G^{\mathrm{even}*}_{s,o}
=G^{\mathrm{even}}_{s,o}
\odot\operatorname{sgn}^*
\left(G^{\mathrm{odd}}_{s,o}\right),
$$

再按相同方向投影。公开代码在复数累加时保留同样语义，并使用尺度权重 $N_s-s+1$。

为避免图像坐标与数学坐标的符号差异造成“公式正确、代码错向”，`Code-Reproduction` 直接锁定公开实现的复数累加：

$$
E_{s,o}
=i\,\operatorname{Im}(G_{s,o})
+\operatorname{Re}(G_{s,o})
\odot\operatorname{sgn}^*
\left(\operatorname{Im}(G_{s,o})\right),
$$

$$
\widetilde G_x
\leftarrow
\widetilde G_x
-E_{s,o}\cos\theta_o(N_s-s+1),
$$

$$
\widetilde G_y
\leftarrow
\widetilde G_y
+E_{s,o}\sin\theta_o(N_s-s+1).
$$

最后以 $\operatorname{Im}(\widetilde G)$ 作为 Odd、$\operatorname{Re}(\widetilde G)$ 作为异化 Even。PyTorch 回归测试必须在这一层直接对齐 MATLAB，而不能只比较最终 IMO。

#### 4.3.2 Sobel 浅层响应

Odd 与 Even 的固定浅层核分别为：

$$
S^{\mathrm{odd}}=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\qquad
S^{\mathrm{even}}=
\frac{2}{3}
\begin{bmatrix}
-1&2&-1\\
-2&4&-2\\
-1&2&-1
\end{bmatrix}.
$$

Deep-Shallow 合成响应为：

$$
G_x^{\mathrm{odd}}
=G_{x,\mathrm{LG}}^{\mathrm{odd}}+I*S^{\mathrm{odd}},
\qquad
G_y^{\mathrm{odd}}
=G_{y,\mathrm{LG}}^{\mathrm{odd}}+I*(S^{\mathrm{odd}})^T,
$$

$$
G_x^{\mathrm{even}}
=G_{x,\mathrm{LG}}^{\mathrm{even}}
+(I*S^{\mathrm{even}})
\odot\operatorname{sgn}^*(G_x^{\mathrm{odd}}),
$$

$$
G_y^{\mathrm{even}}
=G_{y,\mathrm{LG}}^{\mathrm{even}}
+(I*(S^{\mathrm{even}})^T)
\odot\operatorname{sgn}^*(G_y^{\mathrm{odd}}).
$$

这里的 “Even” 是论文定义的异化 Even，不得替换成普通 quadrature 能量。

### 4.4 MASW

对任意二维响应 $(X,Y)$，定义多尺度局部充分统计：

$$
A=\sum_\sigma\lambda_\sigma
\sum_{W_\sigma}(X^2-Y^2),
\qquad
B=\sum_\sigma\lambda_\sigma
\sum_{W_\sigma}(2X\odot Y).
$$

MASW 输出：

$$
M_\rho(X,Y)=A^2+B^2,
\qquad
M_\phi^{\mathrm{paper}}(X,Y)=\frac12\operatorname{atan2}(B,A).
$$

公开 MATLAB 使用轴向坐标约定：

$$
M_\phi^{\mathrm{code}}(X,Y)
=\frac12\operatorname{atan2}(B,A)+\frac{\pi}{2},
$$

并在 Odd/Even 耦合后取模到 $[0,\pi)$。`Code-Reproduction` 必须保留这项 $\pi/2$，不能只照抄论文式而忽略代码坐标约定。

分别代入 Odd 与 Even Deep-Shallow 响应：

$$
m_o=M_\rho(G_x^{\mathrm{odd}},G_y^{\mathrm{odd}}),
\qquad
\phi_o=M_\phi(G_x^{\mathrm{odd}},G_y^{\mathrm{odd}}),
$$

$$
m_e=M_\rho(G_x^{\mathrm{even}},G_y^{\mathrm{even}}),
\qquad
\phi_e=M_\phi(G_x^{\mathrm{even}},G_y^{\mathrm{even}}).
$$

公开代码调用参数为 `R1=1`、`R2=r`、`s=4`。$r$ 由 PolarP patch 决定：

$$
r=\sqrt{\frac{W^2}{2N_A+1}},
\qquad
W=\left\lfloor\frac{\text{patch\_size}}2\right\rfloor.
$$

默认 `patch_size=72`、$N_A=12$，因此 $W=36$、$r=7.2$。代码在半径 $\lfloor r\rfloor$ 的圆窗内叠加 4 个 Gaussian，标准差从 $R_1/6$ 线性变化到 $R_2/6$。

### 4.5 Odd/Even 耦合与有效性

每个像素采用强响应分支的方向：

$$
\phi_{\mathrm{IMO}}(x,y)=
\begin{cases}
\phi_o(x,y),&m_o(x,y)\geq m_e(x,y),\\
\phi_e(x,y),&m_o(x,y)<m_e(x,y),
\end{cases}
\qquad
\phi_{\mathrm{IMO}}\leftarrow
\phi_{\mathrm{IMO}}\bmod\pi.
$$

这一步是 hard Odd/Even coupling，不是软门控。

论文的有效性由 Odd/Even 响应的局部协方差给出：

$$
\mathbf{M}=
\sum_W
\begin{bmatrix}
m_o^2&m_om_e\\
m_om_e&m_e^2
\end{bmatrix}
=
\begin{bmatrix}
A&C\\
C&B
\end{bmatrix},
$$

$$
v_{\mathrm{IMO}}=
\frac{\det\mathbf{M}}{\operatorname{trace}\mathbf{M}}
=\frac{AB-C^2}{A+B+\epsilon}.
$$

公开代码在 `int_flag=1` 时使用固定 `scale=6` Gaussian 圆窗计算该量，并以 `0.1` 二值化为权重图；`int_flag=0` 时则选择 $\max(m_o,m_e)$ 并取四次方根作为幅值。V3 跨模态主配置保留 `int_flag=1` 的 source-faithful 有效性，同时输出未耦合的 $m_o,m_e$ 供紧凑状态使用。

### 4.6 DoFS 的范围

完整 HIMO 还通过尺度间 CoF、Deep-Shallow、MASW 和 IMO 差异构造 DoFS，用于增强关键点检测。V3 是 detector-free dense encoder，不在 V3-Core 中使用 DoFS 选择 anchor；但冻结 HIMO 的回归测试应能生成与公开代码一致的 DoFS。不得把“V3 不使用 DoFS”解释为 DoFS 不属于 HIMO。

## 5. 紧凑 Physical State

> 本节全部属于 `V3-Extension`。原始 HIMO 没有定义 5 通道 Physical State，也没有定义 $r_{oe}$。

每个位置只保留：

$$
P=
\left[
u_{\mathrm{IMO}},
m_o,
m_e,
r_{oe}
\right].
$$

总通道数为：

$$
C_p=2+1+1+1=5.
$$

### 5.1 轴向方向

不要直接保存角度 $\phi$，而保存 doubled-angle 单位向量：

$$
u_{\mathrm{IMO}}=
\begin{bmatrix}
\cos 2\phi\\
\sin 2\phi
\end{bmatrix}.
$$

该表示满足：

$$
\phi\equiv\phi+\pi,
$$

因此符合边缘和轮廓的无符号轴向方向。

当局部 Odd/Even 总强度低于固定阈值时，方向没有可靠物理意义。此时使用固定回退值：

$$
u_{\mathrm{IMO}}=(1,0),
$$

同时保留接近零的 $m_o,m_e$，让后续网络识别该位置缺少有效结构。不得在零向量上直接计算 `atan2(0,0)`。

回退判定优先使用 HIMO 原始 $v_{\mathrm{IMO}}$，而不是另行发明阈值。`v_imo` 作为诊断张量返回，但不计入默认 5 通道。

### 5.2 Odd 与 Even 强度

HIMO 分别输出非负强度：

$$
m_o\geq0,\qquad m_e\geq0.
$$

这里必须使用 MASW 在 hard coupling 之前的 $m_o,m_e$。为避免不同图像动态范围直接控制网络，进入 Physical State 前使用固定、无参数的压缩：

$$
\widehat m=
\frac{m^{1/4}}
{\operatorname{median}_{v_{\mathrm{IMO}}>0}(m^{1/4})+\epsilon},
$$

再裁剪到固定上限。四次方根来自公开 `int_flag=0` 幅值路径；按图像有效区域中位数归一化是 `V3-Extension`，必须单独做数值稳定性消融，不能声称来自论文。

文档后续提到的 $m_o,m_e$ 均指压缩后的稳定强度。

### 5.3 Odd-Even 相对优势

定义：

$$
r_{oe}=
\frac{|m_o-m_e|}
{m_o+m_e+\epsilon}.
$$

其范围为：

$$
r_{oe}\in[0,1].
$$

- $r_{oe}\approx0$：Odd 与 Even 强度接近；
- $r_{oe}\approx1$：其中一种响应占明显优势。

$r_{oe}$ 只表示优势程度，不表示优势属于 Odd 还是 Even；具体类别仍由 $m_o,m_e$ 两个独立通道给出。

由于论文原始有效性为 $v_{\mathrm{IMO}}$，首轮必须比较：

- `P-roe`：$[\cos2\phi,\sin2\phi,m_o,m_e,r_{oe}]$；
- `P-valid`：$[\cos2\phi,\sin2\phi,m_o,m_e,v_{\mathrm{IMO}}]$。

用户提出的 `P-roe` 是 V3 主假设；`P-valid` 是更贴近 HIMO 原文的必要对照。

### 5.4 输出张量

HIMO 保持高分辨率输出：

$$
P_{\mathrm{HIMO}}\in
\mathbb{R}^{B\times5\times H_0\times W_0},
\qquad H_0=W_0\in\{512,256\}.
$$

默认采用：

$$
P_{\mathrm{HIMO}}\in\mathbb{R}^{B\times5\times256\times256}.
$$

若从 `512x512` HIMO 响应构造 `256x256` 状态，方向场不能直接对角度做平均。必须先对 doubled-angle 充分统计量和 Odd/Even 能量做固定 BlurPool，再重新计算并归一化 $u_{\mathrm{IMO}}$。

从这一层开始不再直接把 Physical State 降到 `64x64`。后续尺度变化由可训练但抗混叠的 FPN 完成。

## 6. 原始 PolarP 与 Dense 化边界

### 6.1 原始等面积极坐标分区

原始 PolarP 不是离散的 `1+8+8` 个采样点。对每个 anchor，以半径 $R_2$ 的圆形 patch 统计：

- 中心圆盘 $A_0$；
- 第一圆环的 $N_A$ 个扇区 $A_j^1$；
- 第二圆环的 $N_A$ 个扇区 $A_j^2$。

论文设置各 cell 面积相等：

$$
N_A\pi R_0^2
=\pi(R_1^2-R_0^2)
=\pi(R_2^2-R_1^2).
$$

公开代码使用 `patch_size=72`、$W=R_2=36$、$N_A=12$，并以平方半径实现：

$$
R_0^2=\frac{W^2}{2N_A+1},
\qquad
R_1^2=R_0^2(N_A+1).
$$

因此：

$$
R_0=7.2,\qquad
R_1\approx25.96,\qquad
R_2=36.
$$

每个 cell 对 $\phi_{\mathrm{IMO}}\in[0,\pi)$ 构建 $N_O=12$ 个方向 bin，像素权重来自 HIMO magnitude/validness。基础结构共有：

$$
1+2N_A=25
$$

个空间 cell，而不是 17 个点。

论文文字提到沿角度方向做 Gaussian 加权以缓解 bin 碎片化，但公开 `PolarP_Descriptor.m` 使用 hard bin 累加。`PolarP-CodeExact` 必须采用 hard bin；Gaussian soft binning 只能作为单独的 `PolarP-SoftBin` 消融。

### 6.2 Base Direction

对整个圆形 patch 先按 HIMO 权重累计 $N_O=12$ 维方向直方图。局部峰值超过全局峰值 `0.8` 的方向均作为候选 base direction，并用相邻三个 bin 做抛物线插值：

$$
\hat{o}
=o+\frac12
\frac{h_{o-1}-h_{o+1}}
{h_{o-1}+h_{o+1}-2h_o},
\qquad
\theta_0=\hat{o}\frac{\pi}{N_O}.
$$

公开代码允许同一 anchor 产生多个 base direction。V3 dense 训练默认保留最高峰方向以控制计算量；“只保留一个方向”是 `V3-Extension`，必须与原始多方向版本在小规模数据上比较。

官方 demo 的 `rot_flag=0` 表示已知图像对没有明显旋转，此时跳过 base direction 与 PSD。V3 的在线 Homography 包含旋转，source-faithful PolarP 基线必须使用 `rot_flag=1`；关闭它只能作为无旋转对照。

空间扇区角和 IMO 方向 bin 都相对 $\theta_0$ 旋转：

$$
b_\theta
=
\left\lfloor
\frac{(\theta-\theta_0)N_A}{2\pi}
\right\rfloor\bmod N_A,
$$

$$
b_\phi
=
\left\lfloor
\frac{(\phi_{\mathrm{IMO}}-\theta_0)N_O}{\pi}
\right\rfloor\bmod N_O.
$$

### 6.3 Base 与 deeper PolarP 统计

记中心方向直方图为 $H_0\in\mathbb{R}^{N_O}$，两圈第 $j$ 个扇区为 $H_j^1,H_j^2\in\mathbb{R}^{N_O}$。基础描述长度为：

$$
(1+2N_A)N_O=300.
$$

论文进一步沿径向和角向组合相邻、相隔和全局 cell，形成 $H^3,H^4,H^5,H^6$，再按式 (23) 拼接。公开 MATLAB 对这部分采用一个紧凑实现：

$$
H_j^3=
\frac{
(H_j^1+H_{j+1}^1)w_1
+(H_j^2+H_{j+1}^2)w_2
}{2}
+\frac{H_0}{6},
$$

$$
w_1=\sqrt{\frac{R_0^2}{R_1^2}}
=\frac1{\sqrt{N_A+1}},
\qquad
w_2=\frac{R_0}{R_2}
=\frac1{\sqrt{2N_A+1}}.
$$

代码另外形成：

- `feat_skip`：交错组合两圈奇偶扇区，$2N_O$ 维；
- `feat_all`：$\{H_j^3\}$ 的扇区均值，$N_O$ 维。

代码可复现描述子为：

$$
d_{\mathrm{PolarP-code}}
=
[H_0;\operatorname{vec}(H^1,H^2,H^3);
H_{\mathrm{skip}};H_{\mathrm{all}}],
$$

在 $N_A=N_O=12$ 时维度为：

$$
12+12\times12\times3+24+12=480.
$$

论文式 (22)-(23) 给出更一般的 deeper pyramid 展开，按其维数公式在相同参数下不是 480 维。为保证可复现，V3 的 `PolarP-CodeExact` 基线严格对齐公开 MATLAB 的 480 维输出；论文通用展开另记为 `PolarP-PaperFull`，不得把二者混写。

### 6.4 PSD 反转判别

IMO 方向定义在长度为 $\pi$ 的轴向空间，跨模态旋转可能造成 PolarP descriptor reversal。原文不是简单平均 $\theta_0$ 和 $\theta_0+\pi$ 两个描述子，而是使用 Partition-State Discrimination。

公开代码把 $H^1,H^2,H^3$ 的 12 个空间扇区沿 base direction 分为相对两半，比较对应 cell 方向直方图的方差。若判别状态满足代码条件，则交换两个半区：

$$
\tau_d=
\sum_{i=1}^{3}
\sum_{j=1}^{N_A/2}
\mathbf{1}
\left[
\operatorname{Var}(H_j^i)
-\operatorname{Var}(H_{j+N_A/2}^i)
\geq0
\right].
$$

公开代码在：

$$
\tau_d>\frac{3N_A}{4}=9
$$

时交换两个半区。论文式 (24)-(25) 使用通用 PSD 记法，而公开代码的比较索引和阈值是可直接复现的落地定义。首个 source-faithful 实现必须复现代码的 threshold 和 half swap。原先设想的 $\phi/(\phi+\pi)$ 双分支对称融合只能作为 `V3-SymmetricHalfTurn` 消融，不能再称为原始 PolarP。

### 6.5 Dense 化原则

Dense PolarP 对每个高分辨率 anchor 重复上述固定统计，但不得永久展开所有 patch 像素。实现要求：

1. 用固定卷积/积分统计或 anchor chunk 计算 25 个 cell 的 12-bin histogram；
2. source-faithful 路径先得到 480 维 `PolarP-CodeExact`；
3. 所有空间分区、bin、base direction、deeper 组合和 PSD 均无参数；
4. patch 越过图像边界时只统计有效像素并同步归一化有效面积，不使用 reflection 伪造结构；
5. `256x256` 状态下将 `patch_size=72` 的输入像素范围等比例映射为状态网格半径 18；
6. `512x512` 消融使用原始半径 36。

这使 Dense Polar 编码建立在原文的“区域方向统计”上，而不是凭直觉选取少量坐标。

## 7. 小型 Polar Encoder 初稿

Polar Encoder 不直接读取原图或 SLiM feature，只读取固定 PolarP 统计及可选的 V3 Physical State 区域统计。

### 7.1 `PolarP-CodeExact` 非学习基线

先生成与公开 `PolarP_Descriptor.m` 一致的原始 480 维向量并直接用于复现测试。该函数本身没有执行描述子归一化，公开 `Match_Keypoint.m` 也将切出的原始描述子直接交给 MATLAB `matchFeatures`。因此不得在“code-exact”向量内暗加 L2 或 RootSIFT；神经训练前需要的固定归一化必须标记为 `V3-Extension`。该基线不含神经网络，用于验证 PyTorch HIMO 和 PolarP 是否复现正确。

### 7.2 `PolarP-Projection`

首个可训练基线：

```text
code-exact PolarP [480]
  -> documented fixed input normalization (V3 extension)
  -> Linear 480 -> 128
  -> LayerNorm + GELU
  -> Linear 128 -> Cd
  -> L2 normalize
```

它只学习压缩原始 PolarP，不改变物理统计。

### 7.3 `Structured-Polar Encoder`

主 V3 编码器保留：

$$
T_i\in\mathbb{R}^{25\times12}
$$

的 center/two-ring 结构，并追加代码定义的 deeper、skip 和 global token。建议使用：

- 方向 bin 上的 circular depthwise mixing；
- 12 个空间扇区上的 circular mixing；
- center、inner、outer、deeper、skip、global 的固定 type embedding；
- 1-2 个小型残差 MLP；
- 参数受控的 attention 仅作为后续消融。

若要利用 $m_o,m_e,r_{oe}$，应按同一 25-cell Polar 分区计算区域均值/方差，再作为附加 token feature；不得用 17 点双线性采样替代原始方向直方图。

首轮顺序固定为：

1. `PolarP-CodeExact`；
2. `PolarP-Projection`；
3. `Structured-Polar-LocalOnly`；
4. 确认 Polar 编码本身有效后才接 FPN。

## 8. Physical FPN 后端

Dense Polar Encoder 输出：

$$
D_{\mathrm{local}}\in
\mathbb{R}^{B\times C_d\times H_0\times W_0}.
$$

FPN 的目标是同时输出：

| 层级 | 默认尺寸 | 主要职责 |
|---|---:|---|
| `P2` | `256x256` | 精细边缘与轮廓定位 |
| `P3` | `128x128` | 中等范围结构确认 |
| `P4` | `64x64` | coarse correspondence discovery |

当 Polar Encoder 输出为 `256x256` 时，直接作为 `C2`。当输出为 `512x512` 时，先经过一个抗混叠 stride-2 stem 形成 `C2`，之后两种配置共享完全相同的 FPN。

### 8.1 Bottom-up 路径

默认结构：

```text
C2 [B,32,256,256]
  -> anti-aliased stride-2 residual stage
C3 [B,64,128,128]
  -> anti-aliased stride-2 residual stage
C4 [B,96,64,64]
```

每个 stage 使用：

```text
fixed BlurPool
  -> depthwise Conv3x3, stride 2
  -> pointwise Conv1x1
  -> LayerNorm2d
  -> GELU
  -> depthwise residual block
```

固定 BlurPool 必须位于降采样卷积之前，减少高频物理状态在尺度变化时产生混叠。FPN 只能读取 Polar 编码后的特征，不能绕过 Polar Encoder 重新读取原图。

### 8.2 Top-down 与 lateral fusion

所有 lateral feature 投影到统一维度 $C_f=64$：

$$
L_s=\operatorname{Conv}_{1\times1}(C_s).
$$

自顶向下融合：

$$
P_4=L_4,
$$

$$
P_3=\operatorname{Smooth}
\left(L_3+\operatorname{Up}(P_4)\right),
$$

$$
P_2=\operatorname{Smooth}
\left(L_2+\operatorname{Up}(P_3)\right).
$$

`Up` 使用双线性插值，`Smooth` 使用 depthwise `3x3` 加 pointwise `1x1`。不使用反卷积。

### 8.3 多尺度描述子头

每层使用独立但结构相同的 descriptor head：

$$
D_s=
\operatorname{L2Norm}
\left(
\operatorname{LN2d}
\left(
\operatorname{Conv}_{1\times1}(P_s)
\right)
\right).
$$

默认输出：

| 描述子 | 尺寸 | 通道 |
|---|---:|---:|
| `descriptor_fine` | `256x256` | 64 |
| `descriptor_mid` | `128x128` | 64 |
| `descriptor_coarse` | `64x64` | 64 |

每一层分别按通道 L2 归一化。后续 coarse-to-fine 匹配首先在 `descriptor_coarse` 发现候选，再用 `descriptor_mid` 和 `descriptor_fine` 局部细化，而不是直接对 `256x256` 执行全局两两相似度。

### 8.4 Local-only 对照

保留不使用 FPN 的 Local-only 对照：

```text
D_local
  -> fixed BlurPool to 64x64
  -> Conv1x1
  -> LayerNorm2d
  -> L2 normalize
```

该对照用于判断提升来自 Polar descriptor 本身还是来自多尺度 FPN。

## 9. 前向接口草案

```python
state = frozen_himo(image)

output = {
    "physical_state": state,          # [B,5,H0,W0], frozen
    "himo_validness": validness,       # [B,1,H0,W0], frozen diagnostic
    "polar_base_hist": base_hist,      # 可选 [B,N,25,12]，仅分块诊断
    "polar_code_exact": polar_480,     # 可选 [B,N,480]，仅分块诊断
    "local_descriptor": local,        # [B,Cd,H0,W0]
    "descriptor_fine": descriptor2,   # [B,64,256,256]
    "descriptor_mid": descriptor3,    # [B,64,128,128]
    "descriptor_coarse": descriptor4, # [B,64,64,64]
}
```

`physical_state`、`himo_validness` 和所有固定 PolarP 统计必须 `requires_grad=False`。诊断张量仅在可视化、MATLAB 数值对齐或单元测试模式下返回，训练时不长期保存全图 Polar 统计。

## 10. 张量尺寸

| 阶段 | 张量 | 默认尺寸 |
|---|---|---|
| 输入 | `image` | `[B,1,512,512]` |
| 轴向方向 | `u_imo` | `[B,2,256,256]` |
| Odd 强度 | `m_odd` | `[B,1,256,256]` |
| Even 强度 | `m_even` | `[B,1,256,256]` |
| O/E 优势 | `r_oe` | `[B,1,256,256]` |
| HIMO 原始有效性 | `himo_validness` | `[B,1,256,256]` |
| Physical State | `physical_state` | `[B,5,256,256]` |
| 单 anchor 基础 PolarP | `T_i` | `[B,25,12]` |
| 单 anchor code-exact PolarP | `polar_480_i` | `[B,480]` |
| 全图基础 PolarP | `polar_base_hist` | `[B,65536,25,12]`，只允许分块瞬时生成 |
| 全图 code-exact PolarP | `polar_code_exact` | `[B,65536,480]`，只允许分块瞬时生成 |
| Polar 局部描述 | `local_descriptor` | `[B,32,256,256]` |
| FPN `P2` | `descriptor_fine` | `[B,64,256,256]` |
| FPN `P3` | `descriptor_mid` | `[B,64,128,128]` |
| FPN `P4` | `descriptor_coarse` | `[B,64,64,64]` |

实现不得永久展开并保存完整 `[B,65536,25,12]` 或 `[B,65536,480]` 中间张量。按 anchor chunk 执行区域统计和 Polar Encoder，默认 chunk 候选为 256、512 或 1024。`512x512` 消融具有 262144 个 anchor，只有分块实现通过显存测试后才允许训练。

## 11. 训练草案

### 11.1 第一阶段目标

先训练独立 V3 多尺度 descriptor：

$$
L=
L_{\mathrm{coarse}}
+\lambda_mL_{\mathrm{mid}}
+\lambda_fL_{\mathrm{fine}}.
$$

默认：

$$
\lambda_m=0.5,\qquad \lambda_f=0.25.
$$

`64x64` coarse 层沿用 Homography GT 和 chunked partial dual-softmax focal loss。`128x128` 和 `256x256` 不能构造全局完整相似度矩阵，只在 coarse 候选对应的局部窗口中采样正例与 hard negative，计算局部对比损失。

第一阶段不加入 SLiM recovery/preservation loss，因为 V3 尚未与 SLiM 融合。

### 11.2 数据

首轮沿用当前可复现实验协议：

- GoogleEarth 单幅可见光训练索引；
- 每张基础影像每 epoch 只在线生成一种随机几何变化；
- 验证阶段固定扰动；
- 输入 patch 无黑色填充区域；
- seed 固定并支持恢复训练。

为了检验跨模态迁移，正式多模态测试仍使用 Proposed 和 Expanded MRSI，但不用测试集选择结构或阈值。

### 11.3 参数更新

| 模块 | 是否训练 |
|---|---|
| 强度归一化与 CoF | 否 |
| Log-Gabor 与 Odd/Even Sobel | 否 |
| HIMO Deep-Shallow/MASW/coupling | 否 |
| PolarP 分区、直方图、base direction、PSD | 否 |
| Polar Encoder | 是 |
| Spatial backend | 是 |
| Descriptor head | 是 |

## 12. 首轮消融

按以下顺序运行：

1. `V3-HIMO-PolarP-CodeExact`：固定 HIMO 加公开代码 480 维 PolarP，不训练；
2. `V3-Pointwise-P5`：不做 PolarP，只对每点 5 通道编码；
3. `V3-PolarP-Projection`：固定 480 维 PolarP 加小型投影；
4. `V3-StructuredPolar-LocalOnly`：保留 25-cell、12-bin 和 deeper 结构，不使用 FPN；
5. `V3-NoBaseDirection`：删除原始 base direction 对齐；
6. `V3-NoPSD`：删除原始 PSD half swap；
7. `V3-SymmetricHalfTurn`：用 $\phi/(\phi+\pi)$ 对称融合替换 PSD，仅作对照；
8. `V3-PValid`：以原始 $v_{\mathrm{IMO}}$ 替换 $r_{oe}$；
9. `V3-PolarFPN-256`：`256x256` Polar descriptor 加三层 FPN；
10. `V3-PolarFPN-512`：`512x512` Polar descriptor 加三层 FPN。

这组消融分别回答：

- HIMO 的 5 通道状态本身是否有效；
- PyTorch 是否忠实复现原始 HIMO/PolarP；
- 增益是否来自 PolarP 区域统计和方向对齐；
- 原始 base direction 与 PSD 是否必要；
- 用户提出的 $r_{oe}$ 是否优于原始有效性；
- 神经编码是否优于固定 480 维描述子；
- FPN 的跨 anchor、多尺度空间混合是否必要；
- `512` 高分辨率 Polar 编码是否值得 4 倍计算成本。

## 13. 单元测试与可视化

必须覆盖：

1. HIMO 中不存在可训练参数；
2. 固定测试图上的 CoF、Log-Gabor、Sobel、MASW、coupling 和 validness 与 MATLAB 对齐；
3. `Ns=4`、`No=6`、Sobel 核、MASW 窗和公开参数逐项锁定；
4. 相同输入的 Physical State 逐元素确定；
5. `u_imo` 在有效位置满足单位范数；
6. $m_o,m_e\geq0$，$r_{oe}\in[0,1]$；
7. 弱结构位置不出现 NaN；
8. PolarP 基础空间 cell 数严格为 25，每个 cell 为 12-bin；
9. code-exact 描述子严格为 480 维；
10. base direction 峰值、抛物线插值和多方向输出与 MATLAB 对齐；
11. PSD 判别和 half swap 与 MATLAB 对齐；
12. 固定 anchor 的 PyTorch PolarP 与 MATLAB 描述子相对误差达到约定阈值；
13. 已知旋转下 PolarP 具有预期不变性；
14. descriptor 通道 L2 范数接近 1；
15. 冻结 HIMO/PolarP 始终无梯度，神经编码器可正常反传；
16. anchor chunk 前后输出一致；
17. `256` 和 `512` Polar 输入都能得到严格的 `256/128/64` FPN 输出；
18. 各 FPN 描述子通道 L2 范数接近 1；
19. 抗混叠降采样对棋盘格和高频测试图不产生明显混叠峰值；
20. BF16 神经路径和 FP32 HIMO/PolarP 路径无 NaN。

训练可视化至少包括：

- 输入影像；
- `cos(2phi)` 与 `sin(2phi)`；
- $m_o$；
- $m_e$；
- $r_{oe}$；
- 原始 $v_{\mathrm{IMO}}$；
- 选定 anchor 的中心圆盘、两圈 12 扇区、base direction 和 PSD 状态；
- 选定 anchor 的 25 个方向直方图；
- `P2/P3/P4` 三层特征能量；
- 三层 descriptor norm。

## 14. 与 V2 的关键区别

| 项目 | V2 | V3 初稿 |
|---|---|---|
| HIMO/Log-Gabor | 可学习物理参数 | 按论文与公开代码完全冻结 |
| 物理中间量 | 多尺度高维响应 | 固定 5 通道状态 |
| Pair Interaction | 有 | 首轮无 |
| Polar 输入 | 学习后的高维 physical feature | 原始 PolarP 统计与紧凑 HIMO state |
| SLiM 关系 | 直接预测 additive delta | 首轮独立描述子 |
| 主要风险 | 扩大候选但降低纯度 | 物理状态容量可能不足 |
| 首要问题 | 如何保护 SLiM base | Polar 物理邻域是否真正稳定 |

## 15. 待锁定问题

实现前需要通过小规模统计或 smoke test 确定：

1. 第一 octave 的 layer 1 还是 layer 2 作为默认高分辨率 state；
2. 单 base direction 与原始多 base direction 的精度/成本差异；
3. $m_o,m_e$ 的固定压缩和裁剪范围；
4. 默认第五通道使用 $r_{oe}$ 还是原始 $v_{\mathrm{IMO}}$；
5. 首轮神经编码器使用 480 维投影还是 structured encoder；
6. FPN lateral 统一通道使用 48、64 还是 96；
7. 输出描述子使用 64、96 还是 128 通道；
8. 论文 `PolarP-PaperFull` 与公开代码 `PolarP-CodeExact` 的差异是否值得单独复现。

## 16. 初稿推荐落点

首个可实施版本建议锁定为：

```text
Frozen HIMO
  -> official CoF + Deep-Shallow Odd/Even + MASW + hard coupling
  -> P=[cos2phi, sin2phi, m_odd, m_even, r_oe]
  -> fixed anti-aliased state at 256x256
  -> dense PolarP: center + two 12-sector rings, 12 orientation bins
  -> official base direction + code-faithful deeper statistics + PSD
  -> 480-d code-exact descriptor
  -> small projection or structured Polar encoder
  -> anti-aliased bottom-up stages
  -> top-down FPN with lateral fusion
  -> 256/128/64 three-level L2 descriptors
```

先完成 MATLAB 数值对齐，再用 `HIMO-PolarP-CodeExact`、`Pointwise-P5`、`PolarP-Projection` 和 `StructuredPolar-LocalOnly` 判断复现质量与神经编码贡献。只有 Structured Polar 明确超过固定 480 维和 Pointwise 基线，才训练 `V3-PolarFPN-256`；只有 `256` 版本确认有效且高分辨率细化仍受限，才运行 `V3-PolarFPN-512`。

## 17. 来源与版本锁定

本设计依据：

1. [HIMO 论文（IEEE Xplore）](https://ieeexplore.ieee.org/document/11435911)：`HIMO: Cross-Arbitrary-Modality Image Invariant Feature Transform With Hierarchical Intrinsic Major Orientation`，重点为公式 (1)-(25)；
2. [作者公开仓库](https://github.com/MrPingQi/HIMO_ImgMatching)；
3. 复现锁定 commit：`884297aa36dfb9c89d9a7d5bf66c142bf8707a77`；
4. 关键文件：
   - [`Build_Himo_Pyramid.m`](https://github.com/MrPingQi/HIMO_ImgMatching/blob/884297aa36dfb9c89d9a7d5bf66c142bf8707a77/HIMO_image_matching/func_HIMO/Build_Himo_Pyramid.m)：强度归一化、CoF、Gaussian pyramid、DoFS；
   - [`Intrinsic_Major_Orientation.m`](https://github.com/MrPingQi/HIMO_ImgMatching/blob/884297aa36dfb9c89d9a7d5bf66c142bf8707a77/HIMO_image_matching/func_HIMO/Intrinsic_Major_Orientation.m)：Deep-Shallow、MASW、Odd/Even coupling、validness；
   - [`Base_Direction.m`](https://github.com/MrPingQi/HIMO_ImgMatching/blob/884297aa36dfb9c89d9a7d5bf66c142bf8707a77/HIMO_image_matching/func_HIMO/Base_Direction.m)：主方向直方图、80% 辅助方向和抛物线插值；
   - [`PolarP_Descriptor.m`](https://github.com/MrPingQi/HIMO_ImgMatching/blob/884297aa36dfb9c89d9a7d5bf66c142bf8707a77/HIMO_image_matching/func_HIMO/PolarP_Descriptor.m)：等面积分区、方向统计、deeper extension、PSD 和最终拼接；
   - [`Match_Keypoint.m`](https://github.com/MrPingQi/HIMO_ImgMatching/blob/884297aa36dfb9c89d9a7d5bf66c142bf8707a77/HIMO_image_matching/func_HIMO/Match_Keypoint.m)：公开描述子的实际匹配调用链。

### 17.1 不得混淆的来源边界

- 5 通道 Physical State 是 V3 扩展，不是 HIMO 原始输出；
- $r_{oe}$ 是 V3 扩展，不是论文公式；
- Dense every-pixel PolarP、单主方向约束、小型神经编码器和 FPN 都是 V3 扩展；
- CoF、Deep-Shallow Odd/Even、MASW、hard coupling、validness、PolarP 分区、base direction 和 PSD 来自 HIMO；
- 原先的 `1+8+8` 点采样和简单 half-turn 对称融合不属于 HIMO/PolarP，已从 V3-Core 删除。
