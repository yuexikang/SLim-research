# Physical Encoder V2

> 状态：实现规范。V2只研究跨模态粗对应发现，不训练或评估Fine与Refinement；正式多模态测评另行设计。

## 版本历史

| 版本 | 定位 | 主要更新 | 文档 |
|---|---|---|---|
| V2 | 初始结构版本 | 三尺度物理前端、HIMO Odd/Even、MASW、Pair Transformer、Hard O/E、Polar Descriptor和SLiM残差Adapter | 本文档 |
| V2.1 | 数值稳定性修订 | Pair Linear Attention改用FP32；加入Safe Axial Angle和非有限值三级诊断 | [版本更新记录](./Physical%20Encoder%20V2%20版本更新记录.md) |
| V2.1.1 | 训练可观测性修订 | 每20 step覆盖输出关键物理特征图；固定文件名，不保留历史图片；模型参数拓扑不变 | [版本更新记录](./Physical%20Encoder%20V2%20版本更新记录.md) |
| V2.1.2 | GoogleEarth数据修订 | 移除3MOS训练行；GoogleEarth按pair无泄漏划分并展开为单图；每图每epoch一种在线扰动 | [版本更新记录](./Physical%20Encoder%20V2%20版本更新记录.md) |
| V2.1.3 | 强几何与LG光度增强 | difficulty 0.7、最小区域0.3、roll ±45°；训练目标图启用p=0.95的LG在线增强 | [版本更新记录](./Physical%20Encoder%20V2%20版本更新记录.md) |
| V2.1.4 | 有效区域Rectification | 在原图内采样有效四边形并展开为完整512 patch，取消旋转/透视后的黑色补边 | [版本更新记录](./Physical%20Encoder%20V2%20版本更新记录.md) |

## 1. 研究边界

Physical V2接收一对灰度图像，并在SLiM的`1/8`粗网格上产生物理残差：

$$
(I_0,I_1)\rightarrow \Delta F_0,\Delta F_1,
\qquad F_i^{V2}=F_i^{SLiM}+\Delta F_i.
$$

当前阶段回答：V2能否在保持冻结SLiM粗特征的前提下，提高合成Homography监督下的粗对应发现能力。当前不负责：

- Fine-level跨模态描述；
- 子像素定位；
- V2-only候选的精细化；
- 完整coarse-to-fine联合训练；
- Proposed、MRSI等正式多模态测评。

## 2. 输入输出

输入：`image0/image1: [B,1,512,512]`。

训练前向返回：

| 键 | 尺寸 | 含义 |
|---|---:|---|
| `physical0/1` | `[B,96,64,64]` | L2归一化的最终物理描述子 |
| `delta0/1` | `[B,192,64,64]` | Physical-to-SLiM残差 |
| `enhanced0/1` | `[B,192,64,64]` | `slim + delta` |
| `orientation0/1` | `[B,2,64,64]` | 融合后的`(cos 2phi,sin 2phi)` |
| `reliability0/1` | `[B,1,64,64]` | 参数自由的稳定可靠性 |
| `scale_weights0/1` | `[B,3,64,64]` | 参数自由尺度权重 |
| `oe_selector0/1` | `[B,3,64,64]` | 三尺度Hard Odd/Even选择 |

冻结SLiM的base similarity只在训练loss内部计算，不作为公共前向输出。

## 3. 总体结构

```text
image pair
  -> anti-aliased pyramid: 512 / 256 / 128
  -> shared LDN + shared 3-frequency x 8-orientation Gabor
  -> HIMO-inspired Odd/Even construction
  -> shared MASW and sufficient-statistic downsampling to 64 x 64
  -> shared Odd/Even token projection
  -> per-scale [Self -> Cross] x 2 linear pair transformers
  -> per-scale hard Odd/Even coupling
  -> parameter-free reliability scale fusion
  -> dense polar sampling at phi and phi + pi
  -> shared polar transformer and reversal-invariant fusion
  -> physical descriptor [B,96,64,64]
  -> zero-init 1x1 adapter
  -> delta [B,192,64,64]
```

三个图像尺度共享LDN、Gabor、MASW、Token Projection和Pair Transformer参数，不允许建立三套独立编码器。

## 4. 抗混叠图像金字塔

$$I^{(0)}=I,\quad I^{(1)}=D_2(I),\quad I^{(2)}=D_4(I).$$

`D`使用固定5x5 binomial BlurPool后stride-2采样，不允许用裸bilinear interpolation代替完整下采样。

## 5. 共享物理前端

### 5.1 Local Divisive Normalization

每个尺度独立执行固定LDN：

$$
\mu=G*I,\quad
\sigma=\sqrt{G*(I-\mu)^2+10^{-4}},\quad
I_n=\operatorname{clip}((I-\mu)/\sigma,-5,5).
$$

Gaussian核大小9、`sigma=2`，无可训练参数。

### 5.2 Parametric Quadrature Gabor

每个图像尺度计算三频率、八方向quadrature响应：

$$O_{f,k},E_{f,k},A_{f,k}=\sqrt{O_{f,k}^2+E_{f,k}^2+\epsilon}.$$

- 基础波长：`{3,6,12}`；
- 核大小：`{9,15,25}`；
- 方向：`theta_k=k*pi/8`；
- 只学习受限的`lambda/sigma/gamma`；
- 同频八方向和三个图像尺度共享物理参数；
- 每个解析核去均值并做L2归一化。

## 6. HIMO-inspired Odd/Even构造

Odd XY投影：

$$
G_x^o=\sum_{f,k}O_{f,k}\cos\theta_k,\qquad
G_y^o=\sum_{f,k}O_{f,k}\sin\theta_k.
$$

Odd方向谱：

$$S_k^o=\sum_f|O_{f,k}|.$$

Even严格使用停止梯度的Odd符号修正：

$$
\widetilde E_{f,k}=E_{f,k}\operatorname{sign}^{*}(\operatorname{stopgrad}(O_{f,k})).
$$

零值归入正号；不使用tanh、STE、learnable sign或MLP correction。随后以相同方式计算`Gx_e/Gy_e`和`S_e`。

## 7. Shared MASW

Odd和Even调用同一个固定多邻域方向统计模块。对`Gx/Gy`构造：

$$X=G_x^2-G_y^2,\qquad Y=2G_xG_y.$$

使用`sigma in {1,2,4}`的固定Gaussian，等权聚合：

$$
A=\frac13\sum_\sigma W_\sigma*X,\qquad
B=\frac13\sum_\sigma W_\sigma*Y.
$$

$$
m=\sqrt{A^2+B^2+\epsilon},\qquad
u=(A,B)/(m+\epsilon).
$$

`u=(cos 2phi,sin 2phi)`是轴向方向，天然满足`phi`与`phi+pi`等价。

## 8. 对齐到64x64

512、256、128分支分别下采样3、2、1次。方向场不能直接下采样角度；必须对`A/B`以及谱、phase等充分统计量做固定BlurPool，再重新计算`m/u`。

原始Gabor响应在每个尺度完成聚合后立即释放，长期只保留64x64物理场。

## 9. Reliability与方向谱规范化

每个方向的跨频相干度：

$$
PC_k=\frac{|\sum_f(E_{f,k}+iO_{f,k})|}{\sum_f A_{f,k}+\epsilon}.
$$

以方向幅值加权得到`p_phase`，再结合方向集中度`rho`和局部归一化结构强度`s`：

$$r_{stable}=p_{phase}\rho s.$$

所有routing使用`stopgrad(r_stable)`，防止网络操纵解析可靠性。

Odd/Even方向谱分别归一化，并根据各自MASW方向对八个方向bin做连续循环线性采样。V2只规范化orientation spectrum，不执行V1的全响应canonicalization和OAN。

## 10. 共享Physical Token Projection

定义相对幅值：

$$r_o=m_o/(m_o+m_e+\epsilon),\qquad r_e=m_e/(m_o+m_e+\epsilon).$$

Odd/Even各自形成11通道输入：

$$X_o=[r_o,s,r_{stable},\widehat S^o_{1:8}],\qquad
X_e=[r_e,s,r_{stable},\widehat S^e_{1:8}].$$

二者共享`Conv1x1(11,96) -> LayerNorm2d -> GELU`，确保Hard Coupling前位于同一latent space。

## 11. Pair Transformer

三个尺度分别执行Pair Interaction，全部共享Transformer参数。Odd和Even使用两套独立Transformer，每套默认两轮：

```text
round 1: self -> cross
round 2: self -> cross
```

每轮self与cross参数独立，不同round不共享。图0、图1的更新必须同时从旧状态计算，保证pair exchange equivariance。

Token尺寸为`[B,4096,96]`。Attention使用`Phi(x)=ELU(x)+1`的LoFTR-style linear attention，不构造4096x4096 attention矩阵。固定2D sine-cosine PE只进入Q/K，V保持物理内容。source reliability只乘Key一次。

每层使用固定标准残差和Pre-LayerNorm，不使用LayerScale或learnable residual gate。

## 12. Hard Odd/Even与尺度融合

每尺度使用Attention前、同物理链的幅值：

$$H_s=\mathbf1[m_{o,s}\ge m_{e,s}].$$

$$
P_s=H_sP_{o,s}+(1-H_s)P_{e,s},\quad
u_s=H_su_{o,s}+(1-H_s)u_{e,s}.
$$

尺度质量完全解析：

$$q_s=\operatorname{stopgrad}(g_s r_{stable,s}),\qquad
w_s=\frac{q_s+\epsilon}{\sum_t(q_t+\epsilon)}.$$

若所有`q_s`接近零，退化为`1/3`。不存在Scale MLP或Scale Gate。

$$P_{ms}=\sum_sw_sP_s,\qquad v=\sum_sw_su_s,\qquad u_{ms}=v/(||v||+\epsilon).$$

## 13. Dense Polar Descriptor

每个64x64 anchor根据`u_ms`恢复`phi`，分别围绕`phi`和`phi+pi`采样：

- center：1 token；
- radius 2：8 sectors；
- radius 4：8 sectors。

使用`grid_sample(..., padding_mode="reflection", align_corners=True)`，按1024个anchor分块。每个状态加入CLS、ring embedding和sector embedding，经过共享的一层、4头、MLP ratio 2的标准Transformer。

两个CLS输出`D0/Dpi`进行严格反转不变融合：

$$D_+=D_0+D_\pi,\quad D_-=|D_0-D_\pi|,$$

$$F_{phy}=\operatorname{L2Norm}(\operatorname{LN}(W[D_+;D_-])).$$

## 14. Physical-to-SLiM Adapter

```text
LayerNorm2d(96)
Conv1x1(96,192,bias=False), zero initialization
```

$$\Delta F=Adapter(F_{phy}),\qquad F^{V2}=F^{SLiM}+\Delta F.$$

初始化时必须逐元素满足`F_V2 == F_SLiM`。不使用fusion scalar、alpha或learned gate。

## 15. 冻结SLiM

V2加载`ckpt/megadepth_19epochs.ckpt`。全部SLiM参数`requires_grad=False`并永久保持`eval()`。Backbone特征提取位于`torch.no_grad()`；`F_SLIM + delta -> correlation -> loss`不能置于`no_grad()`。

只提取coarse feature，不运行Fine upsample、Fine Matching或Refinement。

## 16. 训练损失

总损失：

$$L=L_{recover}+\lambda_kL_{keep}+\lambda_pL_{physical}+\lambda_uL_{unary}+\lambda_oL_{ori}.$$

- `L_recover`：用冻结base GT置信度对V2 dual-softmax focal正样本加权，`alpha=2,gamma=2`，权重均值归一化；
- `L_keep`：保护base GT置信度最高30%，hinge margin `delta=0.02`；
- `L_physical`：最终96维物理描述子的PP loss，使用全部有效GT；
- `L_unary`：Attention前六个Odd/Even尺度描述子的平均chunked PP loss，使用25% GT；
- `L_ori`：融合方向的Homography局部Jacobian等变性损失，使用停止梯度的可靠性加权。

前3 epoch：

$$L=L_{recover}+0.25L_{keep}+1.0L_{physical}+0.3L_{unary}+0.1L_{ori}.$$

之后：

$$L=L_{recover}+0.25L_{keep}+0.5L_{physical}+0.2L_{unary}+0.1L_{ori}.$$

## 17. 数据与优化

- 训练manifest：`train_physical_v2_optical_single_ratio30_seed66.jsonl`，16,337行；
- 验证manifest：`val_optical_single_images.jsonl`，5,164行；
- image size 512，seed 66；
- 每基础影像每epoch确定性选择一种扰动；
- 训练期验证每图固定一种扰动；最终best完整展开五种扰动；
- Gabor参数：AdamW，LR `1e-5`，weight decay `0`；
- 其他V2参数：AdamW，LR `1e-4`，weight decay `0.01`；
- Cosine scheduler，20 epoch，BF16，有效batch 6，chunk 256；
- 正式单卡micro batch为6、不做梯度累积，有效batch为6；
- Pair Linear Attention固定使用FP32，其他主路径保持BF16；该设置在RTX 4090上峰值分配显存约16.66 GiB；
- best监控`val/enhanced_r0`，每2 epoch保存阶段权重。

PP与Recovery均使用数学等价的稳定`log_softmax`实现。禁止先计算双向概率乘积再把它裁剪到`1e-6`：随机初始化时，4096个token的双向概率约为`1/4096^2`，旧式裁剪会令Physical和Unary分支落入零梯度区。V0/V1保留原有默认实现，稳定模式只由V2显式启用。

## 18. 消融配置

| 名称 | 变化 |
|---|---|
| `physical_v2_core` | 完整V2 |
| `physical_v2_no_recovery_weight` | Recovery权重固定为1 |
| `physical_v2_no_pair_transformer` | Pair Transformer替换为恒等映射 |
| `physical_v2_no_polar` | Polar替换为共享1x1投影 |
| `physical_v2_soft_oe` | Hard selector改为解析幅值比例 |
| `physical_v2_single_scale_512` | 只使用512图像尺度 |

## 19. Smoke与验收

正式训练前必须验证：

1. 新manifest与V1来源逐字节一致；
2. Gabor参数范围、方向共享和quadrature响应无NaN；
3. MASW输出单位轴向方向；
4. pair exchange后输出严格交换；
5. Hard O/E逐点选择正确；
6. Polar在`phi`与`phi+pi`状态交换时融合结果不变；
7. `physical`为`[B,96,64,64]`且通道L2范数为1；
8. 冻结SLiM无梯度，V2预期参数均可反传；
9. zero-init时enhanced与base coarse feature一致；
10. batch 1/2 BF16前后向、checkpoint保存与恢复成功。

正式训练结束后停止。正式多模态测评、候选协议和Fine兼容性由后续独立任务实现。

## 20. 实施与Pilot记录

实现文件：

- `src/physical/v2_models.py`：冻结SLiM coarse extractor和V2结构；
- `src/physical/v2_losses.py`：Recovery、Preservation及稳定dual-softmax；
- `src/physical/v2_lightning.py`：五项loss、优化器、验证指标和梯度契约；
- `train_physical_v2.py`：独立训练入口；
- `tests/test_physical_v2.py`：V2单元测试。

2026-07-21完成的验收：

| 项目 | 结果 |
|---|---:|
| V0/V1/V2 CPU测试 | 36 passed |
| batch 1 BF16前后向 | 通过 |
| batch 2 BF16前后向与恢复训练 | 通过 |
| 旧BF16 Attention、batch 4峰值显存 | 9.02 GiB |
| 旧BF16 Attention、batch 8峰值显存 | 17.88 GiB（长跑升至约21.8 GiB） |
| FP32 Attention、batch 4峰值显存 | 11.19 GiB |
| FP32 Attention、batch 6峰值显存 | 16.66 GiB（预留约7.3 GiB） |
| 冻结SLiM梯度 | 始终为None |
| V2预期参数梯度 | 全部存在且有限 |
| checkpoint大小 | 约8.3 MiB，不重复保存冻结SLiM |

修正版1%/3 epoch pilot使用163条训练记录和100条验证记录。结果如下：

| epoch | train total | train physical | val physical | val enhanced R@0 |
|---:|---:|---:|---:|---:|
| 0 | 21.6413 | 15.5256 | 15.1436 | 0.86644 |
| 1 | 20.4753 | 14.2455 | 14.7920 | 0.86675 |
| 2 | 19.8615 | 13.7396 | 14.5569 | 0.86690 |

pilot表明主物理loss能够持续下降，Adapter、Pair Transformer、Polar分支和解析Gabor参数均可反传。1%数据只用于流程与数值稳定性验收，不用于判断最终模型收益。

正式训练首次使用BF16 Pair Linear Attention和batch 8时，在epoch 0约global step 699后出现持续NaN。该运行尚未完成epoch，因此没有可恢复checkpoint。修订内容：

1. Pair Linear Attention的Q/K/V、线性注意力累积、分母归一化、输出投影和MLP固定使用FP32，输出再转回主路径dtype；
2. 数值修复后先以micro batch 4、累积2完成跨epoch回归；随后按实验要求测试micro batch 6、不累积，并将正式有效batch设为6；
3. 每个训练step检查各项loss和关键输出；每次反传检查所有V2梯度；每次优化器更新后检查参数；
4. 首次非有限值立即停止，并在实验`paper_logs/nonfinite_failure.json`记录epoch、step、loss、输出、样本ID和扰动类型；
5. 修订后完成120个连续训练batch压力测试，无NaN，loss保持有限并下降；
6. 进一步使用163条基础影像（约1%）、micro batch 4、梯度累积2，完整运行3个epoch。三个epoch的训练总loss分别为21.6586、20.2852、19.6984，所有CSV数值均为有限值；峰值分配显存约11.20 GiB，并正常生成`best-02.ckpt`、`last.ckpt`和逐epoch阶段权重。
7. micro batch 6完成2个训练step及2个验证step，loss、梯度和参数均为有限值，峰值分配/保留显存分别为16.66/17.87 GiB。正式训练据此使用batch 6。

该微量数据回归测试只验证长于单epoch边界的数值稳定性、验证流程和checkpoint链路，不作为V2精度结论。

batch 6正式训练在epoch 0、global step 927被非有限梯度保护主动终止。终止前总loss保持有限并由20.14降至约10，异常参数仅为`gabor.delta_lambda`、`gabor.delta_sigma`和`gabor.gamma_logits`。根因是弱纹理位置产生零方向向量后，V2直接执行`atan2(0, 0)`：前向值有限，但反向导数未定义，非有限梯度最终汇聚到解析Gabor参数。修订后，V2在方向向量模长小于阈值时回退到固定0度，并停止该位置的方向梯度；有效方向仍使用连续`atan2`。零方向CPU反向测试及BF16、batch 6 GPU前后向测试均通过。后续非有限梯度报告还会记录当前batch索引、样本ID和扰动类型。
