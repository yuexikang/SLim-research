# Physical Encoder V3.0.3 P5局部编码消融实验报告

## 结论摘要

- `Pointwise-P5`几乎不能建立稳定对应，说明单点的5通道HIMO状态不足以构成局部描述子。
- 参数量几乎相同的`RectConv-P5`在合成验证和Proposed跨模态测试上均显著优于`Pointwise-P5`，证明空间邻域聚合是必要条件。
- `RectConv-P5`的绝对性能仍低：Proposed上`Pre=0.424%`、`SR=3.75%`。普通矩形卷积解决了“没有上下文”的问题，但没有解决方向对齐、跨模态不变性和强判别性问题。
- 本实验是1%训练数据、3 epoch的结构筛选，不是最终20 epoch性能实验。结果足以淘汰Pointwise路线，但不能据此确定RectConv完整训练后的上限。

## 实验问题

本次只改变5通道Physical State之后的局部编码方式：

| 路线 | 空间聚合 | 可训练参数 |
|---|---|---:|
| Pointwise-P5 | 两层`1x1`卷积，只读取当前位置 | 6,816 |
| RectConv-P5 | `5x5 + 3x3 + 1x1`卷积 | 6,768 |

两者共享冻结HIMO和相同P5：

$$
P_5=[\cos(2\phi),\sin(2\phi),m_o,m_e,r_{oe}],
\qquad
r_{oe}=\frac{|m_o-m_e|}{m_o+m_e+\epsilon}.
$$

输出均为`[B,128,64,64]`的通道L2归一化描述子。参数量差异仅48，约0.71%，因此性能差异主要来自空间聚合，不来自模型容量。

## 公平性控制

- 训练集：`train_GoogleEarth_single.jsonl`的固定1%分层样本，共164张基础影像；
- 两次训练的`selected_train_rows.jsonl` SHA256均为
  `47093ea4f0f0e50f3db14672f67d268d53945c8d255406e286334a2d6f3686d4`；
- seed 66，每张影像每epoch在线选择一种扰动；
- 训练3 epoch，batch 4，有效batch 8；
- 优化器、学习率、PP loss、验证40张基础影像及扰动协议完全相同；
- Proposed测试固定使用V3.0.2抽取的80对影像，每种模态20对；
- 测试采用64x64 dense cosine mutual-nearest-neighbor，不使用RANSAC；
- 原图坐标重投影误差不超过5 px判为正确，`NCM>=20`判为成功。

## 合成验证

| 路线 | 最佳epoch | val R@0 | val R@1 | val PP loss | train PP loss |
|---|---:|---:|---:|---:|---:|
| Pointwise-P5 | 0 | 0.00435 | 0.00795 | 18.538 | 19.608 |
| RectConv-P5 | 2 | **0.07519** | **0.08944** | **15.149** | **15.165** |

RectConv的最佳`val R@0`是Pointwise的约17.29倍。RectConv的正样本相似度为0.9185，hard-negative相似度为0.9790，margin仍为负值`-0.0604`，说明其描述子已学到部分对应结构，但最近的错误匹配仍普遍比真值更相似。

## Proposed跨模态测试

### 总体

| 路线 | NCM | Pre | SR | RMSE | 平均匹配数 |
|---|---:|---:|---:|---:|---:|
| Pointwise-P5 | 0.30 | 0.025% | 0.00% | 10.000 | 1066.80 |
| RectConv-P5 | **4.80** | **0.424%** | **3.75%** | **9.742** | 973.75 |

RectConv相对Pointwise：

- NCM提高16倍；
- Precision提高约16.68倍；
- 从0个成功对提高到3个成功对；
- 输出匹配数略少，但正确匹配更多，提升不是靠增加匹配数量获得的。

### RectConv分模态

| 模态 | NCM | Pre | SR | RMSE |
|---|---:|---:|---:|---:|
| Optical-LiDAR | 5.05 | 0.478% | 0.00% | 10.000 |
| Optical-Map | 0.15 | 0.016% | 0.00% | 10.000 |
| Optical-SAR | **11.75** | **0.950%** | **15.00%** | **8.969** |
| SAR-LiDAR | 2.25 | 0.251% | 0.00% | 10.000 |

收益主要来自Optical-SAR；Optical-Map几乎完全失败。这与此前物理分支在SAR方向相对更有潜力的观察一致，但样本只有20对，暂时只能作为方向性信号。

## 解释

### 为什么Pointwise失败

单个位置的$\phi,m_o,m_e,r_{oe}$只描述局部响应状态。大量不同地点会产生相近的方向和强度组合，网络没有邻域排列、边缘延伸、角点结构或纹理布局可用于消歧。`1x1`网络无论增加多少层，都不能创造输入中不存在的空间信息。

### RectConv证明了什么

`5x5 + 3x3`卷积让每个粗特征看到邻域P5的相对排列，因此能识别局部方向场和Odd/Even结构组合。参数量匹配排除了“更大模型”这一主要混淆因素，所以本实验支持：

> P5必须经过空间邻域编码，不能只做逐点映射。

### RectConv为什么仍然很弱

- 矩形卷积的采样坐标不随IMO主方向对齐，旋转和透视会改变邻域排列；
- 1%数据和3 epoch只够判断结构能否学习，尚未充分收敛；
- HIMO单点状态被压缩到5通道，可能丢失多方向、多尺度和相位信息；
- dense mutual-NN中存在大量外观相似的hard negatives，而当前margin仍为负；
- 不同模态的局部响应统计差异仍直接进入卷积，没有显式模态归一化。

## 决策

1. 淘汰`Pointwise-P5`作为独立描述子路线。
2. 保留`RectConv-P5`作为“常规矩形局部聚合”对照组。
3. V3主路线应继续比较Polar邻域编码与参数量匹配的RectConv，而不是再与Pointwise比较。
4. 下一次正式比较必须复用同一30%训练记录、相同训练轮数和同一dense matching协议；否则不能把差异归因于Polar结构。

## 结果位置

- Pointwise checkpoint：
  `logs/tb_logs/physical_v3_p5_pilot/pointwise_p5_googleearth_ratio1_seed66_ep3/checkpoints/best-00-0.0043.ckpt`
- RectConv checkpoint：
  `logs/tb_logs/physical_v3_p5_pilot/rectconv_p5_googleearth_ratio1_seed66_ep3/checkpoints/best-02-0.0752.ckpt`
- Pointwise测试：
  `outputs/eval_physical_v3_p5_pointwise_pilot_proposed/`
- RectConv测试：
  `outputs/eval_physical_v3_p5_rectconv_pilot_proposed/`
