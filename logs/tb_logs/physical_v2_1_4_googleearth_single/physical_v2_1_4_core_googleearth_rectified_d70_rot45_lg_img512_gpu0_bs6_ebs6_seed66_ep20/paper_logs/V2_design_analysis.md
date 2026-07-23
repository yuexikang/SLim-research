# Physical Encoder V2.1.4 训练结果分析

## 实验身份

- 模型：`physical_v2_core`，711,274个可训练参数。
- 数据：16,402张GoogleEarth训练图；1,822张无泄漏验证图。
- 训练：每图每epoch一种在线几何扰动，20 epoch，batch 6，BF16。
- 训练链：epoch 0-1来自V2.1.3 checkpoint；epoch 2-19使用V2.1.4有效区域rectification和黑图保护。
- 最佳权重：`checkpoints/best-18.ckpt`，epoch 18，global step 51,946。
- 最终权重：`checkpoints/last-v1.ckpt`，epoch 19，global step 54,680。
- `best-02.ckpt`和`last.ckpt`是同目录早期重启留下的旧文件，不用于后续实验。

## 训练与完整验证

训练期验证从epoch 2到epoch 18：

| 指标 | epoch 2 | epoch 18 | 变化 |
|---|---:|---:|---:|
| enhanced R@0 | 82.01% | 84.52% | +2.50 pp |
| enhanced R@1 | 95.96% | 97.86% | +1.90 pp |
| recovery loss | 1.597 | 1.097 | -31.3% |
| physical loss | 3.724 | 2.252 | -39.5% |
| unary loss | 15.255 | 14.383 | -5.7% |
| orientation loss | 0.0493 | 0.0488 | 基本不变 |

best权重在五种扰动完整验证上的训练日志：

| 扰动 | R@0 | R@1 |
|---|---:|---:|
| translation | 90.73% | 99.10% |
| scale | 87.29% | 99.39% |
| yaw | 82.08% | 98.66% |
| pitch | 82.23% | 98.59% |
| roll | 80.81% | 94.42% |
| 总体 | 84.46% | 97.84% |

## Base与Enhanced同协议对照

补算文件：`base_vs_enhanced_full_validation_batch2_logged_protocol.json`。共统计25,630,764个有效GT coarse token：

| 指标 | 冻结SLiM base | V2 enhanced | 净变化 |
|---|---:|---:|---:|
| R@0 | 80.87% | 81.85% | +0.98 pp |
| R@1 | 94.01% | 97.94% | +3.92 pp |

| 扰动 | R@0净变化 | R@1净变化 |
|---|---:|---:|
| translation | +0.02 pp | +2.97 pp |
| scale | +0.65 pp | +2.50 pp |
| yaw | +0.46 pp | +2.33 pp |
| pitch | -0.23 pp | +2.03 pp |
| roll | +3.44 pp | +8.60 pp |

- R@0：V2救回7.49%的token，同时破坏6.51%，净增0.98点。
- R@1：V2救回4.14%的token，只破坏0.22%，净增3.92点。
- FP32相似度复算得到近似相同的相对结论：R@0净增0.92点、R@1净增3.92点。
- 训练结束时自动保存的enhanced R@0为84.46%，但独立使用相同checkpoint、数据、batch 2和原统计函数复算为81.85%；R@1可以复现。该R@0绝对值在用于论文前必须继续审计，当前以同一次补算中的base/enhanced相对变化为准。

## 结构诊断

- 残差路线有效：完整验证的dual-softmax正样本置信度由base 0.639提升到enhanced 0.696。
- Scale fusion未塌缩：归一化尺度熵为0.853，等效使用约2.55个尺度。
- Hard O/E明显偏科：Odd选择比例为94.7%，Even专家在最终融合中利用不足。
- Unary分支学习较弱：loss只下降5.7%；最终收益主要来自Pair Transformer、Polar和Adapter后的physical/recovery路径。
- Orientation loss和梯度都很小，当前方向监督对优化的实际驱动力有限。
- Gabor参数只温和偏离初值，未触碰约束边界；全程无NaN，Gabor梯度清洗触发数为0。
- Delta平均范数约为base的60%，不是微小修正；这带来明显救回能力，也会改变部分原本正确的R@0 token。
- 峰值分配/保留显存约17.66/18.94 GiB，吞吐量约3.60张/秒。

## 结论

V2-Core在同域GoogleEarth合成几何验证上是“部分成功”：

1. 冻结SLiM加物理残差的主路线成立，尤其显著提升正确3x3 coarse邻域内的召回。
2. 对roll的收益最强，说明方向物理先验和Polar结构确实提供旋转鲁棒性。
3. exact R@0净增较小，Pitch略有退化；当前V2更像粗候选鲁棒性增强器，而不是独立精确定位器。
4. Hard O/E几乎退化为Odd单专家，不能据此宣称Odd/Even双专家设计已被验证。
5. 本实验没有真实多模态验证，不能据此判断V2是否达到跨模态目标。

下一步不建议继续增加epoch。应使用`best-18.ckpt`执行真实多模态测试；消融优先级为`SoftOE`、`NoPairTransformer`、`NoPolar`，分别验证O/E偏科、成对交互和旋转鲁棒性来源。
