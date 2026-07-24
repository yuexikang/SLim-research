# Physical Encoder V3.0.2 无训练消融实验报告

## 实验摘要

- 数据：`data/remote_archive/manifests/test_SwinMatcher_proposed_gt.jsonl`
- 分层样本：80对，每模态20对，seed=66
- 输入：512，基础anchor上限1000
- 设备：`cuda:1`，物理GPU 1，启动前空闲显存22201 MiB
- 总耗时：8.10分钟
- 所有路线均为固定算子，无训练参数、优化器或checkpoint。
- 固定样本SHA256：`e6d50bad4ed5c7594813d133857d134a870974729f594932818819bc58ef8167`
- Summary SHA256：`d5b8f9088c6b7772720fac9a79027c8927551f307f8f463b2b4ef6da26d93235`

## 核心结论

1. 冻结HIMO内部存在可用跨模态信息，但V3.0.1将PolarP权重限制为二值
   `vIMO`是主要信息瓶颈。改用连续Odd/Even幅值后，NCM从3.08提高到
   29.61-29.79，Precision从1.02%提高到8.82%-8.90%，SR从3.75%提高到
   42.50%-43.75%。
2. 固定PolarP仍不能直接充当最终描述子。最佳路线的Oracle R@5只有21.88%，
   且所有路线的正负距离margin均为负，说明GT正样本通常仍比hard negative更远。
3. 固定Shi-Tomasi的Repeatability@5为43.44%，只能覆盖不到一半GT对应区域。
   V3训练入口应使用dense/grid anchor，而不是继续优化稀疏检测阈值。
4. HIMO明显偏向包含SAR的模态：Optical-SAR与SAR-LiDAR的Odd/Even相关性、
   Oracle检索和端点指标远高于Optical-Map。当前证据支持将物理分支定位为
   可门控的SAR/结构专家，而不是无条件替代通用视觉描述子。
5. 原始PSD在当前公开复现链路中降低结果，简单SymmetricHalfTurn更差。
   后续网络应读取未折叠的方向/分区状态，自行学习反转处理，暂不把两种固定
   half-turn规则硬编码进最终表示。

## 总体消融

| 变体 | NCM | Pre | SR | RMSE | Oracle R@1 | R@5 | R@10 | Unique target | Max fan-in | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `code_exact_raw_multi_psd` | 3.08 | 1.02% | 3.75% | 9.739 | 1.78% | 4.61% | 6.51% | 13.30% | 364.81 | -73.3365 |
| `l2_multi_no_psd` | 6.47 | 1.95% | 11.25% | 9.207 | 2.91% | 6.99% | 9.69% | 29.41% | 52.79 | -0.1705 |
| `l2_multi_psd` | 4.41 | 1.33% | 5.00% | 9.647 | 1.92% | 5.09% | 7.36% | 29.46% | 52.44 | -0.1777 |
| `l2_multi_psd_even` | 18.35 | 5.75% | 27.50% | 8.057 | 8.42% | 16.44% | 20.88% | 30.59% | 34.98 | -0.2095 |
| `l2_multi_psd_hard_magnitude` | 29.61 | 8.90% | 43.75% | 6.882 | 12.52% | 21.74% | 26.54% | 32.14% | 37.44 | -0.1745 |
| `l2_multi_psd_odd` | 29.79 | 8.89% | 42.50% | 6.959 | 12.66% | 21.88% | 26.63% | 32.31% | 38.58 | -0.1744 |
| `l2_multi_psd_soft_magnitude` | 29.29 | 8.82% | 43.75% | 6.886 | 12.41% | 21.63% | 26.25% | 32.11% | 36.79 | -0.1766 |
| `l2_multi_symmetric_halfturn` | 2.67 | 0.87% | 1.25% | 9.911 | 1.51% | 4.19% | 6.47% | 27.02% | 58.20 | -0.1880 |
| `l2_single_psd` | 3.06 | 1.34% | 3.75% | 9.732 | 1.90% | 5.13% | 7.42% | 29.96% | 43.02 | -0.1939 |
| `root_multi_psd` | 4.72 | 1.77% | 7.50% | 9.461 | 2.12% | 5.08% | 7.35% | 24.09% | 73.39 | -0.1535 |

## 物理状态与检测重复性

| 模态 | Repeat@3 | Repeat@5 | 方向中位误差 | Odd corr | Even corr | rOE corr | vIMO agree | vIMO IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 23.97% | 43.44% | 32.29 deg | 0.352 | 0.341 | 0.080 | 40.61% | 5.77% |
| optical-lidar | 22.45% | 42.54% | 34.27 deg | 0.395 | 0.394 | 0.099 | 19.61% | 4.71% |
| optical-map | 17.66% | 37.34% | 39.80 deg | 0.058 | 0.012 | -0.013 | 7.06% | 5.45% |
| optical-sar | 30.87% | 50.93% | 26.39 deg | 0.583 | 0.600 | 0.131 | 89.50% | 12.70% |
| sar-lidar | 23.87% | 41.76% | 28.70 deg | 0.373 | 0.356 | 0.103 | 46.27% | 0.21% |

## 各模态最优端点结果

| 模态 | 最高Precision变体 | Precision | NCM | Oracle R@5 |
|---|---|---:|---:|---:|
| optical-lidar | `l2_multi_psd_hard_magnitude` | 5.68% | 17.35 | 15.43% |
| optical-map | `l2_multi_psd_odd` | 1.42% | 4.25 | 6.11% |
| optical-sar | `l2_multi_psd_soft_magnitude` | 14.21% | 53.45 | 31.10% |
| sar-lidar | `l2_multi_psd_hard_magnitude` | 14.75% | 44.75 | 33.12% |

## 结果分析

### 归一化与hubness

- raw CodeExact的unique-target ratio只有13.30%，平均最大fan-in达到364.81，
  存在严重descriptor hubness。
- L2将unique-target ratio提高到29.46%，fan-in降低到52.44，但Precision只从
  1.02%提高到1.33%。归一化修复了幅值尺度和部分hubness，却没有解决描述子
  排序问题。
- Root的Precision为1.77%，仍远低于连续Odd/Even幅值路线。

### 主方向与half-turn

- 单主方向与多方向L2路线Precision接近，但单方向NCM从4.41降到3.06，
  Oracle R@10也没有改善，因此没有理由在固定基线中删除辅助方向。
- NoPSD相对L2+PSD将NCM从4.41提高到6.47、SR从5.00%提高到11.25%，说明
  当前PSD交换不稳定。
- SymmetricHalfTurn只有0.87% Precision和1.25% SR，简单平均会抹平有效的
  空间非对称信息，不能作为PSD的直接替代。

### Odd/Even与原始有效性

- Odd-only、Hard magnitude和Soft magnitude三条路线几乎持平，说明决定性
  因素是保留连续结构幅值，而不是某一种复杂O/E选择器。
- Odd-only获得最高NCM 29.79和最高Oracle R@5 21.88%；Hard magnitude获得
  最高总体Precision 8.90%；Soft magnitude获得并列最高SR 43.75%。
- Even-only仍达到5.75% Precision，证明Even分支有独立信息，但总体弱于Odd。
- 原始`vIMO`总体agreement为40.61%，IoU却只有5.77%。Optical-SAR虽有
  89.50% agreement，IoU也只有12.70%，高agreement主要来自双方同时为0，
  不代表有效区域真正重合。

### 模态差异

- Optical-SAR最佳Precision为14.21%、NCM 53.45、SR 80%；SAR-LiDAR最佳
  Precision为14.75%、NCM 44.75、SR 55%-60%。
- Optical-LiDAR有中等信号，最佳Precision为5.68%、NCM 17.35。
- Optical-Map仍几乎失效，最佳Precision仅1.42%，方向中位误差39.80度，
  Odd/Even相关性接近0。
- Optical-SAR的Odd/Even相关性为0.583/0.600且Repeatability@5为50.93%，
  是最适合继续验证物理编码的子任务。

## 限制

- 该实验使用分层80对诊断集，不替代完整测试集最终结果。
- 当前公开实现不包含P-code预处理、DoFS、CDMS或官方MATLAB匹配器。
- `vIMO`是二值有效性图，主要查看agreement和IoU，不单独依赖Pearson相关。
- Oracle指标以B中5 px内存在描述子为条件，不等同于端到端匹配率。

## 下一步

V3首个可训练版本应按以下顺序实现：

1. HIMO保持冻结，输出
   `[cos(2phi), sin(2phi), compressed_odd, compressed_even, r_oe]`；
   原始`vIMO`只作为辅助mask或诊断量，不再作为PolarP唯一权重。
2. 使用dense/grid位置训练，不依赖固定Shi-Tomasi重复检测。
3. 先实现`StructuredPolar-LocalOnly`，保留Odd/Even连续统计和未折叠方向token；
   不把PSD或SymmetricHalfTurn写死。
4. 对低方向置信区域绕过强制主方向对齐，或让编码器同时读取原始和对齐分支。
5. 先确认LocalOnly在Oracle R@1/R@5和多模态端点上超过本报告的固定Odd基线，
   再接256分辨率FPN；暂不实现512 FPN。
6. 模态门控实验必须保留。若提升继续集中在SAR相关组合，V3应作为SAR物理专家
   与通用特征融合，而不是单独承担所有跨模态匹配。
