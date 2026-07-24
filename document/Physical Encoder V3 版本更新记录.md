# Physical Encoder V3 版本更新记录

> 当前实现：V3.0.3。基础结构规范见[Physical Encoder V3](./Physical%20Encoder%20V3.md)。本文件只记录实现修订，不重复完整设计。

## V3.0.3

### 更新摘要

- 新增可训练的Pointwise-P5与参数量匹配的RectConv-P5局部编码消融。
- 使用相同固定训练样本、优化配置和Proposed 80对测试子集，单独检验空间聚合的作用。
- 本版本只进行1%数据、3 epoch结构筛选，不替代后续30%正式实验，也不改变V3.0.0至V3.0.2行为。

### 详细内容

#### 设计与协议

冻结HIMO并构造
`[cos(2phi), sin(2phi), odd, even, r_oe]`五通道状态。
Pointwise仅使用`1x1`卷积；RectConv使用`5x5 + 3x3 + 1x1`卷积。
两者分别为6816和6768个可训练参数，均输出128维、64x64的L2描述子。

训练使用GoogleEarth固定1%基础影像、seed 66、3 epoch和相同在线扰动。
测试复用V3.0.2 Proposed固定80对，采用dense cosine mutual-nearest-neighbor。

#### 运行与输出

- Pointwise最佳`val R@0=0.00435`；
- RectConv最佳`val R@0=0.07519`；
- Proposed上Pointwise为`NCM=0.30/Pre=0.025%/SR=0%`；
- Proposed上RectConv为`NCM=4.80/Pre=0.424%/SR=3.75%`；
- 完整分析见
  `document/Physical Encoder V3.0.3 P5局部编码消融实验报告.md`。

#### 兼容性

V3.0.3使用独立模型、训练入口和测评入口，不修改冻结HIMO、PolarP、
V3.0.2消融协议及已有结果。旧checkpoint与旧命令不受影响。

#### 验证状态

- 3项P5单元测试通过；
- Pointwise与RectConv GPU训练smoke通过；
- 两条路线均完成相同1%数据、3 epoch pilot；
- 两条路线均完成相同Proposed 80对测试；
- 尚未运行30%正式训练和Expanded MRSI测试。

### 文件变更

#### 新增

- `src/physical/v3_p5_models.py`
- `src/physical/v3_p5_lightning.py`
- `train_physical_v3_p5.py`
- `test/evaluate_physical_v3_p5.py`
- `test/test_physical_v3_p5.py`
- `document/Physical Encoder V3.0.3 P5局部编码消融实验报告.md`

#### 修改

- `document/Physical Encoder V3 版本更新记录.md`

#### 删除

- 无

#### 重命名

- 无

## V3.0.2

### 更新摘要

- 新增无需训练的分层消融套件和GT Oracle故障分解。
- 新增NoPSD、SymmetricHalfTurn及Odd/Even固定分支。
- 自动选择空闲显存最多的可见GPU，并保存设备快照和固定抽样记录。
- V3.0.1 CodeExact默认行为保持不变，不增加任何可训练参数。

### 详细内容

#### 诊断指标

除NCM、Precision、SR和RMSE外，新增：

- Detector Repeatability@3/@5；
- descriptor Oracle R@1/@5/@10和MRR@10；
- 正样本距离、top-10 hard-negative距离与margin；
- unique-target ratio和maximum target fan-in；
- IMO轴向方向误差；
- Odd、Even、`r_oe`与原始`vIMO`对应相关性。

#### 固定消融

统一在分层相同影像对上比较raw/L2/Root、单/多主方向、PSD/NoPSD/
SymmetricHalfTurn，以及Odd-only、Even-only、Hard和Soft O/E magnitude。
SymmetricHalfTurn使用$\phi$与$\phi+\pi$两个无PSD描述子的均值，属于V3扩展，
不宣称为原始PolarP。

#### 文件清单

新增：

- `src/physical/v3_ablation.py`
- `test/evaluate_physical_v3_ablations.py`
- `test/report_physical_v3_ablations.py`
- `test/test_physical_v3_ablations.py`
- `document/Physical Encoder V3.0.2 无训练消融实验计划.md`

修改：

- `src/physical/v3_himo.py`
- `src/physical/v3_polarp.py`
- `document/Physical Encoder V3.md`
- `document/Physical Encoder V3 版本更新记录.md`

#### 正式验证

- 在Proposed测试集按四种模态各固定抽取20对，共80对；
- seed为66，所有10条路线使用完全相同的影像对和HIMO状态；
- 启动时自动选择物理GPU 1，空闲显存22201 MiB；
- 10个单元测试和4模态smoke test通过；
- 正式运行80对耗时485.96秒，无NaN或中断；
- 原始输出保存到
  `outputs/eval_physical_v3_0_2_ablations_proposed/`；
- 实验报告保存为
  `document/Physical Encoder V3.0.2 无训练消融实验报告.md`。

## V3.0.1

### 更新摘要

- 修复高动态范围影像经过HIMO中位数归一化后，CoF所需灰度级超过2048而提前终止的问题。
- CoF安全上限提高到8192并开放命令行配置，不压缩、不截断也不重新量化实际灰度级。
- HIMO、PolarP、anchor、匹配协议和评价指标保持V3.0.0不变。

### 详细内容

#### CoF动态灰度级

官方CoF根据当前影像最大整数灰度动态建立共现矩阵。部分多模态影像在非零中位数归一化后需要2169个灰度级，超过V3.0.0人为设置的2048安全上限，但仍属于合理输入。

V3.0.1默认允许最多8192个灰度级。实际矩阵尺寸仍由当前影像动态决定，例如2169级只建立`2169x2169`矩阵。`--max_cooccurrence_levels`只防止异常暗图造成不可控矩阵分配，不会改变上限以内的图像数值。

#### 验证状态

- 新增超过2048灰度级的CoF回归测试；
- 完整单元测试通过后运行正式数据集；
- Proposed和Expanded MRSI正式结果在本版本完成后补充。

### 文件变更

#### 新增

- 无

#### 修改

- `src/physical/v3_himo.py`
- `test/evaluate_physical_v3_baseline.py`
- `test/test_physical_v3_baseline.py`
- `document/Physical Encoder V3.md`
- `document/Physical Encoder V3 版本更新记录.md`

#### 删除

- 无

#### 重命名

- 无

## V3.0.0

### 更新摘要

- 建立首个完全不训练的 Physical V3 基线，固定执行 HIMO 与480维 PolarP。
- HIMO 物理公式和参数锁定到官方仓库 commit `884297aa36dfb9c89d9a7d5bf66c142bf8707a77`。
- 基线不加载 SLiM、不加载 checkpoint、不包含神经网络，也不使用 RANSAC。
- 新增独立多模态评测入口，输出 NCM、Pre、SR、RMSE、逐对结果、分模态结果和随机可视化。
- 明确公开源码可复现范围与官方 P-code、DoFS、CDMS 等未覆盖范围，结果文件会保存该边界。
- 建立 Physical Encoder 后续版本记录的固定格式与文件清单规则。

### 详细内容

#### 冻结 HIMO

实现公开源码可审计的单尺度 HIMO 核心：

1. 非零像素中位数强度归一化；
2. Co-occurrence Filter 与 `0.75/0.25` 残差混合；
3. 四尺度、六方向 Log-Gabor；
4. Odd/Even Sobel 和 Deep-Shallow 复响应；
5. MASW；
6. Hard Odd/Even orientation coupling；
7. 跨模态 `int_flag=1` 有效性图。

模块没有可训练参数，运行在 `eval()` 和 inference mode 下。输入相同则输出确定，不接受任何训练梯度。

#### PolarP

PolarP 使用官方公开代码的落地定义：

- `patch_size=72`；
- 12个空间扇区；
- 12个方向 bin；
- 中心圆盘和两圈等面积扇区；
- 80%辅助主方向阈值和抛物线插值；
- 保留全部满足官方阈值的主方向；
- deeper、skip、global统计；
- PSD half swap；
- 最终描述子维度严格为480。

`descriptor_normalization=none` 是V3.0.0默认基线。`l2`和`root`只作为显式消融，不属于Code-Exact描述子。

#### Anchor与匹配

公开官方入口的`Deal_Extreme.p`和`Preproscessing.p`只有P-code，无法审计；官方DoFS与CDMS也未在本版本复现。V3.0.0使用固定Shi-Tomasi在HIMO结构幅值上检测anchor，再执行PolarP。

描述子匹配采用单向最近邻，并为每个目标描述子只保留距离最小的一个源描述子，对应官方公开匹配代码的唯一目标处理。评测阶段不做RANSAC，正确性只由真值Homography在原图坐标下判定。

#### 评测输出

评测目录包含：

```text
summary.json
pair_metrics.csv
modality_metrics.csv
command.txt
visualizations/
```

`summary.json`同时记录实现版本、官方源码commit、已覆盖链路、未覆盖链路、运行参数和总体/分模态指标。

#### 正式测试命令

Proposed：

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib \
/root/miniconda3/envs/slim/bin/python test/evaluate_physical_v3_baseline.py \
  --manifest_path data/remote_archive/manifests/test_SwinMatcher_proposed_gt.jsonl \
  --manifest_split test \
  --output_dir outputs/eval_physical_v3_0_0_himo_polarp_proposed \
  --device cuda:0 \
  --image_size 512 \
  --max_keypoints 1000 \
  --num_vis_pairs 10
```

Expanded MRSI：

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib \
/root/miniconda3/envs/slim/bin/python test/evaluate_physical_v3_baseline.py \
  --manifest_path data/remote_archive/manifests/test_SwinMatcher_expanded_MRSI_gt.jsonl \
  --manifest_split test \
  --output_dir outputs/eval_physical_v3_0_0_himo_polarp_expanded_mrsi \
  --device cuda:0 \
  --image_size 512 \
  --max_keypoints 1000 \
  --num_vis_pairs 10
```

#### 验证状态

- Python语法检查通过；
- 4项V3单元测试通过；
- Proposed manifest单对128尺寸smoke通过；
- Proposed manifest单对512尺寸smoke通过；
- 尚未运行两个完整多模态数据集。

### 文件变更

#### 新增

- `src/physical/v3_himo.py`
- `src/physical/v3_polarp.py`
- `test/evaluate_physical_v3_baseline.py`
- `test/test_physical_v3_baseline.py`
- `document/Physical Encoder V3 版本更新记录.md`
- `document/实验版本更新记录固定提示.md`

#### 修改

- `document/Physical Encoder V3.md`

#### 删除

- 无

#### 重命名

- 无
