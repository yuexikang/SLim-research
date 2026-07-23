# Physical Encoder V3 版本更新记录

> 当前实现：V3.0.0。基础结构规范见[Physical Encoder V3](./Physical%20Encoder%20V3.md)。本文件只记录实现修订，不重复完整设计。

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
