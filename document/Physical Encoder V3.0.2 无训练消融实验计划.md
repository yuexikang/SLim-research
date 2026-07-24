# Physical Encoder V3.0.2 无训练消融实验计划

## 1. 目标

在实现可训练 Polar Encoder 和 FPN 之前，定位 V3.0.1 无训练基线的主要
故障层：

1. HIMO 物理状态是否跨模态稳定；
2. 固定 Shi-Tomasi anchor 是否具备几何重复性；
3. PolarP 是否能在 GT 正样本存在时找回正确描述子；
4. 匹配失败是否主要来自 descriptor hubness；
5. 主方向、PSD 与 Odd/Even coupling 是否有效。

本实验不包含优化器、反向传播、checkpoint、SLiM 融合或任何可训练模块。

## 2. 数据与抽样

- 数据：`test_SwinMatcher_proposed_gt.jsonl`。
- 按 `modality0-modality1` 分层。
- 每种模态固定抽取 20 对，共 80 对。
- seed 固定为 66。
- 保存 `selected_rows.jsonl`，所有变体逐行使用相同影像对。
- smoke test 每种模态 1 对、最多 100 个基础 anchor。

不得使用 manifest 前 80 行代替分层抽样。

## 3. 第一步：GT Oracle 故障分解

### 3.1 Detector

将 A 的基础 anchor 通过 GT 投影到 B，在原图坐标统计：

- `Detector Repeatability@3`；
- `Detector Repeatability@5`。

### 3.2 Descriptor

只对 B 中存在 5 px GT 正样本的 A 描述子统计：

- `Oracle R@1/R@5/R@10`；
- `MRR@10`；
- positive descriptor distance；
- top-10 hardest-negative distance；
- distance margin。

### 3.3 Matching

统计：

- unique-target ratio；
- maximum target fan-in；
- NCM、Precision、SR、RMSE。

### 3.4 Physical State

在 GT 对应位置统计：

- doubled-angle IMO 方向平均和中位误差；
- Odd、Even、`r_oe`、原始 `vIMO` 的对应相关性。

## 4. 第二步：固定 PolarP 消融

| ID | 归一化 | 主方向 | Half-turn |
|---|---|---|---|
| `code_exact_raw_multi_psd` | 无 | 原始多方向 | 原始 PSD |
| `l2_multi_psd` | L2 | 原始多方向 | 原始 PSD |
| `root_multi_psd` | Root | 原始多方向 | 原始 PSD |
| `l2_single_psd` | L2 | 单主方向 | 原始 PSD |
| `l2_multi_no_psd` | L2 | 原始多方向 | 关闭 |
| `l2_multi_symmetric_halfturn` | L2 | 原始多方向 | 双分支平均 |

除表中单一因素外，HIMO、anchor、Polar patch、bin 和匹配协议保持一致。

## 5. 第三步：Odd/Even 固定分支

使用 L2、多主方向和 PSD，对比：

1. 原始 `vIMO + Hard O/E orientation`；
2. Odd-only；
3. Even-only；
4. Hard O/E magnitude；
5. Soft doubled-angle O/E magnitude。

这些是固定物理分支，不包含学习门控。

## 6. 第四步：架构决策

- Repeatability@5 低：后续使用 dense/grid anchor。
- Repeatability 足够但 Oracle R@5 低：训练 Structured Polar Encoder。
- Oracle retrieval 高但端点匹配低：先处理归一化、hubness 和匹配。
- 仅单一模态稳定：HIMO 定位为 gated specialist。
- 所有模态方向均不稳定：先验证 PyTorch 与官方 HIMO 数值一致性。

阈值只用于诊断，不作为论文最终结论。

## 7. 执行与资源

入口：

```bash
/root/miniconda3/envs/slim/bin/python \
  test/evaluate_physical_v3_ablations.py \
  --manifest_path data/remote_archive/manifests/test_SwinMatcher_proposed_gt.jsonl \
  --manifest_split test \
  --output_dir outputs/eval_physical_v3_0_2_ablations_proposed \
  --device auto \
  --pairs_per_modality 20 \
  --image_size 512 \
  --max_keypoints 1000 \
  --seed 66
```

`--device auto`通过 `nvidia-smi`选择可见 GPU 中空闲显存最多的一张，并将
选择结果写入 summary。`max_cooccurrence_levels`仅为内存安全上限，不作为消融。

## 8. 输出

- `selected_rows.jsonl`
- `pair_variant_metrics.csv`
- `physical_pair_metrics.csv`
- `variant_summary.csv`
- `variant_modality_summary.csv`
- `physical_modality_summary.csv`
- `summary.json`
- `command.txt`
- `document/Physical Encoder V3.0.2 无训练消融实验报告.md`
