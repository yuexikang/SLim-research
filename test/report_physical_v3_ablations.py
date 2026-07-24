# 作用：将Physical V3.0.2无需训练消融的JSON结果整理为紧凑Markdown报告。

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def percent(value):
    return f"{100.0 * float(value):.2f}%"


def number(value, digits=3):
    return f"{float(value):.{digits}f}"


def variant_table(summary):
    lines = [
        "| 变体 | NCM | Pre | SR | RMSE | Oracle R@1 | R@5 | R@10 | Unique target | Max fan-in | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["variant_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    number(row["NCM"], 2),
                    percent(row["Pre"]),
                    percent(row["SR"]),
                    number(row["RMSE"], 3),
                    percent(row["oracle_r1"]),
                    percent(row["oracle_r5"]),
                    percent(row["oracle_r10"]),
                    percent(row["unique_target_ratio"]),
                    number(row["max_target_fan_in"], 2),
                    number(row["distance_margin_top10"], 4),
                ]
            )
            + " |"
        )
    return lines


def physical_table(summary):
    lines = [
        "| 模态 | Repeat@3 | Repeat@5 | 方向中位误差 | Odd corr | Even corr | rOE corr | vIMO agree | vIMO IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {"overall": summary["physical_summary"]}
    groups.update(summary["physical_by_modality"])
    for name, row in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    percent(row["detector_repeatability_r3"]),
                    percent(row["detector_repeatability_r5"]),
                    f'{number(row["orientation_median_error_deg"], 2)} deg',
                    number(row["odd_correlation"], 3),
                    number(row["even_correlation"], 3),
                    number(row["roe_correlation"], 3),
                    percent(row.get("vimo_agreement", 0.0)),
                    percent(row.get("vimo_iou", 0.0)),
                ]
            )
            + " |"
        )
    return lines


def modality_winners(summary):
    modalities = sorted(
        {
            modality
            for groups in summary["variant_by_modality"].values()
            for modality in groups
        }
    )
    lines = [
        "| 模态 | 最高Precision变体 | Precision | NCM | Oracle R@5 |",
        "|---|---|---:|---:|---:|",
    ]
    for modality in modalities:
        candidates = [
            (name, groups[modality])
            for name, groups in summary["variant_by_modality"].items()
            if modality in groups
        ]
        name, row = max(candidates, key=lambda item: item[1]["Pre"])
        lines.append(
            f"| {modality} | `{name}` | {percent(row['Pre'])} | "
            f"{number(row['NCM'], 2)} | {percent(row['oracle_r5'])} |"
        )
    return lines


def diagnosis(summary):
    physical = summary["physical_summary"]
    variants = summary["variant_summary"]
    best_pre_name, best_pre = max(
        variants.items(),
        key=lambda item: item[1]["Pre"],
    )
    best_oracle_name, best_oracle = max(
        variants.items(),
        key=lambda item: item[1]["oracle_r5"],
    )
    statements = []
    repeatability = physical["detector_repeatability_r5"]
    if repeatability < 0.3:
        statements.append(
            "Detector Repeatability@5低于30%，固定稀疏anchor是首要瓶颈之一；"
            "后续应优先使用dense/grid anchor。"
        )
    elif repeatability < 0.5:
        statements.append(
            "Detector Repeatability@5处于30%-50%，稀疏anchor覆盖有限，"
            "仍建议把dense/grid作为主要训练接口。"
        )
    else:
        statements.append(
            "Detector Repeatability@5超过50%，检测覆盖不是当前唯一主导瓶颈。"
        )
    if best_oracle["oracle_r5"] < 0.3:
        statements.append(
            "最佳Oracle R@5仍低于30%，即使GT附近存在目标描述子，"
            "固定PolarP也通常不能找回它，需要可训练Polar编码。"
        )
    elif best_oracle["oracle_r5"] < 0.6:
        statements.append(
            "最佳Oracle R@5处于30%-60%，固定物理统计有可学习信号，"
            "但不足以直接作为最终描述子。"
        )
    else:
        statements.append(
            "最佳Oracle R@5超过60%，固定物理统计具有较强候选检索能力；"
            "应先优化匹配与hubness再增加网络容量。"
        )
    if physical["orientation_median_error_deg"] > 30:
        statements.append(
            "IMO轴向方向中位误差超过30度，直接依赖单一主方向对齐风险较高。"
        )
    statements.append(
        f"端点Precision最高的是`{best_pre_name}`"
        f"（{percent(best_pre['Pre'])}）；Oracle R@5最高的是"
        f"`{best_oracle_name}`（{percent(best_oracle['oracle_r5'])}）。"
    )
    return statements


def main():
    args = parse_args()
    summary = json.loads(args.summary_path.read_text(encoding="utf-8"))
    gpu = summary.get("selected_gpu") or {}
    lines = [
        "# Physical Encoder V3.0.2 无训练消融实验报告",
        "",
        "## 实验摘要",
        "",
        f"- 数据：`{summary['manifest_path']}`",
        f"- 分层样本：{summary['num_pairs']}对，"
        f"每模态{summary['pairs_per_modality']}对，seed={summary['seed']}",
        f"- 输入：{summary['image_size']}，基础anchor上限"
        f"{summary['max_keypoints']}",
        f"- 设备：`{summary['device']}`，物理GPU {gpu.get('index', 'N/A')}，"
        f"启动前空闲显存{gpu.get('memory_free_mib', 'N/A')} MiB",
        f"- 总耗时：{summary['runtime_seconds'] / 60.0:.2f}分钟",
        "- 所有路线均为固定算子，无训练参数、优化器或checkpoint。",
        "",
        "## 总体消融",
        "",
        *variant_table(summary),
        "",
        "## 物理状态与检测重复性",
        "",
        *physical_table(summary),
        "",
        "## 各模态最优端点结果",
        "",
        *modality_winners(summary),
        "",
        "## 自动诊断",
        "",
    ]
    lines.extend(f"- {statement}" for statement in diagnosis(summary))
    lines.extend(
        [
            "",
            "## 限制",
            "",
            f"- 该实验使用分层{summary['num_pairs']}对诊断集，"
            "不替代完整测试集最终结果。",
            "- 当前公开实现不包含P-code预处理、DoFS、CDMS或官方MATLAB匹配器。",
            "- `vIMO`是二值有效性图，主要查看agreement和IoU，不单独依赖Pearson相关。",
            "- Oracle指标以B中5 px内存在描述子为条件，不等同于端到端匹配率。",
            "",
            "## 下一步",
            "",
            "根据上述诊断选择dense anchor、可训练Polar Encoder或匹配修正；"
            "在确定瓶颈前不实现完整FPN。",
            "",
        ]
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
