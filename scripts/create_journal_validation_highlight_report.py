#!/usr/bin/env python3
"""Create a Chinese highlighted report from journal-validation tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
MAIN_CSV = ANALYSIS_DIR / "journal_validation_main_table.csv"
BASELINE_CSV = ANALYSIS_DIR / "journal_validation_baseline_table.csv"
SWEEP_CSV = ANALYSIS_DIR / "rca_component_weight_sweep_aggregate.csv"
OUTPUT_MD = ANALYSIS_DIR / "journal_validation_highlight_zh.md"


GREEN = "#166534"
BLUE = "#1d4ed8"
RED = "#b91c1c"
ORANGE = "#c2410c"
GRAY = "#475569"


def num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    x = num(value)
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.{digits}f}"


def strong(value: Any, color: str = GREEN) -> str:
    return f'<span style="color:{color}"><strong>{fmt(value)}</strong></span>'


def warn(value: Any) -> str:
    return strong(value, RED)


def plain(value: Any) -> str:
    return fmt(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(item).replace("\n", " ").replace("|", "/") for item in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def find_row(df: pd.DataFrame, exp_id: str) -> pd.Series:
    rows = df.loc[df["exp_id"].astype(str).eq(exp_id)]
    if rows.empty:
        raise ValueError(f"missing exp_id: {exp_id}")
    return rows.iloc[-1]


def main_experiment_section(main: pd.DataFrame) -> list[str]:
    rows = []
    for exp_id, note in [
        ("main_wadi_e8", "WADI 快筛主结果，成本低，条件变量 RCA 最好"),
        ("main_wadi_e15", "WADI 长训练确认，变量级 MRR/Hit@1 最好"),
        ("main_swat_e8", "SWaT 当前推荐主结果"),
        ("main_swat_e15", "SWaT 长训练，检测 Raw-F1 提高但 RCA 下降"),
    ]:
        row = find_row(main, exp_id)
        is_selected = exp_id in {"main_wadi_e8", "main_swat_e8"}
        is_best_wadi_15 = exp_id == "main_wadi_e15"
        rows.append(
            [
                row["dataset"],
                row["method"],
                int(row["epochs"]),
                strong(row["Var-MRR"], BLUE if is_best_wadi_15 else GREEN) if is_selected or is_best_wadi_15 else warn(row["Var-MRR"]),
                strong(row["Var-Hit@1"], BLUE if is_best_wadi_15 else GREEN) if is_selected or is_best_wadi_15 else warn(row["Var-Hit@1"]),
                strong(row["Subsys-MRR"], GREEN) if is_selected else plain(row["Subsys-MRR"]),
                strong(row["Cond-Var-MRR"], GREEN) if is_selected else plain(row["Cond-Var-MRR"]),
                strong(row["Aff-F1"], BLUE if exp_id == "main_wadi_e15" else GREEN) if exp_id in {"main_wadi_e8", "main_swat_e8", "main_wadi_e15"} else warn(row["Aff-F1"]),
                note,
            ]
        )
    return [
        "## 主实验结果",
        "",
        "绿色表示当前建议主线结果，蓝色表示单数据集上的最高值，红色表示需要谨慎解释的下降。",
        "",
        *md_table(
            ["数据集", "方法", "epoch", "变量 MRR", "变量 Hit@1", "子系统 MRR", "条件变量 MRR", "Aff-F1", "解释"],
            rows,
        ),
        "",
        "结论：WADI 的 15 epoch 在变量 MRR/Hit@1 上最高，但 SWaT 的 15 epoch RCA 明显下降。因此论文主线更适合固定 8 epoch 作为统一设置，15 epoch 作为 WADI 稳定性补充。",
    ]


def ablation_section(main: pd.DataFrame) -> list[str]:
    pairs = [
        ("no source-propagation", "source-propagation 分解", "验证源变量/传播变量拆分是否必要"),
        ("no source gate", "source gate", "验证模型是否真的学会选择源变量"),
        ("no source bottleneck", "source bottleneck", "验证源变量压缩监督是否必要"),
        ("no mechanism", "mechanism prior", "验证正常机制/通道依赖是否有帮助"),
    ]
    rows = []
    for dataset in ["WADI", "SWaT"]:
        main_row = main[(main["dataset"].eq(dataset)) & (main["method"].eq("LaGraph-main")) & (main["epochs"].astype(int).eq(8))].iloc[0]
        for method, module_name, purpose in pairs:
            ab = main[(main["dataset"].eq(dataset)) & (main["method"].eq(method))]
            if ab.empty:
                continue
            ab_row = ab.iloc[0]
            delta = num(ab_row["Var-MRR"]) - num(main_row["Var-MRR"])
            if delta is None:
                delta_text = "-"
            elif delta < -0.03:
                delta_text = warn(delta)
            elif delta > 0.03:
                delta_text = strong(delta, ORANGE)
            else:
                delta_text = fmt(delta)
            risk = ""
            if method == "no mechanism" and dataset == "SWaT":
                risk = '<span style="color:#b91c1c"><strong>风险：去掉 mechanism 反而更好</strong></span>'
            elif delta is not None and delta < -0.03:
                risk = '<span style="color:#166534"><strong>支持该模块必要性</strong></span>'
            else:
                risk = "影响较小或需谨慎解释"
            rows.append(
                [
                    dataset,
                    module_name,
                    fmt(main_row["Var-MRR"]),
                    fmt(ab_row["Var-MRR"]),
                    delta_text,
                    fmt(ab_row["Var-Hit@1"]),
                    purpose,
                    risk,
                ]
            )
    return [
        "## 消融实验重点",
        "",
        "变量 MRR 的下降越明显，越能说明对应模块不是装饰项。",
        "",
        *md_table(
            ["数据集", "被验证模块", "主模型 MRR", "去掉后 MRR", "变化", "去掉后 Hit@1", "实验目的", "解释"],
            rows,
        ),
        "",
        "结论：source-propagation、source gate、source bottleneck 在 WADI 上证据很强，在 SWaT 上 source gate 也有正向证据。最大问题是 mechanism prior：它提升 WADI，但 SWaT 上固定机制强度可能过强，因此后续需要 adaptive mechanism gate。",
    ]


def adaptive_gate_section(main: pd.DataFrame) -> list[str]:
    rows = []
    for dataset in ["WADI", "SWaT"]:
        base = main[
            (main["dataset"].eq(dataset))
            & (main["method"].eq("LaGraph-main"))
            & (main["epochs"].astype(int).eq(8))
        ].iloc[0]
        candidate = main[
            (main["dataset"].eq(dataset))
            & (main["method"].eq("adaptive mechanism gate"))
            & (main["epochs"].astype(int).eq(8))
        ]
        if candidate.empty:
            continue
        cand = candidate.iloc[0]
        delta = num(cand["Var-MRR"]) - num(base["Var-MRR"])
        if delta is None:
            delta_text = "-"
            decision = "未获得有效对比"
        elif delta > 0.01:
            delta_text = strong(delta, BLUE)
            decision = '<span style="color:#166534"><strong>有提升，可继续验证</strong></span>'
        elif delta < -0.01:
            delta_text = warn(delta)
            decision = '<span style="color:#b91c1c"><strong>下降，不建议作为主模型</strong></span>'
        else:
            delta_text = fmt(delta)
            decision = '<span style="color:#475569"><strong>基本持平，暂不作为主模型</strong></span>'
        rows.append(
            [
                dataset,
                fmt(base["Var-MRR"]),
                fmt(cand["Var-MRR"]),
                delta_text,
                fmt(cand["Var-Hit@1"]),
                fmt(cand["Subsys-MRR"]),
                fmt(cand["Cond-Var-MRR"]),
                fmt(cand["Aff-F1"]),
                decision,
            ]
        )
    if not rows:
        return []
    return [
        "## Adaptive Mechanism Gate 候选结果",
        "",
        "这个候选结构不是简单提高 mechanism prior 权重，而是让机制图只在“残差异常、机制偏离、事件源证据”同时成立时才增强 RCA 分数。通俗地说，它让模型先判断“这次异常像不像真的破坏了正常变量依赖机制”，再决定是否相信机制图。",
        "",
        *md_table(
            ["数据集", "主模型变量 MRR", "Adaptive 变量 MRR", "变化", "Adaptive Hit@1", "Adaptive 子系统 MRR", "Adaptive 条件变量 MRR", "Adaptive Aff-F1", "判断"],
            rows,
        ),
        "",
        "结论：adaptive mechanism gate 第一版没有带来可见提升。WADI 变量 MRR 从 0.5603 小降到 0.5544，SWaT 与主模型完全持平。因此它可以保留为候选解释机制，但目前不应替代 source-bottleneck-specificity-rca 主模型。",
    ]


def source_aware_corefine_section(main: pd.DataFrame) -> list[str]:
    rows = []
    for dataset in ["WADI", "SWaT"]:
        base = main[
            (main["dataset"].eq(dataset))
            & (main["method"].eq("LaGraph-main"))
            & (main["epochs"].astype(int).eq(8))
        ].iloc[0]
        candidate = main[
            (main["dataset"].eq(dataset))
            & (main["method"].eq("source-aware dual corefine"))
            & (main["epochs"].astype(int).eq(8))
        ]
        if candidate.empty:
            continue
        cand = candidate.iloc[0]
        var_delta = num(cand["Var-MRR"]) - num(base["Var-MRR"])
        subsys_delta = num(cand["Subsys-MRR"]) - num(base["Subsys-MRR"])
        aff_delta = num(cand["Aff-F1"]) - num(base["Aff-F1"])
        time_ratio = num(cand["Fit-min"]) / max(num(base["Fit-min"]), 1e-9)
        if var_delta > 0.01:
            decision = '<span style="color:#166534"><strong>变量级提升，值得继续确认</strong></span>'
        elif subsys_delta > 0.01 and var_delta > -0.04:
            decision = '<span style="color:#1d4ed8"><strong>更偏子系统/事件质量，暂不替代主模型</strong></span>'
        else:
            decision = '<span style="color:#b91c1c"><strong>变量级下降且更慢，不建议作为主模型</strong></span>'
        rows.append(
            [
                dataset,
                fmt(base["Var-MRR"]),
                fmt(cand["Var-MRR"]),
                warn(var_delta) if var_delta < -0.01 else strong(var_delta, BLUE),
                fmt(cand["Subsys-MRR"]),
                strong(subsys_delta, BLUE) if subsys_delta > 0.0 else warn(subsys_delta),
                fmt(cand["Cond-Var-MRR"]),
                fmt(cand["Aff-F1"]),
                strong(aff_delta, BLUE) if aff_delta > 0.0 else warn(aff_delta),
                f"{time_ratio:.2f}x",
                decision,
            ]
        )
    if not rows:
        return []
    return [
        "## Source-Aware Dual Corefine 候选结果",
        "",
        "这个候选结构把 source gate 接入双图 refinement：先用通道图找机制邻居，再用 source gate 强调疑似源变量，最后把 source-aware context 送入第二次时序图更新。通俗地说，它不是只在最后给变量打分，而是让“疑似根因变量”参与中间特征更新。",
        "",
        *md_table(
            [
                "数据集",
                "主模型变量 MRR",
                "Corefine 变量 MRR",
                "变量变化",
                "Corefine 子系统 MRR",
                "子系统变化",
                "Corefine 条件变量 MRR",
                "Corefine Aff-F1",
                "Aff-F1 变化",
                "训练时间倍率",
                "判断",
            ],
            rows,
        ),
        "",
        "结论：source-aware dual corefine 没有证明变量级 RCA 收益。WADI 子系统 MRR 和 Aff-F1 略升，但变量 MRR/Hit@1 下降；SWaT 条件变量 MRR 基本持平、子系统 MRR 极小提升，但变量 MRR 下降且训练更慢。因此它更像“深融合可行但过重”的负例，不应替代当前主模型。下一步若继续做双图融合，应改为轻量 gate/attention，而不是再堆第二次图传播。",
    ]


def source_preserving_fusion_section(main: pd.DataFrame) -> list[str]:
    rows = []
    for dataset in ["WADI", "SWaT"]:
        candidates = main[
            (main["dataset"].eq(dataset))
            & (main["method"].eq("source-preserving fusion"))
        ].copy()
        if candidates.empty:
            continue
        candidates["epochs_int"] = candidates["epochs"].astype(int)
        for _, cand in candidates.sort_values("epochs_int").iterrows():
            epoch = int(cand["epochs_int"])
            base_rows = main[
                (main["dataset"].eq(dataset))
                & (main["method"].eq("LaGraph-main"))
                & (main["epochs"].astype(int).eq(epoch))
            ]
            if base_rows.empty:
                base_rows = main[
                    (main["dataset"].eq(dataset))
                    & (main["method"].eq("LaGraph-main"))
                    & (main["epochs"].astype(int).eq(8))
                ]
            if base_rows.empty:
                continue
            base = base_rows.iloc[0]
            var_delta = num(cand["Var-MRR"]) - num(base["Var-MRR"])
            hit1_delta = num(cand["Var-Hit@1"]) - num(base["Var-Hit@1"])
            cond_delta = num(cand["Cond-Var-MRR"]) - num(base["Cond-Var-MRR"])
            aff_delta = num(cand["Aff-F1"]) - num(base["Aff-F1"])
            time_ratio = num(cand["Fit-min"]) / max(num(base["Fit-min"]), 1e-9)
            if var_delta > 0.01 or hit1_delta > 0.01:
                decision = '<span style="color:#166534"><strong>变量级有提升，优先继续确认</strong></span>'
            elif var_delta > -0.01 and cond_delta >= 0.0:
                decision = '<span style="color:#1d4ed8"><strong>基本持平且更有机制解释，可保留候选</strong></span>'
            else:
                decision = '<span style="color:#b91c1c"><strong>没有证明收益，暂不替代主模型</strong></span>'
            rows.append(
                [
                    dataset,
                    str(epoch),
                    fmt(base["Var-MRR"]),
                    fmt(cand["Var-MRR"]),
                    strong(var_delta, BLUE) if var_delta >= 0.0 else warn(var_delta),
                    fmt(cand["Var-Hit@1"]),
                    strong(hit1_delta, BLUE) if hit1_delta >= 0.0 else warn(hit1_delta),
                    fmt(cand["Cond-Var-MRR"]),
                    strong(cond_delta, BLUE) if cond_delta >= 0.0 else warn(cond_delta),
                    fmt(cand["Aff-F1"]),
                    strong(aff_delta, BLUE) if aff_delta >= 0.0 else warn(aff_delta),
                    f"{time_ratio:.2f}x",
                    decision,
                ]
            )
    if not rows:
        return []
    return [
        "## Source-Preserving Mechanism Fusion 候选结果",
        "",
        "这个候选结构让 source gate 直接调节机制上下文注入：疑似源变量少吸收邻居解释，保留自身残差信号；非源变量继续利用机制邻居完成重构。通俗地说，它避免把真正的根因变量“平滑成正常”，从而尽量保留变量级 RCA 的尖锐度。",
        "",
        *md_table(
            [
                "数据集",
                "Epoch",
                "主模型变量 MRR",
                "候选变量 MRR",
                "MRR 变化",
                "候选 Hit@1",
                "Hit@1 变化",
                "候选条件变量 MRR",
                "条件变量变化",
                "候选 Aff-F1",
                "Aff-F1 变化",
                "训练时间倍率",
                "判断",
            ],
            rows,
        ),
        "",
        "结论：该候选的目标不是增加后处理权重，而是让通道机制图在 decoder 内部以 source-aware 的方式参与重构。若它提升变量级 MRR/Hit@1，则说明通道图确实在深度表征中发挥作用；若只持平或下降，则说明当前 source gate 更适合 RCA 评分层，而不适合强行改动重构路径。",
    ]


def baseline_section(main: pd.DataFrame, baselines: pd.DataFrame) -> list[str]:
    rows = []
    for dataset in ["WADI", "SWaT"]:
        main_row = main[(main["dataset"].eq(dataset)) & (main["method"].eq("LaGraph-main")) & (main["epochs"].astype(int).eq(8))].iloc[0]
        rows.append([dataset, "LaGraph-main", "变量级", strong(main_row["Var-MRR"]), strong(main_row["Var-Hit@1"]), strong(main_row["Var-Hit@3"]), strong(main_row["Var-Hit@5"])])
        for method in ["z-score", "residual-only", "random"]:
            base = baselines[
                (baselines["dataset"].eq(dataset))
                & (baselines["epochs"].astype(int).eq(8))
                & (baselines["scope"].eq("variable"))
                & (baselines["method"].eq(method))
            ]
            if base.empty:
                continue
            row = base.iloc[0]
            rows.append([dataset, method, "变量级", plain(row["MRR"]), plain(row["Hit@1"]), plain(row["Hit@3"]), plain(row["Hit@5"])])
    return [
        "## 与统计 baseline 对比",
        "",
        "该表用于回答审稿人可能提出的质疑：结果是否只是 z-score、残差或随机排序带来的。",
        "",
        *md_table(["数据集", "方法", "粒度", "MRR", "Hit@1", "Hit@3", "Hit@5"], rows),
        "",
        "结论：变量级 RCA 上，LaGraph-main 明显高于 random 和 z-score；SWaT 上 residual-only 较强，说明残差本身已经包含部分根因信息，但主模型仍提供图结构与源变量解释。子系统级上 z-score 有时较强，因此论文中应强调变量级 RCA 是主要贡献。",
    ]


def sweep_section(sweep: pd.DataFrame) -> list[str]:
    variable = sweep[sweep["scope"].eq("variable")].copy()
    exported = variable[variable["method"].eq("exported")].iloc[0]
    rescored = variable[~variable["method"].eq("exported")].copy()
    best_mean = rescored.sort_values(["MRR_mean", "MRR_min", "Hit@1_mean"], ascending=False).iloc[0]
    best_min = rescored.sort_values(["MRR_min", "MRR_mean", "Hit@1_mean"], ascending=False).iloc[0]
    rows = [
        ["当前导出主分数", strong(exported["MRR_mean"]), strong(exported["MRR_min"]), strong(exported["Hit@1_mean"]), "平均变量 MRR 最优，作为当前主模型"],
        ["最佳后验重加权", plain(best_mean["MRR_mean"]), plain(best_mean["MRR_min"]), plain(best_mean["Hit@1_mean"]), "未超过当前导出分数"],
        ["最稳健后验重加权", plain(best_min["MRR_mean"]), strong(best_min["MRR_min"], BLUE), plain(best_min["Hit@1_mean"]), "最差数据集更稳，但平均值和 Hit@1 下降"],
    ]
    return [
        "## RootScore 权重搜索",
        "",
        "该实验不重新训练，只在固定 WADI/SWaT 主模型 RCA JSON 上重组分数组件，避免变成数据集定制。",
        "",
        *md_table(["设置", "平均变量 MRR", "最差数据集 MRR", "平均 Hit@1", "解释"], rows),
        "",
        "结论：继续调固定 RootScore 权重收益不足。更有意义的方向是让 mechanism prior 的强度随事件自适应变化，而不是全数据集共享一个固定权重。",
    ]


def write_report() -> None:
    main = pd.read_csv(MAIN_CSV)
    baselines = pd.read_csv(BASELINE_CSV)
    sweep = pd.read_csv(SWEEP_CSV)
    lines = [
        "# Journal Validation 中文高亮汇报表",
        "",
        f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "说明：本文件是汇报用高亮版，不替代原始 CSV。绿色表示当前支持主结论的数据，蓝色表示某项最高或稳健结果，红色表示风险点或明显下降。",
        "",
        *main_experiment_section(main),
        "",
        *ablation_section(main),
        "",
        *adaptive_gate_section(main),
        "",
        *source_aware_corefine_section(main),
        "",
        *source_preserving_fusion_section(main),
        "",
        *baseline_section(main, baselines),
        "",
        *sweep_section(sweep),
        "",
        "## 当前判断",
        "",
        "当前系统已经能证明 source-oriented RCA 的价值：source-propagation、source gate、source bottleneck 在 WADI 上贡献明确，SWaT 上 no-mechanism 虽然更强，但它在 WADI 明显退化，说明完全删掉机制图不适合作为统一框架。source-preserving mechanism fusion 在 8 epoch 快筛中提升了 SWaT 变量级 RCA，但 15 epoch 确认中没有稳定超过主模型，因此它更适合作为“机制图如何进入 decoder”的候选证据，而不是当前主模型。当前主模型仍固定为 source-bottleneck-specificity-rca；后续优化应优先解决机制图监督不足和长训练 RCA 退化，而不是继续叠加后处理权重。",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    write_report()
