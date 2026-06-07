#!/usr/bin/env python3
"""Diagnose why RCA quality changes across training epochs.

The script is intentionally offline: it reads existing RCA JSON exports and the
RCA label registry, then compares true-event and predicted-event RCA behavior.
It does not retrain the model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"
COMPONENTS = [
    "score",
    "base_score",
    "source_score",
    "onset_score",
    "source_gate_score",
    "mechanism_guided_source_score",
    "mechanism_score",
    "graph_score",
    "propagation_score",
]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_hierarchical_rca import (  # noqa: E402
    conditional_variable_ranking,
    flat_variable_ranking,
    group_ranking,
    interval_match_stats,
    metric_row,
    root_cause_group_name,
    selected_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--done-json",
        type=Path,
        nargs="*",
        default=None,
        help="Done JSON files to analyze. By default, scans main and source-preserving e* files.",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    return parser.parse_args()


def read_done(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_done_path"] = str(path)
    return data


def load_rca(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def method_name(exp_id: str, profile: str) -> str:
    if exp_id.startswith("main_"):
        return "LaGraph-main"
    if "source_preserving_fusion" in exp_id:
        return "source-preserving fusion"
    return profile


def dataset_short(name: str) -> str:
    lower = name.lower()
    if "swat" in lower:
        return "SWaT"
    if "wadi" in lower:
        return "WADI"
    if "hai" in lower:
        return "HAI"
    return Path(name).stem


def event_args(registry: Path, event_source: str, k_values: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        registry=registry,
        event_source=event_source,
        prediction_key=None,
        k=k_values,
        group_aggregation="exported",
        group_topk=3,
    )


def rank_by_component(event: dict, component: str) -> list[str]:
    items = list(event.get("channel_ranking", []))
    items = sorted(
        items,
        key=lambda item: float(item.get(component, item.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    return [str(item.get("name", "")) for item in items]


def component_values(event: dict, component: str) -> list[float]:
    return [float(item.get(component, item.get("score", 0.0)) or 0.0) for item in event.get("channel_ranking", [])]


def normalized_entropy(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    min_value = min(vals)
    if min_value < 0:
        vals = [v - min_value for v in vals]
    total = sum(vals)
    if total <= 1e-12:
        return 0.0
    entropy = 0.0
    for value in vals:
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(vals))


def top_margin(values: Iterable[float]) -> tuple[float, float]:
    vals = sorted([float(v) for v in values], reverse=True)
    if not vals:
        return 0.0, 0.0
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0
    margin = top1 - top2
    rel = margin / max(abs(top1), 1e-12)
    return margin, rel


def first_rank(ranking: list[str], roots: set[str]) -> int:
    for idx, name in enumerate(ranking, start=1):
        if name in roots:
            return idx
    return 0


def max_root_score(event: dict, roots: set[str], component: str) -> float:
    best = None
    for item in event.get("channel_ranking", []):
        if str(item.get("name", "")) in roots:
            value = float(item.get(component, item.get("score", 0.0)) or 0.0)
            best = value if best is None else max(best, value)
    return 0.0 if best is None else best


def analyze_done(done: dict, registry: Path, k_values: list[int]) -> list[dict]:
    rca_path = Path(done["rca_json"])
    rca = load_rca(rca_path)
    exp_id = str(done.get("exp_id", rca_path.stem))
    profile = str(done.get("profile", ""))
    method = method_name(exp_id, profile)
    dataset = dataset_short(str(done.get("dataset", rca.get("series_name", ""))))
    epochs = int(done.get("epochs", 0) or 0)
    rows = []

    for event_source in ["true", "predicted"]:
        args = event_args(registry, event_source, k_values)
        pairs, prediction_key = selected_events(rca, args)
        for meta_event, event, best_overlap in pairs:
            root_variables = set(meta_event.get("root_variables", []))
            root_groups = set(meta_event.get("root_groups", []))
            if not root_variables and not root_groups:
                continue

            row = {
                "exp_id": exp_id,
                "dataset": dataset,
                "method": method,
                "profile": profile,
                "epochs": epochs,
                "event_source": event_source,
                "prediction_key": "" if prediction_key is None else str(prediction_key),
                "event_id": meta_event.get("event_id"),
                "event_start": meta_event.get("start"),
                "event_end": meta_event.get("end"),
                "root_variables": ";".join(sorted(root_variables)),
                "root_groups": ";".join(sorted(root_groups)),
                "rca_json": str(rca_path),
                "label_csv": str(done.get("label_csv", "")),
            }

            if event:
                row.update(interval_match_stats(meta_event, event, best_overlap))
                row.update(
                    {
                        "pred_event_id": event.get("event_id"),
                        "pred_start": event.get("start"),
                        "pred_end": event.get("end"),
                        "overlap": best_overlap,
                    }
                )
                flat_rank = flat_variable_ranking(event)
                cond_rank = conditional_variable_ranking(event, root_groups)
                groups = group_ranking(event, "exported", 3)
            else:
                row.update(interval_match_stats(meta_event, None, 0))
                row.update({"pred_event_id": "", "pred_start": "", "pred_end": "", "overlap": 0})
                flat_rank, cond_rank, groups = [], [], []

            row["top_variable"] = flat_rank[0] if flat_rank else ""
            row["top_group"] = groups[0] if groups else ""
            row["root_variable_rank"] = first_rank(flat_rank, root_variables)
            row["root_group_rank"] = first_rank(groups, root_groups)
            if root_variables:
                row.update(metric_row(flat_rank, root_variables, k_values, "variable"))
                row.update(metric_row(cond_rank, root_variables, k_values, "conditional_variable"))
            if root_groups:
                row.update(metric_row(groups, root_groups, k_values, "subsystem"))

            if event:
                for component in COMPONENTS:
                    values = component_values(event, component)
                    margin, rel_margin = top_margin(values)
                    rank = rank_by_component(event, component)
                    row[f"{component}_entropy"] = normalized_entropy(values)
                    row[f"{component}_top_margin"] = margin
                    row[f"{component}_top_rel_margin"] = rel_margin
                    row[f"{component}_top_variable"] = rank[0] if rank else ""
                    row[f"{component}_root_rank"] = first_rank(rank, root_variables)
                    row[f"{component}_root_score"] = max_root_score(event, root_variables, component)
                    row[f"{component}_top_score"] = max(values) if values else 0.0
                    if root_variables:
                        row.update(metric_row(rank, root_variables, k_values, component))
            rows.append(row)

    del rca
    gc.collect()
    return rows


def summarize(per_event: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "method", "profile", "epochs", "event_source", "prediction_key"]
    preferred = [
        "matched",
        "true_coverage",
        "pred_coverage",
        "event_iou",
        "variable_MRR",
        "variable_Hit@1",
        "variable_Hit@3",
        "variable_Hit@5",
        "conditional_variable_MRR",
        "subsystem_MRR",
        "subsystem_Hit@1",
        "score_entropy",
        "score_top_margin",
        "score_top_rel_margin",
        "source_score_entropy",
        "source_score_top_margin",
        "source_score_top_rel_margin",
        "base_score_MRR",
        "source_score_MRR",
        "onset_score_MRR",
        "source_gate_score_MRR",
        "mechanism_guided_source_score_MRR",
        "graph_score_MRR",
        "propagation_score_MRR",
    ]
    numeric = [col for col in preferred if col in per_event.columns]
    out = per_event.groupby(group_cols, dropna=False)[numeric].mean().reset_index()
    counts = per_event.groupby(group_cols, dropna=False).size().reset_index(name="event_count")
    return counts.merge(out, on=group_cols, how="left")


def epoch_delta(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, source), block in summary.groupby(["dataset", "method", "event_source"]):
        if not {8, 15}.issubset(set(block["epochs"].astype(int))):
            continue
        e8 = block[block["epochs"].astype(int).eq(8)].iloc[0]
        e15 = block[block["epochs"].astype(int).eq(15)].iloc[0]
        row = {
            "dataset": dataset,
            "method": method,
            "event_source": source,
            "Var-MRR@8": e8.get("variable_MRR", 0.0),
            "Var-MRR@15": e15.get("variable_MRR", 0.0),
            "Delta Var-MRR": e15.get("variable_MRR", 0.0) - e8.get("variable_MRR", 0.0),
            "Hit@1@8": e8.get("variable_Hit@1", 0.0),
            "Hit@1@15": e15.get("variable_Hit@1", 0.0),
            "Delta Hit@1": e15.get("variable_Hit@1", 0.0) - e8.get("variable_Hit@1", 0.0),
            "Event IoU@8": e8.get("event_iou", 0.0),
            "Event IoU@15": e15.get("event_iou", 0.0),
            "Delta Event IoU": e15.get("event_iou", 0.0) - e8.get("event_iou", 0.0),
            "Score Entropy@8": e8.get("score_entropy", 0.0),
            "Score Entropy@15": e15.get("score_entropy", 0.0),
            "Delta Score Entropy": e15.get("score_entropy", 0.0) - e8.get("score_entropy", 0.0),
            "Top Margin@8": e8.get("score_top_rel_margin", 0.0),
            "Top Margin@15": e15.get("score_top_rel_margin", 0.0),
            "Delta Top Margin": e15.get("score_top_rel_margin", 0.0) - e8.get("score_top_rel_margin", 0.0),
        }
        for component in [
            "base_score_MRR",
            "source_score_MRR",
            "onset_score_MRR",
            "source_gate_score_MRR",
            "mechanism_guided_source_score_MRR",
            "graph_score_MRR",
        ]:
            if component in block.columns:
                row[f"{component}@8"] = e8.get(component, 0.0)
                row[f"{component}@15"] = e15.get(component, 0.0)
                row[f"Delta {component}"] = e15.get(component, 0.0) - e8.get(component, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_event_delta(per_event: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "method", "event_source", "event_id"]
    for key, block in per_event.groupby(keys):
        if not {8, 15}.issubset(set(block["epochs"].astype(int))):
            continue
        e8 = block[block["epochs"].astype(int).eq(8)].iloc[0]
        e15 = block[block["epochs"].astype(int).eq(15)].iloc[0]
        rows.append(
            {
                "dataset": key[0],
                "method": key[1],
                "event_source": key[2],
                "event_id": key[3],
                "root_variables": e8.get("root_variables", ""),
                "top_variable@8": e8.get("top_variable", ""),
                "top_variable@15": e15.get("top_variable", ""),
                "root_rank@8": e8.get("root_variable_rank", 0),
                "root_rank@15": e15.get("root_variable_rank", 0),
                "MRR@8": e8.get("variable_MRR", 0.0),
                "MRR@15": e15.get("variable_MRR", 0.0),
                "Delta MRR": e15.get("variable_MRR", 0.0) - e8.get("variable_MRR", 0.0),
                "score_entropy@8": e8.get("score_entropy", 0.0),
                "score_entropy@15": e15.get("score_entropy", 0.0),
                "Delta score_entropy": e15.get("score_entropy", 0.0) - e8.get("score_entropy", 0.0),
                "score_top_rel_margin@8": e8.get("score_top_rel_margin", 0.0),
                "score_top_rel_margin@15": e15.get("score_top_rel_margin", 0.0),
                "Delta top_rel_margin": e15.get("score_top_rel_margin", 0.0) - e8.get("score_top_rel_margin", 0.0),
                "event_iou@8": e8.get("event_iou", 0.0),
                "event_iou@15": e15.get("event_iou", 0.0),
                "Delta event_iou": e15.get("event_iou", 0.0) - e8.get("event_iou", 0.0),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def write_report(summary: pd.DataFrame, deltas: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    lines: list[str] = [
        "# RCA Epoch Degradation Diagnosis",
        "",
        "本报告用于回答：为什么 8 epoch 的变量级 RCA 有时优于 15 epoch。分析只读取已有 RCA JSON，不重新训练。",
        "",
        "## 1. 8/15 总体变化",
        "",
    ]

    rows = []
    keep = deltas[deltas["method"].eq("LaGraph-main")].copy()
    for _, row in keep.sort_values(["dataset", "event_source"]).iterrows():
        rows.append(
            [
                row["dataset"],
                row["event_source"],
                fmt(row["Var-MRR@8"]),
                fmt(row["Var-MRR@15"]),
                fmt(row["Delta Var-MRR"]),
                fmt(row["Hit@1@8"]),
                fmt(row["Hit@1@15"]),
                fmt(row["Delta Hit@1"]),
                fmt(row["Event IoU@8"]),
                fmt(row["Event IoU@15"]),
                fmt(row["Delta Event IoU"]),
            ]
        )
    lines.extend(
        md_table(
            ["数据集", "事件源", "MRR@8", "MRR@15", "ΔMRR", "Hit@1@8", "Hit@1@15", "ΔHit@1", "IoU@8", "IoU@15", "ΔIoU"],
            rows,
        )
    )

    lines.extend(["", "## 2. 分数组件变化", ""])
    component_rows = []
    for _, row in keep.sort_values(["dataset", "event_source"]).iterrows():
        component_rows.append(
            [
                row["dataset"],
                row["event_source"],
                fmt(row.get("Delta base_score_MRR", 0.0)),
                fmt(row.get("Delta source_score_MRR", 0.0)),
                fmt(row.get("Delta onset_score_MRR", 0.0)),
                fmt(row.get("Delta source_gate_score_MRR", 0.0)),
                fmt(row.get("Delta mechanism_guided_source_score_MRR", 0.0)),
                fmt(row.get("Delta graph_score_MRR", 0.0)),
                fmt(row.get("Delta Score Entropy", 0.0)),
                fmt(row.get("Delta Top Margin", 0.0)),
            ]
        )
    lines.extend(
        md_table(
            [
                "数据集",
                "事件源",
                "ΔResidual/Base MRR",
                "ΔSource MRR",
                "ΔOnset MRR",
                "ΔSourceGate MRR",
                "ΔMechanismGuided MRR",
                "ΔGraph MRR",
                "ΔEntropy",
                "ΔTopMargin",
            ],
            component_rows,
        )
    )

    lines.extend(["", "## 3. 退化事件样例", ""])
    examples = paired[
        (paired["method"].eq("LaGraph-main"))
        & (paired["event_source"].eq("predicted"))
        & (paired["Delta MRR"] < 0)
    ].sort_values(["dataset", "Delta MRR"]).head(12)
    example_rows = []
    for _, row in examples.iterrows():
        example_rows.append(
            [
                row["dataset"],
                row["event_id"],
                row["root_variables"],
                row["top_variable@8"],
                row["top_variable@15"],
                fmt(row["MRR@8"]),
                fmt(row["MRR@15"]),
                fmt(row["Delta MRR"]),
                fmt(row["Delta score_entropy"]),
                fmt(row["Delta top_rel_margin"]),
            ]
        )
    lines.extend(
        md_table(
            ["数据集", "事件ID", "根因变量", "Top@8", "Top@15", "MRR@8", "MRR@15", "ΔMRR", "ΔEntropy", "ΔTopMargin"],
            example_rows,
        )
    )

    lines.extend(["", "## 4. 诊断结论", ""])
    swat_pred = keep[(keep["dataset"].eq("SWaT")) & (keep["event_source"].eq("predicted"))]
    swat_true = keep[(keep["dataset"].eq("SWaT")) & (keep["event_source"].eq("true"))]
    if not swat_pred.empty and not swat_true.empty:
        pred_delta = float(swat_pred.iloc[0]["Delta Var-MRR"])
        true_delta = float(swat_true.iloc[0]["Delta Var-MRR"])
        iou_delta = float(swat_pred.iloc[0]["Delta Event IoU"])
        source_delta = float(swat_pred.iloc[0].get("Delta source_score_MRR", 0.0))
        base_delta = float(swat_pred.iloc[0].get("Delta base_score_MRR", 0.0))
        lines.append(
            f"- SWaT predicted-event 变量 MRR 变化为 {pred_delta:.4f}，true-event 变化为 {true_delta:.4f}。"
        )
        if true_delta < -0.02:
            lines.append("- true-event 也明显下降，说明退化不是单纯由预测事件边界造成，RCA 分数本身或训练表征发生了退化。")
        elif pred_delta < -0.02 and true_delta >= -0.02:
            lines.append("- 主要是 predicted-event 下降，说明检测事件切分/边界质量是主要混杂因素。")
        else:
            lines.append("- true-event 和 predicted-event 均未出现大幅下降，当前差异更可能来自事件样本波动。")
        if iou_delta < -0.02:
            lines.append(f"- 预测事件 IoU 同时下降 {iou_delta:.4f}，事件边界确实会放大 predicted-event RCA 退化。")
        if source_delta < -0.02 or base_delta < -0.02:
            lines.append(
                f"- source/base 组件 MRR 同步下降（source Δ={source_delta:.4f}, base Δ={base_delta:.4f}），更支持“长训练后根因残差/source 证据被重构目标稀释”的解释。"
            )
    lines.append("- 下一步若要补实验，优先跑 5/10/12 epoch trajectory，而不是继续增加新模块。")
    lines.append("- 如果 trajectory 证实 8-10 epoch 是 RCA 峰值，应考虑 RCA-aware early stopping 或轻量 source sharpness 正则。")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done_json = args.done_json
    if done_json is None:
        done_json = sorted(args.output_dir.glob("main_*_e*.done.json")) + sorted(
            args.output_dir.glob("candidate_source_preserving_fusion_*_e*.done.json")
        )
    rows = []
    for done_path in done_json:
        if not done_path.exists():
            continue
        done = read_done(done_path)
        if done.get("status") != "ok" or not done.get("rca_json"):
            continue
        print(f"Analyzing {done_path.name}")
        rows.extend(analyze_done(done, args.registry, args.k))

    if not rows:
        raise SystemExit("No RCA rows were produced.")

    per_event = pd.DataFrame(rows)
    summary = summarize(per_event)
    deltas = epoch_delta(summary)
    paired = paired_event_delta(per_event)

    per_event_path = args.output_dir / "rca_epoch_degradation_per_event.csv"
    summary_path = args.output_dir / "rca_epoch_degradation_summary.csv"
    delta_path = args.output_dir / "rca_epoch_degradation_delta.csv"
    paired_path = args.output_dir / "rca_epoch_degradation_paired_events.csv"
    report_path = args.output_dir / "rca_epoch_degradation_report.md"

    per_event.to_csv(per_event_path, index=False)
    summary.to_csv(summary_path, index=False)
    deltas.to_csv(delta_path, index=False)
    paired.to_csv(paired_path, index=False)
    write_report(summary, deltas, paired, report_path)

    print(f"Wrote {per_event_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {paired_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
