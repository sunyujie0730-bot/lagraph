#!/usr/bin/env python3
"""Diagnose whether RCA failures come from event detection or variable ranking.

The standard predicted-event RCA evaluator treats any temporal overlap as a
match. That is appropriate for a lenient evaluation protocol, but it hides an
important diagnostic distinction: a predicted event may overlap the true event
by only a few points, making the downstream ranking unreliable. This script
keeps the standard metrics and adds stricter diagnostic buckets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchical-csv", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--coverage-threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    return parser.parse_args()


def f(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except Exception:
        return default


def rank_from_mrr(value: float) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value) or value <= 0:
        return ""
    rank = round(1.0 / value)
    return str(rank) if math.isfinite(rank) else ""


def classify(row: pd.Series, coverage_threshold: float, iou_threshold: float) -> str:
    if f(row, "matched") <= 0:
        return "A_event_not_detected"
    if f(row, "true_coverage") < coverage_threshold or f(row, "event_iou") < iou_threshold:
        return "B_event_weak_overlap"
    if f(row, "flat_variable_Hit@1") > 0:
        return "C_rank_ok_top1"
    if f(row, "flat_variable_Hit@3") > 0:
        return "D_rank_near_miss_top3"
    if f(row, "subsystem_Hit@3") <= 0:
        return "E_subsystem_wrong"
    if f(row, "conditional_variable_Hit@3") > 0:
        return "F_subsystem_ok_global_rank_wrong"
    return "G_subsystem_ok_variable_missing"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.hierarchical_csv)
    df = df[df.get("series_name", "") != "MEAN"].copy()
    if df.empty:
        raise SystemExit("No per-event rows found.")

    df["diagnosis"] = df.apply(
        lambda row: classify(row, args.coverage_threshold, args.iou_threshold),
        axis=1,
    )
    df["flat_variable_rank"] = df["flat_variable_MRR"].map(rank_from_mrr)
    df["subsystem_rank"] = df["subsystem_MRR"].map(rank_from_mrr)
    df["conditional_variable_rank"] = df["conditional_variable_MRR"].map(rank_from_mrr)

    method = args.method_name or str(df.get("method", pd.Series([""])).iloc[0])
    counts = df["diagnosis"].value_counts().sort_index()
    detection_problem = df["diagnosis"].isin(["A_event_not_detected", "B_event_weak_overlap"]).mean()
    ranking_problem = df["diagnosis"].isin(
        [
            "D_rank_near_miss_top3",
            "E_subsystem_wrong",
            "F_subsystem_ok_global_rank_wrong",
            "G_subsystem_ok_variable_missing",
        ]
    ).mean()
    adequate = df[~df["diagnosis"].isin(["A_event_not_detected", "B_event_weak_overlap"])]

    metric_cols = [
        "matched",
        "true_coverage",
        "event_iou",
        "subsystem_MRR",
        "subsystem_Hit@1",
        "subsystem_Hit@3",
        "flat_variable_MRR",
        "flat_variable_Hit@1",
        "flat_variable_Hit@3",
        "flat_variable_Hit@5",
        "conditional_variable_MRR",
        "conditional_variable_Hit@1",
        "conditional_variable_Hit@3",
    ]
    means = {col: pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean() for col in metric_cols if col in df.columns}
    adequate_means = {
        col: pd.to_numeric(adequate[col], errors="coerce").fillna(0.0).mean()
        for col in metric_cols
        if col in adequate.columns and not adequate.empty
    }

    detail_cols = [
        "event_id",
        "event_start",
        "event_end",
        "pred_start",
        "pred_end",
        "true_coverage",
        "event_iou",
        "root_groups",
        "root_variables",
        "top_subsystem",
        "subsystem_rank",
        "top_variable",
        "flat_variable_rank",
        "top_conditional_variable",
        "conditional_variable_rank",
        "diagnosis",
    ]
    detail_cols = [col for col in detail_cols if col in df.columns]

    lines = [
        f"# RCA Detection-vs-Ranking Diagnosis: {method}",
        "",
        f"- Source CSV: `{args.hierarchical_csv}`",
        f"- Events: {len(df)}",
        f"- Adequate-match threshold: true_coverage >= {args.coverage_threshold:.2f} and event_iou >= {args.iou_threshold:.2f}",
        "",
        "## Main Finding",
        "",
        f"- Detection/localization problem ratio: **{detection_problem:.3f}**",
        f"- Ranking problem ratio after excluding weak event matches: **{ranking_problem:.3f}**",
        f"- Top-1 correct ratio: **{(df['diagnosis'].eq('C_rank_ok_top1')).mean():.3f}**",
        "",
        "## Diagnosis Buckets",
        "",
        "| Bucket | Count | Ratio |",
        "|---|---:|---:|",
    ]
    for bucket, count in counts.items():
        lines.append(f"| {bucket} | {int(count)} | {count / len(df):.3f} |")

    lines.extend(["", "## Mean Metrics: All Events", "", "| Metric | Value |", "|---|---:|"])
    for key, value in means.items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## Mean Metrics: Adequately Matched Events Only", "", "| Metric | Value |", "|---|---:|"])
    for key, value in adequate_means.items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## Event-Level Diagnosis", "", *markdown_table(df, detail_cols)])

    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
    print(args.save_md)


if __name__ == "__main__":
    main()
