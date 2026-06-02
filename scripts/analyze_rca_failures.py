#!/usr/bin/env python3
"""Summarize RCA failure modes from hierarchical RCA evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchical-csv", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    parser.add_argument("--method-name", type=str, default=None)
    return parser.parse_args()


def as_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except Exception:
        return default


def classify(row: pd.Series) -> str:
    matched = as_float(row, "matched")
    if matched <= 0:
        return "event_not_matched"

    subsystem_h1 = as_float(row, "subsystem_Hit@1")
    subsystem_h3 = as_float(row, "subsystem_Hit@3")
    flat_h1 = as_float(row, "flat_variable_Hit@1")
    flat_h3 = as_float(row, "flat_variable_Hit@3")
    conditional_h1 = as_float(row, "conditional_variable_Hit@1")
    conditional_h3 = as_float(row, "conditional_variable_Hit@3")

    if flat_h1 > 0:
        return "ok_top1"
    if flat_h3 > 0:
        return "variable_near_miss_top3"
    if subsystem_h1 <= 0 and subsystem_h3 <= 0:
        return "subsystem_wrong"
    if conditional_h1 > 0 or conditional_h3 > 0:
        return "subsystem_ok_flat_ranking_wrong"
    return "subsystem_ok_variable_missing"


def metric_mean(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out = {}
    for col in columns:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean()
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.hierarchical_csv)
    df = df[df.get("series_name", "") != "MEAN"].copy()
    if df.empty:
        raise SystemExit("No per-event rows found.")

    df["failure_mode"] = df.apply(classify, axis=1)
    counts = df["failure_mode"].value_counts().sort_index()
    metrics = metric_mean(
        df,
        [
            "matched",
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
            "hierarchical_variable_MRR",
            "hierarchical_variable_Hit@1",
            "hierarchical_variable_Hit@3",
        ],
    )

    method = args.method_name or str(df.get("method", pd.Series([""])).iloc[0])
    lines = [
        f"# RCA Failure Analysis: {method}",
        "",
        f"- Source CSV: `{args.hierarchical_csv}`",
        f"- Events: {len(df)}",
        "",
        "## Mean Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## Failure Modes", "", "| Mode | Count | Ratio |", "|---|---:|---:|"])
    for mode, count in counts.items():
        lines.append(f"| {mode} | {int(count)} | {count / len(df):.3f} |")

    detail_cols = [
        "event_id",
        "matched",
        "root_groups",
        "root_variables",
        "top_subsystem",
        "top_variable",
        "top_conditional_variable",
        "top_hierarchical_variable",
        "failure_mode",
    ]
    detail_cols = [col for col in detail_cols if col in df.columns]
    lines.extend(["", "## Event Details", "", "| " + " | ".join(detail_cols) + " |"])
    lines.append("|" + "|".join(["---"] * len(detail_cols)) + "|")
    for _, row in df[detail_cols].iterrows():
        values = []
        for col in detail_cols:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")

    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.save_md)


if __name__ == "__main__":
    main()
