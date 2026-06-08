#!/usr/bin/env python3
"""Compare LaGraph RCA against residual-only and static RCA baselines.

This analysis is aimed at a reviewer-facing question: if residual-only is close
to LaGraph on SWaT, which events does it solve and where does LaGraph add value?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"

COMPONENTS = [
    "score",
    "base_score",
    "source_score",
    "source_gate_score",
    "onset_score",
    "mechanism_guided_source_score",
    "mechanism_residual_score",
    "graph_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", default="main_swat_e8")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--top-events", type=int, default=8)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--save-md", type=Path, default=None)
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    return pd.read_csv(path)


def mean_row(df: pd.DataFrame, metric: str) -> float:
    rows = df[df[df.columns[0]].astype(str).eq("MEAN")]
    if rows.empty or metric not in df.columns:
        return float("nan")
    return float(rows.iloc[-1][metric])


def non_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[df.columns[0]].astype(str).ne("MEAN")].copy()


def first_rank(ranking: list[str], roots: set[str]) -> int:
    for idx, name in enumerate(ranking, start=1):
        if name in roots:
            return idx
    return 0


def ranking_by_component(event: dict, component: str) -> list[str]:
    items = sorted(
        event.get("channel_ranking", []),
        key=lambda item: float(item.get(component, item.get("score", 0.0))),
        reverse=True,
    )
    return [str(item.get("name", "")) for item in items]


def event_item(event: dict, name: str) -> dict:
    for item in event.get("channel_ranking", []):
        if str(item.get("name", "")) == name:
            return item
    return {}


def load_rca_events(exp_id: str, summary: pd.DataFrame) -> dict[int, dict]:
    row = summary.loc[summary["exp_id"].eq(exp_id)]
    if row.empty:
        return {}
    rca_path = Path(str(row.iloc[-1].get("rca_json", "")))
    if not rca_path.exists():
        return {}
    with rca_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    key = str(report.get("rca_prediction_key", "15"))
    events = report.get("predicted_events_by_key", {}).get(key, report.get("predicted_events", []))
    return {int(event.get("event_id")): event for event in events if "event_id" in event}


def component_ranks(event: dict, roots: set[str]) -> dict[str, Any]:
    out = {}
    for component in COMPONENTS:
        ranking = ranking_by_component(event, component)
        rank = first_rank(ranking, roots)
        out[f"{component}_root_rank"] = rank
        out[f"{component}_top1"] = ranking[0] if ranking else ""
    return out


def component_snapshot(event: dict, top_name: str, roots: set[str]) -> dict[str, Any]:
    top = event_item(event, top_name)
    root_name = next((name for name in ranking_by_component(event, "score") if name in roots), "")
    root = event_item(event, root_name) if root_name else {}
    out = {"root_found_in_export": root_name}
    for component in COMPONENTS:
        out[f"top1_{component}"] = top.get(component, "")
        out[f"root_{component}"] = root.get(component, "")
    return out


def static_rows(df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = non_mean(df)
    rows = rows[rows["method"].astype(str).eq(method)].copy()
    return rows[["event_id", "MRR", "Hit@1", "Hit@3", "Hit@5", "top1"]].rename(
        columns={
            "MRR": f"{method}_MRR",
            "Hit@1": f"{method}_Hit@1",
            "Hit@3": f"{method}_Hit@3",
            "Hit@5": f"{method}_Hit@5",
            "top1": f"{method}_top1",
        }
    )


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    exp = args.exp_id
    summary = read_csv(args.analysis_dir / "journal_validation_summary.csv")
    main = non_mean(read_csv(args.analysis_dir / f"{exp}_channel_pred_key15.csv"))
    residual = non_mean(read_csv(args.analysis_dir / f"{exp}_channel_residual_only_pred_key15.csv"))
    static = read_csv(args.analysis_dir / f"{exp}_channel_static_baselines_pred_key15.csv")
    hier = non_mean(read_csv(args.analysis_dir / f"{exp}_hierarchical_pred_key15.csv"))
    rca_events = load_rca_events(exp, summary)

    cols = [
        "event_id",
        "event_start",
        "event_end",
        "pred_event_id",
        "pred_start",
        "pred_end",
        "overlap",
        "event_iou",
        "roots",
        "top1",
        "MRR",
        "Hit@1",
        "Hit@3",
        "Hit@5",
    ]
    merged = main[cols].rename(
        columns={
            "top1": "lagraph_top1",
            "MRR": "lagraph_MRR",
            "Hit@1": "lagraph_Hit@1",
            "Hit@3": "lagraph_Hit@3",
            "Hit@5": "lagraph_Hit@5",
        }
    )
    residual_cols = residual[["event_id", "top1", "MRR", "Hit@1", "Hit@3", "Hit@5"]].rename(
        columns={
            "top1": "residual_top1",
            "MRR": "residual_MRR",
            "Hit@1": "residual_Hit@1",
            "Hit@3": "residual_Hit@3",
            "Hit@5": "residual_Hit@5",
        }
    )
    merged = merged.merge(residual_cols, on="event_id", how="left")
    for method in ("zscore", "correlation_prior", "random_graph_prior"):
        merged = merged.merge(static_rows(static, method), on="event_id", how="left")
    hcols = ["event_id", "top_subsystem", "subsystem_MRR", "subsystem_Hit@1", "conditional_variable_MRR"]
    hcols = [col for col in hcols if col in hier.columns]
    merged = merged.merge(hier[hcols], on="event_id", how="left")

    detail_rows = []
    for _, row in merged.iterrows():
        roots = {part for part in str(row.get("roots", "")).split(";") if part}
        pred_event_id = row.get("pred_event_id", "")
        try:
            event = rca_events.get(int(float(pred_event_id)), {})
        except (TypeError, ValueError):
            event = {}
        detail = row.to_dict()
        detail["delta_residual_minus_lagraph_MRR"] = float(row.get("residual_MRR", 0.0)) - float(row.get("lagraph_MRR", 0.0))
        if detail["delta_residual_minus_lagraph_MRR"] > 1e-9:
            detail["winner"] = "residual-only"
        elif detail["delta_residual_minus_lagraph_MRR"] < -1e-9:
            detail["winner"] = "LaGraph"
        else:
            detail["winner"] = "tie"
        detail["boundary_risk"] = (
            "high"
            if float(row.get("event_iou", 0.0)) < 0.1
            else "medium"
            if float(row.get("event_iou", 0.0)) < 0.3
            else "low"
        )
        if event:
            detail.update(component_ranks(event, roots))
            detail.update(component_snapshot(event, str(row.get("lagraph_top1", "")), roots))
        detail_rows.append(detail)
    return pd.DataFrame(detail_rows)


def markdown_table(df: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("-")
            elif isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(df: pd.DataFrame, args: argparse.Namespace) -> None:
    metrics = {
        "events": len(df),
        "lagraph_mrr": df["lagraph_MRR"].mean(),
        "residual_mrr": df["residual_MRR"].mean(),
        "lagraph_hit1": df["lagraph_Hit@1"].mean(),
        "residual_hit1": df["residual_Hit@1"].mean(),
        "residual_better_events": int((df["winner"] == "residual-only").sum()),
        "lagraph_better_events": int((df["winner"] == "LaGraph").sum()),
        "tie_events": int((df["winner"] == "tie").sum()),
        "high_boundary_risk_events": int((df["boundary_risk"] == "high").sum()),
    }
    top_cols = [
        "event_id",
        "roots",
        "boundary_risk",
        "event_iou",
        "lagraph_top1",
        "residual_top1",
        "zscore_top1",
        "correlation_prior_top1",
        "lagraph_MRR",
        "residual_MRR",
        "delta_residual_minus_lagraph_MRR",
        "source_score_root_rank",
        "onset_score_root_rank",
        "mechanism_guided_source_score_root_rank",
        "graph_score_root_rank",
    ]
    top_cols = [col for col in top_cols if col in df.columns]
    residual_wins = df.sort_values("delta_residual_minus_lagraph_MRR", ascending=False).head(args.top_events)
    lagraph_wins = df.sort_values("delta_residual_minus_lagraph_MRR", ascending=True).head(args.top_events)

    lines = [
        f"# Residual-Only vs LaGraph RCA Analysis: {args.exp_id}",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {fmt(value) if isinstance(value, float) else value} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `residual-only` is a strong SWaT baseline when the root variable itself has a large reconstruction residual in the predicted event window.",
            "- LaGraph adds value when source/onset/mechanism-guided evidence promotes the root beyond raw residual magnitude.",
            "- If `graph_score_root_rank` is worse than source/onset ranks, graph propagation is likely spreading evidence toward affected variables rather than the initiating source.",
            "- High boundary-risk events should not be over-interpreted as RCA failures because the predicted window only weakly overlaps the true event.",
            "",
            "## Events Where Residual-Only Beats LaGraph",
            "",
            *markdown_table(residual_wins[top_cols]),
            "",
            "## Events Where LaGraph Beats Residual-Only",
            "",
            *markdown_table(lagraph_wins[top_cols]),
        ]
    )
    if args.save_md:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(args.save_md)


def main() -> None:
    args = parse_args()
    if args.save_csv is None:
        args.save_csv = args.analysis_dir / f"{args.exp_id}_residual_vs_lagraph_analysis.csv"
    if args.save_md is None:
        args.save_md = args.analysis_dir / f"{args.exp_id}_residual_vs_lagraph_analysis.md"
    df = build_rows(args)
    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
    print(args.save_csv)
    write_report(df, args)


if __name__ == "__main__":
    main()
