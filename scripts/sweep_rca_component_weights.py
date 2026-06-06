#!/usr/bin/env python3
"""Sweep RCA component weights on exported journal-validation reports.

This script does not retrain LaGraph. It reloads the fixed main-model RCA JSON
files and recomputes predicted-event RCA rankings from component scores. The
selection rule is dataset-shared: one weight setting is scored on both WADI and
SWaT, then ranked by the average variable-level MRR and the worst-dataset MRR.
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import evaluate_rca as ev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
SUMMARY_CSV = ANALYSIS_DIR / "journal_validation_summary.csv"
PREDICTION_KEY = "15"
MAIN_EXP_IDS = ["main_wadi_e8", "main_swat_e8"]


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("-")
            else:
                values.append(str(value).replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def dataset_label(dataset: str) -> str:
    if dataset.startswith("WADI"):
        return "WADI"
    if dataset.startswith("SWAT"):
        return "SWaT"
    return Path(dataset).stem


def load_main_reports() -> list[dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"Missing {SUMMARY_CSV}")
    summary = pd.read_csv(SUMMARY_CSV)
    reports = []
    for exp_id in MAIN_EXP_IDS:
        rows = summary.loc[summary["exp_id"].eq(exp_id)]
        if rows.empty:
            raise SystemExit(f"Missing experiment row: {exp_id}")
        row = rows.iloc[-1]
        rca_path = Path(str(row["rca_json"]))
        if not rca_path.exists():
            raise SystemExit(f"Missing RCA JSON for {exp_id}: {rca_path}")
        rca = ev.load_json(rca_path)
        meta = ev.load_meta_from_registry(ev.DEFAULT_LABEL_REGISTRY, rca["series_name"])
        reports.append(
            {
                "exp_id": exp_id,
                "dataset": dataset_label(str(row["dataset"])),
                "rca_path": rca_path,
                "rca": rca,
                "meta": meta,
            }
        )
    return reports


def args_for(scope: str, config: dict[str, float], score_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        scope=scope,
        k=[1, 3, 5],
        score_mode=score_mode,
        component_base_weight=config.get("base", 1.0),
        component_source_score_weight=config.get("source_score", 0.0),
        component_graph_weight=config.get("graph", 0.0),
        component_mechanism_weight=config.get("mechanism", 0.0),
        component_causal_weight=0.0,
        component_source_gate_weight=config.get("source_gate", 0.0),
        component_synthetic_weight=0.0,
        component_counterfactual_weight=0.0,
        component_onset_weight=config.get("onset", 0.0),
        component_mechanism_residual_weight=config.get("mechanism_residual", 0.0),
        component_contrast_weight=0.0,
        component_source_interaction_weight=config.get("source_interaction", 0.0),
        component_graph_penalty_weight=config.get("graph_penalty", 0.0),
        component_normalize=True,
        baseline="none",
        random_seed=2021,
        random_trials=0,
        random_candidates="prediction",
        _meta=None,
    )


def evaluate_report(report: dict[str, Any], scope: str, config: dict[str, float], method: str, score_mode: str) -> dict:
    args = args_for(scope, config, score_mode)
    args._meta = report["meta"]
    root_key = "root_groups" if scope == "group" else "root_variables"
    pred_events_by_key = report["rca"].get("predicted_events_by_key", {})
    pred_events = pred_events_by_key.get(PREDICTION_KEY, report["rca"].get("predicted_events", []))
    rows = []
    for meta_idx, meta_event in enumerate(report["meta"].get("events", []), start=1):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        pred_event, best_overlap = ev.match_pred_event(meta_event, pred_events)
        ranking = ev.ranking_for_event(pred_event, args)
        metric_values, _ = ev.evaluate_ranking(pred_event, roots, ranking, args)
        metric_values.update(ev.interval_match_stats(meta_event, pred_event, best_overlap))
        rows.append(metric_values)
    if not rows:
        raise RuntimeError(f"No evaluable rows for {report['exp_id']} {scope}")
    mean = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    return {
        "exp_id": report["exp_id"],
        "dataset": report["dataset"],
        "scope": "variable" if scope == "channel" else "subsystem",
        "method": method,
        **config,
        "matched": mean.get("matched"),
        "MRR": mean.get("MRR"),
        "Hit@1": mean.get("Hit@1"),
        "Hit@3": mean.get("Hit@3"),
        "Hit@5": mean.get("Hit@5"),
        "PR@1": mean.get("PR@1"),
        "PR@3": mean.get("PR@3"),
        "PR@5": mean.get("PR@5"),
        "MAP@3": mean.get("MAP@3"),
        "MAP@5": mean.get("MAP@5"),
        "NDCG@3": mean.get("NDCG@3"),
        "NDCG@5": mean.get("NDCG@5"),
    }


def config_name(config: dict[str, float]) -> str:
    return (
        f"b{config['base']}_ss{config['source_score']}_g{config['source_gate']}"
        f"_m{config['mechanism']}_mr{config['mechanism_residual']}"
        f"_on{config['onset']}_int{config['source_interaction']}_gp{config['graph_penalty']}"
    )


def build_grid() -> list[dict[str, float]]:
    grid = []
    for source_score, source_gate, mechanism, mechanism_residual, onset, source_interaction, graph_penalty in itertools.product(
        [0.0, 0.25],
        [0.0, 0.25, 0.50, 0.75],
        [0.0, 0.05, 0.10, 0.20],
        [0.0, 0.10, 0.25],
        [0.50, 0.75, 1.00],
        [0.0, 0.50, 1.00, 1.50, 2.00],
        [0.0, 0.05, 0.10],
    ):
        grid.append(
            {
                "base": 1.0,
                "source_score": source_score,
                "source_gate": source_gate,
                "mechanism": mechanism,
                "mechanism_residual": mechanism_residual,
                "onset": onset,
                "source_interaction": source_interaction,
                "graph_penalty": graph_penalty,
            }
        )
    return grid


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "method",
        "base",
        "source_score",
        "source_gate",
        "mechanism",
        "mechanism_residual",
        "onset",
        "source_interaction",
        "graph_penalty",
        "scope",
    ]
    metrics = ["MRR", "Hit@1", "Hit@3", "Hit@5", "PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5", "NDCG@3", "NDCG@5"]
    grouped = rows.groupby(key_cols, dropna=False)[metrics].agg(["mean", "min", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    return grouped.sort_values(["scope", "MRR_mean", "MRR_min", "Hit@1_mean"], ascending=[True, False, False, False])


def write_outputs(rows: pd.DataFrame, agg: pd.DataFrame) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = ANALYSIS_DIR / "rca_component_weight_sweep_rows.csv"
    agg_path = ANALYSIS_DIR / "rca_component_weight_sweep_aggregate.csv"
    md_path = ANALYSIS_DIR / "rca_component_weight_sweep.md"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    display_cols = [
        "scope",
        "method",
        "source_score",
        "source_gate",
        "mechanism",
        "mechanism_residual",
        "onset",
        "source_interaction",
        "graph_penalty",
        "MRR_mean",
        "MRR_min",
        "Hit@1_mean",
        "Hit@3_mean",
        "Hit@5_mean",
        "MAP@3_mean",
        "NDCG@3_mean",
    ]
    top_var = agg.loc[agg["scope"].eq("variable"), display_cols].head(20).copy()
    top_group = agg.loc[agg["scope"].eq("subsystem"), display_cols].head(10).copy()
    for frame in (top_var, top_group):
        for col in frame.columns:
            if col not in {"scope", "method"}:
                frame[col] = frame[col].map(fmt)

    lines = [
        "# RCA Component Weight Sweep",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This is a post-training sweep on fixed main-model RCA exports. One weight setting is shared across WADI and SWaT.",
        "",
        "## Top Variable-Level Settings",
        "",
        *markdown_table(top_var),
        "",
        "## Top Subsystem-Level Settings",
        "",
        *markdown_table(top_group),
        "",
        f"Full rows: `{rows_path}`",
        f"Aggregate rows: `{agg_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(top_var.head(10).to_string(index=False))


def main() -> None:
    reports = load_main_reports()
    rows = []
    for report in reports:
        for scope in ("channel", "group"):
            rows.append(evaluate_report(report, scope, {}, "exported", "exported"))
    for config in build_grid():
        method = config_name(config)
        for report in reports:
            for scope in ("channel", "group"):
                rows.append(evaluate_report(report, scope, config, method, "components"))
    rows_df = pd.DataFrame(rows)
    agg_df = aggregate(rows_df)
    write_outputs(rows_df, agg_df)


if __name__ == "__main__":
    main()
