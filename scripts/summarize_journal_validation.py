#!/usr/bin/env python3
"""Summarize journal-validation RCA experiments.

This script does not train models. It reads the journal-validation outputs and
creates paper-facing tables plus a small set of RCA case studies.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
SUMMARY_CSV = ANALYSIS_DIR / "journal_validation_summary.csv"

CORE_PROFILE = "source-bottleneck-specificity-rca"

METHOD_LABELS = {
    CORE_PROFILE: "LaGraph-main",
    "source-bottleneck-adaptive-mechanism-rca": "adaptive mechanism gate",
    "source-bottleneck-no-mechanism-rca": "no mechanism",
    "source-bottleneck-no-source-gate-rca": "no source gate",
    "source-bottleneck-no-source-bottleneck-rca": "no source bottleneck",
    "source-bottleneck-no-source-propagation-rca": "no source-propagation",
}

DATASET_LABELS = {
    "WADI_A1_2017_ds10.csv": "WADI",
    "SWAT_A1A2_Physical_v1.csv": "SWaT",
}


@dataclass(frozen=True)
class CaseCandidate:
    dataset: str
    exp_id: str
    rca_json: Path
    channel_csv: Path
    hierarchical_csv: Path


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def seconds_to_minutes(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x / 60.0:.1f}"


def dataset_short(name: str) -> str:
    return DATASET_LABELS.get(name, Path(str(name)).stem)


def method_short(profile_or_method: str) -> str:
    text = str(profile_or_method)
    if text in METHOD_LABELS:
        return METHOD_LABELS[text]
    for prefix in [
        "main_wadi_e8_",
        "main_swat_e8_",
        "main_wadi_e15_",
        "main_swat_e15_",
        "ablation_no_mechanism_wadi_e8_",
        "ablation_no_mechanism_swat_e8_",
        "ablation_no_source_gate_wadi_e8_",
        "ablation_no_source_gate_swat_e8_",
        "ablation_no_source_bottleneck_wadi_e8_",
        "ablation_no_source_bottleneck_swat_e8_",
        "ablation_no_source_propagation_wadi_e8_",
        "ablation_no_source_propagation_swat_e8_",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("_channel", "").replace("_group", "")
    return {
        "random": "random",
        "channel_random": "random",
        "group_random": "random",
        "residual_only": "residual-only",
        "channel_residual_only": "residual-only",
        "group_residual_only": "residual-only",
        "zscore": "z-score",
        "correlation_prior": "static correlation",
        "random_graph_prior": "random graph prior",
    }.get(text, text)


def read_mean(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    first_col = df.columns[0]
    mean = df[df[first_col].astype(str).eq("MEAN")]
    if mean.empty:
        return None
    return mean.iloc[-1].to_dict()


def mean_rows_for_pattern(pattern: str, source_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ANALYSIS_DIR.glob(pattern)):
        mean = read_mean(path)
        if mean is None:
            continue
        method = mean.get("method", path.stem)
        if pd.isna(method) or str(method).strip() in {"", "-", "nan", "None"}:
            if "channel_random" in path.name or "group_random" in path.name:
                method = "random"
            elif "channel_residual_only" in path.name or "group_residual_only" in path.name:
                method = "residual_only"
            else:
                method = path.stem
        exp_id = path.name.split("_channel_")[0].split("_group_")[0].split("_hierarchical_")[0]
        rows.append(
            {
                **mean,
                "exp_id": exp_id,
                "source_type": source_type,
                "file": str(path),
                "method": method_short(str(method)),
            }
        )
    return rows


def write_csv_and_md(df: pd.DataFrame, csv_path: Path, md_path: Path, columns: list[str], title: str) -> None:
    out = df.loc[:, [c for c in columns if c in df.columns]].copy()
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = [f"# {title}", "", f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if out.empty:
        lines.append("No rows found.")
    else:
        lines.extend(markdown_table(out))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def build_training_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"Missing summary CSV: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["status"].astype(str).str.startswith("ok")].copy()
    df["dataset_short"] = df["dataset"].map(dataset_short)
    df["method"] = df["profile"].map(method_short)
    df["fit_minutes"] = df["fit_time"].map(seconds_to_minutes)

    main = df[
        [
            "exp_id",
            "dataset_short",
            "method",
            "epochs",
            "channel_MRR",
            "channel_Hit@1",
            "channel_Hit@3",
            "channel_Hit@5",
            "group_MRR",
            "group_Hit@1",
            "group_Hit@3",
            "group_Hit@5",
            "conditional_variable_MRR",
            "conditional_variable_Hit@1",
            "conditional_variable_Hit@3",
            "conditional_variable_Hit@5",
            "best_affiliation_f",
            "best_raw_f1_at_best_aff",
            "fit_minutes",
            "rca_json",
        ]
    ].copy()
    rename = {
        "dataset_short": "dataset",
        "channel_MRR": "Var-MRR",
        "channel_Hit@1": "Var-Hit@1",
        "channel_Hit@3": "Var-Hit@3",
        "channel_Hit@5": "Var-Hit@5",
        "group_MRR": "Subsys-MRR",
        "group_Hit@1": "Subsys-Hit@1",
        "group_Hit@3": "Subsys-Hit@3",
        "group_Hit@5": "Subsys-Hit@5",
        "conditional_variable_MRR": "Cond-Var-MRR",
        "conditional_variable_Hit@1": "Cond-Var-Hit@1",
        "conditional_variable_Hit@3": "Cond-Var-Hit@3",
        "conditional_variable_Hit@5": "Cond-Var-Hit@5",
        "best_affiliation_f": "Aff-F1",
        "best_raw_f1_at_best_aff": "Raw-F1",
        "fit_minutes": "Fit-min",
    }
    main = main.rename(columns=rename)

    for col in [
        "Var-MRR",
        "Var-Hit@1",
        "Var-Hit@3",
        "Var-Hit@5",
        "Subsys-MRR",
        "Subsys-Hit@1",
        "Subsys-Hit@3",
        "Subsys-Hit@5",
        "Cond-Var-MRR",
        "Cond-Var-Hit@1",
        "Cond-Var-Hit@3",
        "Cond-Var-Hit@5",
        "Aff-F1",
        "Raw-F1",
    ]:
        main[col] = main[col].map(fmt)

    detail = main.copy()
    return main, detail


def build_baseline_table() -> pd.DataFrame:
    rows = []
    rows.extend(mean_rows_for_pattern("*_channel_random_pred_key15.csv", "variable"))
    rows.extend(mean_rows_for_pattern("*_channel_residual_only_pred_key15.csv", "variable"))
    rows.extend(mean_rows_for_pattern("*_channel_static_baselines_pred_key15.csv", "variable"))
    rows.extend(mean_rows_for_pattern("*_group_random_pred_key15.csv", "subsystem"))
    rows.extend(mean_rows_for_pattern("*_group_static_baselines_pred_key15.csv", "subsystem"))

    parsed = []
    summary = pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()
    exp_to_dataset = dict(zip(summary.get("exp_id", []), summary.get("dataset", [])))
    exp_to_epoch = dict(zip(summary.get("exp_id", []), summary.get("epochs", [])))
    for row in rows:
        exp_id = row.get("exp_id", "")
        parsed.append(
            {
                "exp_id": exp_id,
                "dataset": dataset_short(str(exp_to_dataset.get(exp_id, ""))),
                "epochs": exp_to_epoch.get(exp_id, ""),
                "scope": row.get("source_type", ""),
                "method": row.get("method", ""),
                "MRR": fmt(row.get("MRR")),
                "Hit@1": fmt(row.get("Hit@1")),
                "Hit@3": fmt(row.get("Hit@3")),
                "Hit@5": fmt(row.get("Hit@5")),
                "PR@1": fmt(row.get("PR@1", row.get("Precision@1"))),
                "PR@3": fmt(row.get("PR@3", row.get("Precision@3"))),
                "PR@5": fmt(row.get("PR@5", row.get("Precision@5"))),
                "MAP@3": fmt(row.get("MAP@3")),
                "MAP@5": fmt(row.get("MAP@5")),
                "NDCG@3": fmt(row.get("NDCG@3")),
                "NDCG@5": fmt(row.get("NDCG@5")),
            }
        )
    return pd.DataFrame(parsed)


def top_channels(event: dict[str, Any], limit: int = 5) -> str:
    items = event.get("top_channels", [])[:limit]
    chunks = []
    for item in items:
        parts = [
            str(item.get("name", "")),
            f"score={fmt(item.get('score'), 3)}",
            f"base={fmt(item.get('base_score'), 3)}",
            f"gate={fmt(item.get('source_gate_score'), 3)}",
            f"onset={fmt(item.get('onset_score'), 3)}",
        ]
        chunks.append(" / ".join(parts))
    return "; ".join(chunks)


def matched_case_rows(candidate: CaseCandidate, max_cases: int = 2) -> list[dict[str, Any]]:
    channel_df = pd.read_csv(candidate.channel_csv)
    hier_df = pd.read_csv(candidate.hierarchical_csv)
    channel_rows = channel_df[channel_df[channel_df.columns[0]].astype(str).ne("MEAN")].copy()
    channel_rows = channel_rows[channel_rows["matched"].astype(float).gt(0)]
    channel_rows = channel_rows.sort_values(["Hit@1", "MRR", "event_iou"], ascending=[False, False, False])
    selected = channel_rows.head(max_cases)
    if selected.empty:
        return []

    with candidate.rca_json.open("r", encoding="utf-8") as f:
        report = json.load(f)
    pred_events = {int(e["event_id"]): e for e in report.get("predicted_events", [])}

    rows = []
    for _, row in selected.iterrows():
        pred_id = int(row.get("pred_event_id", row.get("event_id")))
        event = pred_events.get(pred_id, {})
        hmatch = hier_df[
            (hier_df[hier_df.columns[0]].astype(str).ne("MEAN"))
            & (hier_df["event_id"].astype(float).eq(float(row.get("event_id"))))
            & (hier_df["pred_event_id"].astype(float).eq(float(pred_id)))
        ]
        hrow = hmatch.iloc[0].to_dict() if not hmatch.empty else {}
        rows.append(
            {
                "dataset": candidate.dataset,
                "exp_id": candidate.exp_id,
                "true_event": int(row.get("event_id")),
                "pred_event": pred_id,
                "true_interval": f"{int(row.get('event_start'))}-{int(row.get('event_end'))}",
                "pred_interval": f"{int(row.get('pred_start'))}-{int(row.get('pred_end'))}",
                "overlap": int(row.get("overlap", 0)),
                "roots": row.get("roots", ""),
                "top1": row.get("top1", ""),
                "Var-MRR": fmt(row.get("MRR")),
                "Var-Hit@1": fmt(row.get("Hit@1")),
                "Subsys-top1": hrow.get("top_subsystem", ""),
                "Cond-top1": hrow.get("top_conditional_variable", ""),
                "top evidence": top_channels(event, limit=5),
            }
        )
    return rows


def build_case_studies(summary_df: pd.DataFrame) -> pd.DataFrame:
    cases: list[dict[str, Any]] = []
    main = summary_df[
        (summary_df["profile"].astype(str).eq(CORE_PROFILE))
        & (summary_df["epochs"].astype(int).eq(8))
        & (summary_df["status"].astype(str).str.startswith("ok"))
    ]
    for _, row in main.iterrows():
        exp_id = str(row["exp_id"])
        dataset = dataset_short(str(row["dataset"]))
        rca_json = Path(str(row["rca_json"]))
        channel_csv = ANALYSIS_DIR / f"{exp_id}_channel_pred_key15.csv"
        hier_csv = ANALYSIS_DIR / f"{exp_id}_hierarchical_pred_key15.csv"
        if rca_json.exists() and channel_csv.exists() and hier_csv.exists():
            cases.extend(matched_case_rows(CaseCandidate(dataset, exp_id, rca_json, channel_csv, hier_csv), max_cases=2))
    return pd.DataFrame(cases)


def write_interpretation(main: pd.DataFrame, baselines: pd.DataFrame, cases: pd.DataFrame) -> None:
    path = ANALYSIS_DIR / "journal_validation_interpretation.md"
    sweep_path = ANALYSIS_DIR / "rca_component_weight_sweep_aggregate.csv"
    lines = [
        "# Journal Validation Interpretation",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Fixed Main Model",
        "",
        "`source-bottleneck-specificity-rca` is treated as the current fixed main model. It combines reconstruction residuals, learned channel graph evidence, source gate evidence, source-effect synthetic supervision, and event-specificity correction for variable-level RCA.",
        "",
        "## Current Evidence",
        "",
    ]

    main_rows = main[(main["method"].eq("LaGraph-main")) & (main["epochs"].astype(str).isin(["8", "15"]))]
    if not main_rows.empty:
        lines.extend(markdown_table(main_rows.drop(columns=["rca_json"], errors="ignore")))
        lines.append("")

    lines.extend(
        [
            "## Key Reading",
            "",
            "- WADI: the 15-epoch main model is currently strongest for variable Hit@1/MRR, while 8 epochs is already close and cheaper.",
            "- SWaT: 8 epochs is better than 15 epochs for RCA under the current protocol, suggesting longer training improves reconstruction smoothness but may weaken root-specific ranking.",
            "- `no source-propagation` strongly hurts variable-level RCA on WADI and also lowers SWaT RCA, so source-oriented event decomposition is necessary.",
            "- `no source gate` hurts WADI variable-level and conditional variable-level RCA and is also lower than the main model on SWaT Var-MRR/Hit@1, so source selection has measurable value beyond formatting the final explanation.",
            "- `no mechanism` remains the main risk: it hurts WADI substantially but outperforms the main model on SWaT. This means the mechanism prior is useful for cross-system structure but still needs adaptive strength if we want it to be uniformly score-improving.",
        ]
    )
    if sweep_path.exists():
        try:
            sweep = pd.read_csv(sweep_path)
            variable = sweep[sweep["scope"].astype(str).eq("variable")].copy()
            exported = variable[variable["method"].astype(str).eq("exported")]
            rescored = variable[~variable["method"].astype(str).eq("exported")].copy()
            best_mean = rescored.sort_values(
                ["MRR_mean", "MRR_min", "Hit@1_mean"],
                ascending=False,
            ).head(1)
            best_min = rescored.sort_values(
                ["MRR_min", "MRR_mean", "Hit@1_mean"],
                ascending=False,
            ).head(1)
            if not exported.empty and not best_mean.empty and not best_min.empty:
                lines.extend(
                    [
                        "",
                        "## RootScore Weight Sweep",
                        "",
                        (
                            "- A post-training shared-weight sweep over the fixed WADI/SWaT main RCA exports found that the exported main score remains best by average variable-level MRR "
                            f"({fmt(exported.iloc[0].get('MRR_mean'))})."
                        ),
                        (
                            "- The best component-rescored setting by average variable-level MRR reaches "
                            f"{fmt(best_mean.iloc[0].get('MRR_mean'))}, which is slightly lower than the exported score."
                        ),
                        (
                            "- A more conservative setting can improve worst-dataset variable MRR to "
                            f"{fmt(best_min.iloc[0].get('MRR_min'))}, but it lowers average MRR and Hit@1, so it is not selected as the current main model."
                        ),
                    ]
                )
        except Exception as exc:  # pragma: no cover - reporting fallback
            lines.extend(["", "## RootScore Weight Sweep", "", f"- Sweep summary could not be loaded: {exc}"])
    if not baselines.empty:
        lines.extend(["", "## Baseline Coverage", "", "The detailed baseline table includes random, residual-only, z-score, static correlation, and random graph prior where available. This is enough to answer whether the RCA result comes from a trivial statistical score."])
    if not cases.empty:
        lines.extend(["", "## Case Study Notes", "", "The case-study table records matched predicted events and the top evidence components used by the model. These rows can be converted into 2-3 qualitative figures later."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global ANALYSIS_DIR, SUMMARY_CSV
    ANALYSIS_DIR = args.analysis_dir
    SUMMARY_CSV = ANALYSIS_DIR / "journal_validation_summary.csv"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    main_table, detail_table = build_training_tables()
    baselines = build_baseline_table()
    summary_df = pd.read_csv(SUMMARY_CSV)
    cases = build_case_studies(summary_df)

    write_csv_and_md(
        main_table.drop(columns=["rca_json"], errors="ignore"),
        ANALYSIS_DIR / "journal_validation_main_table.csv",
        ANALYSIS_DIR / "journal_validation_main_table.md",
        [
            "exp_id",
            "dataset",
            "method",
            "epochs",
            "Var-MRR",
            "Var-Hit@1",
            "Var-Hit@3",
            "Var-Hit@5",
            "Subsys-MRR",
            "Subsys-Hit@1",
            "Subsys-Hit@3",
            "Subsys-Hit@5",
            "Cond-Var-MRR",
            "Cond-Var-Hit@1",
            "Cond-Var-Hit@3",
            "Cond-Var-Hit@5",
            "Aff-F1",
            "Raw-F1",
            "Fit-min",
        ],
        "Journal Validation Main Table",
    )
    write_csv_and_md(
        detail_table,
        ANALYSIS_DIR / "journal_validation_detailed_table.csv",
        ANALYSIS_DIR / "journal_validation_detailed_table.md",
        list(detail_table.columns),
        "Journal Validation Detailed Training Table",
    )
    write_csv_and_md(
        baselines,
        ANALYSIS_DIR / "journal_validation_baseline_table.csv",
        ANALYSIS_DIR / "journal_validation_baseline_table.md",
        [
            "exp_id",
            "dataset",
            "epochs",
            "scope",
            "method",
            "MRR",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "PR@1",
            "PR@3",
            "PR@5",
            "MAP@3",
            "MAP@5",
            "NDCG@3",
            "NDCG@5",
        ],
        "Journal Validation Baseline Table",
    )
    write_csv_and_md(
        cases,
        ANALYSIS_DIR / "journal_validation_case_studies.csv",
        ANALYSIS_DIR / "journal_validation_case_studies.md",
        [
            "dataset",
            "exp_id",
            "true_event",
            "pred_event",
            "true_interval",
            "pred_interval",
            "overlap",
            "roots",
            "top1",
            "Var-MRR",
            "Var-Hit@1",
            "Subsys-top1",
            "Cond-top1",
            "top evidence",
        ],
        "Journal Validation RCA Case Studies",
    )
    write_interpretation(main_table, baselines, cases)
    print(f"Wrote summary artifacts under {ANALYSIS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
