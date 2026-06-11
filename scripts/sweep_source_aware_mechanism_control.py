#!/usr/bin/env python3
"""Sweep source-aware mechanism-control RCA rules on fixed exports.

The goal is reviewer-facing: test whether mechanism evidence can be retained
for WADI while reducing mechanism-induced ranking errors on SWaT, without
dataset-specific branches or retraining.
"""

from __future__ import annotations

import itertools
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_rca as ev  # noqa: E402


ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "source_aware_mechanism_control"
SUMMARY_CSV = PROJECT_ROOT / "result" / "analysis" / "journal_validation" / "journal_validation_summary.csv"
PREDICTION_KEY = "15"
MAIN_EXP_IDS = ["proof_main_wadi_e8", "proof_main_swat_e8"]


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            vals.append("-" if pd.isna(value) else str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def dataset_label(dataset: str) -> str:
    if dataset.startswith("WADI"):
        return "WADI"
    if dataset.startswith("SWAT"):
        return "SWaT"
    return Path(dataset).stem


def normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def load_reports() -> list[dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"Missing summary CSV: {SUMMARY_CSV}")
    summary = pd.read_csv(SUMMARY_CSV)
    reports: list[dict[str, Any]] = []
    for exp_id in MAIN_EXP_IDS:
        rows = summary.loc[summary["exp_id"].eq(exp_id)]
        if rows.empty:
            raise SystemExit(f"Missing experiment row: {exp_id}")
        row = rows.iloc[-1]
        rca_path = Path(str(row["rca_json"]))
        if not rca_path.exists():
            raise SystemExit(f"Missing RCA JSON: {rca_path}")
        rca = ev.load_json(rca_path)
        meta = ev.load_meta_from_registry(ev.DEFAULT_LABEL_REGISTRY, rca["series_name"])
        reports.append(
            {
                "exp_id": exp_id,
                "dataset": dataset_label(str(row["dataset"])),
                "dataset_name": str(row["dataset"]),
                "profile": str(row["profile"]),
                "rca_path": rca_path,
                "rca": rca,
                "meta": meta,
                "baseline_channel_mrr": float(row.get("channel_MRR", 0.0)),
                "baseline_group_mrr": float(row.get("group_MRR", 0.0)),
            }
        )
    return reports


def cases_for(report: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    cache_key = f"_cases_{scope}"
    if cache_key in report:
        return report[cache_key]
    root_key = "root_groups" if scope == "group" else "root_variables"
    pred_events_by_key = report["rca"].get("predicted_events_by_key", {})
    pred_events = pred_events_by_key.get(PREDICTION_KEY, report["rca"].get("predicted_events", []))
    cases: list[dict[str, Any]] = []
    for meta_event in report["meta"].get("events", []):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        pred_event, overlap = ev.match_pred_event(meta_event, pred_events)
        cases.append(
            {
                "meta_event": meta_event,
                "roots": roots,
                "pred_event": pred_event,
                "overlap": overlap,
            }
        )
    report[cache_key] = cases
    return cases


def args_for(scope: str) -> SimpleNamespace:
    return SimpleNamespace(
        scope=scope,
        k=[1, 3, 5],
        score_mode="exported",
        baseline="none",
        random_seed=2021,
        random_trials=0,
        random_candidates="prediction",
        _meta=None,
    )


def component_values(items: list[dict[str, Any]]) -> dict[str, list[float]]:
    values = {
        "exported": [float(item.get("score", 0.0)) for item in items],
        "base": [float(item.get("base_score", item.get("score", 0.0))) for item in items],
        "source": [float(item.get("source_score", 0.0)) for item in items],
        "source_gate": [float(item.get("source_gate_score", 0.0)) for item in items],
        "onset": [float(item.get("onset_score", 0.0)) for item in items],
        "mechanism": [float(item.get("mechanism_score", 0.0)) for item in items],
        "mechanism_residual": [float(item.get("mechanism_residual_score", 0.0)) for item in items],
        "mechanism_guided": [float(item.get("mechanism_guided_source_score", 0.0)) for item in items],
        "graph": [float(item.get("graph_score", 0.0)) for item in items],
        "propagation": [float(item.get("propagation_score", 0.0)) for item in items],
    }
    return {key: normalize(val) for key, val in values.items()}


def topk_members(values: list[float], k: int) -> set[int]:
    if k <= 0:
        return set()
    return {
        idx
        for idx, _ in sorted(
            enumerate(values),
            key=lambda pair: pair[1],
            reverse=True,
        )[: min(k, len(values))]
    }


def support_signal(comp: dict[str, list[float]], idx: int, mode: str) -> float:
    source = max(comp["source"][idx], comp["source_gate"][idx])
    onset = comp["onset"][idx]
    base = comp["base"][idx]
    if mode == "source":
        return source
    if mode == "source_onset_max":
        return max(source, onset)
    if mode == "source_onset_mean":
        return 0.5 * (source + onset)
    if mode == "source_onset_product":
        return math.sqrt(max(0.0, source * onset))
    if mode == "base_source_onset":
        return max(source, math.sqrt(max(0.0, base * onset)))
    raise ValueError(f"Unknown support mode: {mode}")


def mechanism_signal(comp: dict[str, list[float]], idx: int, mode: str) -> float:
    if mode == "mechanism":
        return comp["mechanism"][idx]
    if mode == "residual":
        return comp["mechanism_residual"][idx]
    if mode == "max":
        return max(comp["mechanism"][idx], comp["mechanism_residual"][idx])
    if mode == "guided":
        return max(comp["mechanism"][idx], comp["mechanism_residual"][idx], comp["mechanism_guided"][idx])
    raise ValueError(f"Unknown mechanism mode: {mode}")


def source_aware_scores(items: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[str, str, float]]:
    comp = component_values(items)
    source_rank_signal = [
        max(comp["source"][idx], comp["source_gate"][idx], comp["base"][idx] * comp["onset"][idx])
        for idx in range(len(items))
    ]
    mech_rank_signal = [
        max(comp["mechanism"][idx], comp["mechanism_residual"][idx])
        for idx in range(len(items))
    ]
    source_top = topk_members(source_rank_signal, int(config["agreement_topk"]))
    mechanism_top = topk_members(mech_rank_signal, int(config["agreement_topk"]))
    overlap = source_top & mechanism_top
    event_agreement = len(overlap) / max(1, len(source_top | mechanism_top))

    scored: list[tuple[str, str, float]] = []
    for idx, item in enumerate(items):
        support = support_signal(comp, idx, str(config["support_mode"]))
        mechanism = mechanism_signal(comp, idx, str(config["mechanism_mode"]))
        in_overlap = 1.0 if idx in overlap else 0.0
        in_source_top = 1.0 if idx in source_top else 0.0
        if config["agreement_mode"] == "none":
            agreement = 1.0
        elif config["agreement_mode"] == "event":
            agreement = event_agreement
        elif config["agreement_mode"] == "topk":
            agreement = max(float(config["agreement_floor"]), in_overlap)
        elif config["agreement_mode"] == "source_top":
            agreement = max(float(config["agreement_floor"]), in_source_top)
        else:
            raise ValueError(f"Unknown agreement mode: {config['agreement_mode']}")

        gate = support * agreement
        conflict = max(0.0, mechanism - support)
        score = (
            float(config["exported_weight"]) * comp["exported"][idx]
            + float(config["source_weight"]) * source_rank_signal[idx]
            + float(config["mechanism_weight"]) * gate * mechanism
            - float(config["conflict_penalty"]) * conflict
            - float(config["graph_penalty"]) * comp["graph"][idx]
            - float(config["propagation_penalty"]) * comp["propagation"][idx]
        )
        name = str(item.get("name", ""))
        group = str(item.get("group") or ev.root_cause_group_name(name))
        scored.append((name, group, float(score)))
    return scored


def ranking_for_event(pred_event: dict | None, scope: str, config: dict[str, Any]) -> list[str]:
    if not pred_event:
        return []
    if config["method"] == "exported":
        args = args_for(scope)
        return ev.ranking_for_event(pred_event, args)

    items = list(pred_event.get("channel_ranking", []))
    if not items:
        args = args_for(scope)
        return ev.ranking_for_event(pred_event, args)
    scored = source_aware_scores(items, config)
    if scope == "channel":
        return [name for name, _, _ in sorted(scored, key=lambda x: x[2], reverse=True)]

    group_scores: dict[str, float] = {}
    for _, group, score in scored:
        group_scores[group] = max(group_scores.get(group, float("-inf")), score)
    return [name for name, _ in sorted(group_scores.items(), key=lambda x: x[1], reverse=True)]


def evaluate_report(report: dict[str, Any], scope: str, config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    args = args_for(scope)
    for case in cases_for(report, scope):
        pred_event = case["pred_event"]
        ranking = ranking_for_event(pred_event, scope, config)
        metric_values, _ = ev.evaluate_ranking(pred_event, case["roots"], ranking, args)
        metric_values.update(ev.interval_match_stats(case["meta_event"], pred_event, case["overlap"]))
        rows.append(metric_values)
    if not rows:
        raise RuntimeError(f"No evaluable rows for {report['exp_id']} {scope}")
    mean = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    return {
        "exp_id": report["exp_id"],
        "dataset": report["dataset"],
        "dataset_name": report["dataset_name"],
        "scope": "variable" if scope == "channel" else "subsystem",
        "method": config["method"],
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
        "true_coverage": mean.get("true_coverage"),
        "event_iou": mean.get("event_iou"),
        **{k: v for k, v in config.items() if k != "method"},
    }


def config_name(config: dict[str, Any]) -> str:
    if config["method"] == "exported":
        return "exported"
    return (
        f"{config['support_mode']}-{config['agreement_mode']}-{config['mechanism_mode']}"
        f"-ew{config['exported_weight']}-sw{config['source_weight']}"
        f"-mw{config['mechanism_weight']}-cp{config['conflict_penalty']}"
        f"-gp{config['graph_penalty']}-pp{config['propagation_penalty']}"
    )


def build_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{"method": "exported"}]
    for support_mode, agreement_mode, mechanism_mode, mechanism_weight, conflict_penalty, graph_penalty, propagation_penalty in itertools.product(
        ["source", "source_onset_mean", "source_onset_product", "base_source_onset"],
        ["none", "event", "topk", "source_top"],
        ["mechanism", "residual", "max"],
        [0.05, 0.10, 0.20, 0.35],
        [0.0, 0.05, 0.10, 0.20],
        [0.0, 0.05, 0.10],
        [0.0, 0.05, 0.10],
    ):
        configs.append(
            {
                "method": "source_aware_mechanism_control",
                "exported_weight": 1.0,
                "source_weight": 0.0,
                "support_mode": support_mode,
                "agreement_mode": agreement_mode,
                "agreement_topk": 5,
                "agreement_floor": 0.05,
                "mechanism_mode": mechanism_mode,
                "mechanism_weight": mechanism_weight,
                "conflict_penalty": conflict_penalty,
                "graph_penalty": graph_penalty,
                "propagation_penalty": propagation_penalty,
            }
        )
    return configs


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "method",
        "support_mode",
        "agreement_mode",
        "mechanism_mode",
        "exported_weight",
        "source_weight",
        "mechanism_weight",
        "conflict_penalty",
        "graph_penalty",
        "propagation_penalty",
        "scope",
    ]
    for col in key_cols:
        if col not in rows:
            rows[col] = ""
    metrics = ["MRR", "Hit@1", "Hit@3", "Hit@5", "PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5", "NDCG@3", "NDCG@5"]
    grouped = rows.groupby(key_cols, dropna=False)[metrics].agg(["mean", "min", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    return grouped.sort_values(["scope", "MRR_min", "MRR_mean", "Hit@1_mean"], ascending=[True, False, False, False])


def add_decision_columns(rows: pd.DataFrame) -> pd.DataFrame:
    var = rows.loc[rows["scope"].eq("variable")].copy()
    pivot = var.pivot_table(index="method_name", columns="dataset", values=["MRR", "Hit@1", "Hit@3", "Hit@5"], aggfunc="mean")
    flat = pivot.copy()
    flat.columns = [f"{metric}_{dataset}" for metric, dataset in flat.columns]
    flat = flat.reset_index()
    exported = flat.loc[flat["method_name"].eq("exported")]
    if not exported.empty:
        base_wadi = float(exported.iloc[0].get("MRR_WADI", 0.0))
        base_swat = float(exported.iloc[0].get("MRR_SWaT", 0.0))
    else:
        base_wadi = 0.0
        base_swat = 0.0
    flat["delta_WADI_MRR"] = flat.get("MRR_WADI", 0.0) - base_wadi
    flat["delta_SWaT_MRR"] = flat.get("MRR_SWaT", 0.0) - base_swat
    flat["min_MRR"] = flat[[c for c in ["MRR_WADI", "MRR_SWaT"] if c in flat]].min(axis=1)
    flat["mean_MRR"] = flat[[c for c in ["MRR_WADI", "MRR_SWaT"] if c in flat]].mean(axis=1)
    flat["passes_main_gate"] = (flat["delta_SWaT_MRR"] > 0.005) & (flat["delta_WADI_MRR"] >= -0.02)
    return flat.sort_values(["passes_main_gate", "min_MRR", "mean_MRR", "delta_SWaT_MRR"], ascending=[False, False, False, False])


def write_outputs(rows: pd.DataFrame, agg: pd.DataFrame, decision: pd.DataFrame) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = ANALYSIS_DIR / "source_aware_mechanism_control_rows.csv"
    agg_path = ANALYSIS_DIR / "source_aware_mechanism_control_aggregate.csv"
    decision_path = ANALYSIS_DIR / "source_aware_mechanism_control_decision.csv"
    md_path = ANALYSIS_DIR / "source_aware_mechanism_control_summary.md"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    decision.to_csv(decision_path, index=False, encoding="utf-8-sig")

    display_cols = [
        "method_name",
        "MRR_WADI",
        "MRR_SWaT",
        "delta_WADI_MRR",
        "delta_SWaT_MRR",
        "Hit@1_WADI",
        "Hit@1_SWaT",
        "Hit@3_WADI",
        "Hit@3_SWaT",
        "passes_main_gate",
    ]
    top = decision[[c for c in display_cols if c in decision.columns]].head(20).copy()
    for col in top.columns:
        if col not in {"method_name", "passes_main_gate"}:
            top[col] = top[col].map(fmt)

    lines = [
        "# Source-Aware Mechanism Control Sweep",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Fixed protocol: same formula on WADI and SWaT, predicted-event key=15, no retraining.",
        "",
        "Selection gate: SWaT variable MRR improves by >0.005 and WADI variable MRR drops by no more than 0.02.",
        "",
        "## Top Shared Settings",
        "",
        *markdown_table(top),
        "",
        f"Rows: `{rows_path}`",
        f"Aggregate: `{agg_path}`",
        f"Decision table: `{decision_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(top.head(12).to_string(index=False))


def main() -> None:
    reports = load_reports()
    rows: list[dict[str, Any]] = []
    for config in build_configs():
        method_name = config_name(config)
        for report in reports:
            for scope in ("channel", "group"):
                row = evaluate_report(report, scope, config)
                row["method_name"] = method_name
                rows.append(row)
    rows_df = pd.DataFrame(rows)
    agg_df = aggregate(rows_df.copy())
    decision_df = add_decision_columns(rows_df)
    write_outputs(rows_df, agg_df, decision_df)


if __name__ == "__main__":
    main()
