#!/usr/bin/env python3
"""Evaluate root-cause ranking reports with Hit@K, MRR, and NDCG@K."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "root_cause_meta"
DEFAULT_LABEL_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True, help="Path to *_rca.json exported by LaGraph.")
    parser.add_argument("--meta", type=Path, default=None, help="Root-cause metadata JSON. Defaults by series_name.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_LABEL_REGISTRY,
        help="Unified RCA label registry CSV. Used by default when --meta is not set.",
    )
    parser.add_argument("--scope", choices=["group", "channel"], default="group")
    parser.add_argument(
        "--event-source",
        choices=["true", "predicted"],
        default="true",
        help="Evaluate RCA on true anomaly intervals or model-predicted anomaly intervals.",
    )
    parser.add_argument(
        "--prediction-key",
        type=str,
        default=None,
        help="Prediction key to evaluate when --event-source=predicted, e.g. pot, 0.5, 1.0, 2.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--score-mode",
        choices=["exported", "components"],
        default="exported",
        help="Use exported ranking or recompute ranking from base/graph/mechanism/contrast component scores.",
    )
    parser.add_argument("--component-base-weight", type=float, default=1.0)
    parser.add_argument("--component-graph-weight", type=float, default=0.0)
    parser.add_argument("--component-mechanism-weight", type=float, default=0.0)
    parser.add_argument("--component-causal-weight", type=float, default=0.0)
    parser.add_argument("--component-source-gate-weight", type=float, default=0.0)
    parser.add_argument("--component-synthetic-weight", type=float, default=0.0)
    parser.add_argument("--component-counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--component-onset-weight", type=float, default=0.0)
    parser.add_argument("--component-mechanism-residual-weight", type=float, default=0.0)
    parser.add_argument("--component-contrast-weight", type=float, default=0.0)
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Name written to the output CSV, useful for RCA ablation tables.",
    )
    parser.add_argument("--baseline", choices=["none", "random"], default="none")
    parser.add_argument("--random-seed", type=int, default=2021)
    parser.add_argument(
        "--random-trials",
        type=int,
        default=1000,
        help="Number of shuffled rankings to average for the random baseline.",
    )
    parser.add_argument(
        "--random-candidates",
        choices=["prediction", "meta"],
        default="prediction",
        help="Candidate set used by random baseline. 'prediction' uses the same ranked names exported by LaGraph.",
    )
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_list(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part for part in str(value).split(";") if part]


def load_meta_from_registry(registry_path: Path, series_name: str) -> dict:
    df = pd.read_csv(registry_path)
    rows = df.loc[df["file"] == series_name].copy()
    if rows.empty:
        raise ValueError(f"{series_name} has no RCA labels in {registry_path}")

    events = []
    group_scope = set()
    for _, row in rows.sort_values("event_id").iterrows():
        root_groups = decode_list(row.get("subsystem_root", ""))
        root_variables = decode_list(row.get("variable_roots", ""))
        group_scope.update(root_groups)
        events.append(
            {
                "event_id": int(row["event_id"]),
                "start": int(row["event_start"]),
                "end": int(row["event_end"]),
                "root_groups": root_groups,
                "root_variables": root_variables,
            }
        )

    return {
        "series_name": series_name,
        "task": rows.iloc[0].get("task", ""),
        "time_index": rows.iloc[0].get("time_index", "relative_to_test_segment"),
        "group_scope": sorted(group_scope),
        "events": events,
    }


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def match_meta_event(pred_event: dict, meta_events: list[dict]) -> dict | None:
    best = None
    best_overlap = 0
    for event in meta_events:
        value = overlap(pred_event["start"], pred_event["end"], event["start"], event["end"])
        if value > best_overlap:
            best = event
            best_overlap = value
    return best


def match_pred_event(meta_event: dict, pred_events: list[dict]) -> tuple[dict | None, int]:
    best = None
    best_overlap = 0
    for event in pred_events:
        value = overlap(event["start"], event["end"], meta_event["start"], meta_event["end"])
        if value > best_overlap:
            best = event
            best_overlap = value
    return best, best_overlap


def root_cause_group_name(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        return feature_name
    name = feature_name.strip()
    if re.match(r"^P\d+_", name):
        return name.split("_", 1)[0]
    wadi_match = re.match(r"^([123])(?:[A-Z])?_", name)
    if wadi_match:
        return f"WADI_P{wadi_match.group(1)}"
    match = re.search(r"(\d{3})", name)
    if match:
        return f"P{match.group(1)[0]}"
    return name


def ranking_names(pred_event: dict, scope: str) -> list[str]:
    key = "group_ranking" if scope == "group" else "channel_ranking"
    names = [item["name"] for item in pred_event.get(key, [])]
    if scope == "group":
        return list(dict.fromkeys(root_cause_group_name(name) for name in names))
    return names


def component_ranking_names(
    pred_event: dict,
    scope: str,
    base_weight: float,
    graph_weight: float,
    mechanism_weight: float,
    causal_weight: float,
    source_gate_weight: float,
    synthetic_weight: float,
    counterfactual_weight: float,
    onset_weight: float,
    mechanism_residual_weight: float,
    contrast_weight: float,
) -> list[str]:
    channel_items = pred_event.get("channel_ranking", [])
    if not channel_items:
        return ranking_names(pred_event, scope)

    if scope == "channel":
        scored = []
        for item in channel_items:
            score = (
                base_weight * float(item.get("base_score", item.get("score", 0.0)))
                + graph_weight * float(item.get("graph_score", 0.0))
                + mechanism_weight * float(item.get("mechanism_score", 0.0))
                + causal_weight * float(item.get("causal_score", 0.0))
                + source_gate_weight * float(item.get("source_gate_score", 0.0))
                + synthetic_weight * float(item.get("synthetic_rca_score", 0.0))
                + counterfactual_weight * float(item.get("counterfactual_score", 0.0))
                + onset_weight * float(item.get("onset_score", 0.0))
                + mechanism_residual_weight * float(item.get("mechanism_residual_score", 0.0))
                + contrast_weight * float(item.get("contrast_score", 0.0))
            )
            scored.append((item["name"], score))
        return [name for name, _ in sorted(scored, key=lambda item: item[1], reverse=True)]

    group_scores = {}
    for item in channel_items:
        group = root_cause_group_name(item["name"])
        score = (
            base_weight * float(item.get("base_score", item.get("score", 0.0)))
            + graph_weight * float(item.get("graph_score", 0.0))
            + mechanism_weight * float(item.get("mechanism_score", 0.0))
            + causal_weight * float(item.get("causal_score", 0.0))
            + source_gate_weight * float(item.get("source_gate_score", 0.0))
            + synthetic_weight * float(item.get("synthetic_rca_score", 0.0))
            + counterfactual_weight * float(item.get("counterfactual_score", 0.0))
            + onset_weight * float(item.get("onset_score", 0.0))
            + mechanism_residual_weight * float(item.get("mechanism_residual_score", 0.0))
            + contrast_weight * float(item.get("contrast_score", 0.0))
        )
        group_scores[group] = max(group_scores.get(group, float("-inf")), score)
    return [
        name
        for name, _ in sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
    ]


def meta_candidates(meta: dict, scope: str) -> list[str]:
    if scope == "group":
        return list(meta.get("group_scope", []))
    return sorted(
        {
            variable
            for event in meta.get("events", [])
            for variable in event.get("root_variables", [])
        }
    )


def baseline_ranking(candidates: list[str], roots: set[str], seed: int) -> list[str]:
    names = list(dict.fromkeys(candidates))
    for root in sorted(roots):
        if root not in names:
            names.append(root)
    rng = random.Random(seed)
    rng.shuffle(names)
    return names


def metric_row(ranking: list[str], roots: set[str], k_values: list[int]) -> dict:
    row = {}
    first_rank = None
    for idx, name in enumerate(ranking, start=1):
        if name in roots:
            first_rank = idx
            break
    row["MRR"] = 0.0 if first_rank is None else 1.0 / first_rank

    for k in k_values:
        topk = ranking[:k]
        hits = sum(1 for name in topk if name in roots)
        row[f"Hit@{k}"] = 1.0 if hits > 0 else 0.0
        row[f"Precision@{k}"] = hits / max(k, 1)
        row[f"Recall@{k}"] = hits / max(len(roots), 1)
        dcg = 0.0
        for idx, name in enumerate(topk, start=1):
            if name in roots:
                dcg += 1.0 / math.log2(idx + 1)
        ideal_hits = min(len(roots), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
        row[f"NDCG@{k}"] = 0.0 if idcg == 0 else dcg / idcg
    return row


def mean_metric_row(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def ranking_for_event(pred_event: dict | None, args: argparse.Namespace) -> list[str]:
    if not pred_event:
        return []
    if args.score_mode == "components":
        return component_ranking_names(
            pred_event,
            args.scope,
            args.component_base_weight,
            args.component_graph_weight,
            args.component_mechanism_weight,
            args.component_causal_weight,
            args.component_source_gate_weight,
            args.component_synthetic_weight,
            args.component_counterfactual_weight,
            args.component_onset_weight,
            args.component_mechanism_residual_weight,
            args.component_contrast_weight,
        )
    return ranking_names(pred_event, args.scope)


def evaluate_ranking(
    pred_event: dict | None,
    roots: set[str],
    exported_ranking: list[str],
    args: argparse.Namespace,
) -> tuple[dict, str]:
    if not pred_event:
        return metric_row([], roots, args.k), ""
    if args.baseline == "random":
        if args.random_candidates == "prediction":
            candidates = exported_ranking
        else:
            candidates = meta_candidates(args._meta, args.scope)
        trial_rows = []
        for trial in range(args.random_trials):
            seed = args.random_seed + int(pred_event.get("event_id", 0)) * args.random_trials + trial
            ranking = baseline_ranking(candidates, roots, seed)
            trial_rows.append(metric_row(ranking, roots, args.k))
        return mean_metric_row(trial_rows), f"random_mean_{args.random_trials}"
    metric_values = metric_row(exported_ranking, roots, args.k)
    top1 = exported_ranking[0] if exported_ranking else ""
    return metric_values, top1


def main() -> None:
    args = parse_args()
    rca = load_json(args.rca)
    if args.meta is not None:
        meta = load_json(args.meta)
    elif args.registry is not None and args.registry.exists():
        meta = load_meta_from_registry(args.registry, rca["series_name"])
    else:
        series_stem = Path(rca["series_name"]).stem
        meta_path = DEFAULT_META_DIR / f"{series_stem}.json"
        meta = load_json(meta_path)
    args._meta = meta

    root_key = "root_groups" if args.scope == "group" else "root_variables"
    rows = []
    if args.event_source == "predicted":
        pred_events_by_key = rca.get("predicted_events_by_key", {})
        prediction_key = args.prediction_key or rca.get("rca_prediction_key")
        if prediction_key is not None and str(prediction_key) in pred_events_by_key:
            pred_events = pred_events_by_key[str(prediction_key)]
        else:
            pred_events = rca.get("predicted_events", [])
            prediction_key = rca.get("rca_prediction_key", prediction_key)
        for meta_idx, meta_event in enumerate(meta.get("events", []), start=1):
            roots = set(meta_event.get(root_key, []))
            if not roots:
                continue
            pred_event, best_overlap = match_pred_event(meta_event, pred_events)
            exported_ranking = ranking_for_event(pred_event, args)
            metric_values, top1 = evaluate_ranking(pred_event, roots, exported_ranking, args)
            delay = math.nan
            if pred_event and best_overlap > 0:
                delay = max(0, int(pred_event.get("start", 0)) - int(meta_event.get("start", 0)))
            for k in args.k:
                metric_values[f"RCA_Delay@{k}"] = delay if metric_values.get(f"Hit@{k}", 0.0) > 0 else math.nan
            row = {
                "series_name": rca.get("series_name"),
                "event_source": "predicted",
                "prediction_key": prediction_key,
                "event_id": meta_event.get("event_id", meta_idx),
                "event_start": meta_event.get("start"),
                "event_end": meta_event.get("end"),
                "pred_event_id": "" if not pred_event else pred_event.get("event_id"),
                "pred_start": "" if not pred_event else pred_event.get("start"),
                "pred_end": "" if not pred_event else pred_event.get("end"),
                "overlap": best_overlap,
                "matched": 1.0 if best_overlap > 0 else 0.0,
                "roots": ",".join(sorted(roots)),
                "top1": top1,
                "method": args.method_name or ("random" if args.baseline == "random" else "lagraph"),
            }
            row.update(metric_values)
            rows.append(row)
    else:
        for pred_event in rca.get("events", []):
            meta_event = match_meta_event(pred_event, meta.get("events", []))
            if not meta_event:
                continue
            roots = set(meta_event.get(root_key, []))
            if not roots:
                continue
            exported_ranking = ranking_for_event(pred_event, args)
            metric_values, top1 = evaluate_ranking(pred_event, roots, exported_ranking, args)
            row = {
                "series_name": rca.get("series_name"),
                "event_source": "true",
                "event_id": pred_event.get("event_id"),
                "event_start": pred_event.get("start"),
                "event_end": pred_event.get("end"),
                "roots": ",".join(sorted(roots)),
                "top1": top1,
                "method": args.method_name or ("random" if args.baseline == "random" else "lagraph"),
            }
            row.update(metric_values)
            rows.append(row)

    if not rows:
        raise SystemExit("No evaluable RCA events found. Check metadata roots and event intervals.")

    df = pd.DataFrame(rows)
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@", "RCA_Delay@"))
        or c in {"MRR", "matched"}
    ]
    summary = df[metric_cols].mean().to_frame("mean").T
    print("\nPer-event RCA:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, summary.assign(series_name="MEAN")], ignore_index=True).to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
