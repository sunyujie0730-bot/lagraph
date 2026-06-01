#!/usr/bin/env python3
"""Evaluate hierarchical RCA reports.

This evaluator separates three questions:

1. Did the model locate the correct subsystem?
2. If the subsystem is known/correct, can it rank the root variable inside it?
3. Does a top-down subsystem-first ranking recover the root variable end to end?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--event-source", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--prediction-key", type=str, default=None)
    parser.add_argument(
        "--group-aggregation",
        choices=["exported", "max", "mean", "topk_mean"],
        default="exported",
        help="How to aggregate channel scores into subsystem scores.",
    )
    parser.add_argument("--group-topk", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_list(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part for part in str(value).split(";") if part]


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


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def load_meta(registry: Path, series_name: str) -> list[dict]:
    df = pd.read_csv(registry)
    rows = df.loc[df["file"] == series_name].copy()
    if rows.empty:
        raise ValueError(f"{series_name} has no RCA labels in {registry}")

    events = []
    for _, row in rows.sort_values("event_id").iterrows():
        root_groups = decode_list(row.get("subsystem_root", ""))
        root_variables = decode_list(row.get("variable_roots", ""))
        if not root_groups and root_variables:
            root_groups = sorted({root_cause_group_name(name) for name in root_variables})
        events.append(
            {
                "event_id": int(row["event_id"]),
                "start": int(row["event_start"]),
                "end": int(row["event_end"]),
                "root_groups": root_groups,
                "root_variables": root_variables,
            }
        )
    return events


def metric_row(ranking: list[str], roots: set[str], k_values: list[int], prefix: str) -> dict:
    first_rank = None
    for idx, name in enumerate(ranking, start=1):
        if name in roots:
            first_rank = idx
            break
    row = {f"{prefix}_MRR": 0.0 if first_rank is None else 1.0 / first_rank}
    for k in k_values:
        topk = ranking[:k]
        hits = sum(1 for name in topk if name in roots)
        row[f"{prefix}_Hit@{k}"] = 1.0 if hits > 0 else 0.0
        row[f"{prefix}_Precision@{k}"] = hits / max(k, 1)
        row[f"{prefix}_PR@{k}"] = row[f"{prefix}_Precision@{k}"]
        row[f"{prefix}_Recall@{k}"] = hits / max(len(roots), 1)
        precision_sum = 0.0
        seen_hits = 0
        dcg = 0.0
        for idx, name in enumerate(topk, start=1):
            if name in roots:
                seen_hits += 1
                precision_sum += seen_hits / idx
                dcg += 1.0 / math.log2(idx + 1)
        max_relevant = min(len(roots), k)
        row[f"{prefix}_MAP@{k}"] = 0.0 if max_relevant == 0 else precision_sum / max_relevant
        ideal_hits = min(len(roots), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
        row[f"{prefix}_NDCG@{k}"] = 0.0 if idcg == 0 else dcg / idcg
    return row


def group_ranking(event: dict, aggregation: str = "exported", topk: int = 3) -> list[str]:
    if aggregation != "exported":
        scores: dict[str, list[float]] = {}
        for item in event.get("channel_ranking", []):
            name = item.get("name", "")
            group = root_cause_group_name(item.get("group") or name)
            scores.setdefault(group, []).append(float(item.get("score", 0.0)))
        group_scores = {}
        for group, values in scores.items():
            values = sorted(values, reverse=True)
            if aggregation == "max":
                group_scores[group] = values[0]
            elif aggregation == "mean":
                group_scores[group] = sum(values) / max(len(values), 1)
            elif aggregation == "topk_mean":
                k = max(1, int(topk or 1))
                group_scores[group] = sum(values[:k]) / min(k, len(values))
        return [name for name, _ in sorted(group_scores.items(), key=lambda value: value[1], reverse=True)]

    names = [item.get("name", "") for item in event.get("group_ranking", [])]
    if names:
        return list(dict.fromkeys(root_cause_group_name(name) for name in names))

    scores: dict[str, float] = {}
    for item in event.get("channel_ranking", []):
        group = root_cause_group_name(item.get("group") or item.get("name", ""))
        score = float(item.get("group_score", item.get("score", 0.0)))
        scores[group] = max(scores.get(group, float("-inf")), score)
    return [name for name, _ in sorted(scores.items(), key=lambda value: value[1], reverse=True)]


def flat_variable_ranking(event: dict) -> list[str]:
    return [item.get("name", "") for item in event.get("channel_ranking", [])]


def conditional_variable_ranking(event: dict, root_groups: set[str]) -> list[str]:
    items = []
    for item in event.get("channel_ranking", []):
        name = item.get("name", "")
        group = root_cause_group_name(item.get("group") or name)
        if group in root_groups:
            items.append(item)
    items = sorted(
        items,
        key=lambda item: (
            int(item.get("within_group_rank", 10**9)),
            -float(item.get("score", 0.0)),
        ),
    )
    return [item.get("name", "") for item in items]


def strict_hierarchical_variable_ranking(
    event: dict,
    aggregation: str = "exported",
    topk: int = 3,
) -> list[str]:
    group_order = group_ranking(event, aggregation, topk)
    group_rank = {name: rank for rank, name in enumerate(group_order, start=1)}
    items = sorted(
        event.get("channel_ranking", []),
        key=lambda item: (
            group_rank.get(root_cause_group_name(item.get("group") or item.get("name", "")), 10**9),
            -float(item.get("score", 0.0)),
        ),
    )
    return [item.get("name", "") for item in items]


def match_pred_event(meta_event: dict, pred_events: list[dict]) -> tuple[dict | None, int]:
    best = None
    best_overlap = 0
    for event in pred_events:
        value = overlap(meta_event["start"], meta_event["end"], int(event["start"]), int(event["end"]))
        if value > best_overlap:
            best = event
            best_overlap = value
    return best, best_overlap


def match_true_event(exported_event: dict, meta_events: list[dict]) -> dict | None:
    best = None
    best_overlap = 0
    for event in meta_events:
        value = overlap(int(exported_event["start"]), int(exported_event["end"]), event["start"], event["end"])
        if value > best_overlap:
            best = event
            best_overlap = value
    return best


def selected_events(rca: dict, args: argparse.Namespace) -> tuple[list[tuple[dict, dict | None, int]], str | None]:
    meta_events = load_meta(args.registry, rca["series_name"])
    if args.event_source == "predicted":
        pred_events_by_key = rca.get("predicted_events_by_key", {})
        prediction_key = args.prediction_key or rca.get("rca_prediction_key")
        if prediction_key is not None and str(prediction_key) in pred_events_by_key:
            pred_events = pred_events_by_key[str(prediction_key)]
        else:
            pred_events = rca.get("predicted_events", [])
            prediction_key = rca.get("rca_prediction_key", prediction_key)
        return [
            (meta_event, *match_pred_event(meta_event, pred_events))
            for meta_event in meta_events
        ], prediction_key

    pairs = []
    for exported_event in rca.get("events", []):
        meta_event = match_true_event(exported_event, meta_events)
        if meta_event is not None:
            pairs.append((meta_event, exported_event, overlap(meta_event["start"], meta_event["end"], int(exported_event["start"]), int(exported_event["end"]))))
    return pairs, None


def main() -> None:
    args = parse_args()
    rca = load_json(args.rca)
    pairs, prediction_key = selected_events(rca, args)
    rows = []
    for meta_event, event, best_overlap in pairs:
        root_groups = set(meta_event.get("root_groups", []))
        root_variables = set(meta_event.get("root_variables", []))
        if not root_groups and not root_variables:
            continue

        row = {
            "series_name": rca.get("series_name"),
            "event_source": args.event_source,
            "prediction_key": prediction_key,
            "event_id": meta_event.get("event_id"),
            "event_start": meta_event.get("start"),
            "event_end": meta_event.get("end"),
            "pred_event_id": "" if not event else event.get("event_id"),
            "pred_start": "" if not event else event.get("start"),
            "pred_end": "" if not event else event.get("end"),
            "overlap": best_overlap,
            "matched": 1.0 if event and best_overlap > 0 else 0.0,
            "root_groups": ",".join(sorted(root_groups)),
            "root_variables": ",".join(sorted(root_variables)),
            "method": args.method_name or "lagraph",
        }

        if event and root_groups:
            groups = group_ranking(event, args.group_aggregation, args.group_topk)
            row.update(metric_row(groups, root_groups, args.k, "subsystem"))
            row["top_subsystem"] = groups[0] if groups else ""
        else:
            row.update(metric_row([], root_groups, args.k, "subsystem"))
            row["top_subsystem"] = ""

        if event and root_variables:
            flat_rank = flat_variable_ranking(event)
            conditional_rank = conditional_variable_ranking(event, root_groups)
            hierarchical_rank = strict_hierarchical_variable_ranking(
                event,
                args.group_aggregation,
                args.group_topk,
            )
            row.update(metric_row(flat_rank, root_variables, args.k, "flat_variable"))
            row.update(metric_row(conditional_rank, root_variables, args.k, "conditional_variable"))
            row.update(metric_row(hierarchical_rank, root_variables, args.k, "hierarchical_variable"))
            row["top_variable"] = flat_rank[0] if flat_rank else ""
            row["top_conditional_variable"] = conditional_rank[0] if conditional_rank else ""
            row["top_hierarchical_variable"] = hierarchical_rank[0] if hierarchical_rank else ""
        elif root_variables:
            row.update(metric_row([], root_variables, args.k, "flat_variable"))
            row.update(metric_row([], root_variables, args.k, "conditional_variable"))
            row.update(metric_row([], root_variables, args.k, "hierarchical_variable"))
            row["top_variable"] = ""
            row["top_conditional_variable"] = ""
            row["top_hierarchical_variable"] = ""

        rows.append(row)

    if not rows:
        raise SystemExit("No evaluable hierarchical RCA events found.")

    df = pd.DataFrame(rows)
    metric_cols = [
        column
        for column in df.columns
        if column == "matched"
        or column.endswith("_MRR")
        or "_Hit@" in column
        or "_Precision@" in column
        or "_PR@" in column
        or "_Recall@" in column
        or "_MAP@" in column
        or "_NDCG@" in column
    ]
    summary = df[metric_cols].mean().to_frame("mean").T
    if not args.summary_only:
        print("\nPer-event hierarchical RCA:")
        print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, summary.assign(series_name="MEAN")], ignore_index=True).to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
