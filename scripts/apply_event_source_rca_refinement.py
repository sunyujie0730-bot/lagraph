#!/usr/bin/env python3
"""Apply unified event refinement and source-oriented RCA reranking.

This is a post-training validation tool. It does not use labels or retrain the
model. It transforms an exported RCA JSON with two dataset-agnostic steps:

1. Event refinement: merge fragmented predicted events, optionally expand
   event boundaries, and aggregate their RCA evidence.
2. Source-oriented reranking: rerank variables from normalized component scores
   so source-like variables are preferred over propagated high-residual effects.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPONENT_KEYS = [
    "base_score",
    "graph_score",
    "mechanism_score",
    "source_score",
    "propagation_score",
    "causal_score",
    "source_gate_score",
    "root_score",
    "source_innovation_score",
    "mechanism_guided_source_score",
    "adaptive_mechanism_gate_score",
    "synthetic_rca_score",
    "counterfactual_score",
    "onset_score",
    "mechanism_residual_score",
    "contrast_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-key", type=str, default="15")
    parser.add_argument("--all-prediction-keys", action="store_true")
    parser.add_argument(
        "--event-mode",
        choices=["rerank-only", "refine"],
        default="refine",
        help="rerank-only keeps predicted event boundaries; refine merges/expands fragmented predicted events.",
    )
    parser.add_argument("--merge-gap", type=int, default=30)
    parser.add_argument("--min-event-len", type=int, default=5)
    parser.add_argument("--expand-left", type=int, default=20)
    parser.add_argument("--expand-right", type=int, default=20)
    parser.add_argument("--max-event-len", type=int, default=0, help="0 disables max-length splitting.")
    parser.add_argument(
        "--max-merge-span",
        type=int,
        default=0,
        help="Do not merge another event if the raw cluster span would exceed this value; 0 disables the cap.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--exported-weight",
        type=float,
        default=1.00,
        help="Weight for the exported RCA score. Keep this high for conservative reranking.",
    )
    parser.add_argument("--base-weight", type=float, default=0.45)
    parser.add_argument("--root-weight", type=float, default=0.30)
    parser.add_argument("--source-gate-weight", type=float, default=0.35)
    parser.add_argument("--onset-weight", type=float, default=0.60)
    parser.add_argument("--source-score-weight", type=float, default=0.10)
    parser.add_argument("--mechanism-weight", type=float, default=0.05)
    parser.add_argument("--mechanism-residual-weight", type=float, default=0.10)
    parser.add_argument("--source-interaction-weight", type=float, default=0.75)
    parser.add_argument("--hierarchy-boost", type=float, default=0.15)
    parser.add_argument("--graph-penalty-weight", type=float, default=0.05)
    parser.add_argument("--propagation-penalty-weight", type=float, default=0.05)
    parser.add_argument("--compact-json", action="store_true")
    return parser.parse_args()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if not math.isfinite(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def root_cause_group_name(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        return str(feature_name)
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


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def sorted_channel_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    items = event.get("channel_ranking") or event.get("top_channels") or []
    return [copy.deepcopy(item) for item in items if isinstance(item, dict) and item.get("name")]


def aggregate_channel_items(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        for item in sorted_channel_items(event):
            by_name[str(item.get("name"))].append(item)

    aggregated = []
    for name, items in by_name.items():
        out = copy.deepcopy(max(items, key=lambda item: as_float(item.get("score"))))
        out["name"] = name
        out["group"] = root_cause_group_name(str(out.get("group") or name))
        out["merged_support"] = len(items)
        for key in ["score", "pre_rerank_score", "hierarchical_score", *COMPONENT_KEYS]:
            vals = [as_float(item.get(key)) for item in items]
            if not vals:
                continue
            out[key] = max(vals) if args.aggregation == "max" else sum(vals) / len(vals)
        aggregated.append(out)
    return aggregated


def source_rerank_items(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not items:
        return []

    component = {
        "base": normalize([as_float(item.get("base_score", item.get("score"))) for item in items]),
        "exported": normalize([as_float(item.get("score")) for item in items]),
        "root": normalize([as_float(item.get("root_score")) for item in items]),
        "source_gate": normalize([as_float(item.get("source_gate_score")) for item in items]),
        "onset": normalize([as_float(item.get("onset_score")) for item in items]),
        "source_score": normalize([as_float(item.get("source_score")) for item in items]),
        "mechanism": normalize([as_float(item.get("mechanism_score")) for item in items]),
        "mechanism_residual": normalize([as_float(item.get("mechanism_residual_score")) for item in items]),
        "graph": normalize([as_float(item.get("graph_score")) for item in items]),
        "propagation": normalize([as_float(item.get("propagation_score")) for item in items]),
    }

    group_pre_scores: dict[str, float] = {}
    scored_items = []
    for idx, item in enumerate(items):
        source_evidence = max(component["onset"][idx], component["mechanism_residual"][idx], component["root"][idx])
        score = (
            args.exported_weight * component["exported"][idx]
            + args.base_weight * component["base"][idx]
            + args.root_weight * component["root"][idx]
            + args.source_gate_weight * component["source_gate"][idx]
            + args.onset_weight * component["onset"][idx]
            + args.source_score_weight * component["source_score"][idx]
            + args.mechanism_weight * component["mechanism"][idx]
            + args.mechanism_residual_weight * component["mechanism_residual"][idx]
            + args.source_interaction_weight * component["base"][idx] * source_evidence
            - args.graph_penalty_weight * component["graph"][idx]
            - args.propagation_penalty_weight * component["propagation"][idx]
        )
        updated = copy.deepcopy(item)
        updated["pre_unified_score"] = as_float(item.get("score"))
        updated["score"] = float(score)
        updated["unified_source_score"] = float(score)
        updated["unified_source_evidence"] = float(source_evidence)
        group = root_cause_group_name(str(updated.get("group") or updated.get("name")))
        updated["group"] = group
        group_pre_scores[group] = max(group_pre_scores.get(group, float("-inf")), float(score))
        scored_items.append(updated)

    group_values = list(group_pre_scores.values())
    group_norm = {}
    if group_values:
        lo, hi = min(group_values), max(group_values)
        span = hi - lo
        for group, value in group_pre_scores.items():
            group_norm[group] = 0.0 if span <= 1e-12 else (value - lo) / span

    for item in scored_items:
        group = str(item.get("group"))
        item["group_score"] = float(group_pre_scores.get(group, 0.0))
        item["hierarchical_score"] = float(as_float(item.get("score")) + args.hierarchy_boost * group_norm.get(group, 0.0))

    ranked = sorted(scored_items, key=lambda item: as_float(item.get("hierarchical_score")), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ranked:
        grouped[str(item.get("group"))].append(item)
    for group_items in grouped.values():
        local = sorted(group_items, key=lambda item: as_float(item.get("score")), reverse=True)
        for rank, item in enumerate(local, start=1):
            item["within_group_rank"] = rank

    group_order = sorted(group_pre_scores.items(), key=lambda kv: kv[1], reverse=True)
    group_rank = {group: rank for rank, (group, _) in enumerate(group_order, start=1)}
    for item in ranked:
        item["group_rank"] = int(group_rank.get(str(item.get("group")), 0))

    return ranked[: max(1, args.top_k)]


def group_ranking_from_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for item in items:
        group = root_cause_group_name(str(item.get("group") or item.get("name")))
        scores[group] = max(scores.get(group, float("-inf")), as_float(item.get("group_score", item.get("score"))))
    return [
        {"rank": rank, "name": name, "score": float(score)}
        for rank, (name, score) in enumerate(sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1)
    ]


def build_refined_event(events: list[dict[str, Any]], event_id: int, args: argparse.Namespace) -> dict[str, Any]:
    start = max(0, min(int(event.get("start", 0)) for event in events) - max(0, args.expand_left))
    end = max(int(event.get("end", 0)) for event in events) + max(0, args.expand_right)
    items = aggregate_channel_items(events, args)
    ranked = source_rerank_items(items, args)
    refined = copy.deepcopy(events[0])
    refined.update(
        {
            "event_id": event_id,
            "start": int(start),
            "end": int(end),
            "length": int(max(0, end - start)),
            "score_start": int(start),
            "score_end": int(end),
            "score_length": int(max(0, end - start)),
            "refined_event": True,
            "merged_event_count": len(events),
            "merged_event_ids": [event.get("event_id") for event in events],
            "channel_ranking": ranked,
            "top_channels": ranked,
            "group_ranking": group_ranking_from_items(ranked),
        }
    )
    return refined


def split_long_event(event: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    max_len = int(args.max_event_len or 0)
    if max_len <= 0 or int(event.get("length", 0)) <= max_len:
        return [event]
    out = []
    start = int(event["start"])
    end = int(event["end"])
    cursor = start
    while cursor < end:
        part = copy.deepcopy(event)
        part["start"] = cursor
        part["end"] = min(cursor + max_len, end)
        part["length"] = int(part["end"] - part["start"])
        part["score_start"] = part["start"]
        part["score_end"] = part["end"]
        part["score_length"] = part["length"]
        out.append(part)
        cursor += max_len
    return out


def refine_event_list(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    valid = [
        copy.deepcopy(event)
        for event in events
        if isinstance(event, dict) and int(event.get("end", 0)) > int(event.get("start", 0))
    ]
    valid.sort(key=lambda event: (int(event.get("start", 0)), int(event.get("end", 0))))
    if not valid:
        return []

    clusters: list[list[dict[str, Any]]] = []
    current = [valid[0]]
    current_start = int(valid[0].get("start", 0))
    current_end = int(valid[0].get("end", 0))
    for event in valid[1:]:
        start = int(event.get("start", 0))
        end = int(event.get("end", 0))
        candidate_span = max(current_end, end) - min(current_start, start)
        span_ok = args.max_merge_span <= 0 or candidate_span <= args.max_merge_span
        if start <= current_end + max(0, args.merge_gap) and span_ok:
            current.append(event)
            current_start = min(current_start, start)
            current_end = max(current_end, end)
        else:
            clusters.append(current)
            current = [event]
            current_start = start
            current_end = end
    clusters.append(current)

    refined = []
    for cluster in clusters:
        raw_start = min(int(event.get("start", 0)) for event in cluster)
        raw_end = max(int(event.get("end", 0)) for event in cluster)
        if raw_end - raw_start < max(1, args.min_event_len):
            continue
        refined.extend(split_long_event(build_refined_event(cluster, len(refined) + 1, args), args))
    for event_id, event in enumerate(refined, start=1):
        event["event_id"] = event_id
    return refined


def rerank_predicted_events(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for event_id, event in enumerate(events, start=1):
        updated = copy.deepcopy(event)
        ranked = source_rerank_items(sorted_channel_items(updated), args)
        updated["event_id"] = int(updated.get("event_id", event_id))
        updated["channel_ranking"] = ranked
        updated["top_channels"] = ranked
        updated["group_ranking"] = group_ranking_from_items(ranked)
        updated["refined_event"] = False
        updated["source_rerank_applied"] = True
        out.append(updated)
    return out


def rerank_true_events(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for event in events:
        updated = copy.deepcopy(event)
        ranked = source_rerank_items(sorted_channel_items(updated), args)
        updated["channel_ranking"] = ranked
        updated["top_channels"] = ranked
        updated["group_ranking"] = group_ranking_from_items(ranked)
        out.append(updated)
    return out


def main() -> None:
    args = parse_args()
    report = json.loads(args.rca.read_text(encoding="utf-8"))
    report["score_method"] = "event-refined source-oriented hierarchical RCA"
    report["event_source_refinement"] = {
        "merge_gap": args.merge_gap,
        "min_event_len": args.min_event_len,
        "expand_left": args.expand_left,
        "expand_right": args.expand_right,
        "max_event_len": args.max_event_len,
        "max_merge_span": args.max_merge_span,
        "event_mode": args.event_mode,
        "top_k": args.top_k,
        "aggregation": args.aggregation,
    }
    report["source_rerank_params"] = {
        "base_weight": args.base_weight,
        "exported_weight": args.exported_weight,
        "root_weight": args.root_weight,
        "source_gate_weight": args.source_gate_weight,
        "onset_weight": args.onset_weight,
        "source_score_weight": args.source_score_weight,
        "mechanism_weight": args.mechanism_weight,
        "mechanism_residual_weight": args.mechanism_residual_weight,
        "source_interaction_weight": args.source_interaction_weight,
        "hierarchy_boost": args.hierarchy_boost,
        "graph_penalty_weight": args.graph_penalty_weight,
        "propagation_penalty_weight": args.propagation_penalty_weight,
    }

    if "events" in report:
        report["events"] = rerank_true_events(report.get("events", []), args)
    if "true_events" in report:
        report["true_events"] = rerank_true_events(report.get("true_events", []), args)

    pred_by_key = report.get("predicted_events_by_key")
    if isinstance(pred_by_key, dict):
        keys = list(pred_by_key.keys()) if args.all_prediction_keys else [str(args.prediction_key)]
        for key in keys:
            if key in pred_by_key:
                if args.event_mode == "rerank-only":
                    pred_by_key[key] = rerank_predicted_events(pred_by_key.get(key, []), args)
                else:
                    pred_by_key[key] = refine_event_list(pred_by_key.get(key, []), args)
        if str(args.prediction_key) in pred_by_key:
            report["predicted_events"] = pred_by_key[str(args.prediction_key)]
    elif "predicted_events" in report:
        if args.event_mode == "rerank-only":
            report["predicted_events"] = rerank_predicted_events(report.get("predicted_events", []), args)
        else:
            report["predicted_events"] = refine_event_list(report.get("predicted_events", []), args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.compact_json:
            json.dump(report, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
