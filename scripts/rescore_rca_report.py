#!/usr/bin/env python3
"""Rescore exported LaGraph RCA reports with a fixed evidence-fusion protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-weight", type=float, default=0.25)
    parser.add_argument("--mechanism-weight", type=float, default=0.10)
    parser.add_argument("--onset-weight", type=float, default=0.50)
    parser.add_argument("--source-interaction-weight", type=float, default=2.0)
    parser.add_argument("--graph-penalty-weight", type=float, default=0.10)
    parser.add_argument("--hierarchy-boost", type=float, default=0.20)
    return parser.parse_args()


def normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    span = float(arr.max() - arr.min())
    if span <= 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.min())) / span


def group_name(item: dict) -> str:
    value = str(item.get("group") or item.get("name") or "")
    return value


def rescore_event(event: dict, args: argparse.Namespace) -> dict:
    items = [dict(item) for item in event.get("channel_ranking", [])]
    if not items:
        return event

    base = normalize([float(item.get("base_score", item.get("score", 0.0))) for item in items])
    graph = normalize([float(item.get("graph_score", 0.0)) for item in items])
    mechanism = normalize([float(item.get("mechanism_score", 0.0)) for item in items])
    onset = normalize([float(item.get("onset_score", 0.0)) for item in items])
    mechanism_residual = normalize([float(item.get("mechanism_residual_score", 0.0)) for item in items])
    source_evidence = np.maximum(onset, mechanism_residual)

    scores = (
        args.base_weight * base
        + args.mechanism_weight * mechanism
        + args.onset_weight * onset
        + args.source_interaction_weight * base * source_evidence
        - args.graph_penalty_weight * graph
    )

    group_scores: dict[str, float] = {}
    for item, score in zip(items, scores):
        group = group_name(item)
        group_scores[group] = max(group_scores.get(group, float("-inf")), float(score))

    sorted_groups = sorted(group_scores.items(), key=lambda entry: entry[1], reverse=True)
    group_rank = {name: rank + 1 for rank, (name, _) in enumerate(sorted_groups)}
    group_values = np.asarray([value for _, value in sorted_groups], dtype=np.float64)
    if group_values.size and float(group_values.max() - group_values.min()) > 1e-12:
        group_norm = {
            name: float((value - group_values.min()) / (group_values.max() - group_values.min()))
            for name, value in sorted_groups
        }
    else:
        group_norm = {name: 0.0 for name, _ in sorted_groups}

    for item, score in zip(items, scores):
        group = group_name(item)
        item["score"] = float(score)
        item["source_score"] = float(score)
        item["group_score"] = float(group_scores.get(group, 0.0))
        item["group_rank"] = int(group_rank.get(group, 0))
        item["hierarchical_score"] = float(score + args.hierarchy_boost * group_norm.get(group, 0.0))

    items = sorted(items, key=lambda item: float(item.get("hierarchical_score", item.get("score", 0.0))), reverse=True)
    by_group: dict[str, list[dict]] = {}
    for item in items:
        by_group.setdefault(group_name(item), []).append(item)
    for group_items in by_group.values():
        ordered = sorted(group_items, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for rank, item in enumerate(ordered, start=1):
            item["within_group_rank"] = rank

    updated = dict(event)
    updated["channel_ranking"] = items
    updated["group_ranking"] = [
        {"rank": rank + 1, "name": name, "score": float(score)}
        for rank, (name, score) in enumerate(sorted_groups)
    ]
    return updated


def rescore_events(value, args: argparse.Namespace):
    if isinstance(value, list):
        return [rescore_event(event, args) if isinstance(event, dict) else event for event in value]
    if isinstance(value, dict):
        return {key: rescore_events(events, args) for key, events in value.items()}
    return value


def main() -> None:
    args = parse_args()
    with open(args.rca, "r", encoding="utf-8") as f:
        report = json.load(f)

    report["score_method"] = (
        "rescored mechanism-calibrated RCA: normalized base residual + weak mechanism deviation "
        "+ onset evidence + base*source-evidence interaction - graph propagation penalty + soft hierarchy"
    )
    report["rescore_params"] = {
        "base_weight": args.base_weight,
        "mechanism_weight": args.mechanism_weight,
        "onset_weight": args.onset_weight,
        "source_interaction_weight": args.source_interaction_weight,
        "graph_penalty_weight": args.graph_penalty_weight,
        "hierarchy_boost": args.hierarchy_boost,
    }
    if "true_events" in report:
        report["true_events"] = rescore_events(report["true_events"], args)
    if "predicted_events_by_key" in report:
        report["predicted_events_by_key"] = rescore_events(report["predicted_events_by_key"], args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
