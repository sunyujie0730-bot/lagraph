"""Lightweight post-hoc RCA rerank check on a few defensible weight recipes."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_rca as ev  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ev.DEFAULT_LABEL_REGISTRY)
    parser.add_argument("--prediction-key", type=str, default="15")
    parser.add_argument("--scope", choices=["channel", "group", "both"], default="channel")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip TemporalHM in weight searches. This keeps large SWAT RCA checks fast.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def score_items(items: list[dict], recipe: dict[str, float], normalize_components: bool = True) -> list[float]:
    raw = {
        "original": [1.0 / max(int(item.get("topk_rerank_original_rank", item.get("raw_rank", item.get("rank", idx + 1))) or idx + 1), 1) for idx, item in enumerate(items)],
        "score": [float(item.get("score", 0.0) or 0.0) for item in items],
        "pre": [float(item.get("pre_rerank_score", item.get("raw_score", 0.0)) or 0.0) for item in items],
        "raw": [float(item.get("raw_score", item.get("pre_rerank_score", 0.0)) or 0.0) for item in items],
        "base": [float(item.get("base_score", item.get("score", 0.0)) or 0.0) for item in items],
        "source_gate": [float(item.get("source_gate_score", 0.0) or 0.0) for item in items],
        "onset": [float(item.get("onset_score", 0.0) or 0.0) for item in items],
        "mech_resid": [float(item.get("mechanism_residual_score", 0.0) or 0.0) for item in items],
    }
    raw["source_interaction"] = [
        raw["base"][idx] * max(raw["source_gate"][idx], raw["onset"][idx], raw["mech_resid"][idx])
        for idx in range(len(items))
    ]
    values = {key: normalize(series) for key, series in raw.items()} if normalize_components else raw
    scores = [0.0 for _ in items]
    for key, weight in recipe.items():
        series = values[key]
        for idx, value in enumerate(series):
            scores[idx] += weight * value
    return scores


def rerank_event(pred_event: dict | None, recipe: dict[str, float], top_m: int) -> dict | None:
    if pred_event is None:
        return None
    items = pred_event.get("channel_ranking", []) or []
    if not items:
        return pred_event
    scores = score_items(items, recipe)
    top_m = min(max(top_m, 1), len(items))
    head = list(range(top_m))
    tail = list(range(top_m, len(items)))
    order = sorted(head, key=lambda idx: (scores[idx], -idx), reverse=True) + tail
    adjusted = copy.deepcopy(pred_event)
    adjusted_items = []
    for rank, idx in enumerate(order, start=1):
        item = copy.deepcopy(items[idx])
        item["rank"] = rank
        item["score"] = float(scores[idx])
        item["light_rerank_score"] = float(scores[idx])
        adjusted_items.append(item)
    adjusted["channel_ranking"] = adjusted_items
    return adjusted


def recipes() -> list[tuple[str, dict[str, float], int]]:
    return [
        ("exported", {"score": 1.0}, 1),
        ("old_style_original_base_onset", {"original": 1.0, "base": 0.45, "onset": 0.20}, 20),
        ("old_plus_light_source", {"original": 1.0, "base": 0.45, "onset": 0.20, "source_gate": 0.20}, 20),
        ("balanced_original_source", {"original": 0.75, "base": 0.45, "onset": 0.20, "source_gate": 0.35}, 20),
        ("new_style_evidence", {"base": 0.45, "onset": 0.25, "source_gate": 1.0, "mech_resid": 0.25, "source_interaction": 0.25}, 20),
        ("source_guarded_original", {"original": 1.0, "score": 0.25, "source_gate": 0.25}, 20),
    ]


def evaluate(rca: dict, meta: dict, pred_events: list[dict], scope: str, args: argparse.Namespace, name: str, recipe: dict[str, float], top_m: int) -> dict:
    adjusted_by_id = {id(event): rerank_event(event, recipe, top_m) for event in pred_events}
    adjusted_events = list(adjusted_by_id.values())
    eval_args = argparse.Namespace(score_mode="exported", scope=scope)
    candidate_count = ev.expected_candidate_count(meta, rca, scope)
    root_key = "root_groups" if scope == "group" else "root_variables"
    rows = []
    for idx, meta_event in enumerate(meta.get("events", []), start=1):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        pred_event, best_overlap = ev.match_pred_event(meta_event, pred_events)
        adjusted = adjusted_by_id.get(id(pred_event)) if pred_event is not None else None
        ranking = ev.ranking_for_event(adjusted, eval_args)
        row = ev.metric_row(ranking, roots, args.k)
        row.update(ev.cw_rcs_row(adjusted, roots, args.k, scope, expected_candidates=candidate_count))
        if not args.skip_temporal:
            row.update(ev.temporal_hm_row(meta_event, adjusted_events, roots, args.k, scope, eval_args=eval_args))
        row.update(ev.interval_match_stats(meta_event, adjusted, best_overlap))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = df.mean(numeric_only=True).to_dict()
    out["scope"] = scope
    out["recipe"] = name
    out["top_m"] = top_m
    return out


def main() -> None:
    args = parse_args()
    rca = ev.load_json(args.rca)
    meta = ev.load_meta_from_registry(args.registry, rca["series_name"])
    pred_events = rca.get("predicted_events_by_key", {}).get(str(args.prediction_key), rca.get("predicted_events", []))
    scopes = ["channel", "group"] if args.scope == "both" else [args.scope]
    rows = []
    for scope in scopes:
        for name, recipe, top_m in recipes():
            rows.append(evaluate(rca, meta, pred_events, scope, args, name, recipe, top_m))
    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["scope", "MRR", "Hit@1", "Hit@3", "TopKRecall@3"] if col in df.columns]
    df = df.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
    show_cols = [
        "scope",
        "recipe",
        "MRR",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "TopKRecall@3",
        "CW-RCS@3",
        "TemporalHMRecall@3",
    ]
    show_cols = [col for col in show_cols if col in df.columns]
    print(df[show_cols].to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
