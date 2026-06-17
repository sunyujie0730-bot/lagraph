"""Search event-level adaptive RCA reranking rules on exported RCA reports.

The script is intentionally post-hoc: it tests whether an unsupervised
event-reliability gate can choose between source-evidence fill and base-heavy
ranking before the rule is moved into the LaGraph exporter.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_rca as ev  # noqa: E402


SOURCE_RECIPE = {
    "base": 0.45,
    "onset": 0.25,
    "source_gate": 1.0,
    "mech_resid": 0.25,
    "source_interaction": 0.25,
}
OLD_RECIPE = {"original": 1.0, "base": 0.45, "onset": 0.20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, nargs="+", required=True)
    parser.add_argument("--registry", type=Path, default=ev.DEFAULT_LABEL_REGISTRY)
    parser.add_argument("--prediction-key", type=str, default="15")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--scope", choices=["channel", "group"], default="channel")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result" / "analysis" / "adaptive_rerank_summary.csv")
    parser.add_argument("--save-events", action="store_true")
    return parser.parse_args()


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / span for value in values]


def score_items(items: list[dict], recipe: dict[str, float]) -> list[float]:
    raw = {
        "original": [
            1.0
            / max(
                int(item.get("topk_rerank_original_rank", item.get("raw_rank", item.get("rank", idx + 1))) or idx + 1),
                1,
            )
            for idx, item in enumerate(items)
        ],
        "score": [float(item.get("score", 0.0) or 0.0) for item in items],
        "pre": [float(item.get("pre_rerank_score", item.get("raw_score", 0.0)) or 0.0) for item in items],
        "base": [float(item.get("base_score", item.get("score", 0.0)) or 0.0) for item in items],
        "source_gate": [float(item.get("source_gate_score", 0.0) or 0.0) for item in items],
        "onset": [float(item.get("onset_score", 0.0) or 0.0) for item in items],
        "mech_resid": [float(item.get("mechanism_residual_score", 0.0) or 0.0) for item in items],
    }
    raw["source_interaction"] = [
        raw["base"][idx] * max(raw["source_gate"][idx], raw["onset"][idx], raw["mech_resid"][idx])
        for idx in range(len(items))
    ]
    values = {key: normalize(series) for key, series in raw.items()}
    scores = [0.0 for _ in items]
    for key, weight in recipe.items():
        series = values[key]
        for idx, value in enumerate(series):
            scores[idx] += float(weight) * float(value)
    return scores


def ranks_from_scores(scores: list[float], top_m: int) -> list[int]:
    top_m = min(max(int(top_m), 1), len(scores))
    return sorted(range(top_m), key=lambda idx: (scores[idx], -idx), reverse=True)


def adaptive_features(items: list[dict], top_m: int = 20) -> dict[str, float]:
    if not items:
        return {}
    source_scores = score_items(items, SOURCE_RECIPE)
    old_scores = score_items(items, OLD_RECIPE)
    top_m = min(max(int(top_m), 1), len(items))
    source_order = ranks_from_scores(source_scores, top_m)
    old_order = ranks_from_scores(old_scores, top_m)
    source_top3 = set(source_order[: min(3, len(source_order))])
    old_top3 = set(old_order[: min(3, len(old_order))])
    source_sorted = sorted((source_scores[idx] for idx in range(top_m)), reverse=True)
    old_sorted = sorted((old_scores[idx] for idx in range(top_m)), reverse=True)
    source_top = source_order[0] if source_order else 0
    exported_top = 0
    source_rank_of_exported_top = (
        source_order.index(exported_top) + 1 if exported_top in source_order else top_m + 1
    )
    old_rank_of_source_top = old_order.index(source_top) + 1 if source_top in old_order else top_m + 1
    source_margin = 0.0
    if len(source_sorted) > 1:
        source_margin = source_sorted[0] - source_sorted[1]
    old_margin = 0.0
    if len(old_sorted) > 1:
        old_margin = old_sorted[0] - old_sorted[1]
    return {
        "top3_overlap": len(source_top3 & old_top3) / max(min(3, len(source_top3 | old_top3)), 1),
        "source_margin": float(source_margin),
        "old_margin": float(old_margin),
        "source_rank_of_exported_top": float(source_rank_of_exported_top),
        "old_rank_of_source_top": float(old_rank_of_source_top),
        "source_top_old_score": float(old_scores[source_top]),
        "exported_top_source_score": float(source_scores[exported_top]),
        "exported_top_old_score": float(old_scores[exported_top]),
    }


def rerank_event(pred_event: dict | None, strategy: str, gate: Callable[[dict[str, float]], bool] | None = None) -> dict | None:
    if pred_event is None:
        return None
    items = pred_event.get("channel_ranking", []) or []
    if not items:
        return pred_event
    top_m = min(20, len(items))
    source_scores = score_items(items, SOURCE_RECIPE)
    old_scores = score_items(items, OLD_RECIPE)
    adjusted = copy.deepcopy(pred_event)

    if strategy == "exported":
        order = list(range(len(items)))
        scores = [float(item.get("score", 0.0) or 0.0) for item in items]
    elif strategy == "old_top20":
        head = ranks_from_scores(old_scores, top_m)
        order = head + [idx for idx in range(len(items)) if idx not in set(head)]
        scores = old_scores
    elif strategy == "source_keep_first5":
        head = [0]
        rest = sorted(range(1, min(5, len(items))), key=lambda idx: (source_scores[idx], -idx), reverse=True)
        order = head + rest + list(range(min(5, len(items)), len(items)))
        scores = source_scores
    elif strategy == "adaptive_source_or_old":
        features = adaptive_features(items, top_m=top_m)
        use_source = bool(gate(features)) if gate is not None else False
        if use_source:
            head = [0]
            rest = sorted(range(1, min(5, len(items))), key=lambda idx: (source_scores[idx], -idx), reverse=True)
            order = head + rest + list(range(min(5, len(items)), len(items)))
            scores = source_scores
        else:
            head = ranks_from_scores(old_scores, top_m)
            order = head + [idx for idx in range(len(items)) if idx not in set(head)]
            scores = old_scores
        adjusted["adaptive_rerank_used_source"] = use_source
        adjusted.update({f"adaptive_{key}": value for key, value in features.items()})
    elif strategy == "adaptive_source_or_exported":
        features = adaptive_features(items, top_m=top_m)
        use_source = bool(gate(features)) if gate is not None else False
        if use_source:
            head = [0]
            rest = sorted(range(1, min(5, len(items))), key=lambda idx: (source_scores[idx], -idx), reverse=True)
            order = head + rest + list(range(min(5, len(items)), len(items)))
            scores = source_scores
        else:
            order = list(range(len(items)))
            scores = [float(item.get("score", 0.0) or 0.0) for item in items]
        adjusted["adaptive_rerank_used_source"] = use_source
        adjusted.update({f"adaptive_{key}": value for key, value in features.items()})
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    adjusted_items = []
    for rank, idx in enumerate(order, start=1):
        item = copy.deepcopy(items[idx])
        item["rank"] = int(rank)
        item["score"] = float(scores[idx])
        item["adaptive_rerank_score"] = float(scores[idx])
        item["adaptive_rerank_strategy"] = strategy
        adjusted_items.append(item)
    adjusted["channel_ranking"] = adjusted_items
    adjusted["top_channels"] = adjusted_items[:20]
    return adjusted


def evaluate_strategy(
    rca: dict,
    meta: dict,
    pred_events: list[dict],
    args: argparse.Namespace,
    strategy: str,
    gate_name: str,
    gate: Callable[[dict[str, float]], bool] | None = None,
) -> tuple[dict, list[dict]]:
    adjusted_by_id = {id(event): rerank_event(event, strategy, gate) for event in pred_events}
    adjusted_events = list(adjusted_by_id.values())
    eval_args = argparse.Namespace(score_mode="exported", scope=args.scope)
    candidate_count = ev.expected_candidate_count(meta, rca, args.scope)
    root_key = "root_groups" if args.scope == "group" else "root_variables"
    rows = []
    source_used = 0
    matched_events = 0
    for idx, meta_event in enumerate(meta.get("events", []), start=1):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        pred_event, best_overlap = ev.match_pred_event(meta_event, pred_events)
        adjusted = adjusted_by_id.get(id(pred_event)) if pred_event is not None else None
        ranking = ev.ranking_for_event(adjusted, eval_args)
        row = ev.metric_row(ranking, roots, args.k)
        row.update(ev.cw_rcs_row(adjusted, roots, args.k, args.scope, expected_candidates=candidate_count))
        row.update(ev.interval_match_stats(meta_event, adjusted, best_overlap))
        if adjusted is not None:
            matched_events += 1
            if adjusted.get("adaptive_rerank_used_source"):
                source_used += 1
        row.update(
            {
                "series_name": rca.get("series_name"),
                "event_id": meta_event.get("event_id", idx),
                "roots": ",".join(sorted(roots)),
                "top1": ranking[0] if ranking else "",
                "strategy": strategy,
                "gate": gate_name,
            }
        )
        rows.append(row)
    if not rows:
        raise RuntimeError("No evaluable events.")
    df = pd.DataFrame(rows)
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@"))
        or c.startswith(("PR@", "MAP@", "TopKRecall@", "CW-RCS@"))
        or c
        in {
            "MRR",
            "matched",
            "matched_coverage_10",
            "matched_iou_10",
            "matched_iou_30",
            "true_coverage",
            "pred_coverage",
            "event_iou",
        }
    ]
    summary = df[metric_cols].mean(numeric_only=True).to_dict()
    summary.update(
        {
            "series_name": rca.get("series_name"),
            "strategy": strategy,
            "gate": gate_name,
            "events": len(rows),
            "source_used_rate": source_used / max(matched_events, 1),
        }
    )
    return summary, rows


def gates() -> list[tuple[str, Callable[[dict[str, float]], bool]]]:
    out: list[tuple[str, Callable[[dict[str, float]], bool]]] = []
    for overlap in [0.0, 0.34, 0.67, 1.0]:
        out.append((f"overlap>={overlap:g}", lambda f, overlap=overlap: f.get("top3_overlap", 0.0) >= overlap))
    for rank in [1, 2, 3, 5]:
        out.append((f"source_top_old_rank<={rank}", lambda f, rank=rank: f.get("old_rank_of_source_top", 99.0) <= rank))
    for score in [0.35, 0.50, 0.65, 0.80]:
        out.append((f"source_top_old_score>={score:g}", lambda f, score=score: f.get("source_top_old_score", 0.0) >= score))
    for rank in [1, 2, 3, 5]:
        out.append((f"exported_top_source_rank<={rank}", lambda f, rank=rank: f.get("source_rank_of_exported_top", 99.0) <= rank))
    for margin in [0.05, 0.10, 0.20, 0.35]:
        out.append((f"source_margin>={margin:g}", lambda f, margin=margin: f.get("source_margin", 0.0) >= margin))
    out.append(
        (
            "overlap>=0.34_and_source_old>=0.50",
            lambda f: f.get("top3_overlap", 0.0) >= 0.34 and f.get("source_top_old_score", 0.0) >= 0.50,
        )
    )
    out.append(
        (
            "source_rank<=3_or_overlap>=0.67",
            lambda f: f.get("source_rank_of_exported_top", 99.0) <= 3 or f.get("top3_overlap", 0.0) >= 0.67,
        )
    )
    return out


def main() -> None:
    args = parse_args()
    summaries = []
    all_rows = []
    for rca_path in args.rca:
        rca = ev.load_json(rca_path)
        meta = ev.load_meta_from_registry(args.registry, rca["series_name"])
        pred_events = rca.get("predicted_events_by_key", {}).get(
            str(args.prediction_key),
            rca.get("predicted_events", []),
        )
        if not pred_events:
            raise RuntimeError(f"No predicted events found in {rca_path}")
        for strategy in ["exported", "old_top20", "source_keep_first5"]:
            summary, rows = evaluate_strategy(rca, meta, pred_events, args, strategy, "none")
            summary["rca_path"] = str(rca_path)
            summaries.append(summary)
            all_rows.extend(rows)
        for gate_name, gate in gates():
            for strategy in ["adaptive_source_or_old", "adaptive_source_or_exported"]:
                summary, rows = evaluate_strategy(rca, meta, pred_events, args, strategy, gate_name, gate)
                summary["rca_path"] = str(rca_path)
                summaries.append(summary)
                all_rows.extend(rows)

    df = pd.DataFrame(summaries)
    sort_cols = [col for col in ["series_name", "MRR", "Hit@1", "Hit@3", "TopKRecall@3"] if col in df.columns]
    df = df.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
    show_cols = [
        "series_name",
        "strategy",
        "gate",
        "MRR",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "TopKRecall@3",
        "CW-RCS@3",
        "source_used_rate",
    ]
    show_cols = [col for col in show_cols if col in df.columns]
    print(df[show_cols].head(80).to_string(index=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")
    if args.save_events:
        event_path = args.output.with_name(args.output.stem + "_events.csv")
        pd.DataFrame(all_rows).to_csv(event_path, index=False)
        print(f"Saved {event_path}")


if __name__ == "__main__":
    main()
