"""Search group-aware channel reranking rules on an exported LaGraph RCA file.

This script is intentionally post-hoc: it does not retrain the model.  It tests
whether the exported group evidence and variable-local evidence can recover
channel-level RCA metrics before we make this logic part of the model/exporter.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_rca as ev  # noqa: E402


COMPONENT_KEYS = {
    "final": "score",
    "pre": "pre_rerank_score",
    "base": "base_score",
    "group": "group_score",
    "source_gate": "source_gate_score",
    "onset": "onset_score",
    "mech_resid": "mechanism_residual_score",
    "mechanism_residual": "mechanism_residual_score",
    "event_resp": "event_responsibility_score",
    "event_responsibility": "event_responsibility_score",
    "graph": "graph_score",
    "source": "source_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True, help="Path to *_rca.json.")
    parser.add_argument("--registry", type=Path, default=ev.DEFAULT_LABEL_REGISTRY)
    parser.add_argument("--prediction-key", type=str, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--temporal-hm-beta", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result" / "analysis" / "group_within_rerank")
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help="Expected channel candidate count. 0 uses feature_names from the RCA file.",
    )
    return parser.parse_args()


def minmax(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / span for value in values]


def positive_l1(scores: list[float], top_m: int = 20, power: float = 2.0) -> list[float]:
    if not scores:
        return []
    keep = set(range(len(scores)))
    if top_m > 0:
        keep = {
            idx
            for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[
                : min(top_m, len(scores))
            ]
        }
    kept_min = min((scores[idx] for idx in keep), default=min(scores))
    weights = [
        max(float(score) - kept_min, 0.0) ** float(power) if idx in keep else 0.0
        for idx, score in enumerate(scores)
    ]
    total = sum(weights)
    if total <= 1e-12:
        uniform = 1.0 / max(len(keep), 1)
        return [uniform if idx in keep else 0.0 for idx in range(len(scores))]
    return [value / total for value in weights]


def parse_weight_spec(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid weight spec {part!r}; expected name=value")
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in COMPONENT_KEYS and name not in {"source_interaction", "graph_low"}:
            raise KeyError(f"Unknown component {name!r}")
        weights[name] = float(value)
    return weights


def group_order_from_event(pred_event: dict, items: list[dict]) -> list[str]:
    group_scores: dict[str, float] = {}
    for group_item in pred_event.get("group_ranking", []) or []:
        name = ev.root_cause_group_name(group_item.get("name", ""))
        score = float(group_item.get("score", group_item.get("group_score", 0.0)))
        group_scores[name] = max(group_scores.get(name, float("-inf")), score)
    for item in items:
        group = item_group(item)
        score = float(item.get("group_score", item.get("score", 0.0)))
        group_scores[group] = max(group_scores.get(group, float("-inf")), score)
    return [
        group
        for group, _ in sorted(
            group_scores.items(),
            key=lambda pair: (pair[1], pair[0]),
            reverse=True,
        )
    ]


def item_group(item: dict) -> str:
    return ev.root_cause_group_name(item.get("group") or item.get("name", ""))


def component_scores(items: list[dict], weights: dict[str, float], normalize: bool = True) -> list[float]:
    raw: dict[str, list[float]] = {}
    for name, key in COMPONENT_KEYS.items():
        raw[name] = [float(item.get(key, 0.0)) for item in items]
    raw["graph_low"] = [-value for value in raw["graph"]]
    raw["source_interaction"] = [
        raw["base"][idx] * max(raw["onset"][idx], raw["mech_resid"][idx])
        for idx in range(len(items))
    ]
    values = {name: minmax(series) for name, series in raw.items()} if normalize else raw
    scores = [0.0 for _ in items]
    for name, weight in weights.items():
        series = values[name]
        for idx, value in enumerate(series):
            scores[idx] += float(weight) * float(value)
    return scores


def ranked_items_with_scores(items: list[dict], order: list[int], scores: list[float], method: str) -> list[dict]:
    ranked = []
    ordered_scores = sorted((float(scores[idx]) for idx in order), reverse=True)
    for rank, idx in enumerate(order, start=1):
        item = copy.deepcopy(items[idx])
        original_rank = int(item.get("rank", idx + 1))
        item["name"] = item.get("name", "")
        item["rank"] = rank
        item["score"] = float(ordered_scores[rank - 1]) if rank - 1 < len(ordered_scores) else float(scores[idx])
        item["group_within_rerank_score"] = float(scores[idx])
        item["group_within_rerank_original_rank"] = original_rank
        item["group_within_rerank_method"] = method
        ranked.append(item)
    return ranked


def rerank_event(pred_event: dict, method: str, weights: dict[str, float], group_top: int) -> tuple[list[str], dict[str, float], dict]:
    items = pred_event.get("channel_ranking", []) or []
    if not items:
        return [], {}, copy.deepcopy(pred_event)

    scores = component_scores(items, weights)
    groups = group_order_from_event(pred_event, items)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        buckets[item_group(item)].append(idx)
    for group in buckets:
        buckets[group].sort(key=lambda idx: (scores[idx], -idx), reverse=True)

    if group_top > 0:
        head_groups = groups[:group_top]
        tail_groups = [group for group in groups if group not in set(head_groups)]
    else:
        head_groups = groups
        tail_groups = []

    if method == "component":
        order = sorted(range(len(items)), key=lambda idx: (scores[idx], -idx), reverse=True)
    elif method == "topm_component":
        top_m = min(max(int(group_top or 5), 2), len(items))
        head = list(range(top_m))
        tail = list(range(top_m, len(items)))
        order = sorted(head, key=lambda idx: (scores[idx], -idx), reverse=True) + tail
    elif method == "topm_keep_first":
        top_m = min(max(int(group_top or 5), 2), len(items))
        head = list(range(1, top_m))
        tail = list(range(top_m, len(items)))
        order = [0] + sorted(head, key=lambda idx: (scores[idx], -idx), reverse=True) + tail
    elif method == "group_blend":
        group_rank_score = {
            group: 1.0 - (rank / max(len(groups) - 1, 1))
            for rank, group in enumerate(groups)
        }
        blended = [
            0.65 * group_rank_score.get(item_group(item), 0.0) + 0.35 * scores[idx]
            for idx, item in enumerate(items)
        ]
        scores = blended
        order = sorted(range(len(items)), key=lambda idx: (scores[idx], -idx), reverse=True)
    elif method == "group_block":
        order = []
        for group in head_groups:
            order.extend(buckets.get(group, []))
        tail = [idx for group in tail_groups for idx in buckets.get(group, [])]
        order.extend(sorted(tail, key=lambda idx: (scores[idx], -idx), reverse=True))
    elif method == "group_round_robin":
        order = []
        max_depth = max((len(buckets.get(group, [])) for group in head_groups), default=0)
        for depth in range(max_depth):
            for group in head_groups:
                bucket = buckets.get(group, [])
                if depth < len(bucket):
                    order.append(bucket[depth])
        tail = [idx for group in tail_groups for idx in buckets.get(group, [])]
        order.extend(sorted(tail, key=lambda idx: (scores[idx], -idx), reverse=True))
    else:
        raise ValueError(f"Unknown method {method!r}")

    ranking = [items[idx].get("name", "") for idx in order]
    q_values = positive_l1([scores[idx] for idx in order], top_m=20, power=2.0)
    q_by_name = {ranking[idx]: q_values[idx] for idx in range(len(ranking))}
    adjusted_event = copy.deepcopy(pred_event)
    adjusted_event["channel_ranking"] = ranked_items_with_scores(items, order, scores, method)
    return ranking, q_by_name, adjusted_event


def evaluate_strategy(
    rca: dict,
    meta: dict,
    pred_events: list[dict],
    args: argparse.Namespace,
    method: str,
    weights: dict[str, float],
    weight_name: str,
    group_top: int,
    candidate_count: int,
) -> tuple[dict, list[dict]]:
    root_key = "root_variables"
    adjusted_events = []
    event_rows = []
    eval_args = argparse.Namespace(score_mode="exported", scope="channel")
    for pred_event in pred_events:
        _, _, adjusted = rerank_event(pred_event, method, weights, group_top)
        adjusted_events.append(adjusted)

    for meta_idx, meta_event in enumerate(meta.get("events", []), start=1):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        pred_event, best_overlap = ev.match_pred_event(meta_event, pred_events)
        if pred_event is None:
            ranking = []
            q_by_name = {}
            adjusted_event = None
        else:
            ranking, q_by_name, adjusted_event = rerank_event(pred_event, method, weights, group_top)

        metric_values = ev.metric_row(ranking, roots, args.k)
        metric_values.update(ev.cw_rcs_from_q(ranking, q_by_name, roots, args.k))
        metric_values.update(
            ev.temporal_hm_row(
                meta_event,
                adjusted_events,
                roots,
                args.k,
                "channel",
                beta=args.temporal_hm_beta,
                eval_args=eval_args,
            )
        )
        delay = math.nan
        if pred_event and best_overlap > 0:
            delay = max(0, int(pred_event.get("start", 0)) - int(meta_event.get("start", 0)))
        for k in args.k:
            metric_values[f"RCA_Delay@{k}"] = delay if metric_values.get(f"Hit@{k}", 0.0) > 0 else math.nan
        match_stats = ev.interval_match_stats(meta_event, pred_event, best_overlap)
        top1 = ranking[0] if ranking else ""
        row = {
            "series_name": rca.get("series_name"),
            "event_id": meta_event.get("event_id", meta_idx),
            "event_start": meta_event.get("start"),
            "event_end": meta_event.get("end"),
            "pred_event_id": "" if not pred_event else pred_event.get("event_id"),
            "pred_start": "" if not pred_event else pred_event.get("start"),
            "pred_end": "" if not pred_event else pred_event.get("end"),
            "overlap": best_overlap,
            "roots": ",".join(sorted(roots)),
            "top1": top1,
            "method": method,
            "weight_name": weight_name,
            "weights": ",".join(f"{key}={value:g}" for key, value in weights.items()),
            "group_top": group_top,
            "candidate_count": candidate_count,
        }
        if adjusted_event:
            top_groups = [
                item_group(item)
                for item in adjusted_event.get("channel_ranking", [])[: min(5, len(adjusted_event.get("channel_ranking", [])))]
            ]
            row["top5_groups"] = ",".join(top_groups)
        else:
            row["top5_groups"] = ""
        row.update(match_stats)
        row.update(metric_values)
        event_rows.append(row)

    if not event_rows:
        raise RuntimeError("No evaluable events.")
    df = pd.DataFrame(event_rows)
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@", "RCA_Delay@"))
        or c.startswith(("PR@", "MAP@"))
        or c.startswith(("TopKRecall@", "CW-RCS@", "Temporal"))
        or c in {
            "MRR",
            "CW_RCS_DenomCoverage",
            "CW_RCS_Truncated",
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
            "method": method,
            "weight_name": weight_name,
            "weights": ",".join(f"{key}={value:g}" for key, value in weights.items()),
            "group_top": group_top,
            "events": len(event_rows),
            "candidate_count": candidate_count,
        }
    )
    return summary, event_rows


def default_weight_grid() -> list[tuple[str, dict[str, float]]]:
    specs = [
        ("exported_final", "final=1"),
        ("pre_rerank", "pre=1"),
        ("base_only", "base=1"),
        ("source_gate_only", "source_gate=1"),
        ("source_gate_onset", "source_gate=1,onset=0.25"),
        ("best_no_event_resp", "base=0.45,source_gate=1,onset=0.25,mech_resid=0.25"),
        ("interaction", "base=0.45,source_gate=1,onset=0.25,mech_resid=0.25,source_interaction=0.25"),
        ("event_resp_check", "base=0.45,source_gate=1,onset=0.25,event_resp=0.25"),
    ]
    return [(name, parse_weight_spec(spec)) for name, spec in specs]


def main() -> None:
    args = parse_args()
    rca = ev.load_json(args.rca)
    meta = ev.load_meta_from_registry(args.registry, rca["series_name"])
    pred_events_by_key = rca.get("predicted_events_by_key", {})
    prediction_key = args.prediction_key or rca.get("rca_prediction_key")
    if prediction_key is not None and str(prediction_key) in pred_events_by_key:
        pred_events = pred_events_by_key[str(prediction_key)]
    else:
        pred_events = rca.get("predicted_events", [])
        prediction_key = rca.get("rca_prediction_key", prediction_key)
    if not pred_events:
        raise SystemExit("No predicted events found in RCA report.")

    candidate_count = args.candidate_count or ev.expected_candidate_count(meta, rca, "channel")
    strategies = [
        "component",
        "topm_component",
        "topm_keep_first",
        "group_blend",
        "group_block",
        "group_round_robin",
    ]
    group_tops = [0, 2, 3, 5, 8]
    topm_values = [3, 5, 8, 10, 20]
    summaries = []
    all_rows = []
    for weight_name, weights in default_weight_grid():
        for method in strategies:
            if method in {"component", "group_blend"}:
                tops = [0]
            elif method in {"topm_component", "topm_keep_first"}:
                tops = topm_values
            else:
                tops = group_tops
            for group_top in tops:
                summary, rows = evaluate_strategy(
                    rca,
                    meta,
                    pred_events,
                    args,
                    method,
                    weights,
                    weight_name,
                    group_top,
                    candidate_count,
                )
                summaries.append(summary)
                all_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    sort_cols = [col for col in ["MRR", "Hit@1", "Hit@3", "TopKRecall@3", "CW-RCS@3"] if col in summary_df.columns]
    summary_df = summary_df.sort_values(sort_cols, ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.rca.stem.replace("_rca", "")
    summary_path = args.output_dir / f"{stem}_group_within_summary.csv"
    rows_path = args.output_dir / f"{stem}_group_within_events.csv"
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(all_rows).to_csv(rows_path, index=False)

    show_cols = [
        "method",
        "weight_name",
        "group_top",
        "MRR",
        "Hit@1",
        "Hit@3",
        "TopKRecall@3",
        "CW-RCS@3",
        "TemporalHMRecall@3",
    ]
    show_cols = [col for col in show_cols if col in summary_df.columns]
    print(summary_df[show_cols].head(20).to_string(index=False))
    print(f"\nSaved {summary_path}")
    print(f"Saved {rows_path}")


if __name__ == "__main__":
    main()
