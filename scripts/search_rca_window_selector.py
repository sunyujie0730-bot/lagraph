"""Search unsupervised predicted-window selectors for RCA evaluation.

The exported RCA report can contain several overlapping predicted windows for
one labeled event.  The default evaluator selects the window with maximum time
overlap.  This script checks whether deployment-compatible evidence confidence
signals can select a better RCA window before we wire the rule into the model
exporter or evaluator.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_rca as ev  # noqa: E402


Selector = Callable[[dict, list[dict], str, list[int], set[str]], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True, help="Path to *_rca.json.")
    parser.add_argument("--registry", type=Path, default=ev.DEFAULT_LABEL_REGISTRY)
    parser.add_argument("--prediction-key", type=str, default=None)
    parser.add_argument("--scope", choices=["channel", "group", "both"], default="both")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--temporal-hm-beta", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "result" / "analysis" / "window_selector",
    )
    return parser.parse_args()


def overlap_with_meta(pred_event: dict, meta_event: dict) -> int:
    return ev.overlap(
        int(pred_event.get("start", 0)),
        int(pred_event.get("end", 0)),
        int(meta_event.get("start", 0)),
        int(meta_event.get("end", 0)),
    )


def event_length(pred_event: dict) -> int:
    return max(0, int(pred_event.get("end", 0)) - int(pred_event.get("start", 0)))


def channel_items(pred_event: dict) -> list[dict]:
    return list(pred_event.get("channel_ranking", []) or [])


def top_value(pred_event: dict, key: str, default_key: str = "score") -> float:
    items = channel_items(pred_event)
    if not items:
        return float("-inf")
    return float(items[0].get(key, items[0].get(default_key, 0.0)) or 0.0)


def mean_top(pred_event: dict, key: str, n: int) -> float:
    items = channel_items(pred_event)[:n]
    if not items:
        return float("-inf")
    return sum(float(item.get(key, item.get("score", 0.0)) or 0.0) for item in items) / len(items)


def top_margin(pred_event: dict, key: str = "score") -> float:
    items = channel_items(pred_event)
    if not items:
        return float("-inf")
    first = float(items[0].get(key, items[0].get("score", 0.0)) or 0.0)
    second = float(items[1].get(key, items[1].get("score", 0.0)) or 0.0) if len(items) > 1 else 0.0
    return first - second


def source_concentration(pred_event: dict) -> float:
    items = channel_items(pred_event)
    if not items:
        return float("-inf")
    first = float(items[0].get("source_gate_score", 0.0) or 0.0)
    tail = [float(item.get("source_gate_score", 0.0) or 0.0) for item in items[1:5]]
    return first - (sum(tail) / len(tail) if tail else 0.0)


def normalized(values: list[float]) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.0 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) if math.isfinite(value) else 0.0 for value in values]


def composite_values(candidates: list[dict], min_len: int = 0) -> list[float]:
    topk = normalized([top_value(event, "topk_rerank_score") for event in candidates])
    source = normalized([top_value(event, "source_gate_score") for event in candidates])
    margin = normalized([top_margin(event, "score") for event in candidates])
    length = [min(event_length(event) / 120.0, 1.0) for event in candidates]
    values = []
    for idx, event in enumerate(candidates):
        if min_len > 0 and event_length(event) < min_len:
            values.append(float("-inf"))
        else:
            values.append(0.35 * topk[idx] + 0.30 * source[idx] + 0.20 * margin[idx] + 0.15 * length[idx])
    if all(not math.isfinite(value) for value in values):
        return composite_values(candidates, min_len=0)
    return values


def oracle_metric_value(pred_event: dict, roots: set[str], scope: str, k_values: list[int]) -> float:
    ranking = ev.ranking_names(pred_event, scope)
    row = ev.metric_row(ranking, roots, k_values)
    hit3 = row.get("Hit@3", 0.0)
    recall3 = row.get("TopKRecall@3", 0.0)
    return 10.0 * hit3 + recall3 + row.get("MRR", 0.0)


def selector_specs() -> dict[str, Selector]:
    return {
        "max_overlap": lambda event, candidates, scope, k, roots: float(event["_selector_overlap"]),
        "max_length": lambda event, candidates, scope, k, roots: float(event_length(event)),
        "min_length": lambda event, candidates, scope, k, roots: -float(event_length(event)),
        "max_top1_score": lambda event, candidates, scope, k, roots: top_value(event, "score"),
        "max_top1_pre_rerank": lambda event, candidates, scope, k, roots: top_value(event, "pre_rerank_score"),
        "max_top1_topk_rerank": lambda event, candidates, scope, k, roots: top_value(event, "topk_rerank_score"),
        "max_top1_source_gate": lambda event, candidates, scope, k, roots: top_value(event, "source_gate_score"),
        "max_top1_source_guard": lambda event, candidates, scope, k, roots: top_value(event, "specificity_source_guard_score"),
        "max_top1_mechanism_residual": lambda event, candidates, scope, k, roots: top_value(event, "mechanism_residual_score"),
        "max_top1_onset": lambda event, candidates, scope, k, roots: top_value(event, "onset_score"),
        "max_score_margin": lambda event, candidates, scope, k, roots: top_margin(event, "score"),
        "max_top3_score_mean": lambda event, candidates, scope, k, roots: mean_top(event, "score", 3),
        "max_top3_source_gate_mean": lambda event, candidates, scope, k, roots: mean_top(event, "source_gate_score", 3),
        "max_source_concentration": lambda event, candidates, scope, k, roots: source_concentration(event),
        "max_composite": lambda event, candidates, scope, k, roots: composite_values(candidates)[candidates.index(event)],
        "max_composite_len10": lambda event, candidates, scope, k, roots: composite_values(candidates, 10)[candidates.index(event)],
        "max_composite_len30": lambda event, candidates, scope, k, roots: composite_values(candidates, 30)[candidates.index(event)],
        "oracle_best_ranking": lambda event, candidates, scope, k, roots: oracle_metric_value(event, roots, scope, k),
    }


def choose_event(
    selector_name: str,
    selector: Selector,
    meta_event: dict,
    pred_events: list[dict],
    scope: str,
    k_values: list[int],
    roots: set[str],
) -> tuple[dict | None, int]:
    candidates = []
    for pred_event in pred_events:
        overlap = overlap_with_meta(pred_event, meta_event)
        if overlap <= 0:
            continue
        candidate = dict(pred_event)
        candidate["_selector_overlap"] = overlap
        candidates.append(candidate)
    if not candidates:
        return None, 0

    best = max(
        candidates,
        key=lambda event: (
            selector(event, candidates, scope, k_values, roots),
            event.get("_selector_overlap", 0),
            -int(event.get("event_id", 0) or 0),
        ),
    )
    return best, int(best.get("_selector_overlap", 0))


def evaluate_selector(
    rca: dict,
    meta: dict,
    pred_events: list[dict],
    selector_name: str,
    selector: Selector,
    scope: str,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    root_key = "root_groups" if scope == "group" else "root_variables"
    candidate_count = ev.expected_candidate_count(meta, rca, scope)
    chosen_events: list[dict] = []
    choices: dict[int, tuple[dict | None, int]] = {}
    for meta_idx, meta_event in enumerate(meta.get("events", []), start=1):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        chosen, best_overlap = choose_event(selector_name, selector, meta_event, pred_events, scope, args.k, roots)
        choices[meta_idx] = (chosen, best_overlap)
        if chosen is not None:
            chosen_events.append(chosen)

    eval_args = argparse.Namespace(score_mode="exported", scope=scope)
    rows = []
    for meta_idx, meta_event in enumerate(meta.get("events", []), start=1):
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        pred_event, best_overlap = choices.get(meta_idx, (None, 0))
        ranking = ev.ranking_for_event(pred_event, eval_args)
        metric_values = ev.metric_row(ranking, roots, args.k)
        top1 = ranking[0] if ranking else ""
        metric_values.update(
            ev.cw_rcs_row(
                pred_event,
                roots,
                args.k,
                scope,
                score_key="score",
                expected_candidates=candidate_count,
            )
        )
        metric_values.update(
            ev.temporal_hm_row(
                meta_event,
                chosen_events,
                roots,
                args.k,
                scope,
                beta=args.temporal_hm_beta,
                eval_args=eval_args,
            )
        )
        delay = math.nan
        if pred_event and best_overlap > 0:
            delay = max(0, int(pred_event.get("start", 0)) - int(meta_event.get("start", 0)))
        for k in args.k:
            metric_values[f"RCA_Delay@{k}"] = delay if metric_values.get(f"Hit@{k}", 0.0) > 0 else math.nan
        row = {
            "series_name": rca.get("series_name"),
            "scope": scope,
            "selector": selector_name,
            "event_id": meta_event.get("event_id", meta_idx),
            "event_start": meta_event.get("start"),
            "event_end": meta_event.get("end"),
            "pred_event_id": "" if not pred_event else pred_event.get("event_id"),
            "pred_start": "" if not pred_event else pred_event.get("start"),
            "pred_end": "" if not pred_event else pred_event.get("end"),
            "overlap": best_overlap,
            "roots": ",".join(sorted(roots)),
            "top1": top1,
            "candidate_count": candidate_count,
        }
        row.update(ev.interval_match_stats(meta_event, pred_event, best_overlap))
        row.update(metric_values)
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [
        col
        for col in df.columns
        if col.startswith(("Hit@", "Precision@", "Recall@", "NDCG@", "RCA_Delay@"))
        or col.startswith(("PR@", "MAP@"))
        or col.startswith(("TopKRecall@", "CW-RCS@", "Temporal"))
        or col
        in {
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
    summary.update({"scope": scope, "selector": selector_name, "events": len(rows)})
    return summary, rows


def main() -> None:
    args = parse_args()
    rca = ev.load_json(args.rca)
    meta = ev.load_meta_from_registry(args.registry, rca["series_name"])
    prediction_key = args.prediction_key or rca.get("rca_prediction_key")
    pred_events_by_key = rca.get("predicted_events_by_key", {})
    if prediction_key is not None and str(prediction_key) in pred_events_by_key:
        pred_events = pred_events_by_key[str(prediction_key)]
    else:
        pred_events = rca.get("predicted_events", [])
        prediction_key = rca.get("rca_prediction_key", prediction_key)
    if not pred_events:
        raise SystemExit("No predicted events found in RCA report.")

    scopes = ["channel", "group"] if args.scope == "both" else [args.scope]
    summaries = []
    all_rows = []
    for scope in scopes:
        for selector_name, selector in selector_specs().items():
            summary, rows = evaluate_selector(rca, meta, pred_events, selector_name, selector, scope, args)
            summaries.append(summary)
            all_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    sort_cols = [col for col in ["scope", "MRR", "Hit@1", "Hit@3", "TopKRecall@3"] if col in summary_df.columns]
    summary_df = summary_df.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.rca.stem.replace("_rca", "")
    summary_path = args.output_dir / f"{stem}_window_selector_summary.csv"
    rows_path = args.output_dir / f"{stem}_window_selector_events.csv"
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(all_rows).to_csv(rows_path, index=False)

    show_cols = [
        "scope",
        "selector",
        "MRR",
        "Hit@1",
        "Hit@3",
        "TopKRecall@3",
        "CW-RCS@3",
        "TemporalHMRecall@3",
        "true_coverage",
        "event_iou",
    ]
    show_cols = [col for col in show_cols if col in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))
    print(f"\nSaved {summary_path}")
    print(f"Saved {rows_path}")


if __name__ == "__main__":
    main()
