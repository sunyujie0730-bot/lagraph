#!/usr/bin/env python3
"""Diagnose variable-level RCA evidence and test sparse responsibility scores."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import evaluate_rca as ev


DEFAULT_OUTPUT_DIR = (
    ev.PROJECT_ROOT / "result" / "analysis" / "journal_validation" / "responsibility"
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    weights: dict[str, float]
    dist: str = "positive_l1"
    top_m: int = 0
    tau: float = 1.0
    power: float = 1.0


COMPONENT_KEYS = {
    "final": "score",
    "pre": "pre_rerank_score",
    "base": "base_score",
    "source_gate": "source_gate_score",
    "root": "root_score",
    "event_resp": "event_responsibility_score",
    "event_responsibility": "event_responsibility_score",
    "onset": "onset_score",
    "mech_resid": "mechanism_residual_score",
    "graph": "graph_score",
    "source": "source_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ev.DEFAULT_LABEL_REGISTRY)
    parser.add_argument("--prediction-key", type=str, default="15")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument(
        "--max-diagnostic-events",
        type=int,
        default=50,
        help="Maximum per-event diagnostics rows to save for the selected method.",
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    rca = ev.load_json(args.rca)
    meta = ev.load_meta_from_registry(args.registry, rca["series_name"])
    pred_events = rca.get("predicted_events_by_key", {}).get(str(args.prediction_key))
    if pred_events is None:
        raise ValueError(f"Prediction key {args.prediction_key!r} not found in {args.rca}")
    return rca, meta, pred_events


def minmax(values: list[float]) -> list[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / span for value in values]


def collect_components(items: list[dict]) -> dict[str, list[float]]:
    raw = {
        name: [float(item.get(key, 0.0)) for item in items]
        for name, key in COMPONENT_KEYS.items()
    }
    raw["graph_low"] = [-value for value in raw["graph"]]
    raw["source_interaction"] = [
        raw["base"][idx] * max(raw["onset"][idx], raw["mech_resid"][idx])
        for idx in range(len(items))
    ]
    raw["response_penalty"] = raw["graph"]
    return raw


def evidence_scores(items: list[dict], spec: MethodSpec) -> list[float]:
    components = collect_components(items)
    normalized = {key: minmax(values) for key, values in components.items()}
    scores = [0.0 for _ in items]
    for key, weight in spec.weights.items():
        values = normalized.get(key)
        if values is None:
            raise KeyError(f"Unknown component {key!r} in method {spec.name}")
        for idx, value in enumerate(values):
            scores[idx] += weight * value
    return scores


def responsibility(scores: list[float], spec: MethodSpec) -> list[float]:
    if not scores:
        return []
    keep = set(range(len(scores)))
    if spec.top_m and spec.top_m > 0:
        keep = {
            idx
            for idx, _ in sorted(
                enumerate(scores),
                key=lambda item: item[1],
                reverse=True,
            )[: spec.top_m]
        }

    if spec.dist == "softmax":
        tau = max(float(spec.tau), 1e-6)
        kept_scores = [scores[idx] / tau for idx in keep]
        max_score = max(kept_scores) if kept_scores else 0.0
        weights = [
            math.exp((scores[idx] / tau) - max_score) if idx in keep else 0.0
            for idx in range(len(scores))
        ]
    elif spec.dist == "positive_l1":
        kept_min = min(scores[idx] for idx in keep) if keep else min(scores)
        weights = [
            max(scores[idx] - kept_min, 0.0) ** max(float(spec.power), 1e-6)
            if idx in keep
            else 0.0
            for idx in range(len(scores))
        ]
    else:
        raise ValueError(f"Unsupported distribution {spec.dist!r}")

    total = sum(weights)
    if total <= 1e-12:
        uniform = 1.0 / max(len(keep), 1)
        return [uniform if idx in keep else 0.0 for idx in range(len(scores))]
    return [value / total for value in weights]


def rank_event(event: dict, spec: MethodSpec) -> tuple[list[str], dict[str, float], list[float]]:
    items = event.get("channel_ranking", [])
    names = [item.get("name", "") for item in items]
    scores = evidence_scores(items, spec)
    probs = responsibility(scores, spec)
    order = sorted(range(len(items)), key=lambda idx: scores[idx], reverse=True)
    ranking = [names[idx] for idx in order]
    q_by_name = {names[idx]: probs[idx] for idx in range(len(items))}
    return ranking, q_by_name, scores


def cw_rcs_from_q(
    ranking: list[str],
    q_by_name: dict[str, float],
    roots: set[str],
    k_values: Iterable[int],
) -> dict[str, float]:
    row = {}
    for k in k_values:
        topk = set(ranking[:k])
        row[f"CW-RCS@{k}"] = (
            sum(q_by_name.get(root, 0.0) for root in roots if root in topk)
            / max(len(roots), 1)
        )
    return row


def precompute_temporal_matches(
    meta_events: list[dict],
    pred_events: list[dict],
) -> dict[int, list[int | None]]:
    cache = {}
    for meta_event in meta_events:
        start = int(meta_event["start"])
        end = int(meta_event["end"])
        indices = []
        for t in range(start, end):
            best_idx = None
            best_overlap = -1
            for idx, pred_event in enumerate(pred_events):
                pred_start = int(pred_event.get("start", 0))
                pred_end = int(pred_event.get("end", 0))
                if pred_start <= t < pred_end:
                    value = ev.overlap(pred_start, pred_end, start, end)
                    if value > best_overlap:
                        best_idx = idx
                        best_overlap = value
            indices.append(best_idx)
        cache[int(meta_event["event_id"])] = indices
    return cache


def temporal_row_for_method(
    meta_event: dict,
    rankings_by_pred: list[list[str]],
    temporal_matches: dict[int, list[int | None]],
    roots: set[str],
    k_values: list[int],
) -> dict[str, float]:
    length = max(0, int(meta_event["end"]) - int(meta_event["start"]))
    row = {}
    if length <= 0:
        return row
    pred_indices = temporal_matches[int(meta_event["event_id"])]
    for k in k_values:
        strict_hits = []
        recall_values = []
        first_strict = None
        first_any = None
        for offset, pred_idx in enumerate(pred_indices):
            if pred_idx is None:
                strict_hit = 0.0
                recall = 0.0
            else:
                topk = set(rankings_by_pred[pred_idx][:k])
                hits = sum(1 for root in roots if root in topk)
                strict_hit = 1.0 if roots and hits == len(roots) else 0.0
                recall = hits / max(len(roots), 1)
            if strict_hit > 0.0 and first_strict is None:
                first_strict = offset
            if recall > 0.0 and first_any is None:
                first_any = offset
            strict_hits.append(strict_hit)
            recall_values.append(recall)
        strict_early = 0.0 if first_strict is None else max(0.0, 1.0 - first_strict / length)
        recall_early = 0.0 if first_any is None else max(0.0, 1.0 - first_any / length)
        strict_persistence = sum(strict_hits) / length
        recall_persistence = sum(recall_values) / length
        row[f"TemporalHM@{k}"] = ev._temporal_hm(strict_early, strict_persistence, 1.0)
        row[f"TemporalHMRecall@{k}"] = ev._temporal_hm(recall_early, recall_persistence, 1.0)
    return row


def evaluate_method(
    spec: MethodSpec,
    meta: dict,
    pred_events: list[dict],
    temporal_matches: dict[int, list[int | None]],
    k_values: list[int],
) -> tuple[dict, list[dict], list[list[str]], list[dict[str, float]]]:
    rankings_by_pred = []
    q_by_pred = []
    for event in pred_events:
        ranking, q_by_name, _ = rank_event(event, spec)
        rankings_by_pred.append(ranking)
        q_by_pred.append(q_by_name)

    rows = []
    for meta_event in meta.get("events", []):
        roots = set(meta_event.get("root_variables", []))
        if not roots:
            continue
        pred_event, best_overlap = ev.match_pred_event(meta_event, pred_events)
        if pred_event is None:
            ranking = []
            q_by_name = {}
        else:
            pred_idx = pred_events.index(pred_event)
            ranking = rankings_by_pred[pred_idx]
            q_by_name = q_by_pred[pred_idx]
        metric_values = ev.metric_row(ranking, roots, k_values)
        metric_values.update(cw_rcs_from_q(ranking, q_by_name, roots, k_values))
        metric_values.update(
            temporal_row_for_method(
                meta_event,
                rankings_by_pred,
                temporal_matches,
                roots,
                k_values,
            )
        )
        match_stats = ev.interval_match_stats(meta_event, pred_event, best_overlap)
        row = {
            "method": spec.name,
            "event_id": meta_event.get("event_id"),
            "roots": ",".join(sorted(roots)),
            "top1": ranking[0] if ranking else "",
        }
        row.update(match_stats)
        row.update(metric_values)
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [
        col
        for col in df.columns
        if col.startswith(("Hit@", "Precision@", "Recall@", "TopKRecall@", "MAP@", "NDCG@"))
        or col.startswith(("CW-RCS@", "TemporalHM"))
        or col in {"MRR", "event_iou", "true_coverage", "pred_coverage"}
    ]
    summary = df[metric_cols].mean().to_dict()
    summary.update(
        {
            "method": spec.name,
            "dist": spec.dist,
            "top_m": spec.top_m,
            "tau": spec.tau,
            "power": spec.power,
            "weights": json.dumps(spec.weights, sort_keys=True),
        }
    )
    return summary, rows, rankings_by_pred, q_by_pred


def single_component_methods() -> list[MethodSpec]:
    methods = []
    for key in [
        "final",
        "pre",
        "base",
        "source_gate",
        "root",
        "event_resp",
        "onset",
        "mech_resid",
        "source",
        "graph_low",
    ]:
        methods.append(MethodSpec(name=f"component_{key}_l1_all", weights={key: 1.0}))
    return methods


def grid_methods() -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    base_weights = [0.45, 0.70]
    gate_weights = [0.35, 0.70, 1.00]
    onset_weights = [0.00, 0.25]
    mech_weights = [0.00, 0.25]
    graph_penalties = [0.00, -0.05]
    interaction_weights = [0.00, 0.75]
    top_ms = [0, 20]
    powers = [1.0, 2.0]

    idx = 0
    for base in base_weights:
        for gate in gate_weights:
            for onset in onset_weights:
                for mech in mech_weights:
                    for graph in graph_penalties:
                        for inter in interaction_weights:
                            weights = {
                                "base": base,
                                "source_gate": gate,
                                "onset": onset,
                                "mech_resid": mech,
                                "response_penalty": graph,
                                "source_interaction": inter,
                            }
                            for top_m in top_ms:
                                for power in powers:
                                    idx += 1
                                    methods.append(
                                        MethodSpec(
                                            name=f"resp_grid_{idx:04d}",
                                            weights=weights,
                                            dist="positive_l1",
                                            top_m=top_m,
                                            power=power,
                                        )
                                    )
                            for tau in [0.70]:
                                idx += 1
                                methods.append(
                                    MethodSpec(
                                        name=f"resp_grid_{idx:04d}",
                                        weights=weights,
                                        dist="softmax",
                                        top_m=20,
                                        tau=tau,
                                    )
                                )
    return methods


def select_best(summary_df: pd.DataFrame) -> pd.Series:
    baseline = summary_df.loc[summary_df["method"] == "component_final_l1_all"]
    if baseline.empty:
        return summary_df.sort_values("CW-RCS@3", ascending=False).iloc[0]
    base_mrr = float(baseline.iloc[0]["MRR"])
    base_hit3 = float(baseline.iloc[0]["Hit@3"])
    filtered = summary_df.loc[
        (summary_df["MRR"] >= base_mrr - 0.03)
        & (summary_df["Hit@3"] >= base_hit3 - 0.05)
    ].copy()
    if filtered.empty:
        filtered = summary_df.copy()
    return filtered.sort_values(
        ["CW-RCS@3", "TopKRecall@3", "MRR"],
        ascending=[False, False, False],
    ).iloc[0]


def component_rank(name: str, items: list[dict], component: str) -> int | None:
    key = COMPONENT_KEYS.get(component, component)
    scored = sorted(
        items,
        key=lambda item: float(item.get(key, 0.0)),
        reverse=(component != "graph_low"),
    )
    for idx, item in enumerate(scored, start=1):
        if item.get("name") == name:
            return idx
    return None


def diagnostic_rows(
    spec: MethodSpec,
    meta: dict,
    pred_events: list[dict],
    limit: int,
) -> list[dict]:
    rows = []
    for meta_event in meta.get("events", []):
        roots = set(meta_event.get("root_variables", []))
        if not roots:
            continue
        pred_event, best_overlap = ev.match_pred_event(meta_event, pred_events)
        if not pred_event:
            continue
        ranking, q_by_name, scores = rank_event(pred_event, spec)
        items = pred_event.get("channel_ranking", [])
        name_to_item = {item.get("name"): item for item in items}
        root_ranks = {
            root: (ranking.index(root) + 1 if root in ranking else None)
            for root in sorted(roots)
        }
        root_q = {root: q_by_name.get(root, 0.0) for root in sorted(roots)}
        top1 = ranking[0] if ranking else ""
        top1_item = name_to_item.get(top1, {})
        root_component_ranks = {}
        for root in sorted(roots):
            root_component_ranks[root] = {
                component: component_rank(root, items, component)
                for component in ["pre", "base", "source_gate", "onset", "mech_resid", "root", "graph_low"]
            }
        rows.append(
            {
                "event_id": meta_event.get("event_id"),
                "event_start": meta_event.get("start"),
                "event_end": meta_event.get("end"),
                "pred_event_id": pred_event.get("event_id"),
                "overlap": best_overlap,
                "roots": ",".join(sorted(roots)),
                "top1": top1,
                "top5": ",".join(ranking[:5]),
                "root_ranks": json.dumps(root_ranks, sort_keys=True),
                "root_responsibility": json.dumps(root_q, sort_keys=True),
                "top1_is_root": float(top1 in roots),
                "top1_base": top1_item.get("base_score", 0.0),
                "top1_source_gate": top1_item.get("source_gate_score", 0.0),
                "top1_onset": top1_item.get("onset_score", 0.0),
                "top1_mech_resid": top1_item.get("mechanism_residual_score", 0.0),
                "top1_graph": top1_item.get("graph_score", 0.0),
                "root_component_ranks": json.dumps(root_component_ranks, sort_keys=True),
            }
        )
    return rows[:limit]


def main() -> None:
    args = parse_args()
    rca, meta, pred_events = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.rca).stem
    temporal_matches = precompute_temporal_matches(meta.get("events", []), pred_events)

    methods = single_component_methods() + grid_methods()
    summaries = []
    all_rows_by_method = {}
    for idx, spec in enumerate(methods, start=1):
        summary, rows, _, _ = evaluate_method(
            spec,
            meta,
            pred_events,
            temporal_matches,
            args.k,
        )
        summaries.append(summary)
        if spec.name.startswith("component_"):
            all_rows_by_method[spec.name] = rows
        if idx % 500 == 0:
            print(f"evaluated {idx}/{len(methods)} methods")

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / f"{prefix}_responsibility_grid_summary.csv"
    summary_df.sort_values(
        ["CW-RCS@3", "TopKRecall@3", "MRR"],
        ascending=[False, False, False],
    ).to_csv(summary_path, index=False)

    component_df = summary_df.loc[summary_df["method"].str.startswith("component_")].copy()
    component_path = args.output_dir / f"{prefix}_component_summary.csv"
    component_df.sort_values("MRR", ascending=False).to_csv(component_path, index=False)

    best = select_best(summary_df)
    best_spec = next(spec for spec in methods if spec.name == best["method"])
    diag = diagnostic_rows(best_spec, meta, pred_events, args.max_diagnostic_events)
    diag_path = args.output_dir / f"{prefix}_best_diagnostics.csv"
    pd.DataFrame(diag).to_csv(diag_path, index=False)

    best_rows = evaluate_method(
        best_spec,
        meta,
        pred_events,
        temporal_matches,
        args.k,
    )[1]
    event_path = args.output_dir / f"{prefix}_best_per_event.csv"
    pd.DataFrame(best_rows).to_csv(event_path, index=False)

    print("\nTop methods by CW-RCS@3:")
    cols = [
        "method",
        "MRR",
        "Hit@3",
        "TopKRecall@3",
        "CW-RCS@3",
        "TemporalHMRecall@3",
        "dist",
        "top_m",
        "power",
        "tau",
        "weights",
    ]
    print(
        summary_df.sort_values(
            ["CW-RCS@3", "TopKRecall@3", "MRR"],
            ascending=[False, False, False],
        )[cols]
        .head(10)
        .to_string(index=False)
    )
    print("\nSelected conservative best:")
    print(best[cols].to_string())
    print(f"\nSaved {summary_path}")
    print(f"Saved {component_path}")
    print(f"Saved {diag_path}")
    print(f"Saved {event_path}")


if __name__ == "__main__":
    main()
