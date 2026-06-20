"""Fast paired WADI/SWAT post-hoc RCA rerank weight grid.

This script loads each large RCA JSON once, extracts only the matched event
rankings, and then evaluates many small weight recipes in memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_rca as ev  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wadi-rca", type=Path, required=True)
    parser.add_argument("--swat-rca", type=Path, required=True)
    parser.add_argument("--prediction-key", type=str, default="15")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result" / "analysis" / "lightweight_rerank" / "wadi_swat_weight_grid.csv")
    return parser.parse_args()


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def extract_dataset(rca_path: Path, prediction_key: str) -> list[dict]:
    rca = ev.load_json(rca_path)
    meta = ev.load_meta_from_registry(ev.DEFAULT_LABEL_REGISTRY, rca["series_name"])
    pred_events = rca.get("predicted_events_by_key", {}).get(str(prediction_key), rca.get("predicted_events", []))
    extracted = []
    for meta_event in meta.get("events", []):
        roots = set(meta_event.get("root_variables", []))
        if not roots:
            continue
        pred_event, _ = ev.match_pred_event(meta_event, pred_events)
        items = [] if pred_event is None else (pred_event.get("channel_ranking", []) or [])
        names = [item.get("name", "") for item in items]
        components = {
            "original": [
                1.0
                / max(
                    int(item.get("topk_rerank_original_rank", item.get("raw_rank", item.get("rank", idx + 1))) or idx + 1),
                    1,
                )
                for idx, item in enumerate(items)
            ],
            "score": [float(item.get("score", 0.0) or 0.0) for item in items],
            "base": [float(item.get("base_score", item.get("score", 0.0)) or 0.0) for item in items],
            "source_gate": [float(item.get("source_gate_score", 0.0) or 0.0) for item in items],
            "onset": [float(item.get("onset_score", 0.0) or 0.0) for item in items],
            "mech": [float(item.get("mechanism_residual_score", 0.0) or 0.0) for item in items],
            "root": [float(item.get("root_score", 0.0) or 0.0) for item in items],
            "event": [float(item.get("event_responsibility_score", 0.0) or 0.0) for item in items],
            "evidence_fusion": [float(item.get("evidence_fusion_score", 0.0) or 0.0) for item in items],
            "source_interaction_head": [float(item.get("source_interaction_head_score", 0.0) or 0.0) for item in items],
            "source_consistency": [float(item.get("source_consistency_head_score", 0.0) or 0.0) for item in items],
            "graph": [float(item.get("graph_score", 0.0) or 0.0) for item in items],
            "response_suppressor": [float(item.get("response_suppressor_score", 0.0) or 0.0) for item in items],
        }
        components["interaction"] = [
            components["base"][idx]
            * max(components["source_gate"][idx], components["onset"][idx], components["mech"][idx])
            for idx in range(len(items))
        ]
        components["learned_interaction"] = [
            components["source_interaction_head"][idx]
            if abs(components["source_interaction_head"][idx]) > 1e-12
            else components["base"][idx] * max(components["onset"][idx], components["mech"][idx])
            for idx in range(len(items))
        ]
        components["graph_low"] = [-value for value in components["graph"]]
        components["response_suppressor_low"] = [
            -value for value in components["response_suppressor"]
        ]
        normalized = {key: normalize(value) for key, value in components.items()}
        extracted.append({"roots": roots, "names": names, "components": normalized})
    return extracted


def ranking_for(extracted_event: dict, weights: dict[str, float], top_m: int = 20) -> list[str]:
    names = extracted_event["names"]
    if not names:
        return []
    components = extracted_event["components"]
    scores = [0.0 for _ in names]
    for key, weight in weights.items():
        values = components[key]
        for idx, value in enumerate(values):
            scores[idx] += weight * value
    top_m = min(max(top_m, 1), len(names))
    head = list(range(top_m))
    tail = list(range(top_m, len(names)))
    order = sorted(head, key=lambda idx: (scores[idx], -idx), reverse=True) + tail
    return [names[idx] for idx in order]


def evaluate_dataset(events: list[dict], weights: dict[str, float]) -> dict:
    rows = [ev.metric_row(ranking_for(event, weights), event["roots"], [1, 3, 5]) for event in events]
    df = pd.DataFrame(rows)
    return {key: df[key].mean() for key in ["MRR", "Hit@1", "Hit@3", "Hit@5", "TopKRecall@3"]}


def recipe_grid() -> list[tuple[str, dict[str, float]]]:
    recipes = [
        ("exported", {"score": 1.0}),
        ("old", {"original": 1.0, "base": 0.45, "onset": 0.20}),
        ("new", {"base": 0.45, "onset": 0.25, "source_gate": 1.0, "mech": 0.25, "interaction": 0.25}),
        (
            "learned_no_source_gate",
            {"base": 0.35, "onset": 0.25, "root": 0.25, "event": 0.20, "evidence_fusion": 0.20},
        ),
        (
            "learned_with_light_source",
            {
                "base": 0.35,
                "onset": 0.25,
                "source_gate": 0.20,
                "root": 0.25,
                "event": 0.20,
                "evidence_fusion": 0.20,
            },
        ),
        (
            "learned_graph_penalized",
            {
                "base": 0.35,
                "onset": 0.25,
                "root": 0.25,
                "event": 0.20,
                "evidence_fusion": 0.20,
                "graph_low": 0.05,
            },
        ),
    ]
    for root in [0.0, 0.15, 0.25, 0.40]:
        for event in [0.0, 0.10, 0.20, 0.35]:
            for fusion in [0.0, 0.10, 0.20, 0.35]:
                for source_gate in [0.0, 0.10, 0.20, 0.35]:
                    weights = {
                        "base": 0.35,
                        "onset": 0.25,
                        "source_gate": source_gate,
                        "root": root,
                        "event": event,
                        "evidence_fusion": fusion,
                    }
                    recipes.append(
                        (
                            f"learned_r{root:g}_e{event:g}_f{fusion:g}_sg{source_gate:g}",
                            weights,
                        )
                    )
    for original in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for source_gate in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
            for mech in [0.0, 0.1, 0.25]:
                for interaction in [0.0, 0.1, 0.25]:
                    weights = {
                        "original": original,
                        "base": 0.45,
                        "onset": 0.25,
                        "source_gate": source_gate,
                        "mech": mech,
                        "interaction": interaction,
                    }
                    recipes.append((f"o{original:g}_s{source_gate:g}_m{mech:g}_i{interaction:g}", weights))
    deduped = {}
    for name, weights in recipes:
        deduped[name] = weights
    return list(deduped.items())


def main() -> None:
    args = parse_args()
    wadi_events = extract_dataset(args.wadi_rca, args.prediction_key)
    swat_events = extract_dataset(args.swat_rca, args.prediction_key)
    rows = []
    for name, weights in recipe_grid():
        wadi = evaluate_dataset(wadi_events, weights)
        swat = evaluate_dataset(swat_events, weights)
        row = {"recipe": name, "weights": ",".join(f"{k}={v:g}" for k, v in weights.items())}
        row.update({f"wadi_{key}": value for key, value in wadi.items()})
        row.update({f"swat_{key}": value for key, value in swat.items()})
        row["mean_MRR"] = (row["wadi_MRR"] + row["swat_MRR"]) / 2.0
        row["min_MRR"] = min(row["wadi_MRR"], row["swat_MRR"])
        rows.append(row)
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    for title, sort_cols in {
        "best_mean": ["mean_MRR", "wadi_MRR", "swat_MRR"],
        "best_min": ["min_MRR", "wadi_MRR", "swat_MRR"],
        "best_wadi": ["wadi_MRR", "swat_MRR"],
        "best_swat": ["swat_MRR", "wadi_MRR"],
    }.items():
        print(f"\n{title}")
        print(
            df.sort_values(sort_cols, ascending=False)
            .head(15)[
                [
                    "recipe",
                    "wadi_MRR",
                    "wadi_Hit@1",
                    "wadi_Hit@3",
                    "swat_MRR",
                    "swat_Hit@1",
                    "swat_Hit@3",
                    "mean_MRR",
                    "min_MRR",
                ]
            ]
            .to_string(index=False)
        )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
