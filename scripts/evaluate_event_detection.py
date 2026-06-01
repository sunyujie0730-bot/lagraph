#!/usr/bin/env python3
"""Evaluate event-level detection quality from an exported RCA report.

The goal is to measure whether predicted anomaly intervals are good RCA inputs.
This is intentionally event-level: a predicted event is correct if it overlaps
at least one registered RCA event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--prediction-key", type=str, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def load_true_events(registry: Path, series_name: str) -> list[dict]:
    df = pd.read_csv(registry)
    rows = df.loc[df["file"] == series_name].copy()
    if rows.empty:
        raise ValueError(f"{series_name} has no RCA events in {registry}")
    events = []
    for _, row in rows.sort_values("event_id").iterrows():
        events.append(
            {
                "event_id": int(row["event_id"]),
                "start": int(row["event_start"]),
                "end": int(row["event_end"]),
            }
        )
    return events


def evaluate_key(rca: dict, true_events: list[dict], key: str) -> dict:
    pred_events = rca.get("predicted_events_by_key", {}).get(str(key), [])
    if not pred_events and str(key) == str(rca.get("rca_prediction_key")):
        pred_events = rca.get("predicted_events", [])

    true_matches = 0
    true_coverages = []
    true_delays = []
    for true_event in true_events:
        best = 0
        best_pred = None
        for pred in pred_events:
            value = overlap(true_event["start"], true_event["end"], int(pred["start"]), int(pred["end"]))
            if value > best:
                best = value
                best_pred = pred
        if best > 0:
            true_matches += 1
            true_len = max(1, true_event["end"] - true_event["start"])
            true_coverages.append(best / true_len)
            true_delays.append(max(0, int(best_pred["start"]) - true_event["start"]))

    pred_matches = 0
    pred_coverages = []
    for pred in pred_events:
        best = 0
        for true_event in true_events:
            value = overlap(true_event["start"], true_event["end"], int(pred["start"]), int(pred["end"]))
            best = max(best, value)
        if best > 0:
            pred_matches += 1
            pred_len = max(1, int(pred["end"]) - int(pred["start"]))
            pred_coverages.append(best / pred_len)

    recall = true_matches / max(len(true_events), 1)
    precision = pred_matches / max(len(pred_events), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "series_name": rca.get("series_name", ""),
        "prediction_key": str(key),
        "true_events": len(true_events),
        "pred_events": len(pred_events),
        "matched_true_events": true_matches,
        "matched_pred_events": pred_matches,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "mean_true_coverage": sum(true_coverages) / max(len(true_coverages), 1),
        "mean_pred_coverage": sum(pred_coverages) / max(len(pred_coverages), 1),
        "mean_detection_delay": sum(true_delays) / max(len(true_delays), 1),
    }


def main() -> None:
    args = parse_args()
    rca = load_json(args.rca)
    true_events = load_true_events(args.registry, rca["series_name"])
    keys = list(rca.get("predicted_events_by_key", {}).keys())
    if args.prediction_key is not None:
        keys = [args.prediction_key]
    elif not keys and rca.get("rca_prediction_key") is not None:
        keys = [str(rca.get("rca_prediction_key"))]
    if not keys:
        raise SystemExit("No predicted event keys found.")

    rows = [evaluate_key(rca, true_events, key) for key in keys]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
