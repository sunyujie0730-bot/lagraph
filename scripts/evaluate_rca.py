#!/usr/bin/env python3
"""Evaluate root-cause ranking reports with Hit@K, MRR, and NDCG@K."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "root_cause_meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, required=True, help="Path to *_rca.json exported by LaGraph.")
    parser.add_argument("--meta", type=Path, default=None, help="Root-cause metadata JSON. Defaults by series_name.")
    parser.add_argument("--scope", choices=["group", "channel"], default="group")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--baseline", choices=["none", "random"], default="none")
    parser.add_argument("--random-seed", type=int, default=2021)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def match_meta_event(pred_event: dict, meta_events: list[dict]) -> dict | None:
    best = None
    best_overlap = 0
    for event in meta_events:
        value = overlap(pred_event["start"], pred_event["end"], event["start"], event["end"])
        if value > best_overlap:
            best = event
            best_overlap = value
    return best


def ranking_names(pred_event: dict, scope: str) -> list[str]:
    key = "group_ranking" if scope == "group" else "channel_ranking"
    return [item["name"] for item in pred_event.get(key, [])]


def baseline_ranking(meta: dict, scope: str, seed: int) -> list[str]:
    if scope == "group":
        names = list(meta.get("group_scope", []))
    else:
        names = sorted(
            {
                variable
                for event in meta.get("events", [])
                for variable in event.get("root_variables", [])
            }
        )
    rng = random.Random(seed)
    rng.shuffle(names)
    return names


def metric_row(ranking: list[str], roots: set[str], k_values: list[int]) -> dict:
    row = {}
    first_rank = None
    for idx, name in enumerate(ranking, start=1):
        if name in roots:
            first_rank = idx
            break
    row["MRR"] = 0.0 if first_rank is None else 1.0 / first_rank

    for k in k_values:
        topk = ranking[:k]
        hits = sum(1 for name in topk if name in roots)
        row[f"Hit@{k}"] = 1.0 if hits > 0 else 0.0
        row[f"Precision@{k}"] = hits / max(k, 1)
        row[f"Recall@{k}"] = hits / max(len(roots), 1)
        dcg = 0.0
        for idx, name in enumerate(topk, start=1):
            if name in roots:
                dcg += 1.0 / math.log2(idx + 1)
        ideal_hits = min(len(roots), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
        row[f"NDCG@{k}"] = 0.0 if idcg == 0 else dcg / idcg
    return row


def main() -> None:
    args = parse_args()
    rca = load_json(args.rca)
    meta_path = args.meta
    if meta_path is None:
        series_stem = Path(rca["series_name"]).stem
        meta_path = DEFAULT_META_DIR / f"{series_stem}.json"
    meta = load_json(meta_path)

    root_key = "root_groups" if args.scope == "group" else "root_variables"
    rows = []
    for pred_event in rca.get("events", []):
        meta_event = match_meta_event(pred_event, meta.get("events", []))
        if not meta_event:
            continue
        roots = set(meta_event.get(root_key, []))
        if not roots:
            continue
        if args.baseline == "random":
            ranking = baseline_ranking(meta, args.scope, args.random_seed + int(pred_event.get("event_id", 0)))
        else:
            ranking = ranking_names(pred_event, args.scope)
        row = {
            "series_name": rca.get("series_name"),
            "event_id": pred_event.get("event_id"),
            "event_start": pred_event.get("start"),
            "event_end": pred_event.get("end"),
            "roots": ",".join(sorted(roots)),
            "top1": ranking[0] if ranking else "",
            "method": "random" if args.baseline == "random" else "lagraph",
        }
        row.update(metric_row(ranking, roots, args.k))
        rows.append(row)

    if not rows:
        raise SystemExit("No evaluable RCA events found. Check metadata roots and event intervals.")

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@")) or c == "MRR"]
    summary = df[metric_cols].mean().to_frame("mean").T
    print("\nPer-event RCA:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, summary.assign(series_name="MEAN")], ignore_index=True).to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
