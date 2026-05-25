#!/usr/bin/env python3
"""Evaluate a simple HAI subsystem RCA baseline using train-normal z-score deviation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "raw" / "HAI" / "hai-21.03"
DEFAULT_META_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "root_cause_meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--train-file", default="train1.csv.gz")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def group_name(column: str) -> str:
    if len(column) >= 2 and column[0] == "P" and column[1].isdigit():
        return column.split("_", 1)[0]
    return column


def metric_row(ranking: list[str], roots: set[str], k_values: list[int]) -> dict:
    row = {}
    first_rank = next((idx for idx, name in enumerate(ranking, start=1) if name in roots), None)
    row["MRR"] = 0.0 if first_rank is None else 1.0 / first_rank
    for k in k_values:
        topk = ranking[:k]
        hits = sum(1 for name in topk if name in roots)
        row[f"Hit@{k}"] = 1.0 if hits else 0.0
        row[f"Precision@{k}"] = hits / max(k, 1)
        row[f"Recall@{k}"] = hits / max(len(roots), 1)
        dcg = sum(1.0 / math.log2(idx + 1) for idx, name in enumerate(topk, start=1) if name in roots)
        ideal = min(len(roots), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal + 1))
        row[f"NDCG@{k}"] = 0.0 if idcg == 0 else dcg / idcg
    return row


def main() -> None:
    args = parse_args()
    label_cols = {"attack", "attack_P1", "attack_P2", "attack_P3"}
    time_cols = {"time", "timestamp", "datetime"}

    train = pd.read_csv(args.raw_dir / args.train_file)
    feature_cols = [c for c in train.columns if c not in label_cols and c not in time_cols]
    train_x = train[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0).replace(0, 1.0)

    test = pd.read_csv(args.raw_dir / args.test_file)
    test_x = test[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    z = ((test_x - center) / scale).abs()

    meta_path = args.meta_dir / f"HAI_21_03_{Path(args.test_file).name.replace('.csv.gz', '')}.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = []
    for event in meta["events"]:
        roots = set(event["root_groups"])
        event_z = z.iloc[event["start"] : event["end"]].mean(axis=0)
        group_scores = {}
        for column, value in event_z.items():
            group = group_name(column)
            group_scores[group] = max(group_scores.get(group, float("-inf")), float(value))
        ranking = [name for name, _ in sorted(group_scores.items(), key=lambda item: item[1], reverse=True)]
        row = {
            "series_name": meta["series_name"],
            "event_id": event["event_id"],
            "event_start": event["start"],
            "event_end": event["end"],
            "roots": ",".join(sorted(roots)),
            "top1": ranking[0] if ranking else "",
            "method": "zscore",
        }
        row.update(metric_row(ranking, roots, args.k))
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@")) or c == "MRR"]
    summary = df[metric_cols].mean().to_frame("mean").T
    print("\nPer-event z-score RCA:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, summary.assign(series_name="MEAN", method="zscore")], ignore_index=True).to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
