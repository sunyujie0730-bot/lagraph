#!/usr/bin/env python3
"""Evaluate static HAI RCA baselines from normal training statistics.

Baselines:
- zscore: rank subsystems by event-time absolute z-score deviation.
- correlation_prior: propagate z-score evidence over the normal-training
  absolute correlation graph, then rank subsystems.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "raw" / "HAI" / "hai-21.03"
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"
DEFAULT_SAVE_CSV = PROJECT_ROOT / "result" / "rca" / "hai_static_rca_baselines.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--train-file", default="train1.csv.gz")
    parser.add_argument("--test-files", nargs="+", default=[f"test{i}.csv.gz" for i in range(1, 6)])
    parser.add_argument("--event-source", choices=["true", "predicted"], default="true")
    parser.add_argument("--rca", type=Path, nargs="*", default=[])
    parser.add_argument("--prediction-key", default="1.0")
    parser.add_argument("--corr-topk", type=int, default=5)
    parser.add_argument("--corr-weight", type=float, default=1.0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--save-csv", type=Path, default=DEFAULT_SAVE_CSV)
    return parser.parse_args()


def group_name(column: str) -> str:
    if isinstance(column, str) and re.match(r"^P\d+_", column):
        return column.split("_", 1)[0]
    return str(column)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"time", "timestamp", "datetime", "attack", "attack_P1", "attack_P2", "attack_P3", "label"}
    return [col for col in frame.columns if col not in excluded]


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


def normalize_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)


def topk_abs_corr(train_x: pd.DataFrame, topk: int) -> np.ndarray:
    corr = train_x.corr().abs().fillna(0.0).to_numpy(dtype=np.float32)
    np.fill_diagonal(corr, 0.0)
    if topk > 0 and topk < corr.shape[1]:
        keep = np.zeros_like(corr)
        idx = np.argpartition(-corr, kth=topk - 1, axis=1)[:, :topk]
        row_idx = np.arange(corr.shape[0])[:, None]
        keep[row_idx, idx] = corr[row_idx, idx]
        corr = keep
    row_sum = corr.sum(axis=1, keepdims=True)
    return np.divide(corr, row_sum, out=np.zeros_like(corr), where=row_sum > 0)


def rank_groups(feature_scores: np.ndarray, columns: list[str]) -> list[str]:
    group_scores = {}
    for column, score in zip(columns, feature_scores):
        group = group_name(column)
        group_scores[group] = max(group_scores.get(group, float("-inf")), float(score))
    return [name for name, _ in sorted(group_scores.items(), key=lambda item: item[1], reverse=True)]


def series_name(test_file: str) -> str:
    return f"HAI_21_03_{Path(test_file).stem.replace('.csv', '')}.csv"


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def load_predicted_events(rca_paths: list[Path], prediction_key: str) -> dict[str, list[dict]]:
    events = {}
    for path in rca_paths:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        by_key = report.get("predicted_events_by_key", {})
        selected = by_key.get(str(prediction_key), report.get("predicted_events", []))
        events[report["series_name"]] = selected
    return events


def match_pred_event(meta_event: dict, pred_events: list[dict]) -> tuple[dict | None, int]:
    best = None
    best_overlap = 0
    for event in pred_events:
        value = overlap(int(event["start"]), int(event["end"]), int(meta_event["event_start"]), int(meta_event["event_end"]))
        if value > best_overlap:
            best = event
            best_overlap = value
    return best, best_overlap


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry)
    predicted_by_series = load_predicted_events(args.rca, args.prediction_key) if args.event_source == "predicted" else {}

    train = pd.read_csv(args.raw_dir / args.train_file)
    columns = feature_columns(train)
    train_x = normalize_features(train, columns)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0).replace(0, 1.0)
    corr = topk_abs_corr(train_x, args.corr_topk)

    rows = []
    for test_file in args.test_files:
        name = series_name(test_file)
        label_rows = registry.loc[registry["file"] == name].copy()
        if label_rows.empty:
            continue
        test = pd.read_csv(args.raw_dir / test_file)
        test_x = normalize_features(test, columns)
        z = ((test_x - center) / scale).abs().to_numpy(dtype=np.float32)

        for _, event in label_rows.sort_values("event_id").iterrows():
            roots = set(str(event["subsystem_root"]).split(";"))
            roots.discard("")
            if not roots:
                continue

            matched = math.nan
            pred_event_id = ""
            event_z = None
            if args.event_source == "predicted":
                pred_event, best_overlap = match_pred_event(event, predicted_by_series.get(name, []))
                matched = 1.0 if best_overlap > 0 else 0.0
                if pred_event is None:
                    start = int(event["event_start"])
                    end = int(event["event_start"])
                else:
                    start = int(pred_event["start"])
                    end = int(pred_event["end"])
                    pred_event_id = pred_event.get("event_id", "")
            else:
                start = int(event["event_start"])
                end = int(event["event_end"])

            if event_z is None and end > start:
                start = max(0, min(start, len(z)))
                end = max(start, min(end, len(z)))
                event_z = z[start:end].mean(axis=0)
            score_items = []
            if event_z is not None:
                corr_score = event_z + args.corr_weight * corr.dot(event_z)
                score_items = [
                    ("zscore", event_z),
                    ("correlation_prior", corr_score),
                ]
            else:
                score_items = [("zscore", None), ("correlation_prior", None)]

            for method, scores in score_items:
                ranking = [] if scores is None else rank_groups(scores, columns)
                row = {
                    "series_name": name,
                    "event_id": int(event["event_id"]),
                    "event_start": start,
                    "event_end": end,
                    "event_source": args.event_source,
                    "prediction_key": args.prediction_key if args.event_source == "predicted" else "",
                    "pred_event_id": pred_event_id,
                    "matched": matched,
                    "roots": ";".join(sorted(roots)),
                    "top1": ranking[0] if ranking else "",
                    "method": method,
                }
                row.update(metric_row(ranking, roots, args.k))
                rows.append(row)

    if not rows:
        raise SystemExit("No evaluable HAI RCA rows found.")

    df = pd.DataFrame(rows)
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@")) or c in {"MRR", "matched"}
    ]
    summary = (
        df.groupby("method", as_index=False)[metric_cols]
        .mean()
        .assign(series_name="MEAN")
    )
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, summary], ignore_index=True).to_csv(args.save_csv, index=False)
    print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
