#!/usr/bin/env python3
"""Evaluate simple non-neural RCA baselines on registered anomaly datasets.

Baselines:
- zscore: rank variables by absolute normal-train z-score during an event.
- correlation_prior: propagate z-score evidence over a normal-train absolute
  correlation graph before ranking.
- granger_prior: propagate z-score evidence over a normal-train directed
  pairwise Granger graph before ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.data.utils import read_data

DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data"
DEFAULT_DETECT_META = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True, help="Registered data file, e.g. SWAT_A1A2_Physical_v1.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--detect-meta", type=Path, default=DEFAULT_DETECT_META)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--scope", choices=["group", "channel"], default="group")
    parser.add_argument("--event-source", choices=["true", "predicted"], default="true")
    parser.add_argument("--rca", type=Path, default=None, help="Required for predicted-event baselines.")
    parser.add_argument("--prediction-key", default="1.0")
    parser.add_argument("--corr-topk", type=int, default=5)
    parser.add_argument("--corr-weight", type=float, default=1.0)
    parser.add_argument("--include-granger", action="store_true")
    parser.add_argument("--granger-max-lag", type=int, default=3)
    parser.add_argument("--granger-topk", type=int, default=5)
    parser.add_argument("--granger-weight", type=float, default=1.0)
    parser.add_argument(
        "--granger-train-points",
        type=int,
        default=50000,
        help="Use the last N normal points for pairwise Granger estimation; 0 uses all training points.",
    )
    parser.add_argument("--granger-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--event-head-ratio",
        type=float,
        default=1.0,
        help="Use only the first fraction of each event for static RCA scores.",
    )
    parser.add_argument(
        "--event-head-points",
        type=int,
        default=0,
        help="Use at most this many leading event points for static RCA scores.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def decode_list(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part for part in str(value).split(";") if part]


def root_cause_group_name(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        return str(feature_name)
    name = feature_name.strip()
    if len(name) >= 2 and name[0] == "P" and name[1].isdigit():
        return name.split("_", 1)[0]
    wadi_match = re.match(r"^([123])(?:[A-Z])?_", name)
    if wadi_match:
        return f"WADI_P{wadi_match.group(1)}"
    match = re.search(r"(\d{3})", name)
    if match:
        return f"P{match.group(1)[0]}"
    return name


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def match_pred_event(meta_event: dict, pred_events: list[dict]) -> tuple[dict | None, int]:
    best = None
    best_overlap = 0
    for event in pred_events:
        value = overlap(int(event["start"]), int(event["end"]), int(meta_event["start"]), int(meta_event["end"]))
        if value > best_overlap:
            best = event
            best_overlap = value
    return best, best_overlap


def event_score_bounds(start: int, end: int, head_ratio: float = 1.0, head_points: int = 0) -> tuple[int, int]:
    length = max(0, int(end) - int(start))
    if length <= 0:
        return int(start), int(end)
    score_len = length
    if 0.0 < float(head_ratio) < 1.0:
        score_len = min(score_len, int(math.ceil(length * float(head_ratio))))
    if int(head_points or 0) > 0:
        score_len = min(score_len, int(head_points))
    return int(start), int(start) + max(1, score_len)


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
        ideal_hits = min(len(roots), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
        row[f"NDCG@{k}"] = 0.0 if idcg == 0 else dcg / idcg
    return row


def topk_abs_corr(train_x: pd.DataFrame, topk: int) -> np.ndarray:
    corr = train_x.corr().abs().fillna(0.0).to_numpy(dtype=np.float32)
    np.fill_diagonal(corr, 0.0)
    if 0 < topk < corr.shape[1]:
        keep = np.zeros_like(corr)
        idx = np.argpartition(-corr, kth=topk - 1, axis=1)[:, :topk]
        keep[np.arange(corr.shape[0])[:, None], idx] = corr[np.arange(corr.shape[0])[:, None], idx]
        corr = keep
    row_sum = corr.sum(axis=1, keepdims=True)
    return np.divide(corr, row_sum, out=np.zeros_like(corr), where=row_sum > 0)


def _ridge_rss(design: np.ndarray, y: np.ndarray, ridge: float) -> float:
    xtx = design.T @ design
    xty = design.T @ y
    if ridge > 0:
        reg = np.eye(xtx.shape[0], dtype=xtx.dtype) * float(ridge)
        reg[0, 0] = 0.0
        xtx = xtx + reg
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    return float(np.dot(resid, resid))


def pairwise_granger_graph(
    train_x: pd.DataFrame,
    max_lag: int,
    topk: int,
    train_points: int,
    ridge: float,
) -> np.ndarray:
    """Return row-normalized directed scores A[source, target]."""
    x = train_x.to_numpy(dtype=np.float64, copy=True)
    if train_points and train_points > 0 and len(x) > train_points:
        x = x[-train_points:]
    if x.shape[0] <= max_lag + 2:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float32)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    scale = np.nanstd(x, axis=0, keepdims=True)
    x = x / np.where(scale > 1e-8, scale, 1.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n, c = x.shape
    max_lag = max(1, int(max_lag))
    lagged = np.stack([x[max_lag - lag:n - lag] for lag in range(1, max_lag + 1)], axis=2)
    y_all = x[max_lag:]
    ones = np.ones((y_all.shape[0], 1), dtype=np.float64)
    scores = np.zeros((c, c), dtype=np.float64)
    eps = 1e-12

    for target in range(c):
        y = y_all[:, target]
        target_lags = lagged[:, target, :]
        restricted = np.concatenate([ones, target_lags], axis=1)
        rss_restricted = _ridge_rss(restricted, y, ridge)
        for source in range(c):
            if source == target:
                continue
            source_lags = lagged[:, source, :]
            unrestricted = np.concatenate([restricted, source_lags], axis=1)
            rss_unrestricted = _ridge_rss(unrestricted, y, ridge)
            scores[source, target] = max(0.0, math.log((rss_restricted + eps) / (rss_unrestricted + eps)))

    if 0 < topk < c:
        keep = np.zeros_like(scores)
        idx = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        keep[np.arange(c)[:, None], idx] = scores[np.arange(c)[:, None], idx]
        scores = keep
    row_sum = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, row_sum, out=np.zeros_like(scores), where=row_sum > 0).astype(np.float32)


def rank_scores(scores: np.ndarray, columns: list[str], scope: str) -> list[str]:
    if scope == "channel":
        pairs = list(zip(columns, scores))
    else:
        group_scores = {}
        for column, score in zip(columns, scores):
            group = root_cause_group_name(column)
            group_scores[group] = max(group_scores.get(group, float("-inf")), float(score))
        pairs = list(group_scores.items())
    return [name for name, _ in sorted(pairs, key=lambda item: item[1], reverse=True)]


def load_events(registry_path: Path, series: str, scope: str) -> list[dict]:
    root_col = "subsystem_root" if scope == "group" else "variable_roots"
    rows = pd.read_csv(registry_path)
    rows = rows.loc[rows["file"] == series].copy()
    events = []
    for _, row in rows.sort_values("event_id").iterrows():
        roots = decode_list(row.get(root_col, ""))
        if roots:
            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "start": int(row["event_start"]),
                    "end": int(row["event_end"]),
                    "roots": roots,
                }
            )
    return events


def load_predicted_events(rca_path: Path, prediction_key: str) -> list[dict]:
    with open(rca_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    by_key = report.get("predicted_events_by_key", {})
    return by_key.get(str(prediction_key), report.get("predicted_events", []))


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.detect_meta)
    meta_row = meta.loc[meta["file_name"] == args.series]
    if meta_row.empty:
        raise SystemExit(f"{args.series} not found in {args.detect_meta}")
    train_lens = int(meta_row.iloc[0]["train_lens"])

    events = load_events(args.registry, args.series, args.scope)
    if not events:
        raise SystemExit(f"No {args.scope} RCA labels found for {args.series}")

    if args.event_source == "predicted" and args.rca is None:
        raise SystemExit("--rca is required when --event-source=predicted")
    pred_events = load_predicted_events(args.rca, args.prediction_key) if args.event_source == "predicted" else []

    frame = read_data(str(args.data_dir / args.series))
    if "label" in frame.columns:
        frame = frame.drop(columns=["label"])
    frame = frame.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    columns = list(frame.columns)
    train_x = frame.iloc[:train_lens]
    test_x = frame.iloc[train_lens:].reset_index(drop=True)

    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0).replace(0, 1.0)
    z = ((test_x - center) / scale).abs().to_numpy(dtype=np.float32)
    corr = topk_abs_corr(train_x, args.corr_topk)
    granger = None
    if args.include_granger:
        print(
            f"Estimating pairwise Granger graph: max_lag={args.granger_max_lag}, "
            f"topk={args.granger_topk}, train_points={args.granger_train_points}"
        )
        granger = pairwise_granger_graph(
            ((train_x - center) / scale).astype(np.float32),
            args.granger_max_lag,
            args.granger_topk,
            args.granger_train_points,
            args.granger_ridge,
        )

    rows = []
    for event in events:
        roots = set(event["roots"])
        pred_event = None
        best_overlap = math.nan
        pred_event_id = ""
        if args.event_source == "predicted":
            pred_event, best_overlap = match_pred_event(event, pred_events)
            start = int(event["start"]) if pred_event is None else int(pred_event["start"])
            end = int(event["start"]) if pred_event is None else int(pred_event["end"])
            pred_event_id = "" if pred_event is None else pred_event.get("event_id", "")
        else:
            start = int(event["start"])
            end = int(event["end"])

        start = max(0, min(start, len(z)))
        end = max(start, min(end, len(z)))
        score_start, score_end = event_score_bounds(start, end, args.event_head_ratio, args.event_head_points)
        has_evaluable_window = end > start
        if args.event_source == "predicted" and (pred_event is None or best_overlap <= 0):
            has_evaluable_window = False
        event_z = z[score_start:score_end].mean(axis=0) if has_evaluable_window else None
        score_items = [
            ("zscore", event_z),
            (
                "correlation_prior",
                None if event_z is None else event_z + args.corr_weight * corr.dot(event_z),
            ),
        ]
        if granger is not None:
            score_items.append(
                (
                    "granger_prior",
                    None if event_z is None else event_z + args.granger_weight * granger.dot(event_z),
                )
            )

        for method, scores in score_items:
            ranking = [] if scores is None else rank_scores(scores, columns, args.scope)
            metric_values = metric_row(ranking, roots, args.k)
            if args.event_source == "predicted":
                delay = math.nan
                if pred_event and best_overlap > 0:
                    delay = max(0, int(pred_event.get("start", 0)) - int(event["start"]))
                for k in args.k:
                    metric_values[f"RCA_Delay@{k}"] = delay if metric_values.get(f"Hit@{k}", 0.0) > 0 else math.nan
            row = {
                "series_name": args.series,
                "event_source": args.event_source,
                "prediction_key": args.prediction_key if args.event_source == "predicted" else "",
                "event_id": event["event_id"],
                "event_start": start,
                "event_end": end,
                "score_start": score_start,
                "score_end": score_end,
                "pred_event_id": pred_event_id,
                "overlap": "" if args.event_source == "true" else best_overlap,
                "matched": math.nan if args.event_source == "true" else (1.0 if best_overlap > 0 else 0.0),
                "roots": ",".join(sorted(roots)),
                "top1": ranking[0] if ranking else "",
                "method": method,
            }
            row.update(metric_values)
            rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("Hit@", "Precision@", "Recall@", "NDCG@", "RCA_Delay@"))
        or c in {"MRR", "matched"}
    ]
    summary = df.groupby("method", as_index=False)[metric_cols].mean().assign(series_name="MEAN")
    print("\nPer-event static RCA baseline:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df, summary], ignore_index=True).to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
