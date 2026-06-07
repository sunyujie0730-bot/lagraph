#!/usr/bin/env python3
"""Export RCA reports for reconstruction-based anomaly detection baselines.

This script turns a baseline model's per-channel reconstruction residual into
the same RCA JSON format used by LaGraph, so hierarchical RCA metrics can be
computed under the existing evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.self_impl.ModernTCN.ModernTCN import ModernTCN
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.data.utils import read_data


DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data"
DEFAULT_DETECT_META = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "rca_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True, help="Registered data file, e.g. WADI_A1_2017_ds10.csv")
    parser.add_argument("--model", choices=["ModernTCN"], default="ModernTCN")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--detect-meta", type=Path, default=DEFAULT_DETECT_META)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--event-head-ratio", type=float, default=0.3)
    parser.add_argument("--event-head-points", type=int, default=30)
    parser.add_argument("--prediction-key", default="15")
    parser.add_argument("--anomaly-ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0])
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def decode_list(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part for part in str(value).split(";") if part]


def root_cause_group_name(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        return str(feature_name)
    name = feature_name.strip()
    if re.match(r"^P\d+_", name):
        return name.split("_", 1)[0]
    wadi_match = re.match(r"^([123])(?:[A-Z])?_", name)
    if wadi_match:
        return f"WADI_P{wadi_match.group(1)}"
    match = re.search(r"(\d{3})", name)
    if match:
        return f"P{match.group(1)[0]}"
    return name


def event_score_bounds(start: int, end: int, head_ratio: float, head_points: int) -> tuple[int, int]:
    length = max(0, int(end) - int(start))
    if length <= 0:
        return int(start), int(end)
    score_len = length
    if 0.0 < float(head_ratio) < 1.0:
        score_len = min(score_len, int(math.ceil(length * float(head_ratio))))
    if int(head_points or 0) > 0:
        score_len = min(score_len, int(head_points))
    return int(start), int(start) + max(1, score_len)


def normalize_key(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(args.detect_meta)
    row = meta.loc[meta["file_name"] == args.series]
    if row.empty:
        raise SystemExit(f"{args.series} not found in {args.detect_meta}")
    train_lens = int(row.iloc[0]["train_lens"])

    frame = read_data(str(args.data_dir / args.series))
    if "label" in frame.columns:
        frame = frame.drop(columns=["label"])
    frame = frame.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    train_x = frame.iloc[:train_lens].copy()
    test_x = frame.iloc[train_lens:].reset_index(drop=True).copy()
    return frame, train_x, test_x


def load_true_events(registry_path: Path, series: str) -> list[dict]:
    rows = pd.read_csv(registry_path)
    rows = rows.loc[rows["file"] == series].copy()
    if rows.empty:
        raise SystemExit(f"No RCA labels found for {series} in {registry_path}")
    events = []
    for _, row in rows.sort_values("event_id").iterrows():
        roots = decode_list(row.get("variable_roots", ""))
        groups = decode_list(row.get("subsystem_root", ""))
        if not groups and roots:
            groups = sorted({root_cause_group_name(name) for name in roots})
        events.append(
            {
                "event_id": int(row["event_id"]),
                "start": int(row["event_start"]),
                "end": int(row["event_end"]),
                "variable_roots": roots,
                "subsystem_root": groups,
            }
        )
    return events


def fit_model(args: argparse.Namespace, train_x: pd.DataFrame, test_x: pd.DataFrame) -> ModernTCN:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ModernTCN(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seq_len=args.seq_len,
        horizon=args.seq_len,
        label_len=max(1, args.seq_len // 2),
        lr=args.lr,
    )
    model.detect_fit(train_x, test_x)
    return model


def reconstruction_residual_matrix(model: ModernTCN, test_x: pd.DataFrame) -> np.ndarray:
    config = model.config
    test_scaled = pd.DataFrame(
        model.scaler.transform(test_x.values),
        columns=test_x.columns,
        index=test_x.index,
    )
    model.model.load_state_dict(model.early_stopping.check_point)
    model.model.to(model.device)
    model.model.eval()

    loader = anomaly_detection_data_provider(
        test_scaled,
        batch_size=config.batch_size,
        win_size=config.seq_len,
        step=1,
        mode="thre",
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )
    residual = np.zeros((len(test_scaled), test_scaled.shape[1]), dtype=np.float64)
    counts = np.zeros((len(test_scaled), 1), dtype=np.float64)
    window_id = 0
    criterion = torch.nn.MSELoss(reduction="none")
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.float().to(model.device)
            outputs = model.model(batch_x, None, None, None)
            outputs = outputs[:, -config.horizon :, :]
            true = batch_x[:, -config.horizon :, :]
            err = criterion(true, outputs).detach().cpu().numpy()
            for b in range(err.shape[0]):
                start = (window_id + b) * config.seq_len
                end = min(start + err.shape[1], len(test_scaled))
                if start >= len(test_scaled):
                    break
                residual[start:end] += err[b, : end - start]
                counts[start:end] += 1.0
            window_id += err.shape[0]
    np.divide(residual, counts, out=residual, where=counts > 0)
    return residual.astype(np.float32)


def rank_event_scores(
    scores: np.ndarray,
    columns: list[str],
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    order = np.argsort(-scores)
    if top_k > 0:
        order = order[:top_k]
    group_best: dict[str, float] = {}
    channel_items = []
    for rank, idx in enumerate(order, start=1):
        name = columns[int(idx)]
        score = float(scores[int(idx)])
        group = root_cause_group_name(name)
        group_best[group] = max(group_best.get(group, float("-inf")), score)
        channel_items.append(
            {
                "rank": rank,
                "name": name,
                "score": score,
                "hierarchical_score": score,
                "group": group,
                "base_score": score,
                "source_score": score,
                "propagation_score": 0.0,
                "mechanism_score": 0.0,
                "causal_score": 0.0,
            }
        )

    group_items = [
        {"rank": rank, "name": name, "score": float(score)}
        for rank, (name, score) in enumerate(
            sorted(group_best.items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    ]
    group_rank = {item["name"]: item["rank"] for item in group_items}
    group_score = {item["name"]: item["score"] for item in group_items}
    within_group_seen: dict[str, int] = {}
    for item in channel_items:
        group = item["group"]
        within_group_seen[group] = within_group_seen.get(group, 0) + 1
        item["group_rank"] = group_rank.get(group, 10**9)
        item["group_score"] = group_score.get(group, item["score"])
        item["within_group_rank"] = within_group_seen[group]
    return channel_items, group_items


def build_event(
    event_id: int,
    start: int,
    end: int,
    residual: np.ndarray,
    columns: list[str],
    args: argparse.Namespace,
) -> dict:
    start = max(0, min(int(start), residual.shape[0]))
    end = max(start, min(int(end), residual.shape[0]))
    score_start, score_end = event_score_bounds(start, end, args.event_head_ratio, args.event_head_points)
    if score_end <= score_start:
        scores = np.zeros(residual.shape[1], dtype=np.float32)
    else:
        scores = residual[score_start:score_end].mean(axis=0)
    channel_ranking, group_ranking = rank_event_scores(scores, columns, args.top_k)
    return {
        "event_id": int(event_id),
        "start": int(start),
        "end": int(end),
        "length": int(max(0, end - start)),
        "aligned_start": int(start),
        "score_start": int(score_start),
        "score_end": int(score_end),
        "score_length": int(max(0, score_end - score_start)),
        "top_channels": channel_ranking,
        "channel_ranking": channel_ranking,
        "group_ranking": group_ranking,
    }


def labels_to_events(labels: np.ndarray) -> list[dict]:
    events = []
    in_event = False
    start = 0
    event_id = 1
    for idx, value in enumerate(labels.astype(int).tolist() + [0]):
        if value and not in_event:
            start = idx
            in_event = True
        elif not value and in_event:
            events.append({"event_id": event_id, "start": start, "end": idx, "length": idx - start})
            event_id += 1
            in_event = False
    return events


def build_report(
    args: argparse.Namespace,
    residual: np.ndarray,
    columns: list[str],
    true_events: list[dict],
) -> dict:
    true_report_events = [
        build_event(event["event_id"], event["start"], event["end"], residual, columns, args)
        for event in true_events
    ]

    event_score = residual.mean(axis=1)
    predicted_by_key = {}
    for ratio in args.anomaly_ratios:
        threshold = np.percentile(event_score, 100.0 - float(ratio))
        labels = (event_score > threshold).astype(np.int32)
        predicted_by_key[normalize_key(ratio)] = [
            build_event(event["event_id"], event["start"], event["end"], residual, columns, args)
            for event in labels_to_events(labels)
        ]

    return {
        "series_name": args.series,
        "dataset_name": Path(args.series).stem,
        "model": f"{args.model}-RCA",
        "score_method": "reconstruction_residual",
        "rca_export_lite": True,
        "rca_export_top_k": args.top_k,
        "rca_event_head_ratio": args.event_head_ratio,
        "rca_event_head_points": args.event_head_points,
        "rca_prediction_key": str(args.prediction_key),
        "feature_names": columns,
        "events": true_report_events,
        "predicted_events": predicted_by_key.get(str(args.prediction_key), []),
        "predicted_events_by_key": predicted_by_key,
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    _, train_x, test_x = load_frame(args)
    true_events = load_true_events(args.registry, args.series)
    print(
        f"Training {args.model}-RCA on {args.series}: "
        f"train={len(train_x)}, test={len(test_x)}, channels={test_x.shape[1]}, epochs={args.epochs}"
    )
    model = fit_model(args, train_x, test_x)
    residual = reconstruction_residual_matrix(model, test_x)
    report = build_report(args, residual, list(test_x.columns), true_events)

    out_dir = args.output_dir / Path(args.series).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.model.lower()}_rca.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    elapsed_min = (time.time() - start_time) / 60.0
    print(json.dumps({"rca_json": str(out_path), "fit_export_min": round(elapsed_min, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
