#!/usr/bin/env python3
"""Evaluate whether LaGraph RCA and channel graph explanations are faithful.

This script tests a reviewer-facing question:

    Does the learned channel graph provide evidence that changes model behavior,
    or is it only a post-hoc visualization?

Protocol:
1. Train LaGraph on the normal prefix of a selected anomaly dataset.
2. For each metadata anomaly event, build event windows.
3. Rank channels by reconstruction contribution.
4. Mask top-RCA channels, graph-neighbor channels, random channels, and
   non-neighbor channels.
5. Measure the score/error change caused by each masking strategy.

The expected pass condition is not "higher F1"; it is that explanation-aligned
masking changes the model output more than random/non-neighbor masking.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.self_impl.LaGraph.LaGraph import LaGraph  # noqa: E402
from ts_benchmark.common.constant import ANOMALY_DETECT_DATASET_PATH  # noqa: E402
from ts_benchmark.data.utils import process_data_df  # noqa: E402


DEFAULT_DATA_DIR = Path(ANOMALY_DETECT_DATASET_PATH) / "data"
DEFAULT_META_PATH = Path(ANOMALY_DETECT_DATASET_PATH) / "DETECT_META.csv"
DEFAULT_RCA_META_DIR = Path(ANOMALY_DETECT_DATASET_PATH) / "root_cause_meta"


@dataclass
class EventResult:
    series_name: str
    event_id: int
    start: int
    end: int
    root_groups: str
    baseline_score: float
    baseline_recon_error: float
    top_channels: str
    neighbor_source: str
    graph_neighbors: str
    random_channels: str
    non_neighbors: str
    top_delta_score: float
    neighbor_delta_score: float
    random_delta_score: float
    non_neighbor_delta_score: float
    top_delta_recon_error: float
    neighbor_delta_recon_error: float
    random_delta_recon_error: float
    non_neighbor_delta_recon_error: float
    graph_neighbor_support: float
    top_vs_random_score_margin: float
    neighbor_vs_random_score_margin: float
    top_vs_random_recon_margin: float
    neighbor_vs_random_recon_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HAI_21_03_test1.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--detect-meta", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--rca-meta", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Use only the first N normal training rows for smoke tests. 0 means full training prefix.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--event-limit", type=int, default=5)
    parser.add_argument("--window-samples", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument(
        "--channel-prior-align",
        type=float,
        default=None,
        help="Override normal-structure alignment loss weight for channel-prior profiles",
    )
    parser.add_argument(
        "--channel-prior-weight",
        type=float,
        default=None,
        help="Override hard normal-prior mixing weight for channel-prior profiles",
    )
    parser.add_argument(
        "--channel-prior-bias",
        type=float,
        default=None,
        help="Override normal-prior logit bias for channel-prior profiles",
    )
    parser.add_argument(
        "--channel-prior-topk",
        type=int,
        default=None,
        help="Override top-k edges retained in the normal channel correlation prior",
    )
    parser.add_argument(
        "--neighbor-source",
        choices=["learned", "correlation", "group"],
        default="learned",
        help=(
            "Source used to select explanation neighbors. 'learned' uses the model channel graph, "
            "'correlation' uses absolute normal-training correlation, and 'group' uses same-subsystem variables."
        ),
    )
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--mask-mode", choices=["baseline", "zero"], default="baseline")
    parser.add_argument(
        "--arch-profile",
        choices=[
            "full",
            "prior-guided-graph",
            "structure-consistent",
            "structure-biased",
            "mechanism-graph",
            "channel-only",
            "temporal-only",
            "reconstruction",
        ],
        default="full",
    )
    parser.add_argument("--save-csv", type=Path, default=PROJECT_ROOT / "result" / "analysis" / "graph_faithfulness.csv")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, int]:
    data_path = args.data_dir / args.dataset
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    detect_meta = pd.read_csv(args.detect_meta)
    row = detect_meta.loc[detect_meta["file_name"] == args.dataset]
    if row.empty:
        raise ValueError(f"{args.dataset} is not listed in {args.detect_meta}")
    train_len = int(row.iloc[0]["train_lens"])
    df = pd.read_csv(data_path)
    if {"date", "data", "cols"}.issubset(df.columns):
        df = process_data_df(df)
    if "label" not in df.columns:
        raise ValueError(f"{data_path} must contain a label column.")
    labels = df["label"].astype(int)
    features = df.drop(columns=["label"])
    return features, labels, train_len


def load_rca_meta(args: argparse.Namespace) -> dict:
    path = args.rca_meta
    if path is None:
        path = DEFAULT_RCA_META_DIR / f"{Path(args.dataset).stem}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def arch_switches(profile: str) -> dict:
    if profile == "full":
        return {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        }
    if profile == "prior-guided-graph":
        return {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.30,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
        }
    if profile == "structure-consistent":
        return {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.0,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.001,
        }
    if profile == "structure-biased":
        return {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.0,
            "channel_corr_prior_bias": 0.5,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.0,
        }
    if profile == "mechanism-graph":
        return {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "lambda_channel_mechanism": 0.02,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.05,
        }
    if profile == "channel-only":
        return {
            "use_channel_graph": True,
            "use_temporal_graph": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        }
    if profile == "temporal-only":
        return {
            "use_channel_graph": False,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        }
    return {
        "use_channel_graph": False,
        "use_temporal_graph": False,
        "use_vq_bypass": False,
        "use_multi_scale_scorer": False,
    }


def build_model(args: argparse.Namespace, n_features: int) -> LaGraph:
    switches = arch_switches(args.arch_profile)
    if args.channel_prior_align is not None:
        switches["lambda_channel_prior_align"] = max(0.0, args.channel_prior_align)
        if args.channel_prior_align > 0:
            switches["use_channel_corr_prior"] = True
    if args.channel_prior_weight is not None:
        switches["channel_corr_prior_weight"] = min(max(float(args.channel_prior_weight), 0.0), 1.0)
        switches["use_channel_corr_prior"] = True
    if args.channel_prior_bias is not None:
        switches["channel_corr_prior_bias"] = max(0.0, float(args.channel_prior_bias))
        switches["use_channel_corr_prior"] = True
    if args.channel_prior_topk is not None:
        switches["channel_corr_prior_topk"] = max(1, args.channel_prior_topk)
        switches["use_channel_corr_prior"] = True
    kwargs = {
        "num_epochs": args.epochs,
        "n_gpus": 1,
        "batch_size": args.batch_size,
        "dataloader_num_workers": args.num_workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "export_rca": False,
        "dataset_name": Path(args.dataset).stem,
        "input_c": n_features,
        "output_c": n_features,
        **switches,
    }
    return LaGraph(**kwargs)


def event_window_starts(start: int, end: int, win_size: int, total_length: int, samples: int) -> list[int]:
    lo = max(0, end - win_size)
    hi = min(start, total_length - win_size)
    if hi < lo:
        center = max(0, min(total_length - win_size, (start + end - win_size) // 2))
        return [center]
    if hi == lo or samples <= 1:
        return [lo]
    starts = np.linspace(lo, hi, num=min(samples, hi - lo + 1), dtype=int)
    return sorted(set(int(v) for v in starts))


def channel_names(names: Iterable[str], indices: Iterable[int]) -> str:
    names = list(names)
    return ",".join(names[i] for i in indices)


def root_group_name(feature_name: str) -> str:
    if isinstance(feature_name, str) and len(feature_name) >= 2 and feature_name[0] == "P" and feature_name[1].isdigit():
        return feature_name.split("_", 1)[0]
    return feature_name


def select_random_channels(candidates: list[int], k: int, rng: random.Random) -> list[int]:
    if not candidates:
        return []
    k = min(k, len(candidates))
    return rng.sample(candidates, k)


def average_adjacency(adjs: list[np.ndarray]) -> np.ndarray | None:
    valid = [a for a in adjs if a is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


@torch.no_grad()
def forward_event(
    model: LaGraph,
    windows: np.ndarray,
    event_start: int,
    event_end: int,
    window_starts: list[int],
) -> dict:
    raw_model = model._get_raw_model()
    device = model.device
    x = torch.as_tensor(windows, dtype=torch.float32, device=device)
    rec, adj, _, _, _, _, _ = raw_model(x)
    scores, _ = raw_model.multi_scale_forward(x)
    channel_err = torch.abs(rec - x)
    if hasattr(raw_model, "_normalize_score_error"):
        channel_err = raw_model._normalize_score_error(channel_err)

    event_channel_err = []
    event_scores = []
    for i, win_start in enumerate(window_starts):
        local_start = max(0, event_start - win_start)
        local_end = min(windows.shape[1], event_end - win_start)
        if local_end <= local_start:
            continue
        event_channel_err.append(channel_err[i, local_start:local_end].mean(dim=0).detach().cpu().numpy())
        local_scores = scores[i, local_start:local_end]
        event_scores.append(float(local_scores.mean().detach().cpu().item()))

    if not event_channel_err:
        raise RuntimeError("No event overlap inside selected windows.")

    adj_mean = None
    if adj is not None:
        adj_mean = adj.detach().cpu().numpy().mean(axis=0)

    return {
        "score": float(np.mean(event_scores)),
        "recon_error": float(np.mean([row.mean() for row in event_channel_err])),
        "channel_error": np.mean(np.stack(event_channel_err, axis=0), axis=0),
        "adjacency": adj_mean,
    }


def mask_windows(
    windows: np.ndarray,
    channel_idx: list[int],
    baseline: np.ndarray,
    mode: str,
) -> np.ndarray:
    masked = windows.copy()
    if not channel_idx:
        return masked
    if mode == "zero":
        masked[:, :, channel_idx] = 0.0
    else:
        masked[:, :, channel_idx] = baseline[channel_idx].reshape(1, 1, -1)
    return masked


def graph_neighbors(adjacency: np.ndarray | None, top_channels: list[int], k: int, n_channels: int) -> tuple[list[int], float]:
    if adjacency is None or not top_channels:
        return [], 0.0
    support = np.zeros(n_channels, dtype=np.float64)
    for idx in top_channels:
        outgoing = adjacency[idx].astype(np.float64)
        incoming = adjacency[:, idx].astype(np.float64)
        support += outgoing + incoming
    support[top_channels] = -np.inf
    order = np.argsort(-support)
    selected = [int(i) for i in order if np.isfinite(support[i]) and support[i] > 0][:k]
    finite_support = support[np.isfinite(support)]
    mean_support = float(np.mean(finite_support[finite_support > 0])) if np.any(finite_support > 0) else 0.0
    return selected, mean_support


def correlation_graph(train_scaled: pd.DataFrame) -> np.ndarray:
    values = train_scaled.to_numpy(dtype=np.float32)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32)


def group_neighbors(feature_names: list[str], top_channels: list[int], k: int) -> tuple[list[int], float]:
    if not top_channels:
        return [], 0.0
    top_groups = {root_group_name(feature_names[i]) for i in top_channels}
    candidates = [
        i
        for i, name in enumerate(feature_names)
        if i not in set(top_channels) and root_group_name(name) in top_groups
    ]
    return candidates[:k], 1.0 if candidates else 0.0


def mean_masked_result(
    model: LaGraph,
    windows: np.ndarray,
    channel_sets: list[list[int]],
    baseline: np.ndarray,
    mode: str,
    event_start: int,
    event_end: int,
    window_starts: list[int],
) -> tuple[float, float]:
    scores = []
    errors = []
    for channel_idx in channel_sets:
        masked = mask_windows(windows, channel_idx, baseline, mode)
        result = forward_event(model, masked, event_start, event_end, window_starts)
        scores.append(result["score"])
        errors.append(result["recon_error"])
    return float(np.mean(scores)), float(np.mean(errors))


def evaluate_events(args: argparse.Namespace) -> pd.DataFrame:
    set_seed(args.seed)
    features, labels, train_len = load_dataset(args)
    rca_meta = load_rca_meta(args)

    train_data = features.iloc[:train_len].reset_index(drop=True)
    test_data = features.iloc[train_len:].reset_index(drop=True)
    train_label = labels.iloc[:train_len].reset_index(drop=True)
    if args.train_limit and args.train_limit > 0:
        train_data = train_data.iloc[: args.train_limit].reset_index(drop=True)
        train_label = train_label.iloc[: args.train_limit].reset_index(drop=True)
    if train_label.sum() > 0:
        print(f"[WARN] Training prefix contains {int(train_label.sum())} labeled anomaly points.")

    model = build_model(args, train_data.shape[1])
    model.detect_fit(train_data, test_data)
    model._get_raw_model().load_state_dict(model.early_stopping.check_point)
    model.model.to(model.device)
    model.model.eval()

    scaled_test = pd.DataFrame(
        model.scaler.transform(test_data.values),
        columns=test_data.columns,
        index=test_data.index,
    )
    scaled_train = pd.DataFrame(
        model.scaler.transform(train_data.values),
        columns=train_data.columns,
        index=train_data.index,
    )
    corr_graph = correlation_graph(scaled_train) if args.neighbor_source == "correlation" else None
    train_baseline = model.scaler.transform(train_data.values).mean(axis=0).astype(np.float32)
    feature_names = list(test_data.columns)
    rng = random.Random(args.seed)
    rows: list[EventResult] = []

    events = rca_meta.get("events", [])
    if args.event_limit > 0:
        events = events[: args.event_limit]

    for event in events:
        start = int(event["start"])
        end = int(event["end"])
        if start >= len(scaled_test):
            continue
        end = min(end, len(scaled_test))
        starts = event_window_starts(start, end, model.win_size, len(scaled_test), args.window_samples)
        windows = np.stack(
            [scaled_test.iloc[s:s + model.win_size].values.astype(np.float32) for s in starts],
            axis=0,
        )

        base = forward_event(model, windows, start, end, starts)
        channel_error = base["channel_error"]
        n_channels = len(channel_error)
        top_channels = [int(i) for i in np.argsort(-channel_error)[: args.top_k]]
        if args.neighbor_source == "group":
            neighbors, neighbor_support = group_neighbors(feature_names, top_channels, args.neighbor_k)
        elif args.neighbor_source == "correlation":
            neighbors, neighbor_support = graph_neighbors(corr_graph, top_channels, args.neighbor_k, n_channels)
        else:
            neighbors, neighbor_support = graph_neighbors(base["adjacency"], top_channels, args.neighbor_k, n_channels)
        excluded = set(top_channels) | set(neighbors)
        non_neighbors = [i for i in range(n_channels) if i not in excluded]
        random_candidates = [i for i in range(n_channels) if i not in set(top_channels)]
        random_sets = [
            select_random_channels(random_candidates, max(1, len(top_channels)), rng)
            for _ in range(max(1, args.random_trials))
        ]
        non_neighbor_sets = [
            select_random_channels(non_neighbors, max(1, len(neighbors) or len(top_channels)), rng)
            for _ in range(max(1, args.random_trials))
        ]

        top_score, top_err = mean_masked_result(
            model, windows, [top_channels], train_baseline, args.mask_mode, start, end, starts
        )
        neighbor_score, neighbor_err = mean_masked_result(
            model, windows, [neighbors], train_baseline, args.mask_mode, start, end, starts
        )
        random_score, random_err = mean_masked_result(
            model, windows, random_sets, train_baseline, args.mask_mode, start, end, starts
        )
        non_score, non_err = mean_masked_result(
            model, windows, non_neighbor_sets, train_baseline, args.mask_mode, start, end, starts
        )

        base_score = base["score"]
        base_err = base["recon_error"]
        top_delta_score = abs(top_score - base_score)
        neighbor_delta_score = abs(neighbor_score - base_score)
        random_delta_score = abs(random_score - base_score)
        non_delta_score = abs(non_score - base_score)
        top_delta_err = abs(top_err - base_err)
        neighbor_delta_err = abs(neighbor_err - base_err)
        random_delta_err = abs(random_err - base_err)
        non_delta_err = abs(non_err - base_err)

        rows.append(
            EventResult(
                series_name=args.dataset,
                event_id=int(event.get("event_id", len(rows) + 1)),
                start=start,
                end=end,
                root_groups=",".join(event.get("root_groups", [])),
                baseline_score=base_score,
                baseline_recon_error=base_err,
                top_channels=channel_names(feature_names, top_channels),
                neighbor_source=args.neighbor_source,
                graph_neighbors=channel_names(feature_names, neighbors),
                random_channels=channel_names(feature_names, random_sets[0]),
                non_neighbors=channel_names(feature_names, non_neighbor_sets[0]),
                top_delta_score=top_delta_score,
                neighbor_delta_score=neighbor_delta_score,
                random_delta_score=random_delta_score,
                non_neighbor_delta_score=non_delta_score,
                top_delta_recon_error=top_delta_err,
                neighbor_delta_recon_error=neighbor_delta_err,
                random_delta_recon_error=random_delta_err,
                non_neighbor_delta_recon_error=non_delta_err,
                graph_neighbor_support=neighbor_support,
                top_vs_random_score_margin=top_delta_score - random_delta_score,
                neighbor_vs_random_score_margin=neighbor_delta_score - random_delta_score,
                top_vs_random_recon_margin=top_delta_err - random_delta_err,
                neighbor_vs_random_recon_margin=neighbor_delta_err - random_delta_err,
            )
        )
        print(
            f"[event {rows[-1].event_id}] top_margin={rows[-1].top_vs_random_score_margin:.6f}, "
            f"neighbor_margin={rows[-1].neighbor_vs_random_score_margin:.6f}, "
            f"top={rows[-1].top_channels}, neighbors={rows[-1].graph_neighbors}"
        )

    if not rows:
        raise SystemExit("No events were evaluated.")

    df = pd.DataFrame([row.__dict__ for row in rows])
    summary_cols = [
        "top_delta_score",
        "neighbor_delta_score",
        "random_delta_score",
        "non_neighbor_delta_score",
        "top_delta_recon_error",
        "neighbor_delta_recon_error",
        "random_delta_recon_error",
        "non_neighbor_delta_recon_error",
        "top_vs_random_score_margin",
        "neighbor_vs_random_score_margin",
        "top_vs_random_recon_margin",
        "neighbor_vs_random_recon_margin",
    ]
    summary = df[summary_cols].mean().to_frame("mean").T
    summary.insert(0, "series_name", "MEAN")
    summary.insert(1, "event_id", math.nan)
    out = pd.concat([df, summary], ignore_index=True)
    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.save_csv, index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved {args.save_csv}")
    return out


def main() -> None:
    args = parse_args()
    evaluate_events(args)


if __name__ == "__main__":
    main()
