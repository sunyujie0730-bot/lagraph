#!/usr/bin/env python3
"""Convert HAI 21.03 CSV.GZ files to LaGraph long CSV format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "raw" / "HAI" / "hai-21.03"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"

BASE_METADATA_COLUMNS = [
    "file_name",
    "length",
    "dataset_name",
    "size",
    "freq",
    "if_univariate",
    "trend",
    "seasonal",
    "stationary",
    "transition",
    "shifting",
    "correlation",
]

EXTRA_METADATA_COLUMNS = [
    "train_lens",
    "n_features",
    "mode",
    "fault_id",
    "source_dataset",
    "source_version",
    "pattern",
]

LABEL_COLUMNS = {"attack", "attack_P1", "attack_P2", "attack_P3"}
TIME_COLUMNS = {"time", "timestamp", "datetime"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--train-files", nargs="+", default=["train1.csv.gz"])
    parser.add_argument("--test-files", nargs="+", default=["test1.csv.gz"])
    parser.add_argument("--output-name", default="HAI_21_03_test1.csv")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=1)
    return parser.parse_args()


def read_hai_file(path: Path, max_rows: int | None, downsample: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, nrows=max_rows)
    if downsample > 1:
        frame = frame.iloc[::downsample].reset_index(drop=True)
    return frame


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = LABEL_COLUMNS | TIME_COLUMNS
    return [column for column in frame.columns if column not in excluded]


def normalize_features(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.ffill().bfill().fillna(0.0)
    return features.to_numpy(dtype=np.float32, copy=False)


def extract_labels(frame: pd.DataFrame) -> np.ndarray:
    if "attack" in frame.columns:
        labels = pd.to_numeric(frame["attack"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return (labels > 0).astype(np.float32)
    return np.zeros(frame.shape[0], dtype=np.float32)


def to_long_frame(values: np.ndarray, labels: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    dates = np.arange(values.shape[0], dtype=np.int64)
    wide = pd.DataFrame(values, columns=feature_names)
    wide.insert(0, "date", dates)
    long_values = wide.melt(id_vars="date", var_name="cols", value_name="data")
    label_frame = pd.DataFrame({"date": dates, "data": labels, "cols": "label"})
    return pd.concat([long_values[["date", "data", "cols"]], label_frame[["date", "data", "cols"]]], ignore_index=True)


def update_metadata(metadata_path: Path, row: dict) -> None:
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.DataFrame(columns=BASE_METADATA_COLUMNS)

    for column in BASE_METADATA_COLUMNS + EXTRA_METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = pd.NA

    metadata = metadata[metadata["file_name"] != row["file_name"]]
    metadata = pd.concat([metadata, pd.DataFrame([row])], ignore_index=True)

    for column in ["length", "train_lens", "n_features", "mode", "fault_id"]:
        if column in metadata.columns:
            metadata[column] = pd.to_numeric(metadata[column], errors="coerce").astype("Int64")

    ordered_columns = BASE_METADATA_COLUMNS + EXTRA_METADATA_COLUMNS
    other_columns = [c for c in metadata.columns if c not in ordered_columns]
    metadata = metadata[ordered_columns + other_columns]
    metadata.to_csv(metadata_path, index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = args.metadata.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frames = [
        read_hai_file(raw_dir / name, args.max_train_rows, args.downsample)
        for name in args.train_files
    ]
    test_frames = [
        read_hai_file(raw_dir / name, args.max_test_rows, args.downsample)
        for name in args.test_files
    ]

    reference_columns = infer_feature_columns(train_frames[0])
    for frame in train_frames + test_frames:
        missing = [column for column in reference_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns in HAI file: {missing[:5]}")

    train_values = np.vstack([normalize_features(frame, reference_columns) for frame in train_frames])
    test_values = np.vstack([normalize_features(frame, reference_columns) for frame in test_frames])
    train_labels = np.zeros(train_values.shape[0], dtype=np.float32)
    test_labels = np.concatenate([extract_labels(frame) for frame in test_frames])

    values = np.vstack([train_values, test_values])
    labels = np.concatenate([train_labels, test_labels])
    long_frame = to_long_frame(values, labels, reference_columns)

    output_path = output_dir / args.output_name
    long_frame.to_csv(output_path, index=False)

    row = {
        "file_name": args.output_name,
        "length": int(values.shape[0]),
        "dataset_name": "HAI_21_03",
        "size": "large" if values.shape[0] > 10000 else "small",
        "freq": "other",
        "if_univariate": False,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
        "train_lens": int(train_values.shape[0]),
        "n_features": int(values.shape[1]),
        "source_dataset": "HAI",
        "source_version": "21.03",
        "pattern": f"train={'+'.join(args.train_files)};test={'+'.join(args.test_files)};downsample={args.downsample}",
    }
    update_metadata(metadata_path, row)

    logging.info(
        "Converted %s: length=%s, train=%s, features=%s, anomalies=%s",
        args.output_name,
        row["length"],
        row["train_lens"],
        row["n_features"],
        int(labels.sum()),
    )
    logging.info("Updated metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
