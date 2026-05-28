#!/usr/bin/env python3
"""Convert WADI A2 CSV files to LaGraph long CSV format."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "anomaly_detect"
    / "data"
    / "raw"
    / "WADI"
    / "WaDi.A2_19_Nov_2019"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"

NORMAL_FILE = "WADI_14days_new.csv"
ATTACK_FILE = "WADI_attackdataLABLE.csv"
LABEL_COLUMN = "Attack LABLE (1:No Attack, -1:Attack)"
NON_FEATURE_COLUMNS = {"Row", "Date", "Time"}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--downsample", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument(
        "--min-train-std",
        type=float,
        default=0.0,
        help="Drop feature columns whose normal-training standard deviation is below this value.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="Additional feature columns to exclude after reading the official WADI files.",
    )
    return parser.parse_args()


def strip_columns(columns: list[str]) -> list[str]:
    return [str(column).strip() for column in columns]


def read_attack_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return strip_columns(next(reader))


def infer_columns(raw_dir: Path) -> tuple[list[str], list[str]]:
    normal_path = raw_dir / NORMAL_FILE
    attack_path = raw_dir / ATTACK_FILE
    if not normal_path.exists():
        raise FileNotFoundError(normal_path)
    if not attack_path.exists():
        raise FileNotFoundError(attack_path)

    normal_columns = strip_columns(pd.read_csv(normal_path, nrows=0).columns.tolist())
    attack_columns = read_attack_header(attack_path)
    if attack_columns[-1] != LABEL_COLUMN:
        raise ValueError(f"Unexpected WADI label column: {attack_columns[-1]!r}")
    if normal_columns != attack_columns[:-1]:
        raise ValueError("Normal and attack WADI feature columns do not match")

    feature_columns = [column for column in normal_columns if column not in NON_FEATURE_COLUMNS]
    return normal_columns, feature_columns


def read_part(
    path: Path,
    columns: list[str],
    feature_columns: list[str],
    *,
    downsample: int,
    chunk_size: int,
    attack: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    row_offset = 0
    read_kwargs = {"chunksize": chunk_size, "low_memory": False}
    if attack:
        read_kwargs["skiprows"] = [0]

    for chunk in pd.read_csv(path, **read_kwargs):
        chunk.columns = strip_columns(chunk.columns.tolist())
        if attack and LABEL_COLUMN not in chunk.columns:
            chunk.columns = columns + [LABEL_COLUMN]

        row_numbers = np.arange(row_offset, row_offset + len(chunk))
        keep = (row_numbers % downsample) == 0
        chunk = chunk.loc[keep].reset_index(drop=True)

        if len(chunk) > 0:
            frames.append(chunk.loc[:, feature_columns].copy())
            if attack:
                raw_label = pd.to_numeric(chunk[LABEL_COLUMN], errors="coerce").fillna(1)
                labels.append((raw_label.to_numpy() == -1).astype(np.float32))
            else:
                labels.append(np.zeros(len(chunk), dtype=np.float32))

        row_offset += len(keep)

    if not frames:
        raise ValueError(f"No rows selected from {path}")

    values = pd.concat(frames, ignore_index=True)
    values = values.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    return values.astype(np.float32), np.concatenate(labels)


def write_long_csv(path: Path, values: pd.DataFrame, labels: np.ndarray) -> None:
    dates = np.arange(len(values), dtype=np.int64)
    first = True
    for column in values.columns:
        out = pd.DataFrame({"date": dates, "data": values[column].to_numpy(), "cols": column})
        out.to_csv(path, mode="w" if first else "a", index=False, header=first)
        first = False

    label_frame = pd.DataFrame({"date": dates, "data": labels.astype(np.float32), "cols": "label"})
    label_frame.to_csv(path, mode="a", index=False, header=False)


def filter_feature_columns(
    normal_values: pd.DataFrame,
    attack_values: pd.DataFrame,
    *,
    min_train_std: float,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selected = list(normal_values.columns)
    if min_train_std > 0:
        train_std = normal_values.std()
        selected = [column for column in selected if float(train_std[column]) >= min_train_std]

    explicit_drop = set(drop_columns)
    missing_drop = sorted(explicit_drop.difference(normal_values.columns))
    if missing_drop:
        raise ValueError(f"Requested WADI drop columns are missing: {missing_drop}")
    selected = [column for column in selected if column not in explicit_drop]

    if not selected:
        raise ValueError("No WADI feature columns remain after filtering")
    return normal_values.loc[:, selected], attack_values.loc[:, selected], selected


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
    other_columns = [column for column in metadata.columns if column not in ordered_columns]
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

    output_name = args.output_name or f"WADI_A2_2019_ds{args.downsample}.csv"
    output_path = output_dir / output_name

    columns, feature_columns = infer_columns(raw_dir)
    normal_values, normal_labels = read_part(
        raw_dir / NORMAL_FILE,
        columns,
        feature_columns,
        downsample=args.downsample,
        chunk_size=args.chunk_size,
        attack=False,
    )
    attack_values, attack_labels = read_part(
        raw_dir / ATTACK_FILE,
        columns,
        feature_columns,
        downsample=args.downsample,
        chunk_size=args.chunk_size,
        attack=True,
    )
    normal_values, attack_values, selected_columns = filter_feature_columns(
        normal_values,
        attack_values,
        min_train_std=args.min_train_std,
        drop_columns=args.drop_columns,
    )

    values = pd.concat([normal_values, attack_values], ignore_index=True)
    labels = np.concatenate([normal_labels, attack_labels])
    write_long_csv(output_path, values, labels)

    row = {
        "file_name": output_name,
        "length": int(len(values)),
        "dataset_name": "WADI_A2",
        "size": "large" if len(values) > 10000 else "small",
        "freq": "other",
        "if_univariate": False,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
        "train_lens": int(len(normal_values)),
        "n_features": int(len(selected_columns)),
        "source_dataset": "WADI",
        "source_version": "A2_19_Nov_2019",
        "pattern": (
            f"normal={NORMAL_FILE};attack={ATTACK_FILE};downsample={args.downsample};"
            f"min_train_std={args.min_train_std};drop_columns={'+'.join(args.drop_columns)}"
        ),
    }
    update_metadata(metadata_path, row)

    logging.info(
        "Converted %s: length=%s, train=%s, features=%s, anomalies=%s",
        output_name,
        row["length"],
        row["train_lens"],
        row["n_features"],
        int(labels.sum()),
    )
    logging.info("Output: %s", output_path)
    logging.info("Updated metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
