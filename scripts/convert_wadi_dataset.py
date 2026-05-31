#!/usr/bin/env python3
"""Convert WADI A1/A2 raw CSV files to LaGraph long CSV format.

The default mode targets the public WADI A1 2017 files:
  - WADI_14days.csv
  - WADI_attackdata.csv

WADI A1 attack labels are not stored as a label column in the attack CSV, so
the labels are reconstructed from the official attack-description table.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path(r"D:\WaDi.A1_9 Oct 2017\WADI.A1_9 Oct 2017")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"

NORMAL_FILE_A1 = "WADI_14days.csv"
ATTACK_FILE_A1 = "WADI_attackdata.csv"
NORMAL_SKIPROWS_A1 = 4
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


@dataclass(frozen=True)
class AttackWindow:
    attack_id: str
    start_second: int
    end_second: int


# Seconds are relative to the beginning of WADI_attackdata.csv
# (2017-10-09 18:00:00). Attack 3 and 4 are a single overlapping event in the
# official table and are kept as one RCA event.
WADI_A1_ATTACK_WINDOWS = [
    AttackWindow("1", 5100, 6616),
    AttackWindow("2", 59050, 59640),
    AttackWindow("3-4", 60900, 62640),
    AttackWindow("5", 63040, 63890),
    AttackWindow("6", 70770, 71440),
    AttackWindow("7", 74897, 75595),
    AttackWindow("8", 85200, 85780),
    AttackWindow("9", 147300, 147387),
    AttackWindow("10", 148674, 149480),
    AttackWindow("11", 149791, 150420),
    AttackWindow("12", 151140, 151500),
    AttackWindow("13", 151650, 151852),
    AttackWindow("14", 152160, 152736),
    AttackWindow("15", 163590, 164220),
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
        help="Additional cleaned feature columns to exclude.",
    )
    return parser.parse_args()


def clean_column(column: str) -> str:
    value = str(column).strip().strip('"')
    if "\\" in value:
        value = value.split("\\")[-1]
    if "/" in value:
        value = value.split("/")[-1]
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    return value


def read_columns(path: Path, *, skiprows: int = 0) -> list[str]:
    columns = pd.read_csv(path, nrows=0, skiprows=skiprows).columns.tolist()
    cleaned = [clean_column(column) for column in columns]
    duplicates = sorted({column for column in cleaned if cleaned.count(column) > 1})
    if duplicates:
        raise ValueError(f"Duplicate cleaned WADI columns in {path}: {duplicates}")
    return cleaned


def infer_columns(raw_dir: Path) -> tuple[list[str], list[str]]:
    normal_path = raw_dir / NORMAL_FILE_A1
    attack_path = raw_dir / ATTACK_FILE_A1
    if not normal_path.exists():
        raise FileNotFoundError(normal_path)
    if not attack_path.exists():
        raise FileNotFoundError(attack_path)

    normal_columns = read_columns(normal_path, skiprows=NORMAL_SKIPROWS_A1)
    attack_columns = read_columns(attack_path)
    if normal_columns != attack_columns:
        raise ValueError("Normal and attack WADI A1 columns do not match after cleaning")

    feature_columns = [column for column in normal_columns if column not in NON_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError("No WADI feature columns found")
    return normal_columns, feature_columns


def labels_from_attack_seconds(row_numbers: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(row_numbers), dtype=np.float32)
    for window in WADI_A1_ATTACK_WINDOWS:
        labels[(row_numbers >= window.start_second) & (row_numbers <= window.end_second)] = 1.0
    return labels


def read_part(
    path: Path,
    columns: list[str],
    feature_columns: list[str],
    *,
    downsample: int,
    chunk_size: int,
    skiprows: int,
    attack: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    row_offset = 0
    read_kwargs = {
        "chunksize": chunk_size,
        "low_memory": False,
        "skiprows": skiprows,
    }

    for chunk in pd.read_csv(path, **read_kwargs):
        chunk.columns = [clean_column(column) for column in chunk.columns.tolist()]
        if chunk.columns.tolist() != columns:
            chunk.columns = columns

        row_numbers = np.arange(row_offset, row_offset + len(chunk))
        keep = (row_numbers % downsample) == 0
        selected_rows = row_numbers[keep]
        chunk = chunk.loc[keep, feature_columns].copy().reset_index(drop=True)

        if not chunk.empty:
            values = chunk.apply(pd.to_numeric, errors="coerce")
            frames.append(values)
            if attack:
                labels.append(labels_from_attack_seconds(selected_rows))
            else:
                labels.append(np.zeros(len(chunk), dtype=np.float32))

        row_offset += len(row_numbers)

    if not frames:
        raise ValueError(f"No rows selected from {path}")

    values = pd.concat(frames, ignore_index=True)
    values = values.ffill().bfill().fillna(0.0).astype(np.float32)
    return values, np.concatenate(labels)


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

    output_name = args.output_name or f"WADI_A1_2017_ds{args.downsample}.csv"
    output_path = output_dir / output_name

    columns, feature_columns = infer_columns(raw_dir)
    normal_values, normal_labels = read_part(
        raw_dir / NORMAL_FILE_A1,
        columns,
        feature_columns,
        downsample=args.downsample,
        chunk_size=args.chunk_size,
        skiprows=NORMAL_SKIPROWS_A1,
        attack=False,
    )
    attack_values, attack_labels = read_part(
        raw_dir / ATTACK_FILE_A1,
        columns,
        feature_columns,
        downsample=args.downsample,
        chunk_size=args.chunk_size,
        skiprows=0,
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
        "dataset_name": "WADI_A1",
        "size": "large" if len(values) > 10000 else "small",
        "freq": f"{args.downsample}second",
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
        "source_version": "A1_9_Oct_2017",
        "pattern": (
            f"normal={NORMAL_FILE_A1};attack={ATTACK_FILE_A1};downsample={args.downsample};"
            f"labels=attack_description.xlsx/table_WADI.pdf;"
            f"min_train_std={args.min_train_std};drop_columns={'+'.join(args.drop_columns)}"
        ),
    }
    update_metadata(metadata_path, row)

    logging.info(
        "Converted %s: length=%s, train=%s, test=%s, features=%s, anomaly_points=%s",
        output_name,
        row["length"],
        row["train_lens"],
        int(len(attack_values)),
        row["n_features"],
        int(labels.sum()),
    )
    logging.info("Output: %s", output_path)
    logging.info("Updated metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
