#!/usr/bin/env python3
"""Convert official SWaT A1/A2 physical XLSX files to LaGraph long CSV format."""

from __future__ import annotations

import argparse
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
    / "SWAT"
    / "SWaT.A1_A2_Dec_2015"
    / "Physical"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"

NORMAL_FILE = "SWaT_Dataset_Normal_v1.xlsx"
ATTACK_FILE = "SWaT_Dataset_Attack_v0.xlsx"
LABEL_COLUMN = "Normal/Attack"
TIME_COLUMN = "Timestamp"

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
    parser.add_argument("--output-name", default="SWAT_A1A2_Physical_v1.csv")
    parser.add_argument("--downsample", type=int, default=1)
    return parser.parse_args()


def strip_columns(columns: list[object]) -> list[str]:
    return [str(column).strip() for column in columns]


def read_physical_xlsx(path: Path, *, attack: bool, downsample: int) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_excel(path, sheet_name=0, header=1, engine="openpyxl")
    frame.columns = strip_columns(frame.columns.tolist())
    frame = frame.loc[:, ~pd.Index(frame.columns).str.startswith("Unnamed")]
    if TIME_COLUMN in frame.columns:
        frame = frame.drop(columns=[TIME_COLUMN])
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"{path.name} does not contain {LABEL_COLUMN!r}")

    if downsample > 1:
        frame = frame.iloc[::downsample].reset_index(drop=True)

    label_text = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    if attack:
        labels = label_text.eq("attack").astype(np.float32).to_numpy()
    else:
        labels = np.zeros(len(frame), dtype=np.float32)

    values = frame.drop(columns=[LABEL_COLUMN])
    values.columns = strip_columns(values.columns.tolist())
    values = values.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    return values.astype(np.float32), labels


def write_long_csv(path: Path, values: pd.DataFrame, labels: np.ndarray) -> None:
    dates = np.arange(len(values), dtype=np.int64)
    first = True
    for column in values.columns:
        out = pd.DataFrame({"date": dates, "data": values[column].to_numpy(), "cols": column})
        out.to_csv(path, mode="w" if first else "a", index=False, header=first)
        first = False
    label_frame = pd.DataFrame({"date": dates, "data": labels.astype(np.float32), "cols": "label"})
    label_frame.to_csv(path, mode="a", index=False, header=False)


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

    ordered = BASE_METADATA_COLUMNS + EXTRA_METADATA_COLUMNS
    other = [column for column in metadata.columns if column not in ordered]
    metadata = metadata[ordered + other]
    metadata.to_csv(metadata_path, index=False)


def main() -> None:
    args = parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_values, normal_labels = read_physical_xlsx(
        raw_dir / NORMAL_FILE, attack=False, downsample=args.downsample
    )
    attack_values, attack_labels = read_physical_xlsx(
        raw_dir / ATTACK_FILE, attack=True, downsample=args.downsample
    )

    if list(normal_values.columns) != list(attack_values.columns):
        raise ValueError("SWaT normal and attack feature columns do not match")

    values = pd.concat([normal_values, attack_values], ignore_index=True)
    labels = np.concatenate([normal_labels, attack_labels])
    output_path = output_dir / args.output_name
    write_long_csv(output_path, values, labels)

    row = {
        "file_name": args.output_name,
        "length": int(len(values)),
        "dataset_name": "SWAT_A1A2",
        "size": "large",
        "freq": "second",
        "if_univariate": False,
        "trend": pd.NA,
        "seasonal": pd.NA,
        "stationary": pd.NA,
        "transition": pd.NA,
        "shifting": pd.NA,
        "correlation": pd.NA,
        "train_lens": int(len(normal_values)),
        "n_features": int(values.shape[1]),
        "mode": pd.NA,
        "fault_id": pd.NA,
        "source_dataset": "SWaT A1/A2",
        "source_version": "Dec 2015 Physical Normal_v1+Attack_v0",
        "pattern": f"normal_v1_train+attack_v0_test;downsample={args.downsample}",
    }
    update_metadata(args.metadata.resolve(), row)
    print(
        f"Wrote {output_path} with {len(values)} rows, "
        f"train_lens={len(normal_values)}, n_features={values.shape[1]}"
    )


if __name__ == "__main__":
    main()
