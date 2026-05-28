#!/usr/bin/env python3
"""Convert SWaT A6 Dec 2019 historian data to LaGraph long CSV format."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, time
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
    / "SWaT.A6_Dec_2019"
)
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

PROCESS_ATTACK_WINDOWS = [
    ("12:30:00", "12:32:59"),
    ("12:43:00", "12:45:59"),
    ("12:56:00", "12:58:59"),
    ("13:09:00", "13:11:59"),
    ("13:22:00", "13:24:59"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-name", default="SWAT_A6_Dec2019_process.csv")
    return parser.parse_args()


def normalize_feature_name(name: str) -> str:
    name = str(name).strip()
    for suffix in [".Pv", ".Status", ".Alarm"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace(" ", "_")


def read_swat_a6_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, sheet_name="Dec2019", header=9, engine="openpyxl")
    frame = frame.dropna(how="all").reset_index(drop=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "t_stamp" not in frame.columns:
        raise ValueError("Expected SWaT A6 timestamp column 't_stamp'")
    frame["t_stamp"] = pd.to_datetime(frame["t_stamp"])
    frame = frame.sort_values("t_stamp").reset_index(drop=True)
    return frame


def parse_clock(value: str) -> time:
    return datetime.strptime(value, "%H:%M:%S").time()


def build_process_labels(timestamps: pd.Series) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=np.float32)
    clock = timestamps.dt.time
    for start_text, end_text in PROCESS_ATTACK_WINDOWS:
        start = parse_clock(start_text)
        end = parse_clock(end_text)
        mask = clock.between(start, end)
        labels[mask.to_numpy()] = 1.0
    return labels


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = frame.drop(columns=["t_stamp"]).copy()
    feature_frame.columns = [normalize_feature_name(column) for column in feature_frame.columns]
    feature_frame = feature_frame.replace({"Inactive": 0.0, "Active": 1.0}).infer_objects(copy=False)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    duplicated = feature_frame.columns[feature_frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate normalized SWaT A6 feature names: {duplicated}")
    return feature_frame.astype(np.float32)


def infer_train_lens(timestamps: pd.Series) -> int:
    split_time = parse_clock(PROCESS_ATTACK_WINDOWS[0][0])
    return int((timestamps.dt.time < split_time).sum())


def write_long_csv(path: Path, values: pd.DataFrame, labels: np.ndarray) -> None:
    dates = np.arange(len(values), dtype=np.int64)
    first = True
    for column in values.columns:
        out = pd.DataFrame({"date": dates, "data": values[column].to_numpy(), "cols": column})
        out.to_csv(path, mode="w" if first else "a", index=False, header=first)
        first = False
    label_frame = pd.DataFrame({"date": dates, "data": labels, "cols": "label"})
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

    ordered_columns = BASE_METADATA_COLUMNS + EXTRA_METADATA_COLUMNS
    other_columns = [column for column in metadata.columns if column not in ordered_columns]
    metadata = metadata[ordered_columns + other_columns]
    metadata.to_csv(metadata_path, index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = args.metadata.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_swat_a6_excel(raw_dir / "Dec2019.xlsx")
    labels = build_process_labels(frame["t_stamp"])
    features = build_feature_frame(frame)
    train_lens = infer_train_lens(frame["t_stamp"])

    if labels[:train_lens].sum() != 0:
        raise ValueError("SWaT A6 inferred train segment contains process attack labels")

    output_path = output_dir / args.output_name
    write_long_csv(output_path, features, labels)

    row = {
        "file_name": args.output_name,
        "length": int(len(features)),
        "dataset_name": "SWAT_A6",
        "size": "small" if len(features) <= 100000 else "large",
        "freq": "second",
        "if_univariate": False,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
        "train_lens": int(train_lens),
        "n_features": int(features.shape[1]),
        "source_dataset": "SWaT",
        "source_version": "A6_Dec_2019",
        "pattern": (
            "label=process_disrupt_sensor_actuator_only;"
            "train=before_12:30:00;source=Log.docx"
        ),
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
    logging.info("Output: %s", output_path)
    logging.info("Updated metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
