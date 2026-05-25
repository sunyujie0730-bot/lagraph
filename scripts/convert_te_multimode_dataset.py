#!/usr/bin/env python3
"""Convert multi-mode Tennessee Eastman .mat files to LaGraph long CSV format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover - clearer runtime error
    raise SystemExit("scipy is required to read TE .mat files. Install scipy in the conda env.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "raw" / "TE_multimode"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--faults", type=int, nargs="+", default=[1, 6, 14, 21])
    parser.add_argument("--feature-count", type=int, default=53)
    parser.add_argument(
        "--train-normal-ratio",
        type=float,
        default=0.8,
        help=(
            "Fraction of the normal d00 run used for training. The remaining normal "
            "points stay in the test segment as label=0 before the fault segment."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Convert all 6 modes and 28 faults.")
    return parser.parse_args()


def load_te_array(raw_dir: Path, mode: int, fault: int) -> np.ndarray:
    mat_path = raw_dir / f"M{mode}" / f"m{mode}d{fault:02d}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = loadmat(mat_path)
    key = f"m{mode}d{fault:02d}"
    if key not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        if len(keys) != 1:
            raise KeyError(f"Cannot identify data key in {mat_path}; keys={keys}")
        key = keys[0]

    arr = np.asarray(mat[key])
    if arr.ndim != 2:
        raise ValueError(f"{mat_path} must be 2-D, got shape={arr.shape}")
    return arr


def to_long_frame(values: np.ndarray, labels: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    dates = np.arange(values.shape[0], dtype=np.int64)
    wide = pd.DataFrame(values.astype(np.float32, copy=False), columns=feature_names)
    wide.insert(0, "date", dates)
    long_values = wide.melt(id_vars="date", var_name="cols", value_name="data")
    label_frame = pd.DataFrame(
        {
            "date": dates,
            "cols": "label",
            "data": labels.astype(np.float32, copy=False),
        }
    )
    return pd.concat([long_values[["date", "data", "cols"]], label_frame[["date", "data", "cols"]]], ignore_index=True)


def update_metadata(metadata_path: Path, rows: list[dict]) -> None:
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.DataFrame(columns=BASE_METADATA_COLUMNS)

    for column in BASE_METADATA_COLUMNS + EXTRA_METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = pd.NA

    file_names = {row["file_name"] for row in rows}
    metadata = metadata[~metadata["file_name"].isin(file_names)]
    metadata = pd.concat([metadata, pd.DataFrame(rows)], ignore_index=True)

    for column in ["length", "train_lens", "n_features", "mode", "fault_id"]:
        if column in metadata.columns:
            metadata[column] = pd.to_numeric(metadata[column], errors="coerce").astype("Int64")

    ordered_columns = BASE_METADATA_COLUMNS + EXTRA_METADATA_COLUMNS
    other_columns = [c for c in metadata.columns if c not in ordered_columns]
    metadata = metadata[ordered_columns + other_columns]
    metadata.to_csv(metadata_path, index=False)


def convert_one(
    raw_dir: Path,
    output_dir: Path,
    mode: int,
    fault: int,
    feature_count: int,
    train_normal_ratio: float,
) -> dict:
    normal = load_te_array(raw_dir, mode, 0)
    faulty = load_te_array(raw_dir, mode, fault)
    if normal.shape[1] < feature_count or faulty.shape[1] < feature_count:
        raise ValueError(
            f"feature-count={feature_count} exceeds available columns: "
            f"normal={normal.shape[1]}, faulty={faulty.shape[1]}"
        )

    normal = normal[:, :feature_count]
    faulty = faulty[:, :feature_count]
    values = np.vstack([normal, faulty])
    labels = np.concatenate(
        [
            np.zeros(normal.shape[0], dtype=np.float32),
            np.ones(faulty.shape[0], dtype=np.float32),
        ]
    )
    feature_names = [f"x{i:02d}" for i in range(feature_count)]
    long_frame = to_long_frame(values, labels, feature_names)
    train_lens = int(normal.shape[0] * train_normal_ratio)
    if train_lens <= 0 or train_lens >= normal.shape[0]:
        raise ValueError(
            "--train-normal-ratio must leave both a normal training segment and "
            "a normal test prefix. Use a value in (0, 1), such as 0.8."
        )

    file_name = f"TE_MM_M{mode}_d{fault:02d}.csv"
    output_path = output_dir / file_name
    long_frame.to_csv(output_path, index=False)

    length = int(values.shape[0])
    return {
        "file_name": file_name,
        "length": length,
        "dataset_name": "TE_multimode",
        "size": "large" if length > 10000 else "small",
        "freq": "other",
        "if_univariate": False,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
        "train_lens": int(train_lens),
        "n_features": int(feature_count),
        "mode": int(mode),
        "fault_id": int(fault),
        "source_dataset": "Multi-mode Tennessee Eastman",
        "source_version": "v1.0",
        "pattern": f"mode_{mode}_fault_{fault:02d};normal_train_ratio={train_normal_ratio:g}",
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = args.metadata.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = list(range(1, 7)) if args.all else args.modes
    faults = list(range(1, 29)) if args.all else args.faults

    rows = []
    for mode in modes:
        for fault in faults:
            row = convert_one(raw_dir, output_dir, mode, fault, args.feature_count, args.train_normal_ratio)
            rows.append(row)
            logging.info(
                "Converted %s: length=%s, train=%s, features=%s",
                row["file_name"],
                row["length"],
                row["train_lens"],
                row["n_features"],
            )

    update_metadata(metadata_path, rows)
    logging.info("Updated metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
