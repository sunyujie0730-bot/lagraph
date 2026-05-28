#!/usr/bin/env python3
"""Build WADI A2 RCA label-source templates from converted labels and attack table notes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WADI = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "WADI_A2_2019_ds10.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "label_sources"
DEFAULT_METADATA = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"

ATTACK_TABLE = [
    {
        "attack_identifier": "1",
        "target_tags": "1_MV_001_STATUS",
        "variable_roots": "1_MV_001_STATUS;1_LT_001_PV;1_FIT_001_PV",
        "subsystem_root": "WADI_P1",
        "attack_type": "Maliciously open motorized valve MV001; overflow should affect LT001/FIT001",
    },
    {
        "attack_identifier": "2",
        "target_tags": "1_FIT_001_PV",
        "variable_roots": "1_FIT_001_PV",
        "subsystem_root": "WADI_P1",
        "attack_type": "False reading / shut off flow indication transmitter FIT001",
    },
    {
        "attack_identifier": "3-4",
        "target_tags": "2_LT_002_PV;1_AIT_001_PV",
        "variable_roots": "2_LT_002_PV;1_AIT_001_PV",
        "subsystem_root": "WADI_P2",
        "attack_type": "Stealthy manipulation to drain elevated reservoir and alter quality reading",
    },
    {
        "attack_identifier": "5",
        "target_tags": "2_MCV_101_CO;2_MCV_201_CO;2_MCV_301_CO;2_MCV_401_CO;2_MCV_501_CO;2_MCV_601_CO",
        "variable_roots": "2_MCV_101_CO;2_MCV_201_CO;2_MCV_301_CO;2_MCV_401_CO;2_MCV_501_CO;2_MCV_601_CO",
        "subsystem_root": "WADI_P2",
        "attack_type": "Turn off consumer valves",
    },
    {
        "attack_identifier": "6",
        "target_tags": "2_MCV_101_CO;2_MCV_201_CO",
        "variable_roots": "2_MCV_101_CO;2_MCV_201_CO",
        "subsystem_root": "WADI_P2",
        "attack_type": "Maliciously turn on consumer valves MCV101/MCV201",
    },
    {
        "attack_identifier": "7",
        "target_tags": "1_AIT_002_PV;2_MV_003_STATUS",
        "variable_roots": "1_AIT_002_PV;2_MV_003_STATUS",
        "subsystem_root": "WADI_P1",
        "attack_type": "Contaminated-water attack using AIT002 and MV003",
    },
    {
        "attack_identifier": "8",
        "target_tags": "2_MCV_007_CO",
        "variable_roots": "2_MCV_007_CO;2_PIT_002_PV;2_FIT_002_PV",
        "subsystem_root": "WADI_P2",
        "attack_type": "Open MCV007 to create leakage; reflected in PIT002/FIT002",
    },
    {
        "attack_identifier": "9",
        "target_tags": "1_P_006_STATUS",
        "variable_roots": "1_P_006_STATUS",
        "subsystem_root": "WADI_P1",
        "attack_type": "Turn on pump P006 to cause pipe burst",
    },
    {
        "attack_identifier": "10",
        "target_tags": "1_MV_001_STATUS",
        "variable_roots": "1_MV_001_STATUS;1_P_001_STATUS;1_P_002_STATUS",
        "subsystem_root": "WADI_P1",
        "attack_type": "Damage MV001 and raw-water pump to drain elevated reservoir",
    },
    {
        "attack_identifier": "11",
        "target_tags": "2_MCV_007_CO",
        "variable_roots": "2_MCV_007_CO;2_PIT_002_PV;2_FIT_002_PV",
        "subsystem_root": "WADI_P2",
        "attack_type": "Similar to attack 8",
    },
    {
        "attack_identifier": "12",
        "target_tags": "2_MCV_007_CO",
        "variable_roots": "2_MCV_007_CO;2_PIT_002_PV;2_FIT_002_PV",
        "subsystem_root": "WADI_P2",
        "attack_type": "Similar to attack 8",
    },
    {
        "attack_identifier": "13",
        "target_tags": "2_PIC_003_SP",
        "variable_roots": "2_PIC_003_SP;2_FIT_003_PV;2_PIT_003_PV",
        "subsystem_root": "WADI_P2",
        "attack_type": "Reduce booster set-point pressure; reflected in FIT003/PIT003",
    },
    {
        "attack_identifier": "14",
        "target_tags": "",
        "variable_roots": "",
        "subsystem_root": "WADI_P1",
        "attack_type": "Stop chemical dosing to raw water",
    },
    {
        "attack_identifier": "15",
        "target_tags": "2_LT_002_PV;1_AIT_001_PV",
        "variable_roots": "2_LT_002_PV;1_AIT_001_PV",
        "subsystem_root": "WADI_P2",
        "attack_type": "Stealthy attack; inverse of attack 3",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wadi", type=Path, default=DEFAULT_WADI)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--chunk-size", type=int, default=500000)
    return parser.parse_args()


def read_label_series(path: Path, chunk_size: int) -> pd.Series:
    labels: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=["date", "data", "cols"], chunksize=chunk_size):
        label_chunk = chunk[chunk["cols"] == "label"][["date", "data"]]
        if not label_chunk.empty:
            labels.append(label_chunk)
    if not labels:
        raise ValueError(f"No label rows found in {path}")
    label_frame = pd.concat(labels, ignore_index=True)
    return label_frame.sort_values("date").set_index("date")["data"].astype(int)


def segments_from_labels(labels: pd.Series) -> list[tuple[int, int]]:
    arr = (labels.to_numpy() > 0).astype(int)
    segments: list[tuple[int, int]] = []
    prev = 0
    start = 0
    for idx, value in enumerate(arr):
        if value == 1 and prev == 0:
            start = idx
        if value == 0 and prev == 1:
            segments.append((start, idx - 1))
        prev = value
    if prev == 1:
        segments.append((start, len(arr) - 1))
    return segments


def read_train_lens(metadata_path: Path, file_name: str) -> int:
    metadata = pd.read_csv(metadata_path)
    rows = metadata.loc[metadata["file_name"] == file_name]
    if rows.empty:
        raise ValueError(f"{file_name} is missing from {metadata_path}")
    value = rows.iloc[0].get("train_lens")
    if pd.isna(value):
        raise ValueError(f"{file_name} has no train_lens in {metadata_path}")
    return int(value)


def build_stage_map(feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for tag in feature_names:
        if tag == "label":
            continue
        if tag.startswith("1_"):
            stage = "WADI_P1"
        elif tag.startswith(("2_", "2A_", "2B_")):
            stage = "WADI_P2"
        elif tag.startswith("3_"):
            stage = "WADI_P3"
        else:
            stage = "WADI_PLANT"
        rows.append({"tag": tag, "subsystem": stage, "source": "tag_prefix_rule", "label_status": "needs_review"})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.wadi.exists():
        raise FileNotFoundError(args.wadi)

    file_name = args.wadi.name
    train_lens = read_train_lens(args.metadata, file_name)
    labels = read_label_series(args.wadi, args.chunk_size)
    segments = [(start - train_lens, end - train_lens) for start, end in segments_from_labels(labels)]
    if any(start < 0 for start, _ in segments):
        raise ValueError("WADI attack segment starts before train_lens; check metadata split")
    if len(segments) != len(ATTACK_TABLE):
        raise ValueError(
            f"WADI label segments ({len(segments)}) do not match attack-table rows ({len(ATTACK_TABLE)})"
        )

    rows = []
    for idx, ((start, end), attack) in enumerate(zip(segments, ATTACK_TABLE), start=1):
        rows.append(
            {
                "file": file_name,
                "event_id": idx,
                "attack_identifier": attack["attack_identifier"],
                "event_start": start,
                "event_end": end,
                "target_tags": attack["target_tags"],
                "subsystem_root": attack["subsystem_root"],
                "variable_roots": attack["variable_roots"],
                "attack_type": attack["attack_type"],
                "source": "table_WADI.pdf",
                "label_status": "needs_review",
            }
        )

    attack_template = args.out_dir / "wadi_attack_targets_template.csv"
    pd.DataFrame(rows).to_csv(attack_template, index=False)

    cols = []
    for chunk in pd.read_csv(args.wadi, usecols=["cols"], chunksize=args.chunk_size):
        cols.extend(chunk["cols"].drop_duplicates().tolist())
    stage_map = args.out_dir / "wadi_tag_stage_map.csv"
    build_stage_map(sorted(set(cols))).to_csv(stage_map, index=False)

    print(f"Wrote {attack_template}")
    print(f"Wrote {stage_map}")


if __name__ == "__main__":
    main()
