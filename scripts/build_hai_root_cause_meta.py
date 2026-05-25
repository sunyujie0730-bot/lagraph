#!/usr/bin/env python3
"""Build event-level HAI root-cause metadata from attack_P* labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_TEST = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "raw" / "HAI" / "hai-21.03" / "test1.csv.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "root_cause_meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-test", type=Path, default=DEFAULT_RAW_TEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--series-name", default="HAI_21_03_test1.csv")
    parser.add_argument("--test-files", nargs="+", default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=1)
    return parser.parse_args()


def binary_segments(mask) -> list[tuple[int, int]]:
    values = [bool(v) for v in mask]
    segments = []
    in_segment = False
    for idx, value in enumerate(values):
        if value and not in_segment:
            start = idx
            in_segment = True
        elif in_segment and not value:
            segments.append((start, idx))
            in_segment = False
    if in_segment:
        segments.append((start, len(values)))
    return segments


def build_one(raw_test: Path, output: Path, series_name: str, max_test_rows: int | None, downsample: int) -> None:
    label_cols = ["attack", "attack_P1", "attack_P2", "attack_P3"]
    labels = pd.read_csv(raw_test, usecols=label_cols, nrows=max_test_rows)
    if downsample > 1:
        labels = labels.iloc[::downsample].reset_index(drop=True)

    events = []
    for event_id, (start, end) in enumerate(binary_segments(labels["attack"].to_numpy()), start=1):
        segment = labels.iloc[start:end]
        root_groups = [
            group
            for group in ["P1", "P2", "P3"]
            if int(segment[f"attack_{group}"].max()) > 0
        ]
        events.append(
            {
                "event_id": event_id,
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "root_groups": root_groups,
                "root_variables": [],
            }
        )

    payload = {
        "series_name": series_name,
        "task": "subsystem_root_cause_ranking",
        "time_index": "relative_to_test_segment",
        "group_scope": ["P1", "P2", "P3"],
        "events": events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output} with {len(events)} events")


def main() -> None:
    args = parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")

    if args.test_files:
        raw_dir = args.raw_test.parent
        for test_file in args.test_files:
            stem = Path(test_file).name.replace(".csv.gz", "")
            series_name = f"HAI_21_03_{stem}.csv"
            output = args.output_dir / f"HAI_21_03_{stem}.json"
            build_one(
                raw_dir / test_file,
                output,
                series_name,
                args.max_test_rows,
                args.downsample,
            )
        return

    output = args.output or (args.output_dir / f"{Path(args.series_name).stem}.json")
    build_one(args.raw_test, output, args.series_name, args.max_test_rows, args.downsample)


if __name__ == "__main__":
    main()
