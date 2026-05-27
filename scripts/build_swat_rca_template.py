#!/usr/bin/env python3
"""Build a SWaT RCA annotation template from binary anomaly labels.

The SWaT CSV in this project only contains point-wise anomaly labels. It does
not contain attack target tags, so this script deliberately creates a template
instead of writing ground-truth RCA labels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWAT = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "swat.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "label_sources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swat", type=Path, default=DEFAULT_SWAT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def infer_stage(tag: str) -> str:
    match = re.search(r"(\d{3})", tag)
    if not match:
        return ""
    return f"P{match.group(1)[0]}"


def binary_segments(values: list[int]) -> list[tuple[int, int]]:
    segments = []
    in_segment = False
    for idx, value in enumerate(values):
        active = bool(value)
        if active and not in_segment:
            start = idx
            in_segment = True
        elif in_segment and not active:
            segments.append((start, idx))
            in_segment = False
    if in_segment:
        segments.append((start, len(values)))
    return segments


def read_long_swat(path: Path, chunk_size: int) -> tuple[list[int], list[str]]:
    labels_by_date = {}
    feature_tags = set()
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=["date", "data", "cols"]):
        label_rows = chunk["cols"].eq("label")
        if label_rows.any():
            labels = chunk.loc[label_rows, ["date", "data"]]
            for date, value in labels.itertuples(index=False):
                labels_by_date[int(date)] = int(float(value))
        feature_tags.update(str(tag).strip() for tag in chunk.loc[~label_rows, "cols"].unique())

    if not labels_by_date:
        raise ValueError(f"No label rows found in {path}")
    max_date = max(labels_by_date)
    labels = [int(labels_by_date.get(idx, 0)) for idx in range(max_date + 1)]
    return labels, sorted(feature_tags)


def main() -> None:
    args = parse_args()
    labels, feature_tags = read_long_swat(args.swat, args.chunk_size)
    segments = binary_segments(labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    attack_template = args.out_dir / "swat_attack_targets_template.csv"
    stage_map = args.out_dir / "swat_tag_stage_map.csv"

    pd.DataFrame(
        [
            {
                "file": "swat.csv",
                "event_id": idx,
                "event_start": start,
                "event_end": end,
                "target_tags": "",
                "subsystem_root": "",
                "variable_roots": "",
                "attack_type": "",
                "source": "TODO: official SWaT attack list",
                "label_status": "needs_manual_source",
            }
            for idx, (start, end) in enumerate(segments, start=1)
        ]
    ).to_csv(attack_template, index=False)

    pd.DataFrame(
        [
            {
                "tag": tag,
                "stage": infer_stage(tag),
                "source": "tag numeric prefix heuristic; verify against SWaT process documentation",
            }
            for tag in feature_tags
        ]
    ).to_csv(stage_map, index=False)

    print(f"Wrote {attack_template} with {len(segments)} anomaly events")
    print(f"Wrote {stage_map} with {len(feature_tags)} feature tags")


if __name__ == "__main__":
    main()
