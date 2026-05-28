#!/usr/bin/env python3
"""Build a SWaT RCA annotation template from binary labels and attack metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWAT = PROJECT_ROOT / "dataset" / "anomaly_detect" / "data" / "swat.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "label_sources"
DEFAULT_METADATA = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"
DEFAULT_ATTACK_LIST = (
    PROJECT_ROOT
    / "dataset"
    / "anomaly_detect"
    / "data"
    / "raw"
    / "SWAT"
    / "SWaT.A1_A2_Dec_2015"
    / "List_of_attacks_Final.xlsx"
)
SWAT_TEST_START = pd.Timestamp("2015-12-28 10:00:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swat", type=Path, default=DEFAULT_SWAT)
    parser.add_argument("--file-name", default=None)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--attack-list", type=Path, default=DEFAULT_ATTACK_LIST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def infer_stage(tag: str) -> str:
    match = re.search(r"(\d{3})", tag)
    if not match:
        return ""
    return f"P{match.group(1)[0]}"


def normalize_tag(tag: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(tag).upper())


def normalize_attack_tags(raw_value: object, feature_tags: set[str]) -> list[str]:
    if pd.isna(raw_value):
        return []
    tags = []
    for part in re.split(r"[,;/]+|\band\b", str(raw_value), flags=re.IGNORECASE):
        tag = normalize_tag(part)
        if not tag or "NOPHYSICAL" in tag:
            continue
        if tag == "DIT301" and "DPIT301" in feature_tags:
            tag = "DPIT301"
        tags.append(tag)
    return sorted(dict.fromkeys(tags))


def combine_end_time(start_time: pd.Timestamp, end_value: object) -> pd.Timestamp:
    if isinstance(end_value, dt.datetime):
        end_time = pd.Timestamp(end_value)
    elif isinstance(end_value, dt.time):
        end_time = pd.Timestamp(dt.datetime.combine(start_time.date(), end_value))
    else:
        end_time = pd.to_datetime(end_value)
    if end_time < start_time:
        end_time += pd.Timedelta(days=1)
    return end_time


def normalize_start_time(value: object) -> pd.Timestamp:
    start_time = pd.Timestamp(value)
    # The final five SWaT rows are stored in the workbook as 2015-01-02, but
    # the benchmark interval and binary labels place them on 2016-01-02.
    if start_time < SWAT_TEST_START:
        start_time = start_time.replace(year=start_time.year + 1)
    return start_time


def overlap_len(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start) + 1)


def read_official_attacks(path: Path, feature_tags: set[str]) -> list[dict]:
    if not path.exists():
        return []

    frame = pd.read_excel(path)
    required = {"Attack #", "Start Time", "End Time", "Attack Point", "Attack"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    rows = []
    physical = frame[
        frame["End Time"].notna()
        & ~frame["Attack Point"].astype(str).str.contains("No Physical", case=False, na=False)
    ].copy()
    for _, row in physical.iterrows():
        start_time = normalize_start_time(row["Start Time"])
        end_time = combine_end_time(start_time, row["End Time"])
        target_tags = normalize_attack_tags(row["Attack Point"], feature_tags)
        variable_roots = [tag for tag in target_tags if tag in feature_tags]
        subsystem_root = sorted(dict.fromkeys(stage for stage in map(infer_stage, target_tags) if stage))
        rows.append(
            {
                "attack_identifier": str(int(row["Attack #"])),
                "start_sec": int(round((start_time - SWAT_TEST_START).total_seconds())),
                "end_sec": int(round((end_time - SWAT_TEST_START).total_seconds())),
                "target_tags": target_tags,
                "variable_roots": variable_roots,
                "subsystem_root": subsystem_root,
                "attack_type": str(row["Attack"]).strip() if not pd.isna(row["Attack"]) else "",
            }
        )
    return rows


def annotate_segment(
    event_start: int,
    event_end: int,
    official_attacks: list[dict],
    attack_list_path: Path,
) -> dict:
    hits = []
    for attack in official_attacks:
        overlap = overlap_len(event_start, event_end, attack["start_sec"], attack["end_sec"])
        attack_duration = max(1, attack["end_sec"] - attack["start_sec"] + 1)
        if overlap / attack_duration >= 0.10 or overlap >= 60:
            hits.append(attack)
    if not hits:
        return {
            "attack_identifier": "",
            "target_tags": "",
            "subsystem_root": "",
            "variable_roots": "",
            "attack_type": "",
            "source": f"no overlapping physical row in {attack_list_path.name}",
            "label_status": "needs_review",
        }

    attack_ids = ";".join(hit["attack_identifier"] for hit in hits)
    target_tags = sorted(dict.fromkeys(tag for hit in hits for tag in hit["target_tags"]))
    variable_roots = sorted(dict.fromkeys(tag for hit in hits for tag in hit["variable_roots"]))
    subsystem_root = sorted(dict.fromkeys(stage for hit in hits for stage in hit["subsystem_root"]))
    attack_type = " | ".join(hit["attack_type"] for hit in hits if hit["attack_type"])
    return {
        "attack_identifier": attack_ids,
        "target_tags": ";".join(target_tags),
        "subsystem_root": ";".join(subsystem_root),
        "variable_roots": ";".join(variable_roots),
        "attack_type": attack_type,
        "source": f"{attack_list_path.name}: attack #{attack_ids}",
        "label_status": "verified",
    }


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


def read_train_lens(metadata_path: Path, file_name: str) -> int:
    if not metadata_path.exists():
        return 0
    metadata = pd.read_csv(metadata_path)
    rows = metadata.loc[metadata["file_name"] == file_name]
    if rows.empty or "train_lens" not in rows.columns:
        return 0
    value = rows.iloc[0]["train_lens"]
    if pd.isna(value):
        return 0
    return int(value)


def main() -> None:
    args = parse_args()
    file_name = args.file_name or args.swat.name
    train_lens = read_train_lens(args.metadata, file_name)
    labels, feature_tags = read_long_swat(args.swat, args.chunk_size)
    segments = [
        (start - train_lens, end - train_lens)
        for start, end in binary_segments(labels)
        if end >= train_lens
    ]
    if any(start < 0 for start, _ in segments):
        raise ValueError(f"{file_name} has an anomaly segment before train_lens={train_lens}")
    feature_tag_set = set(feature_tags)
    official_attacks = read_official_attacks(args.attack_list, feature_tag_set)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or "swat_attack_targets_template.csv"
    attack_template = args.out_dir / output_name
    stage_map = args.out_dir / "swat_tag_stage_map.csv"

    rows = []
    for idx, (start, end) in enumerate(segments, start=1):
        if official_attacks:
            annotation = annotate_segment(start, end, official_attacks, args.attack_list)
        else:
            annotation = {
                "attack_identifier": "",
                "target_tags": "",
                "subsystem_root": "",
                "variable_roots": "",
                "attack_type": "",
                "source": "TODO: official SWaT attack list",
                "label_status": "needs_manual_source",
            }
        rows.append(
            {
                "file": file_name,
                "event_id": idx,
                "event_start": start,
                "event_end": end,
                **annotation,
            }
        )
    pd.DataFrame(rows).to_csv(attack_template, index=False)

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

    verified = sum(1 for row in rows if row["label_status"] == "verified")
    print(f"Wrote {attack_template} with {len(segments)} anomaly events ({verified} verified)")
    print(f"Wrote {stage_map} with {len(feature_tags)} feature tags")


if __name__ == "__main__":
    main()
