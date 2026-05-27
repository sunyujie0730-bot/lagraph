#!/usr/bin/env python3
"""Validate RCA label coverage and event bounds for anomaly datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECT_META = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "result" / "rca" / "rca_registry_coverage.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect-meta", type=Path, default=DEFAULT_DETECT_META)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--save-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def split_roots(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [item for item in str(value).split(";") if item]


def main() -> None:
    args = parse_args()
    detect_meta = pd.read_csv(args.detect_meta)
    registry = pd.read_csv(args.registry) if args.registry.exists() else pd.DataFrame()

    rows = []
    for _, meta_row in detect_meta.iterrows():
        file_name = meta_row["file_name"]
        length = int(meta_row["length"])
        label_rows = registry.loc[registry["file"] == file_name] if not registry.empty else pd.DataFrame()
        has_rca = not label_rows.empty

        invalid_bounds = 0
        empty_roots = 0
        granularities = []
        if has_rca:
            for _, label_row in label_rows.iterrows():
                start = int(label_row["event_start"])
                end = int(label_row["event_end"])
                if start < 0 or end <= start or end > length:
                    invalid_bounds += 1
                roots = split_roots(label_row.get("subsystem_root", "")) + split_roots(label_row.get("variable_roots", ""))
                if not roots:
                    empty_roots += 1
            granularities = sorted(label_rows["granularity"].dropna().astype(str).unique().tolist())

        rows.append(
            {
                "file": file_name,
                "dataset_name": meta_row.get("dataset_name", ""),
                "length": length,
                "has_rca_labels": bool(has_rca),
                "rca_events": int(len(label_rows)),
                "granularity": ";".join(granularities),
                "invalid_event_bounds": int(invalid_bounds),
                "empty_root_events": int(empty_roots),
                "recommended_role": "detection+RCA" if has_rca else "detection_only_until_labels_added",
            }
        )

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.save_csv, index=False)
    print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
