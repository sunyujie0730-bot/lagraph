#!/usr/bin/env python3
"""Import verified RCA label-source templates into the canonical registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"
DEFAULT_LABEL_SOURCE_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "label_sources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_LABEL_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def normalize_roots(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return ";".join(part.strip() for part in str(value).replace(",", ";").split(";") if part.strip())


def rows_from_verified_sources(source_dir: Path) -> list[dict]:
    rows = []
    source_files = [
        source_dir / "swat_attack_targets_template.csv",
        source_dir / "te_mm_fault_mapping_template.csv",
        source_dir / "wadi_attack_targets_template.csv",
    ]
    for source_path in source_files:
        if not source_path.exists():
            continue
        frame = pd.read_csv(source_path)
        if "label_status" not in frame.columns:
            continue
        verified = frame.loc[frame["label_status"].astype(str).str.lower().eq("verified")].copy()
        for _, row in verified.iterrows():
            file_name = row["file"]
            subsystem_root = normalize_roots(row.get("subsystem_root", ""))
            variable_roots = normalize_roots(row.get("variable_roots", ""))
            if not subsystem_root and not variable_roots:
                continue
            event_id = int(row["event_id"]) if "event_id" in row and not pd.isna(row["event_id"]) else 1
            rows.append(
                {
                    "dataset": str(file_name).rsplit(".", 1)[0],
                    "file": file_name,
                    "event_id": event_id,
                    "event_start": int(row["event_start"]),
                    "event_end": int(row["event_end"]),
                    "subsystem_root": subsystem_root,
                    "variable_roots": variable_roots,
                    "granularity": "variable" if variable_roots else "subsystem",
                    "task": "root_cause_ranking",
                    "time_index": "relative_to_test_segment",
                    "source": str(source_path.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry) if args.registry.exists() else pd.DataFrame()
    new_rows = rows_from_verified_sources(args.source_dir)
    if not new_rows:
        print("No verified RCA source rows found; registry unchanged.")
        return

    new_frame = pd.DataFrame(new_rows)
    if not registry.empty:
        keys = set(zip(new_frame["file"], new_frame["event_id"]))
        registry = registry[
            ~registry.apply(lambda row: (row["file"], row["event_id"]) in keys, axis=1)
        ]
    merged = pd.concat([registry, new_frame], ignore_index=True)
    merged.sort_values(["file", "event_id"]).to_csv(args.output, index=False)
    print(f"Imported {len(new_rows)} verified rows into {args.output}")


if __name__ == "__main__":
    main()
