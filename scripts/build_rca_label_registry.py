#!/usr/bin/env python3
"""Build the unified RCA label registry from curated metadata files.

The registry is intentionally conservative: it only records labels that already
exist in curated metadata. Datasets without trustworthy root-cause annotation
should be absent until their label source is inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "root_cause_meta"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "anomaly_detect" / "rca_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def encode_list(values: list[str]) -> str:
    return ";".join(str(value) for value in values if str(value))


def rows_from_meta(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    series_name = meta["series_name"]
    dataset = series_name.rsplit(".", 1)[0]
    task = meta.get("task", "")
    rows = []
    for event in meta.get("events", []):
        root_groups = event.get("root_groups", [])
        root_variables = event.get("root_variables", [])
        rows.append(
            {
                "dataset": dataset,
                "file": series_name,
                "event_id": int(event["event_id"]),
                "event_start": int(event["start"]),
                "event_end": int(event["end"]),
                "subsystem_root": encode_list(root_groups),
                "variable_roots": encode_list(root_variables),
                "granularity": "variable" if root_variables else "subsystem",
                "task": task,
                "time_index": meta.get("time_index", "relative_to_test_segment"),
                "source": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.meta_dir.glob("*.json")):
        rows.extend(rows_from_meta(path))

    if not rows:
        raise SystemExit(f"No RCA metadata JSON files found in {args.meta_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "event_id"]).to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(rows)} events")


if __name__ == "__main__":
    main()
