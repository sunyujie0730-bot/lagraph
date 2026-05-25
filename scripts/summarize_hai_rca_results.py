#!/usr/bin/env python3
"""Summarize HAI detection and subsystem-level RCA CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca-root", type=Path, default=PROJECT_ROOT / "result" / "rca")
    parser.add_argument("--save-csv", type=Path, default=PROJECT_ROOT / "result" / "rca" / "hai_rca_summary.csv")
    return parser.parse_args()


def load_summary(path: Path, method: str) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["series_name"] == "MEAN"]
    if row.empty:
        metric_cols = [c for c in df.columns if c.startswith(("Hit@", "NDCG@")) or c == "MRR"]
        values = df[metric_cols].mean().to_dict()
    else:
        values = row.iloc[0].to_dict()
    values["method"] = method
    return values


def main() -> None:
    args = parse_args()
    rows = []
    for dataset_dir in sorted(args.rca_root.glob("HAI_21_03_test*")):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for filename, method in [
            ("lagraph_group_eval.csv", "LaGraph"),
            ("lagraph_contrast075_group_eval.csv", "LaGraph-contrast"),
            ("zscore_group_eval.csv", "z-score"),
            ("random_group_eval.csv", "random"),
            ("latest_group_eval.csv", "LaGraph"),
        ]:
            summary = load_summary(dataset_dir / filename, method)
            if summary is None:
                continue
            summary["dataset"] = dataset
            rows.append(summary)

    if not rows:
        raise SystemExit("No RCA evaluation CSVs found.")

    out = pd.DataFrame(rows)
    keep = [
        "dataset",
        "method",
        "MRR",
        "Hit@1",
        "NDCG@1",
        "Hit@3",
        "NDCG@3",
        "Hit@5",
        "NDCG@5",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].drop_duplicates(subset=["dataset", "method"], keep="first")
    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.save_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
