#!/usr/bin/env python3
"""Summarize predicted-event RCA evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca-root", type=Path, default=PROJECT_ROOT / "result" / "rca")
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=PROJECT_ROOT / "result" / "rca" / "hai_predicted_rca_summary.csv",
    )
    return parser.parse_args()


def load_mean(path: Path) -> dict | None:
    df = pd.read_csv(path)
    row = df[df["series_name"] == "MEAN"]
    if row.empty:
        return None
    values = row.iloc[0].to_dict()
    return values


def main() -> None:
    args = parse_args()
    rows = []
    for dataset_dir in sorted(args.rca_root.glob("HAI_21_03_test*")):
        if not dataset_dir.is_dir():
            continue
        for csv_path in sorted(dataset_dir.glob("pred_*_group_eval.csv")):
            values = load_mean(csv_path)
            if values is None:
                continue
            key = csv_path.name.removeprefix("pred_").removesuffix("_group_eval.csv")
            values["dataset"] = dataset_dir.name
            values["prediction_key"] = key
            rows.append(values)

    if not rows:
        raise SystemExit("No predicted-event RCA CSVs found.")

    out = pd.DataFrame(rows)
    keep = [
        "dataset",
        "prediction_key",
        "matched",
        "MRR",
        "Hit@1",
        "NDCG@3",
        "NDCG@5",
        "RCA_Delay@1",
        "RCA_Delay@3",
        "RCA_Delay@5",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out[keep].sort_values(["dataset", "prediction_key"])
    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.save_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
