#!/usr/bin/env python3
"""Run RCA component ablations for exported LaGraph RCA reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_RCA = PROJECT_ROOT / "scripts" / "evaluate_rca.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, nargs="+", required=True, help="One or more *_rca.json files.")
    parser.add_argument("--scope", choices=["group", "channel"], default="group")
    parser.add_argument(
        "--event-source",
        choices=["true", "predicted", "both"],
        default="both",
        help="Whether to evaluate true-event RCA, predicted-event RCA, or both.",
    )
    parser.add_argument(
        "--prediction-key",
        nargs="+",
        default=["pot", "0.5", "1.0"],
        help="Prediction keys used when event-source includes predicted.",
    )
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "result" / "rca" / "ablation",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Combined CSV path. Defaults to <out-dir>/rca_ablation_summary.csv.",
    )
    return parser.parse_args()


def run_eval(args: list[str], csv_path: Path) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(EVALUATE_RCA), *args, "--save-csv", str(csv_path)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    df = pd.read_csv(csv_path)
    return df[df["series_name"] == "MEAN"].copy()


def dataset_name(rca_path: Path) -> str:
    return rca_path.parent.name if rca_path.parent.name else rca_path.stem.replace("_rca", "")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    summary_csv = args.summary_csv or out_dir / "rca_ablation_summary.csv"

    component_methods = [
        ("exported", ["--score-mode", "exported"]),
        ("base_only", ["--score-mode", "components", "--component-base-weight", "1.0"]),
        (
            "mechanism_only",
            [
                "--score-mode",
                "components",
                "--component-base-weight",
                "0.0",
                "--component-mechanism-weight",
                "1.0",
            ],
        ),
        (
            "base_plus_mechanism",
            [
                "--score-mode",
                "components",
                "--component-base-weight",
                "1.0",
                "--component-mechanism-weight",
                "1.0",
            ],
        ),
        (
            "mechanism_residual_only",
            [
                "--score-mode",
                "components",
                "--component-base-weight",
                "0.0",
                "--component-mechanism-residual-weight",
                "1.0",
            ],
        ),
        (
            "base_plus_mechanism_residual",
            [
                "--score-mode",
                "components",
                "--component-base-weight",
                "1.0",
                "--component-mechanism-residual-weight",
                "1.0",
            ],
        ),
    ]

    event_sources = ["true", "predicted"] if args.event_source == "both" else [args.event_source]
    rows = []
    for rca_path in args.rca:
        dataset = dataset_name(rca_path)
        for event_source in event_sources:
            prediction_keys = args.prediction_key if event_source == "predicted" else [None]
            for prediction_key in prediction_keys:
                for method_name, method_args in component_methods:
                    name_parts = [dataset, event_source, prediction_key or "true", method_name]
                    csv_path = out_dir / ("_".join(name_parts) + ".csv")
                    eval_args = [
                        "--rca",
                        str(rca_path),
                        "--scope",
                        args.scope,
                        "--event-source",
                        event_source,
                        "--method-name",
                        method_name,
                        *method_args,
                    ]
                    if prediction_key is not None:
                        eval_args.extend(["--prediction-key", prediction_key])
                    mean = run_eval(eval_args, csv_path)
                    mean.insert(0, "dataset", dataset)
                    mean.insert(1, "event_source_eval", event_source)
                    mean.insert(2, "prediction_key_eval", prediction_key or "")
                    mean.insert(3, "ablation", method_name)
                    rows.append(mean)

                random_name = "random"
                name_parts = [dataset, event_source, prediction_key or "true", random_name]
                csv_path = out_dir / ("_".join(name_parts) + ".csv")
                eval_args = [
                    "--rca",
                    str(rca_path),
                    "--scope",
                    args.scope,
                    "--event-source",
                    event_source,
                    "--baseline",
                    "random",
                    "--random-trials",
                    str(args.random_trials),
                    "--method-name",
                    random_name,
                ]
                if prediction_key is not None:
                    eval_args.extend(["--prediction-key", prediction_key])
                mean = run_eval(eval_args, csv_path)
                mean.insert(0, "dataset", dataset)
                mean.insert(1, "event_source_eval", event_source)
                mean.insert(2, "prediction_key_eval", prediction_key or "")
                mean.insert(3, "ablation", random_name)
                rows.append(mean)

    summary = pd.concat(rows, ignore_index=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    keep = [
        "dataset",
        "event_source_eval",
        "prediction_key_eval",
        "ablation",
        "matched",
        "MRR",
        "Hit@1",
        "Hit@3",
        "NDCG@3",
        "RCA_Delay@1",
        "RCA_Delay@3",
    ]
    keep = [col for col in keep if col in summary.columns]
    print(summary[keep].to_string(index=False))
    print(f"\nSaved {summary_csv}")


if __name__ == "__main__":
    main()
