#!/usr/bin/env python3
"""Run paper-oriented LaGraph experiments and summarize outputs.

This queue is intentionally conservative: it uses short 5-8 epoch screening runs,
keeps RCA exports in lite mode, and writes all artifacts under logs/ and result/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUN_SINGLE = PROJECT_ROOT / "ts_benchmark" / "run_single.py"
LOG_DIR = PROJECT_ROOT / "logs" / "architecture_experiments"
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis"
LABEL_DIR = PROJECT_ROOT / "result" / "label" / "LaGraph"
RCA_DIR = PROJECT_ROOT / "result" / "rca"


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    purpose: str
    dataset: str
    profile: str
    epochs: int
    export_rca: bool = False
    eval_config: str = "all_detect_label_config.json"
    extra_args: tuple[str, ...] = ()


CORE_EXPERIMENTS = [
    Experiment(
        "E01_wadi_soft_hier_e8",
        "WADI variable-level RCA main candidate; confirm detection/RCA under lite export.",
        "WADI_A1_2017_ds10.csv",
        "soft-hierarchical-rca",
        8,
        export_rca=True,
    ),
    Experiment(
        "E02_swat_soft_hier_e8",
        "SWaT variable-level RCA main candidate; confirm detection/RCA under lite export.",
        "SWAT_A1A2_Physical_v1.csv",
        "soft-hierarchical-rca",
        8,
        export_rca=True,
    ),
    Experiment(
        "E03_hai1_soft_hier_e5",
        "HAI subsystem-level extension check; short run because labels are subsystem-oriented.",
        "HAI_21_03_test1.csv",
        "soft-hierarchical-rca",
        5,
        export_rca=True,
        eval_config="unfixed_detect_label_config.json",
    ),
    Experiment(
        "E04_wadi_channel_only_e5",
        "Detection ablation: channel graph only on WADI.",
        "WADI_A1_2017_ds10.csv",
        "channel-only",
        5,
    ),
    Experiment(
        "E05_wadi_temporal_only_e5",
        "Detection ablation: temporal graph only on WADI.",
        "WADI_A1_2017_ds10.csv",
        "temporal-only",
        5,
    ),
    Experiment(
        "E06_wadi_full_e5",
        "Detection ablation: original full dual graph on WADI.",
        "WADI_A1_2017_ds10.csv",
        "full",
        5,
    ),
    Experiment(
        "E07_swat_channel_only_e5",
        "Detection ablation: channel graph only on SWaT.",
        "SWAT_A1A2_Physical_v1.csv",
        "channel-only",
        5,
    ),
    Experiment(
        "E08_swat_temporal_only_e5",
        "Detection ablation: temporal graph only on SWaT.",
        "SWAT_A1A2_Physical_v1.csv",
        "temporal-only",
        5,
    ),
    Experiment(
        "E09_swat_full_e5",
        "Detection ablation: original full dual graph on SWaT.",
        "SWAT_A1A2_Physical_v1.csv",
        "full",
        5,
    ),
]


FOLLOWUP_EXPERIMENTS = [
    Experiment(
        "F01_wadi_channel_only_rca_e8",
        "Compatibility check: WADI channel-only detection configuration with RCA export.",
        "WADI_A1_2017_ds10.csv",
        "channel-only",
        8,
        export_rca=True,
    ),
    Experiment(
        "F02_wadi_temporal_only_rca_e8",
        "Compatibility check: WADI temporal-only detection configuration with RCA export.",
        "WADI_A1_2017_ds10.csv",
        "temporal-only",
        8,
        export_rca=True,
    ),
    Experiment(
        "F03_wadi_full_rca_e8",
        "Compatibility check: WADI full dual-graph detection configuration with RCA export.",
        "WADI_A1_2017_ds10.csv",
        "full",
        8,
        export_rca=True,
    ),
    Experiment(
        "F04_swat_channel_only_rca_e8",
        "Compatibility check: SWaT channel-only detection configuration with RCA export.",
        "SWAT_A1A2_Physical_v1.csv",
        "channel-only",
        8,
        export_rca=True,
    ),
    Experiment(
        "F05_swat_temporal_only_rca_e8",
        "Compatibility check: SWaT temporal-only detection configuration with RCA export.",
        "SWAT_A1A2_Physical_v1.csv",
        "temporal-only",
        8,
        export_rca=True,
    ),
    Experiment(
        "F06_swat_full_rca_e8",
        "Compatibility check: SWaT full dual-graph detection configuration with RCA export.",
        "SWAT_A1A2_Physical_v1.csv",
        "full",
        8,
        export_rca=True,
    ),
]


EXPERIMENT_SUITES = {
    "core": CORE_EXPERIMENTS,
    "followup": FOLLOWUP_EXPERIMENTS,
    "all": CORE_EXPERIMENTS + FOLLOWUP_EXPERIMENTS,
}


def list_files(root: Path, pattern: str) -> set[Path]:
    if not root.exists():
        return set()
    return set(root.rglob(pattern))


def newest_after(before: set[Path], root: Path, pattern: str) -> list[Path]:
    after = list_files(root, pattern)
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    return created


def command_for(exp: Experiment) -> list[str]:
    command = [
        str(PYTHON),
        str(RUN_SINGLE),
        "--epochs",
        str(exp.epochs),
        "--datasets",
        exp.dataset,
        "--arch-profile",
        exp.profile,
        "--num-workers",
        "2",
        "--prefetch-factor",
        "2",
        "--eval-config",
        exp.eval_config,
    ]
    if exp.export_rca:
        command.extend(["--export-rca", "--rca-export-lite", "--rca-export-top-k", "20"])
    command.extend(exp.extra_args)
    return command


def summarize_label_csv(path: Path, dataset: str) -> dict:
    df = pd.read_csv(path)
    if "file_name" in df.columns:
        df = df[df["file_name"] == dataset]
    if df.empty:
        return {"label_csv": str(path), "error": f"no rows for {dataset}"}

    def best(metric: str) -> dict:
        if metric not in df.columns:
            return {}
        row = df.loc[df[metric].astype(float).idxmax()]
        return {
            "ratio": row.get("typical_anomaly_ratio", ""),
            "raw_f1": float(row.get("f_score", 0.0)),
            "adjusted_f1": float(row.get("adjust_f_score", 0.0)),
            "affiliation_f1": float(row.get("affiliation_f", 0.0)),
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
        }

    return {
        "label_csv": str(path),
        "best_raw": best("f_score"),
        "best_adjusted": best("adjust_f_score"),
        "best_affiliation": best("affiliation_f"),
        "fit_time_sec": float(df["fit_time"].iloc[0]) if "fit_time" in df.columns else None,
        "inference_time_sec": float(df["inference_time"].iloc[0]) if "inference_time" in df.columns else None,
    }


def run_eval(command: list[str], save_csv: Path, log_file: Path) -> dict:
    started = time.time()
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "save_csv": str(save_csv),
    }


def evaluate_rca(rca_path: Path, exp: Experiment, out_prefix: str, log_file: Path) -> dict:
    results = {}
    hierarchical_csv = ANALYSIS_DIR / f"{out_prefix}_hierarchical_pred.csv"
    flat_csv = ANALYSIS_DIR / f"{out_prefix}_flat_channel_pred.csv"
    static_csv = ANALYSIS_DIR / f"{out_prefix}_static_pred.csv"
    random_csv = ANALYSIS_DIR / f"{out_prefix}_random_pred.csv"

    base = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "evaluate_hierarchical_rca.py"),
        "--rca",
        str(rca_path),
        "--event-source",
        "predicted",
        "--prediction-key",
        "15",
        "--method-name",
        exp.profile,
        "--summary-only",
        "--save-csv",
        str(hierarchical_csv),
    ]
    results["hierarchical"] = run_eval(base, hierarchical_csv, log_file)

    flat = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "evaluate_rca.py"),
        "--rca",
        str(rca_path),
        "--scope",
        "channel",
        "--event-source",
        "predicted",
        "--prediction-key",
        "15",
        "--method-name",
        exp.profile,
        "--save-csv",
        str(flat_csv),
    ]
    results["flat_channel"] = run_eval(flat, flat_csv, log_file)

    random = flat.copy()
    random[random.index("--method-name") + 1] = "random"
    random.extend(["--baseline", "random", "--random-trials", "1000"])
    random[random.index(str(flat_csv))] = str(random_csv)
    results["random"] = run_eval(random, random_csv, log_file)

    static = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "evaluate_static_rca_baselines.py"),
        "--series",
        exp.dataset,
        "--rca",
        str(rca_path),
        "--scope",
        "channel",
        "--event-source",
        "predicted",
        "--prediction-key",
        "15",
        "--save-csv",
        str(static_csv),
    ]
    if (PROJECT_ROOT / "scripts" / "evaluate_static_rca_baselines.py").exists():
        results["static"] = run_eval(static, static_csv, log_file)
    return results


def extract_mean_metrics(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "series_name" in df.columns and (df["series_name"] == "MEAN").any():
        row = df[df["series_name"] == "MEAN"].iloc[-1]
    else:
        numeric = df.select_dtypes("number")
        if numeric.empty:
            return {}
        row = numeric.mean()
    wanted = [
        "matched",
        "subsystem_MRR",
        "subsystem_Hit@1",
        "subsystem_Hit@3",
        "subsystem_Hit@5",
        "flat_variable_MRR",
        "flat_variable_Hit@1",
        "flat_variable_Hit@3",
        "flat_variable_Hit@5",
        "flat_variable_PR@1",
        "flat_variable_PR@3",
        "flat_variable_PR@5",
        "flat_variable_MAP@3",
        "flat_variable_MAP@5",
        "conditional_variable_MRR",
        "conditional_variable_Hit@1",
        "conditional_variable_Hit@3",
        "conditional_variable_Hit@5",
    ]
    return {key: float(row[key]) for key in wanted if key in row and pd.notna(row[key])}


def write_summary(markdown_path: Path, records: list[dict]) -> None:
    lines = [
        "# Paper Experiment Queue Summary",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment List",
        "",
    ]
    for item in records:
        exp = item["experiment"]
        status = "OK" if item.get("returncode") == 0 else "FAILED"
        lines.append(
            f"- `{exp['exp_id']}` [{status}] `{exp['dataset']}` `{exp['profile']}` "
            f"epochs={exp['epochs']} export_rca={exp['export_rca']} - {exp['purpose']}"
        )
    lines.extend(["", "## Detection Summary", ""])
    lines.append("| ID | Dataset | Profile | Best Raw F1 | Best Adj F1 | Best Aff F1 | Fit(s) | Infer(s) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for item in records:
        exp = item["experiment"]
        det = item.get("detection", {})
        raw = det.get("best_raw", {}).get("raw_f1")
        adj = det.get("best_adjusted", {}).get("adjusted_f1")
        aff = det.get("best_affiliation", {}).get("affiliation_f1")
        fit = det.get("fit_time_sec")
        infer = det.get("inference_time_sec")
        lines.append(
            f"| {exp['exp_id']} | {exp['dataset']} | {exp['profile']} | "
            f"{raw if raw is not None else ''} | {adj if adj is not None else ''} | "
            f"{aff if aff is not None else ''} | {fit if fit is not None else ''} | "
            f"{infer if infer is not None else ''} |"
        )
    lines.extend(["", "## RCA Summary", ""])
    lines.append("| ID | Dataset | Profile | Matched | Subsys MRR | Subsys H@1 | Flat Var MRR | Flat Var H@1 | Cond Var MRR |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for item in records:
        exp = item["experiment"]
        metrics = item.get("rca_metrics", {})
        lines.append(
            f"| {exp['exp_id']} | {exp['dataset']} | {exp['profile']} | "
            f"{metrics.get('matched', '')} | {metrics.get('subsystem_MRR', '')} | "
            f"{metrics.get('subsystem_Hit@1', '')} | {metrics.get('flat_variable_MRR', '')} | "
            f"{metrics.get('flat_variable_Hit@1', '')} | {metrics.get('conditional_variable_MRR', '')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- TE is not in the current local DETECT_META/data list, so it is left pending until the dataset is restored.")
    lines.append("- RCA exports use lite mode to avoid 100MB-1GB JSON files.")
    lines.append("- Short 5-8 epoch runs are screening runs; any clear winner should be confirmed at 15-20 epochs.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=sorted(EXPERIMENT_SUITES),
        default="core",
        help="Experiment suite to run.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected experiment IDs.")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"paper_queue_{run_id}.log"
    jsonl_file = ANALYSIS_DIR / f"paper_experiment_queue_{run_id}.jsonl"
    markdown_file = ANALYSIS_DIR / f"paper_experiment_summary_{run_id}.md"

    selected = EXPERIMENT_SUITES[args.suite]
    if args.only:
        wanted = set(args.only)
        selected = [exp for exp in selected if exp.exp_id in wanted]
        missing = wanted - {exp.exp_id for exp in selected}
        if missing:
            raise SystemExit(f"Unknown experiment IDs: {sorted(missing)}")

    records: list[dict] = []
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"Paper experiment queue started at {datetime.now()}\n")
        log.write(f"Python: {PYTHON}\n")
        log.write(f"Project: {PROJECT_ROOT}\n")

    for exp in selected:
        label_before = list_files(LABEL_DIR, "*.csv")
        rca_before = list_files(RCA_DIR, "*_rca.json")
        command = command_for(exp)
        started = time.time()
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 88 + "\n")
            log.write(f"{exp.exp_id}: {exp.purpose}\n")
            log.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = round(time.time() - started, 3)
        labels = newest_after(label_before, LABEL_DIR, "*.csv")
        rcas = newest_after(rca_before, RCA_DIR, "*_rca.json")
        record = {
            "experiment": exp.__dict__,
            "command": command,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "labels": [str(path) for path in labels],
            "rcas": [str(path) for path in rcas],
            "log_file": str(log_file),
        }
        if labels:
            record["detection"] = summarize_label_csv(labels[-1], exp.dataset)
        if rcas:
            evals = evaluate_rca(rcas[-1], exp, f"{run_id}_{exp.exp_id}", log_file)
            record["rca_evaluations"] = evals
            hier_csv = ANALYSIS_DIR / f"{run_id}_{exp.exp_id}_hierarchical_pred.csv"
            record["rca_metrics"] = extract_mean_metrics(hier_csv)
        records.append(record)
        with jsonl_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_summary(markdown_file, records)
        if proc.returncode != 0:
            with log_file.open("a", encoding="utf-8") as log:
                log.write(f"\nStopping queue because {exp.exp_id} failed with {proc.returncode}.\n")
            break

    write_summary(markdown_file, records)
    print(f"Log: {log_file}")
    print(f"JSONL: {jsonl_file}")
    print(f"Summary: {markdown_file}")
    return 0 if all(item.get("returncode") == 0 for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
