#!/usr/bin/env python3
"""Run mechanism-interference isolation experiments for LaGraph.

This queue is intentionally small. It tests whether SWaT degradation comes from
channel masked modeling or from using mechanism evidence in RCA, while checking
WADI under the same e8 protocol.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"D:\Anaconda3\envs\lagraph5070\python.exe")
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "mechanism_isolation"
LOG_DIR = PROJECT_ROOT / "logs" / "architecture_experiments"
RCA_ROOT = PROJECT_ROOT / "result" / "rca"
SUMMARY_PATH = ANALYSIS_DIR / "mechanism_isolation_summary.csv"
MD_PATH = ANALYSIS_DIR / "mechanism_isolation_summary.md"


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    dataset: str
    profile: str
    epochs: int = 8


EXPERIMENTS = [
    Experiment(
        "no_channel_masked_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-no-channel-masked-rca",
    ),
    Experiment(
        "no_channel_masked_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-no-channel-masked-rca",
    ),
    Experiment(
        "mechanism_train_only_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-mechanism-train-only-rca",
    ),
    Experiment(
        "mechanism_train_only_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-mechanism-train-only-rca",
    ),
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def newest_rca_after(dataset: str, started: float) -> Path | None:
    folder = RCA_ROOT / dataset_stem(dataset)
    if not folder.exists():
        return None
    candidates = [
        path
        for path in folder.glob("*_rca.json")
        if path.stat().st_mtime >= started - 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_command(cmd: list[str], log_path: Path) -> int:
    append_log(log_path, f"[{now()}] CMD: {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    append_log(log_path, f"[{now()}] RETURNCODE: {proc.returncode}")
    return int(proc.returncode)


def mean_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.mean(numeric_only=True).to_dict()


def evaluate(exp: Experiment, rca_json: Path, log_path: Path) -> dict[str, Any]:
    outputs: dict[str, Path] = {
        "channel": ANALYSIS_DIR / f"{exp.exp_id}_channel_pred_key15.csv",
        "group": ANALYSIS_DIR / f"{exp.exp_id}_group_pred_key15.csv",
        "hier": ANALYSIS_DIR / f"{exp.exp_id}_hier_mean_pred_key15.csv",
    }
    for scope in ("channel", "group"):
        run_command(
            [
                str(PYTHON),
                "scripts/evaluate_rca.py",
                "--rca",
                str(rca_json),
                "--event-source",
                "predicted",
                "--prediction-key",
                "15",
                "--scope",
                scope,
                "--method-name",
                exp.exp_id,
                "--save-csv",
                str(outputs[scope]),
            ],
            log_path,
        )
    run_command(
        [
            str(PYTHON),
            "scripts/evaluate_hierarchical_rca.py",
            "--rca",
            str(rca_json),
            "--event-source",
            "predicted",
            "--prediction-key",
            "15",
            "--group-aggregation",
            "mean",
            "--method-name",
            exp.exp_id,
            "--save-csv",
            str(outputs["hier"]),
        ],
        log_path,
    )

    channel = mean_csv(outputs["channel"])
    group = mean_csv(outputs["group"])
    hier = mean_csv(outputs["hier"])
    return {
        "exp_id": exp.exp_id,
        "dataset": exp.dataset,
        "profile": exp.profile,
        "epochs": exp.epochs,
        "rca_json": str(rca_json),
        "channel_csv": str(outputs["channel"]),
        "group_csv": str(outputs["group"]),
        "hierarchical_csv": str(outputs["hier"]),
        "channel_MRR": channel.get("MRR"),
        "channel_Hit@1": channel.get("Hit@1"),
        "channel_Hit@3": channel.get("Hit@3"),
        "channel_Hit@5": channel.get("Hit@5"),
        "group_MRR": group.get("MRR"),
        "group_Hit@1": group.get("Hit@1"),
        "group_Hit@3": group.get("Hit@3"),
        "group_Hit@5": group.get("Hit@5"),
        "conditional_variable_MRR": hier.get("conditional_variable_MRR"),
        "hierarchical_variable_MRR": hier.get("hierarchical_variable_MRR"),
        "true_coverage": hier.get("true_coverage"),
        "event_iou": hier.get("event_iou"),
    }


def read_existing_summary() -> list[dict[str, Any]]:
    if not SUMMARY_PATH.exists():
        return []
    with SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(rows: list[dict[str, Any]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    display_cols = [
        "exp_id",
        "dataset",
        "profile",
        "channel_MRR",
        "channel_Hit@1",
        "channel_Hit@3",
        "channel_Hit@5",
        "group_MRR",
        "conditional_variable_MRR",
        "hierarchical_variable_MRR",
    ]
    display = df[[c for c in display_cols if c in df.columns]].copy()
    for col in display.columns:
        if col not in {"exp_id", "dataset", "profile"}:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(
                lambda x: "-" if pd.isna(x) else f"{x:.4f}"
            )
    lines = [
        "# Mechanism Isolation Summary",
        "",
        f"Updated: {now()}",
        "",
        "Fixed protocol: epochs=8, predicted-event key=15, num-workers=2, inference-num-workers=0.",
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in display.columns) + " |")
    lines.extend(["", f"CSV: `{SUMMARY_PATH}`"])
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(exp: Experiment) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{exp.exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    started = time.time()
    cmd = [
        str(PYTHON),
        "-u",
        "ts_benchmark/run_single.py",
        "--epochs",
        str(exp.epochs),
        "--datasets",
        exp.dataset,
        "--arch-profile",
        exp.profile,
        "--export-rca",
        "--rca-export-lite",
        "--num-workers",
        "2",
        "--prefetch-factor",
        "2",
        "--inference-num-workers",
        "0",
        "--save-dir",
        f"label\\mechanism_isolation\\{exp.exp_id}",
    ]
    row: dict[str, Any] = {
        "exp_id": exp.exp_id,
        "dataset": exp.dataset,
        "profile": exp.profile,
        "epochs": exp.epochs,
        "status": "started",
        "log": str(log_path),
        "started_at": now(),
    }
    code = run_command(cmd, log_path)
    row["returncode"] = code
    row["finished_at"] = now()
    if code != 0:
        row["status"] = "train_failed"
        return row
    rca_json = newest_rca_after(exp.dataset, started)
    if not rca_json:
        row["status"] = "missing_rca"
        return row
    row.update(evaluate(exp, rca_json, log_path))
    row["status"] = "ok"
    return row


def main() -> None:
    rows = read_existing_summary()
    done = {row.get("exp_id") for row in rows if row.get("status") == "ok"}
    for exp in EXPERIMENTS:
        if exp.exp_id in done:
            continue
        row = run_experiment(exp)
        rows = [old for old in rows if old.get("exp_id") != exp.exp_id]
        rows.append(row)
        write_summary(rows)
    write_summary(rows)
    print(MD_PATH)


if __name__ == "__main__":
    main()
