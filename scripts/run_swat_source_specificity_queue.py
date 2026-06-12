#!/usr/bin/env python3
"""Run SWaT source-specificity quick screens.

The goal is narrow: test whether source-specific training constraints can keep
the mechanism graph while recovering the SWaT variable-level RCA sharpness that
no-mechanism achieved.
"""

from __future__ import annotations

import csv
import os
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
LOG_DIR = PROJECT_ROOT / "logs" / "architecture_experiments"
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "mechanism_gap"
RCA_ROOT = PROJECT_ROOT / "result" / "rca"
PREDICTION_KEY = "15"

SUMMARY_CSV = ANALYSIS_DIR / "swat_source_specificity_summary.csv"
SUMMARY_MD = ANALYSIS_DIR / "swat_source_specificity_summary.md"


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    dataset: str
    profile: str
    purpose: str
    epochs: int = 8

    @property
    def stem(self) -> str:
        return Path(self.dataset).stem


EXPERIMENTS = [
    Experiment(
        exp_id="swat_trained_specificity_e8",
        dataset="SWAT_A1A2_Physical_v1.csv",
        profile="source-bottleneck-trained-specificity-rca",
        purpose="Sharpen source gate distribution during source-bottleneck training.",
    ),
    Experiment(
        exp_id="swat_gate_specificity_e8",
        dataset="SWAT_A1A2_Physical_v1.csv",
        profile="source-bottleneck-gate-specificity-rca",
        purpose="Sharpen synthetic source-effect gate supervision.",
    ),
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text.rstrip("\n") + "\n")


def run_command(cmd: list[str | Path], log_path: Path, title: str) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")
    append_log(log_path, f"\n[{now()}] BEGIN {title}")
    append_log(log_path, "COMMAND " + " ".join(str(part) for part in cmd))
    proc = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        append_log(log_path, line)
    rc = proc.wait()
    append_log(log_path, f"[{now()}] END {title} rc={rc}")
    return int(rc)


def newest_file_after(directory: Path, pattern: str, started: float) -> Path | None:
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob(pattern) if p.stat().st_mtime >= started - 5]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def mean_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.mean(numeric_only=True).to_dict()


def train(exp: Experiment, log_path: Path) -> tuple[int, Path | None, Path | None]:
    started = time.time()
    cmd: list[str | Path] = [
        PYTHON,
        "-u",
        "ts_benchmark/run_single.py",
        "--epochs",
        str(exp.epochs),
        "--checkpoint-policy",
        "best-val",
        "--datasets",
        exp.dataset,
        "--arch-profile",
        exp.profile,
        "--export-rca",
        "--rca-export-lite",
        "--rca-export-top-k",
        "20",
        "--num-workers",
        "2",
        "--prefetch-factor",
        "2",
        "--inference-num-workers",
        "0",
        "--save-dir",
        f"label/journal_validation/{exp.exp_id}",
    ]
    rc = run_command(cmd, log_path, f"train {exp.exp_id}")
    rca_json = newest_file_after(RCA_ROOT / exp.stem, "*_rca.json", started)
    label_csv = newest_file_after(
        PROJECT_ROOT / "result" / "label" / "journal_validation" / exp.exp_id,
        "*.csv",
        started,
    )
    return rc, rca_json, label_csv


def evaluate(exp: Experiment, rca_json: Path, log_path: Path) -> dict[str, Any]:
    channel_csv = ANALYSIS_DIR / f"{exp.exp_id}_channel_pred_key15.csv"
    group_csv = ANALYSIS_DIR / f"{exp.exp_id}_group_pred_key15.csv"
    hier_csv = ANALYSIS_DIR / f"{exp.exp_id}_hierarchical_pred_key15.csv"

    jobs: list[tuple[str, list[str | Path]]] = [
        (
            "variable RCA",
            [
                PYTHON,
                "scripts/evaluate_rca.py",
                "--rca",
                rca_json,
                "--scope",
                "channel",
                "--event-source",
                "predicted",
                "--prediction-key",
                PREDICTION_KEY,
                "--method-name",
                exp.exp_id,
                "--save-csv",
                channel_csv,
            ],
        ),
        (
            "subsystem RCA",
            [
                PYTHON,
                "scripts/evaluate_rca.py",
                "--rca",
                rca_json,
                "--scope",
                "group",
                "--event-source",
                "predicted",
                "--prediction-key",
                PREDICTION_KEY,
                "--method-name",
                exp.exp_id,
                "--save-csv",
                group_csv,
            ],
        ),
        (
            "hierarchical RCA",
            [
                PYTHON,
                "scripts/evaluate_hierarchical_rca.py",
                "--rca",
                rca_json,
                "--event-source",
                "predicted",
                "--prediction-key",
                PREDICTION_KEY,
                "--method-name",
                exp.exp_id,
                "--summary-only",
                "--save-csv",
                hier_csv,
            ],
        ),
    ]
    for title, cmd in jobs:
        rc = run_command(cmd, log_path, f"evaluate {exp.exp_id} {title}")
        if rc != 0:
            raise RuntimeError(f"{exp.exp_id} {title} failed rc={rc}")

    channel = mean_row(channel_csv)
    group = mean_row(group_csv)
    hier = mean_row(hier_csv)
    return {
        "channel_csv": str(channel_csv),
        "group_csv": str(group_csv),
        "hierarchical_csv": str(hier_csv),
        "channel_MRR": channel.get("MRR"),
        "channel_Hit@1": channel.get("Hit@1"),
        "channel_Hit@3": channel.get("Hit@3"),
        "channel_Hit@5": channel.get("Hit@5"),
        "group_MRR": group.get("MRR"),
        "group_Hit@1": group.get("Hit@1"),
        "group_Hit@3": group.get("Hit@3"),
        "group_Hit@5": group.get("Hit@5"),
        "subsystem_MRR": hier.get("subsystem_MRR"),
        "subsystem_Hit@1": hier.get("subsystem_Hit@1"),
        "flat_variable_MRR": hier.get("flat_variable_MRR"),
        "conditional_variable_MRR": hier.get("conditional_variable_MRR"),
        "hierarchical_variable_MRR": hier.get("hierarchical_variable_MRR"),
        "hierarchical_variable_Hit@1": hier.get("hierarchical_variable_Hit@1"),
        "event_iou": hier.get("event_iou"),
        "true_coverage": hier.get("true_coverage"),
    }


def read_rows() -> list[dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, Any]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    display_cols = [
        "exp_id",
        "profile",
        "status",
        "channel_MRR",
        "channel_Hit@1",
        "channel_Hit@3",
        "group_MRR",
        "hierarchical_variable_MRR",
        "hierarchical_variable_Hit@1",
        "event_iou",
        "purpose",
    ]
    df = pd.DataFrame(rows)
    display = df[[col for col in display_cols if col in df.columns]].copy()
    for col in display.columns:
        if col not in {"exp_id", "profile", "status", "purpose"}:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(
                lambda x: "-" if pd.isna(x) else f"{x:.4f}"
            )
    lines = [
        "# SWaT Source-Specificity Quick Screen",
        "",
        f"Updated: {now()}",
        "",
        "Protocol: SWaT e8, predicted-event key=15, checkpoint=best-val, "
        "num-workers=2, inference-num-workers=0.",
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    lines.extend(["", f"CSV: `{SUMMARY_CSV}`"])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exp(exp: Experiment) -> dict[str, Any]:
    log_path = LOG_DIR / f"{exp.exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    started_at = now()
    started = time.time()
    row: dict[str, Any] = {
        "exp_id": exp.exp_id,
        "dataset": exp.dataset,
        "profile": exp.profile,
        "epochs": exp.epochs,
        "purpose": exp.purpose,
        "status": "started",
        "log": str(log_path),
        "started_at": started_at,
    }
    rc, rca_json, label_csv = train(exp, log_path)
    row["returncode"] = rc
    row["runtime_sec"] = round(time.time() - started, 1)
    row["finished_at"] = now()
    row["label_csv"] = str(label_csv) if label_csv else ""
    row["rca_json"] = str(rca_json) if rca_json else ""
    if rc != 0:
        row["status"] = "train_failed"
        return row
    if rca_json is None:
        row["status"] = "missing_rca"
        return row
    row.update(evaluate(exp, rca_json, log_path))
    row["status"] = "ok"
    return row


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    done = {row.get("exp_id") for row in rows if row.get("status") == "ok"}
    for exp in EXPERIMENTS:
        if exp.exp_id in done:
            continue
        row = run_exp(exp)
        rows = [old for old in rows if old.get("exp_id") != exp.exp_id]
        rows.append(row)
        write_rows(rows)
    write_rows(rows)
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
