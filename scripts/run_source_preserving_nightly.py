#!/usr/bin/env python3
"""Nightly source-preserving mechanism fusion validation.

This queue tests one architectural hypothesis only:
the previous source-preserving decoder may have been too strong.  We therefore
compare weaker source-preserving initialization values under the same predicted
event RCA protocol and only continue to longer confirmation runs if both SWaT
and WADI pass a conservative quick-screen gate.
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
PYTHON = Path(sys.executable)
RUN_SINGLE = PROJECT_ROOT / "ts_benchmark" / "run_single.py"
EVAL_RCA = PROJECT_ROOT / "scripts" / "evaluate_rca.py"
EVAL_HIER = PROJECT_ROOT / "scripts" / "evaluate_hierarchical_rca.py"

LOG_DIR = PROJECT_ROOT / "logs" / "architecture_experiments"
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "mechanism_gap"
LABEL_ROOT = PROJECT_ROOT / "result" / "label"
RCA_ROOT = PROJECT_ROOT / "result" / "rca"

PREDICTION_KEY = "15"
SUMMARY_CSV = ANALYSIS_DIR / "source_preserving_nightly_20260613.csv"
SUMMARY_MD = ANALYSIS_DIR / "source_preserving_nightly_20260613.md"
LOG_PATH = LOG_DIR / "source_preserving_nightly_20260613.log"


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    dataset: str
    profile: str
    epochs: int
    purpose: str

    @property
    def stem(self) -> str:
        return Path(self.dataset).stem


@dataclass(frozen=True)
class Candidate:
    name: str
    profile: str
    description: str


CANDIDATES = [
    Candidate(
        name="light035",
        profile="source-preserving-light-rca",
        description="source_preserving_init=0.35",
    ),
    Candidate(
        name="ultra020",
        profile="source-preserving-ultralight-rca",
        description="source_preserving_init=0.20",
    ),
]

DATASETS = {
    "swat": "SWAT_A1A2_Physical_v1.csv",
    "wadi": "WADI_A1_2017_ds10.csv",
}

BASELINE = {
    "swat": {
        "flat_variable_MRR": 0.506938,
        "flat_variable_Hit@1": 0.393939,
        "flat_variable_Hit@3": 0.515152,
        "hierarchical_variable_MRR": 0.500771,
        "subsystem_MRR": 0.738725,
    },
    "wadi": {
        "flat_variable_MRR": 0.596056,
        "flat_variable_Hit@1": 0.500000,
        "flat_variable_Hit@3": 0.714286,
        "hierarchical_variable_MRR": 0.475099,
        "subsystem_MRR": 0.755952,
    },
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(text: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text.rstrip("\n") + "\n")


def run_command(cmd: list[str | Path], title: str) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")
    append_log(f"\n[{now()}] BEGIN {title}")
    append_log("COMMAND " + " ".join(str(part) for part in cmd))
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
        append_log(line)
    rc = int(proc.wait())
    append_log(f"[{now()}] END {title} rc={rc}")
    return rc


def newest_file_after(directory: Path, pattern: str, started: float) -> Path | None:
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob(pattern) if p.stat().st_mtime >= started - 5]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def mean_row(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.mean(numeric_only=True).to_dict()


def train(exp: Experiment) -> tuple[int, Path | None, Path | None]:
    started = time.time()
    cmd: list[str | Path] = [
        PYTHON,
        "-u",
        RUN_SINGLE,
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
    rc = run_command(cmd, f"train {exp.exp_id}")
    rca_json = newest_file_after(RCA_ROOT / exp.stem, "*_rca.json", started)
    label_csv = newest_file_after(
        LABEL_ROOT / "journal_validation" / exp.exp_id,
        "*.csv",
        started,
    )
    return rc, rca_json, label_csv


def evaluate(exp: Experiment, rca_json: Path) -> dict[str, Any]:
    channel_csv = ANALYSIS_DIR / f"{exp.exp_id}_channel_pred_key15.csv"
    group_csv = ANALYSIS_DIR / f"{exp.exp_id}_group_pred_key15.csv"
    hier_csv = ANALYSIS_DIR / f"{exp.exp_id}_hierarchical_pred_key15.csv"

    jobs: list[tuple[str, list[str | Path]]] = [
        (
            "variable RCA",
            [
                PYTHON,
                EVAL_RCA,
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
                EVAL_RCA,
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
                EVAL_HIER,
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
        rc = run_command(cmd, f"evaluate {exp.exp_id} {title}")
        if rc != 0:
            raise RuntimeError(f"{exp.exp_id} {title} failed rc={rc}")

    channel = mean_row(channel_csv)
    group = mean_row(group_csv)
    hier = mean_row(hier_csv)
    return {
        "channel_csv": str(channel_csv),
        "group_csv": str(group_csv),
        "hierarchical_csv": str(hier_csv),
        "flat_variable_MRR": hier.get("flat_variable_MRR"),
        "flat_variable_Hit@1": hier.get("flat_variable_Hit@1"),
        "flat_variable_Hit@3": hier.get("flat_variable_Hit@3"),
        "flat_variable_Hit@5": hier.get("flat_variable_Hit@5"),
        "subsystem_MRR": hier.get("subsystem_MRR"),
        "subsystem_Hit@1": hier.get("subsystem_Hit@1"),
        "conditional_variable_MRR": hier.get("conditional_variable_MRR"),
        "hierarchical_variable_MRR": hier.get("hierarchical_variable_MRR"),
        "hierarchical_variable_Hit@1": hier.get("hierarchical_variable_Hit@1"),
        "event_iou": hier.get("event_iou"),
        "true_coverage": hier.get("true_coverage"),
        "channel_MRR": channel.get("MRR"),
        "channel_Hit@1": channel.get("Hit@1"),
        "channel_Hit@3": channel.get("Hit@3"),
        "group_MRR": group.get("MRR"),
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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(rows: list[dict[str, Any]]) -> None:
    metric_cols = [
        "exp_id",
        "dataset_key",
        "profile",
        "epochs",
        "status",
        "flat_variable_MRR",
        "flat_variable_Hit@1",
        "flat_variable_Hit@3",
        "subsystem_MRR",
        "hierarchical_variable_MRR",
        "event_iou",
        "decision",
    ]
    lines = [
        "# Source-Preserving Nightly Validation 2026-06-13",
        "",
        "Protocol: predicted-event RCA, prediction key 15, same `num-workers=2`, `inference-num-workers=0`, best-val checkpoint.",
        "",
        "| " + " | ".join(metric_cols) + " |",
        "| " + " | ".join(["---"] * len(metric_cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in metric_cols) + " |")
    lines.extend(
        [
            "",
            "## Decision Rules",
            "",
            "- SWaT quick-screen passes if flat variable MRR is not below the current main baseline and Hit@3 improves by at least about 0.03, or flat variable MRR alone improves by at least about 0.005.",
            "- WADI quick-screen passes if flat variable MRR stays within about 0.02 of the current main baseline and Hit@3 does not drop.",
            "- e15 confirmation is run only for candidates that pass both SWaT and WADI e8 gates.",
            "",
            "## Current Baselines Used By This Queue",
            "",
            "| dataset | flat MRR | Hit@1 | Hit@3 | subsystem MRR | hierarchical MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, base in BASELINE.items():
        lines.append(
            f"| {key} | {base['flat_variable_MRR']:.6f} | {base['flat_variable_Hit@1']:.6f} | "
            f"{base['flat_variable_Hit@3']:.6f} | {base['subsystem_MRR']:.6f} | "
            f"{base['hierarchical_variable_MRR']:.6f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def existing_success(rows: list[dict[str, Any]], exp_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("exp_id") == exp_id and row.get("status") == "success":
            return row
    return None


def swat_pass(row: dict[str, Any]) -> bool:
    base = BASELINE["swat"]
    flat = float(row.get("flat_variable_MRR") or 0.0)
    hit3 = float(row.get("flat_variable_Hit@3") or 0.0)
    return (flat >= base["flat_variable_MRR"] and hit3 >= base["flat_variable_Hit@3"] + 0.03) or (
        flat >= base["flat_variable_MRR"] + 0.005
    )


def wadi_pass(row: dict[str, Any]) -> bool:
    base = BASELINE["wadi"]
    flat = float(row.get("flat_variable_MRR") or 0.0)
    hit3 = float(row.get("flat_variable_Hit@3") or 0.0)
    return flat >= base["flat_variable_MRR"] - 0.02 and hit3 >= base["flat_variable_Hit@3"] - 1e-6


def run_experiment(exp: Experiment, dataset_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cached = existing_success(rows, exp.exp_id)
    if cached is not None:
        append_log(f"[{now()}] SKIP {exp.exp_id}: already in summary")
        return cached

    rc, rca_json, label_csv = train(exp)
    row: dict[str, Any] = {
        "exp_id": exp.exp_id,
        "dataset_key": dataset_key,
        "dataset": exp.dataset,
        "profile": exp.profile,
        "epochs": exp.epochs,
        "purpose": exp.purpose,
        "status": "train_failed" if rc != 0 else "trained",
        "rca_json": str(rca_json) if rca_json else "",
        "label_csv": str(label_csv) if label_csv else "",
    }
    if rc == 0 and rca_json is not None:
        row.update(evaluate(exp, rca_json))
        row["status"] = "success"
    elif rc == 0:
        row["status"] = "missing_rca_json"
    rows.append(row)
    write_rows(rows)
    return row


def choose_best_candidate(results: list[tuple[Candidate, dict[str, Any], dict[str, Any] | None]]) -> Candidate | None:
    valid: list[tuple[float, Candidate]] = []
    for candidate, swat_row, wadi_row in results:
        if wadi_row is None:
            continue
        if not swat_pass(swat_row) or not wadi_pass(wadi_row):
            continue
        score = float(swat_row.get("flat_variable_MRR") or 0.0) + float(wadi_row.get("flat_variable_MRR") or 0.0)
        valid.append((score, candidate))
    if not valid:
        return None
    valid.sort(key=lambda item: item[0], reverse=True)
    return valid[0][1]


def main() -> int:
    rows = read_rows()
    append_log(f"\n[{now()}] Nightly queue start")
    append_log(f"Python: {PYTHON}")

    results: list[tuple[Candidate, dict[str, Any], dict[str, Any] | None]] = []
    for candidate in CANDIDATES:
        swat_exp = Experiment(
            exp_id=f"nightly_swat_source_preserving_{candidate.name}_e8",
            dataset=DATASETS["swat"],
            profile=candidate.profile,
            epochs=8,
            purpose=f"SWaT e8 quick-screen for {candidate.description}.",
        )
        swat_row = run_experiment(swat_exp, "swat", rows)
        swat_row["decision"] = "swat_pass" if swat_pass(swat_row) else "swat_fail"
        write_rows(rows)

        wadi_row = None
        if swat_pass(swat_row):
            wadi_exp = Experiment(
                exp_id=f"nightly_wadi_source_preserving_{candidate.name}_e8",
                dataset=DATASETS["wadi"],
                profile=candidate.profile,
                epochs=8,
                purpose=f"WADI e8 generalization check for {candidate.description}.",
            )
            wadi_row = run_experiment(wadi_exp, "wadi", rows)
            wadi_row["decision"] = "wadi_pass" if wadi_pass(wadi_row) else "wadi_fail"
            write_rows(rows)
        else:
            append_log(f"[{now()}] {candidate.name}: skip WADI e8 because SWaT gate failed")
        results.append((candidate, swat_row, wadi_row))

    best = choose_best_candidate(results)
    if best is None:
        append_log(f"[{now()}] No candidate passed both e8 gates; skip e15 confirmation")
        write_rows(rows)
        return 0

    append_log(f"[{now()}] Best candidate for e15 confirmation: {best.name} / {best.profile}")
    for dataset_key in ("swat", "wadi"):
        exp = Experiment(
            exp_id=f"nightly_{dataset_key}_source_preserving_{best.name}_e15",
            dataset=DATASETS[dataset_key],
            profile=best.profile,
            epochs=15,
            purpose=f"{dataset_key.upper()} e15 confirmation for selected {best.description}.",
        )
        row = run_experiment(exp, dataset_key, rows)
        row["decision"] = "e15_confirmation"
        write_rows(rows)

    append_log(f"[{now()}] Nightly queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
