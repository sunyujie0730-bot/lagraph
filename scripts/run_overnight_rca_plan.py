#!/usr/bin/env python3
"""Run the overnight RCA validation queue.

The queue is intentionally conservative:
1. wait for the currently running LaGraph SWaT e12 job to finish;
2. summarize/diagnose journal-validation results;
3. if SWaT e12 failed, retry it once;
4. run ModernTCN reconstruction-RCA baselines on WADI and SWaT;
5. evaluate true-event and predicted-event RCA with the same metrics.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LOG_DIR = PROJECT_ROOT / "logs" / "architecture_experiments"
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
DONE_SWAT_E12 = ANALYSIS_DIR / "main_swat_e12.done.json"


BASELINE_RUNS = [
    ("modern_tcn_wadi_e8", "WADI_A1_2017_ds10.csv", 8),
    ("modern_tcn_swat_e8", "SWAT_A1A2_Physical_v1.csv", 8),
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def python_process_commands() -> list[str]:
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python' } | "
        "Select-Object -ExpandProperty CommandLine"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def swat_e12_in_progress() -> bool:
    for command in python_process_commands():
        command_norm = command.replace("/", "\\")
        if "main_swat_e12" not in command_norm:
            continue
        if "run_journal_validation_queue.py" in command_norm or "ts_benchmark\\run_single.py" in command_norm:
            return True
    return False


def run_command(command: list[str], log_path: Path, title: str) -> tuple[int, list[str]]:
    append(log_path, f"\n[{now()}] BEGIN {title}")
    append(log_path, "COMMAND " + " ".join(str(part) for part in command))
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        append(log_path, line)
    rc = process.wait()
    append(log_path, f"[{now()}] END {title} rc={rc}")
    return rc, lines


def wait_for_swat_e12(log_path: Path) -> None:
    if not swat_e12_in_progress():
        append(log_path, f"[{now()}] No active main_swat_e12 job detected.")
        return
    append(log_path, f"[{now()}] Waiting for active main_swat_e12 job to finish.")
    while swat_e12_in_progress():
        time.sleep(120)
        append(log_path, f"[{now()}] Still waiting for main_swat_e12.")
    append(log_path, f"[{now()}] Active main_swat_e12 job finished.")


def swat_e12_ok() -> bool:
    if not DONE_SWAT_E12.exists():
        return False
    try:
        data = json.loads(DONE_SWAT_E12.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return str(data.get("status", "")).startswith("ok")


def retry_swat_e12_if_needed(log_path: Path) -> None:
    if swat_e12_ok():
        append(log_path, f"[{now()}] main_swat_e12 already ok.")
        return
    append(log_path, f"[{now()}] main_swat_e12 is missing/failed; retrying once.")
    run_command(
        [
            PYTHON,
            "-u",
            "scripts/run_journal_validation_queue.py",
            "--suite",
            "confirmation",
            "--only",
            "main_swat_e12",
        ],
        log_path,
        "retry main_swat_e12",
    )


def latest_rca_baseline(stem: str, after: float) -> Path | None:
    directory = PROJECT_ROOT / "result" / "rca_baselines" / stem
    if not directory.exists():
        return None
    candidates = [
        path
        for path in directory.glob("*_moderntcn_rca.json")
        if path.stat().st_mtime >= after - 5
    ]
    if not candidates:
        candidates = list(directory.glob("*_moderntcn_rca.json"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def parse_rca_json_from_output(lines: list[str]) -> Path | None:
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = data.get("rca_json")
        if path:
            return Path(path)
    return None


def run_modern_tcn_baseline(exp_id: str, series: str, epochs: int, log_path: Path) -> None:
    stem = Path(series).stem
    started = time.time()
    rc, lines = run_command(
        [
            PYTHON,
            "-u",
            "scripts/run_reconstruction_rca_baseline.py",
            "--series",
            series,
            "--model",
            "ModernTCN",
            "--epochs",
            str(epochs),
            "--batch-size",
            "256",
            "--num-workers",
            "2",
            "--prefetch-factor",
            "2",
            "--prediction-key",
            "15",
        ],
        log_path,
        exp_id,
    )
    if rc != 0:
        append(log_path, f"[{now()}] {exp_id} failed before RCA evaluation.")
        return

    rca_json = parse_rca_json_from_output(lines) or latest_rca_baseline(stem, started)
    if rca_json is None or not rca_json.exists():
        append(log_path, f"[{now()}] {exp_id} produced no RCA JSON.")
        return

    for source in ["true", "predicted"]:
        save_csv = ANALYSIS_DIR / f"{exp_id}_hierarchical_{source}.csv"
        command = [
            PYTHON,
            "scripts/evaluate_hierarchical_rca.py",
            "--rca",
            rca_json,
            "--event-source",
            source,
            "--method-name",
            f"ModernTCN-RCA-{epochs}ep",
            "--summary-only",
            "--save-csv",
            save_csv,
        ]
        if source == "predicted":
            command.extend(["--prediction-key", "15"])
        run_command(command, log_path, f"evaluate {exp_id} {source}")


def mean_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df[df["series_name"].astype(str).eq("MEAN")]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def write_baseline_summary(log_path: Path) -> None:
    rows = []
    for exp_id, series, epochs in BASELINE_RUNS:
        for source in ["true", "predicted"]:
            path = ANALYSIS_DIR / f"{exp_id}_hierarchical_{source}.csv"
            row = mean_row(path)
            if row is None:
                continue
            rows.append(
                {
                    "exp_id": exp_id,
                    "dataset": "WADI" if "WADI" in series else "SWaT",
                    "method": f"ModernTCN-RCA-{epochs}ep",
                    "event_source": source,
                    "Var-MRR": fmt(row.get("flat_variable_MRR")),
                    "Var-Hit@1": fmt(row.get("flat_variable_Hit@1")),
                    "Var-Hit@3": fmt(row.get("flat_variable_Hit@3")),
                    "Var-Hit@5": fmt(row.get("flat_variable_Hit@5")),
                    "Var-PR@3": fmt(row.get("flat_variable_PR@3")),
                    "Var-MAP@5": fmt(row.get("flat_variable_MAP@5")),
                    "Subsys-MRR": fmt(row.get("subsystem_MRR")),
                    "Subsys-Hit@1": fmt(row.get("subsystem_Hit@1")),
                    "matched": fmt(row.get("matched")),
                    "event_iou": fmt(row.get("event_iou")),
                    "csv": str(path),
                }
            )
    out_csv = ANALYSIS_DIR / "reconstruction_rca_baseline_summary.csv"
    out_md = ANALYSIS_DIR / "reconstruction_rca_baseline_summary.md"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    lines = [
        "# Reconstruction RCA Baseline Summary",
        "",
        f"Updated: {now()}",
        "",
        "| exp_id | dataset | method | event_source | Var-MRR | Var-Hit@1 | Var-Hit@3 | Var-Hit@5 | Var-PR@3 | Var-MAP@5 | Subsys-MRR | Subsys-Hit@1 | matched | event_iou |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[col])
                for col in [
                    "exp_id",
                    "dataset",
                    "method",
                    "event_source",
                    "Var-MRR",
                    "Var-Hit@1",
                    "Var-Hit@3",
                    "Var-Hit@5",
                    "Var-PR@3",
                    "Var-MAP@5",
                    "Subsys-MRR",
                    "Subsys-Hit@1",
                    "matched",
                    "event_iou",
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append(log_path, f"[{now()}] Wrote {out_csv} and {out_md}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"overnight_rca_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    append(log_path, f"[{now()}] Overnight RCA plan started.")

    wait_for_swat_e12(log_path)
    run_command([PYTHON, "scripts/summarize_journal_validation.py"], log_path, "summarize journal validation")
    run_command([PYTHON, "scripts/diagnose_rca_epoch_degradation.py"], log_path, "diagnose RCA epoch degradation")
    retry_swat_e12_if_needed(log_path)
    run_command([PYTHON, "scripts/summarize_journal_validation.py"], log_path, "resummarize after optional retry")
    run_command([PYTHON, "scripts/diagnose_rca_epoch_degradation.py"], log_path, "rediagnose after optional retry")

    for exp_id, series, epochs in BASELINE_RUNS:
        run_modern_tcn_baseline(exp_id, series, epochs, log_path)
        write_baseline_summary(log_path)

    append(log_path, f"[{now()}] Overnight RCA plan finished.")


if __name__ == "__main__":
    main()
