#!/usr/bin/env python3
"""Run staged RCA weight search on WADI, then confirm on SWaT.

The queue uses one-factor-at-a-time stages:
1. RootScore on WADI.
2. Source gate on WADI with the best RootScore fixed.
3. Onset on WADI with the best RootScore/source gate fixed.
4. SWaT confirmation for the center setting and WADI-best setting.

Each experiment trains independently, exports RCA in lite mode, evaluates with
the same predicted-event protocol, and writes a cumulative summary.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LOG_DIR = PROJECT_ROOT / "logs" / "architecture_experiments"
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
SUMMARY_CSV = ANALYSIS_DIR / "wadi_swat_weight_search_summary.csv"
SUMMARY_MD = ANALYSIS_DIR / "wadi_swat_weight_search_summary.md"
PREDICTION_KEY = "15"
PROFILE = "source-bottleneck-root-score-w03-rca"
EPOCHS = 8
CENTER_ROOT = 0.30
CENTER_SOURCE_GATE = 0.35
CENTER_ONSET = 0.60


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    dataset: str
    stage: str
    root_score: float
    source_gate: float
    onset: float
    purpose: str

    @property
    def stem(self) -> str:
        return Path(self.dataset).stem


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_command(command: list[str | Path], log_path: Path, title: str) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")
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
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        append(log_path, line.rstrip("\n"))
    rc = process.wait()
    append(log_path, f"[{now()}] END {title} rc={rc}")
    return rc


def done_path(exp_id: str) -> Path:
    return ANALYSIS_DIR / f"{exp_id}.done.json"


def load_done(exp_id: str) -> dict[str, Any] | None:
    path = done_path(exp_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if str(data.get("status", "")).startswith("ok"):
        return data
    return None


def latest_file(directory: Path, pattern: str, after: float) -> Path | None:
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob(pattern) if p.stat().st_mtime >= after - 5]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    for row in reversed(rows):
        if str(row.get("series_name", "")).upper() == "MEAN":
            return row
    return rows[-1] if rows else {}


def summarize_label(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    rows = read_rows(path)
    if not rows:
        return {}
    ratio_1 = next((r for r in rows if str(r.get("typical_anomaly_ratio", "")) == "1.0"), rows[0])
    best_aff = max(rows, key=lambda r: as_float(r, "affiliation_f", -1.0))
    return {
        "label_csv": str(path),
        "aff_f_ratio_1": as_float(ratio_1, "affiliation_f"),
        "event_recall_ratio_1": as_float(ratio_1, "adjust_recall"),
        "point_f1_ratio_1": as_float(ratio_1, "f_score"),
        "best_aff_f": as_float(best_aff, "affiliation_f"),
        "best_aff_ratio": best_aff.get("typical_anomaly_ratio", ""),
    }


def summarize_rca(path: Path, prefix: str) -> dict[str, Any]:
    row = mean_row(path)
    return {
        f"{prefix}_MRR": as_float(row, "MRR"),
        f"{prefix}_Hit@1": as_float(row, "Hit@1"),
        f"{prefix}_Hit@3": as_float(row, "Hit@3"),
        f"{prefix}_Hit@5": as_float(row, "Hit@5"),
        f"{prefix}_PR@1": as_float(row, "PR@1"),
        f"{prefix}_PR@3": as_float(row, "PR@3"),
        f"{prefix}_PR@5": as_float(row, "PR@5"),
        f"{prefix}_MAP@3": as_float(row, "MAP@3"),
        f"{prefix}_MAP@5": as_float(row, "MAP@5"),
        f"{prefix}_NDCG@3": as_float(row, "NDCG@3"),
        f"{prefix}_NDCG@5": as_float(row, "NDCG@5"),
        f"{prefix}_matched": as_float(row, "matched"),
    }


def summarize_hier(path: Path) -> dict[str, Any]:
    row = mean_row(path)
    keys = [
        "subsystem_MRR",
        "subsystem_Hit@1",
        "subsystem_Hit@3",
        "subsystem_Hit@5",
        "flat_variable_MRR",
        "flat_variable_Hit@1",
        "flat_variable_Hit@3",
        "flat_variable_Hit@5",
        "conditional_variable_MRR",
        "hierarchical_variable_MRR",
    ]
    return {key: as_float(row, key) for key in keys}


def train_experiment(exp: Experiment, log_path: Path) -> tuple[int, Path | None, Path | None]:
    started = time.time()
    command: list[str | Path] = [
        PYTHON,
        "-u",
        "ts_benchmark/run_single.py",
        "--epochs",
        str(EPOCHS),
        "--checkpoint-policy",
        "best-val",
        "--datasets",
        exp.dataset,
        "--arch-profile",
        PROFILE,
        "--export-rca",
        "--rca-export-lite",
        "--rca-export-top-k",
        "20",
        "--num-workers",
        "2",
        "--prefetch-factor",
        "2",
        "--save-dir",
        f"label/journal_validation/{exp.exp_id}",
        "--rca-root-score-weight",
        f"{exp.root_score:.4f}",
        "--rca-source-gate-weight",
        f"{exp.source_gate:.4f}",
        "--rca-onset-weight",
        f"{exp.onset:.4f}",
    ]
    rc = run_command(command, log_path, f"train {exp.exp_id}")
    rca_json = latest_file(PROJECT_ROOT / "result" / "rca" / exp.stem, "*_rca.json", started)
    label_csv = latest_file(
        PROJECT_ROOT / "result" / "label" / "journal_validation" / exp.exp_id,
        "*.csv",
        started,
    )
    return rc, rca_json, label_csv


def evaluate_experiment(exp: Experiment, rca_json: Path, log_path: Path) -> dict[str, Any]:
    channel_csv = ANALYSIS_DIR / f"{exp.exp_id}_channel_pred_key15.csv"
    group_csv = ANALYSIS_DIR / f"{exp.exp_id}_group_pred_key15.csv"
    hier_csv = ANALYSIS_DIR / f"{exp.exp_id}_hierarchical_pred_key15.csv"

    eval_jobs = [
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
    for title, command in eval_jobs:
        rc = run_command(command, log_path, f"evaluate {exp.exp_id} {title}")
        if rc != 0:
            raise RuntimeError(f"{exp.exp_id} {title} failed rc={rc}")

    return {
        "channel_csv": str(channel_csv),
        "group_csv": str(group_csv),
        "hierarchical_csv": str(hier_csv),
        **summarize_rca(channel_csv, "var"),
        **summarize_rca(group_csv, "subsys"),
        **summarize_hier(hier_csv),
    }


def row_from_done(data: dict[str, Any]) -> dict[str, Any]:
    row = {
        "exp_id": data.get("exp_id"),
        "dataset": data.get("dataset"),
        "stage": data.get("stage"),
        "root_score": data.get("root_score"),
        "source_gate": data.get("source_gate"),
        "onset": data.get("onset"),
        "purpose": data.get("purpose"),
        "status": data.get("status"),
        "runtime_sec": data.get("runtime_sec"),
    }
    row.update(data.get("metrics", {}))
    return row


def all_done_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ANALYSIS_DIR.glob("weight_search_*.done.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(row_from_done(data))
    return rows


def write_summary() -> None:
    rows = all_done_rows()
    if not rows:
        return
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "exp_id",
        "dataset",
        "stage",
        "root_score",
        "source_gate",
        "onset",
        "var_MRR",
        "var_Hit@1",
        "var_Hit@3",
        "var_Hit@5",
        "var_PR@3",
        "var_MAP@3",
        "subsys_MRR",
        "subsys_Hit@1",
        "subsystem_MRR",
        "flat_variable_MRR",
        "conditional_variable_MRR",
        "hierarchical_variable_MRR",
        "aff_f_ratio_1",
        "best_aff_f",
        "status",
        "runtime_sec",
        "purpose",
    ]
    extra_columns = sorted({key for row in rows for key in row} - set(columns))
    columns += extra_columns
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "-"

    display = [
        "exp_id",
        "dataset",
        "stage",
        "root_score",
        "source_gate",
        "onset",
        "var_MRR",
        "var_Hit@1",
        "var_Hit@3",
        "var_Hit@5",
        "subsys_MRR",
        "aff_f_ratio_1",
        "purpose",
    ]
    lines = [
        "# WADI/SWaT RCA Weight Search Summary",
        "",
        f"Updated: {now()}",
        "",
        "统一设置：epochs=8, checkpoint-policy=best-val, predicted-event key=15, RCA export lite top-k=20。",
        "",
        "| " + " | ".join(display) + " |",
        "| " + " | ".join("---" for _ in display) + " |",
    ]
    for row in rows:
        values = []
        for key in display:
            value = row.get(key, "")
            if key.startswith(("var_", "subsys_", "aff_")):
                values.append(fmt(value))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(exp: Experiment, master_log: Path) -> dict[str, Any]:
    existing = load_done(exp.exp_id)
    if existing:
        append(master_log, f"[{now()}] SKIP {exp.exp_id}: done file exists.")
        return existing

    started = time.time()
    exp_log = LOG_DIR / f"{exp.exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    data: dict[str, Any] = {
        "exp_id": exp.exp_id,
        "dataset": exp.dataset,
        "stage": exp.stage,
        "root_score": exp.root_score,
        "source_gate": exp.source_gate,
        "onset": exp.onset,
        "purpose": exp.purpose,
        "started_at": now(),
        "log": str(exp_log),
    }
    append(master_log, f"[{now()}] RUN {exp.exp_id} root={exp.root_score} gate={exp.source_gate} onset={exp.onset}")
    try:
        rc, rca_json, label_csv = train_experiment(exp, exp_log)
        data["train_returncode"] = rc
        data["rca_json"] = str(rca_json) if rca_json else ""
        data["label_csv"] = str(label_csv) if label_csv else ""
        if rc != 0:
            raise RuntimeError(f"training failed rc={rc}")
        if not rca_json or not rca_json.exists():
            raise RuntimeError("RCA JSON not found")
        metrics = {}
        metrics.update(summarize_label(label_csv))
        metrics.update(evaluate_experiment(exp, rca_json, exp_log))
        data["metrics"] = metrics
        data["status"] = "ok"
    except Exception as exc:  # keep the overnight queue moving
        data["status"] = f"failed: {exc}"
        append(exp_log, f"[{now()}] ERROR {exc}")
        append(master_log, f"[{now()}] ERROR {exp.exp_id}: {exc}")
    finally:
        data["finished_at"] = now()
        data["runtime_sec"] = round(time.time() - started, 3)
        done_path(exp.exp_id).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        write_summary()
    return data


def score_for_selection(data: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = data.get("metrics", {})
    return (
        float(metrics.get("var_MRR", 0.0)),
        float(metrics.get("var_Hit@1", 0.0)),
        float(metrics.get("var_Hit@3", 0.0)),
        float(metrics.get("subsys_MRR", 0.0)),
    )


def choose_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in results if str(item.get("status", "")).startswith("ok")]
    if not ok:
        raise RuntimeError("No successful experiments in stage")
    return sorted(ok, key=score_for_selection, reverse=True)[0]


def exp_id_for(dataset_label: str, stage: str, root: float, gate: float, onset: float) -> str:
    return (
        f"weight_search_{dataset_label}_{stage}"
        f"_r{int(round(root * 100)):02d}"
        f"_g{int(round(gate * 100)):02d}"
        f"_o{int(round(onset * 100)):02d}_e8"
    )


def make_exp(dataset: str, dataset_label: str, stage: str, root: float, gate: float, onset: float, purpose: str) -> Experiment:
    return Experiment(
        exp_id=exp_id_for(dataset_label, stage, root, gate, onset),
        dataset=dataset,
        stage=stage,
        root_score=root,
        source_gate=gate,
        onset=onset,
        purpose=purpose,
    )


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__.strip())
        print("\nOptions:\n  --dry-run    Print the staged plan without running training.")
        return 0
    dry_run = "--dry-run" in sys.argv[1:]

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    master_log = LOG_DIR / f"wadi_swat_weight_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    append(master_log, f"[{now()}] Weight search started.")

    wadi = "WADI_A1_2017_ds10.csv"
    swat = "SWAT_A1A2_Physical_v1.csv"

    if dry_run:
        print("Stage 1 WADI RootScore: 0.20, 0.30, 0.40")
        print("Stage 2 WADI Source gate: 0.25, 0.35, 0.45 with best RootScore")
        print("Stage 3 WADI Onset: 0.40, 0.60, 0.80 with best RootScore/source gate")
        print("Stage 4 SWaT: center setting and WADI-best setting")
        print(f"Summary will be written to {SUMMARY_MD}")
        return 0

    root_results = []
    for root in [0.20, 0.30, 0.40]:
        root_results.append(
            run_experiment(
                make_exp(
                    wadi,
                    "wadi",
                    "root",
                    root,
                    CENTER_SOURCE_GATE,
                    CENTER_ONSET,
                    "RootScore 单因素筛选；Source gate 和 Onset 固定中心值。",
                ),
                master_log,
            )
        )
    best_root = choose_best(root_results)
    root_value = float(best_root["root_score"])
    append(master_log, f"[{now()}] Best RootScore={root_value} from {best_root['exp_id']}")

    gate_results = []
    for gate in [0.25, 0.35, 0.45]:
        gate_results.append(
            run_experiment(
                make_exp(
                    wadi,
                    "wadi",
                    "source_gate",
                    root_value,
                    gate,
                    CENTER_ONSET,
                    "Source gate 单因素筛选；RootScore 使用上一阶段最优值，Onset 固定中心值。",
                ),
                master_log,
            )
        )
    best_gate = choose_best(gate_results)
    gate_value = float(best_gate["source_gate"])
    append(master_log, f"[{now()}] Best SourceGate={gate_value} from {best_gate['exp_id']}")

    onset_results = []
    for onset in [0.40, 0.60, 0.80]:
        onset_results.append(
            run_experiment(
                make_exp(
                    wadi,
                    "wadi",
                    "onset",
                    root_value,
                    gate_value,
                    onset,
                    "Onset 单因素筛选；RootScore 和 Source gate 使用前两阶段最优值。",
                ),
                master_log,
            )
        )
    best_onset = choose_best(onset_results)
    onset_value = float(best_onset["onset"])
    append(master_log, f"[{now()}] Best Onset={onset_value} from {best_onset['exp_id']}")

    # SWaT confirmation: center setting and WADI-best setting.
    swat_exps = [
        make_exp(
            swat,
            "swat",
            "center_confirm",
            CENTER_ROOT,
            CENTER_SOURCE_GATE,
            CENTER_ONSET,
            "SWaT 中心参数确认；用于和 WADI 最优参数对照。",
        ),
        make_exp(
            swat,
            "swat",
            "wadi_best_confirm",
            root_value,
            gate_value,
            onset_value,
            "SWaT 跨数据集确认；验证 WADI 选择的参数是否泛化。",
        ),
    ]
    seen: set[str] = set()
    for exp in swat_exps:
        key = f"{exp.root_score:.4f}-{exp.source_gate:.4f}-{exp.onset:.4f}"
        if key in seen:
            append(master_log, f"[{now()}] SKIP duplicate SWaT setting {key}")
            continue
        seen.add(key)
        run_experiment(exp, master_log)

    write_summary()
    append(master_log, f"[{now()}] Weight search finished. Summary: {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
