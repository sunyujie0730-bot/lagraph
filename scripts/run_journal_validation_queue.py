#!/usr/bin/env python3
"""Run journal-validation experiments for LaGraph RCA.

The queue is designed for paper evidence rather than architecture exploration:
it fixes a main profile, runs confirmation experiments, evaluates RCA under a
single predicted-event protocol, and then runs reviewer-facing ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
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
EVAL_RCA = PROJECT_ROOT / "scripts" / "evaluate_rca.py"
EVAL_HIER = PROJECT_ROOT / "scripts" / "evaluate_hierarchical_rca.py"
EVAL_STATIC = PROJECT_ROOT / "scripts" / "evaluate_static_rca_baselines.py"
LOG_DIR = PROJECT_ROOT / "logs" / "journal_validation"
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
LABEL_ROOT = PROJECT_ROOT / "result" / "label"
RCA_ROOT = PROJECT_ROOT / "result" / "rca"


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    dataset: str
    profile: str
    epochs: int
    purpose: str
    run_static_baselines: bool = True


MAIN_PROFILE = "source-bottleneck-specificity-rca"
PREDICTION_KEY = "15"


CONFIRMATION = [
    Experiment(
        "main_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        MAIN_PROFILE,
        8,
        "Main profile confirmation on WADI at 8 epochs.",
    ),
    Experiment(
        "main_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        MAIN_PROFILE,
        8,
        "Main profile confirmation on SWaT at 8 epochs.",
    ),
    Experiment(
        "main_wadi_e10",
        "WADI_A1_2017_ds10.csv",
        MAIN_PROFILE,
        10,
        "Epoch trajectory point for WADI RCA/detection trade-off.",
    ),
    Experiment(
        "main_swat_e10",
        "SWAT_A1A2_Physical_v1.csv",
        MAIN_PROFILE,
        10,
        "Epoch trajectory point for SWaT RCA/detection trade-off.",
    ),
    Experiment(
        "main_wadi_e12",
        "WADI_A1_2017_ds10.csv",
        MAIN_PROFILE,
        12,
        "Epoch trajectory point for WADI RCA/detection trade-off.",
    ),
    Experiment(
        "main_swat_e12",
        "SWAT_A1A2_Physical_v1.csv",
        MAIN_PROFILE,
        12,
        "Epoch trajectory point for SWaT RCA/detection trade-off.",
    ),
    Experiment(
        "main_wadi_e15",
        "WADI_A1_2017_ds10.csv",
        MAIN_PROFILE,
        15,
        "Longer WADI confirmation for stability.",
    ),
    Experiment(
        "main_swat_e15",
        "SWAT_A1A2_Physical_v1.csv",
        MAIN_PROFILE,
        15,
        "Longer SWaT confirmation for stability.",
    ),
]


ABLATIONS = [
    Experiment(
        "ablation_no_mechanism_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-no-mechanism-rca",
        8,
        "Remove mechanism modeling to test mechanism graph necessity.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_mechanism_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-no-mechanism-rca",
        8,
        "Remove mechanism modeling to test mechanism graph necessity.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_source_gate_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-no-source-gate-rca",
        8,
        "Remove only the source gate to test whether gate-based source selection is necessary.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_source_gate_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-no-source-gate-rca",
        8,
        "Remove only the source gate to test whether gate-based source selection is necessary.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_source_bottleneck_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-no-source-bottleneck-rca",
        8,
        "Remove source gate/bottleneck to test source compression necessity.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_source_bottleneck_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-no-source-bottleneck-rca",
        8,
        "Remove source gate/bottleneck to test source compression necessity.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_source_propagation_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-no-source-propagation-rca",
        8,
        "Disable source-propagation RCA scoring to test source-oriented ranking.",
        run_static_baselines=False,
    ),
    Experiment(
        "ablation_no_source_propagation_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-no-source-propagation-rca",
        8,
        "Disable source-propagation RCA scoring to test source-oriented ranking.",
        run_static_baselines=False,
    ),
]

P0_RISK_CHECKS = [
    Experiment(
        "p0_no_mechanism_wadi_e12",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-no-mechanism-rca",
        12,
        "P0 risk check: rerun no-mechanism on WADI at 12 epochs to test cross-dataset stability.",
        run_static_baselines=False,
    ),
    Experiment(
        "p0_no_mechanism_swat_e12",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-no-mechanism-rca",
        12,
        "P0 risk check: rerun no-mechanism on SWaT at 12 epochs to test whether the e8 gain is stable.",
        run_static_baselines=False,
    ),
]

CANDIDATES = [
    Experiment(
        "candidate_adaptive_mechanism_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-bottleneck-adaptive-mechanism-rca",
        8,
        "Test event-adaptive mechanism gating on WADI before longer cross-dataset validation.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_adaptive_mechanism_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-bottleneck-adaptive-mechanism-rca",
        8,
        "Test event-adaptive mechanism gating on SWaT after WADI sanity validation.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_source_aware_dual_corefine_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-aware-dual-corefine-rca",
        8,
        "Test whether source gate can directly improve channel-temporal co-refinement on WADI.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_source_aware_dual_corefine_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-aware-dual-corefine-rca",
        8,
        "Test whether source gate can directly improve channel-temporal co-refinement on SWaT.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_source_preserving_fusion_wadi_e8",
        "WADI_A1_2017_ds10.csv",
        "source-preserving-mechanism-fusion-rca",
        8,
        "Test source-preserving mechanism fusion on WADI without extra graph propagation.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_source_preserving_fusion_swat_e8",
        "SWAT_A1A2_Physical_v1.csv",
        "source-preserving-mechanism-fusion-rca",
        8,
        "Test source-preserving mechanism fusion on SWaT without extra graph propagation.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_source_preserving_fusion_wadi_e15",
        "WADI_A1_2017_ds10.csv",
        "source-preserving-mechanism-fusion-rca",
        15,
        "Confirm source-preserving mechanism fusion on WADI at 15 epochs.",
        run_static_baselines=False,
    ),
    Experiment(
        "candidate_source_preserving_fusion_swat_e15",
        "SWAT_A1A2_Physical_v1.csv",
        "source-preserving-mechanism-fusion-rca",
        15,
        "Confirm source-preserving mechanism fusion on SWaT at 15 epochs.",
        run_static_baselines=False,
    ),
]


SUITES = {
    "confirmation": CONFIRMATION,
    "ablation": ABLATIONS,
    "p0-risk": P0_RISK_CHECKS,
    "candidate": CANDIDATES,
    "all": CONFIRMATION + ABLATIONS + P0_RISK_CHECKS + CANDIDATES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument("--only", nargs="*", default=None, help="Run only the listed experiment ids.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit.")
    return parser.parse_args()


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def list_files(root: Path, pattern: str) -> set[Path]:
    if not root.exists():
        return set()
    return set(root.rglob(pattern))


def newest_after(before: set[Path], root: Path, pattern: str) -> list[Path]:
    after = list_files(root, pattern)
    return sorted(after - before, key=lambda p: p.stat().st_mtime)


def completed_marker(exp_id: str) -> Path:
    return ANALYSIS_DIR / f"{exp_id}.done.json"


def run_command(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=out,
            stderr=err,
            text=True,
        )
        return proc.wait()


def training_command(exp: Experiment) -> list[str]:
    save_dir = f"label/journal_validation/{exp.exp_id}"
    return [
        str(PYTHON),
        "-u",
        str(RUN_SINGLE),
        "--epochs",
        str(exp.epochs),
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
        save_dir,
    ]


def eval_rca(rca_path: Path, exp_id: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for scope in ("channel", "group"):
        out_csv = ANALYSIS_DIR / f"{exp_id}_{scope}_pred_key{PREDICTION_KEY}.csv"
        command = [
            str(PYTHON),
            str(EVAL_RCA),
            "--rca",
            str(rca_path),
            "--scope",
            scope,
            "--event-source",
            "predicted",
            "--prediction-key",
            PREDICTION_KEY,
            "--method-name",
            f"{exp_id}_{scope}",
            "--save-csv",
            str(out_csv),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        outputs[scope] = out_csv

        random_csv = ANALYSIS_DIR / f"{exp_id}_{scope}_random_pred_key{PREDICTION_KEY}.csv"
        random_command = [
            str(PYTHON),
            str(EVAL_RCA),
            "--rca",
            str(rca_path),
            "--scope",
            scope,
            "--event-source",
            "predicted",
            "--prediction-key",
            PREDICTION_KEY,
            "--method-name",
            f"{exp_id}_{scope}_random",
            "--baseline",
            "random",
            "--random-trials",
            "1000",
            "--save-csv",
            str(random_csv),
        ]
        subprocess.run(random_command, cwd=PROJECT_ROOT, check=True)
        outputs[f"{scope}_random"] = random_csv

    residual_csv = ANALYSIS_DIR / f"{exp_id}_channel_residual_only_pred_key{PREDICTION_KEY}.csv"
    residual_command = [
        str(PYTHON),
        str(EVAL_RCA),
        "--rca",
        str(rca_path),
        "--scope",
        "channel",
        "--event-source",
        "predicted",
        "--prediction-key",
        PREDICTION_KEY,
        "--score-mode",
        "components",
        "--component-base-weight",
        "1.0",
        "--component-graph-weight",
        "0.0",
        "--component-source-gate-weight",
        "0.0",
        "--component-source-interaction-weight",
        "0.0",
        "--component-onset-weight",
        "0.0",
        "--method-name",
        f"{exp_id}_residual_only",
        "--save-csv",
        str(residual_csv),
    ]
    subprocess.run(residual_command, cwd=PROJECT_ROOT, check=True)
    outputs["channel_residual_only"] = residual_csv

    hier_csv = ANALYSIS_DIR / f"{exp_id}_hierarchical_pred_key{PREDICTION_KEY}.csv"
    hier_command = [
        str(PYTHON),
        str(EVAL_HIER),
        "--rca",
        str(rca_path),
        "--event-source",
        "predicted",
        "--prediction-key",
        PREDICTION_KEY,
        "--method-name",
        f"{exp_id}_hierarchical",
        "--save-csv",
        str(hier_csv),
    ]
    subprocess.run(hier_command, cwd=PROJECT_ROOT, check=True)
    outputs["hierarchical"] = hier_csv
    return outputs


def eval_static_baselines(rca_path: Path, exp: Experiment) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for scope in ("channel", "group"):
        out_csv = ANALYSIS_DIR / f"{exp.exp_id}_{scope}_static_baselines_pred_key{PREDICTION_KEY}.csv"
        command = [
            str(PYTHON),
            str(EVAL_STATIC),
            "--series",
            exp.dataset,
            "--scope",
            scope,
            "--event-source",
            "predicted",
            "--rca",
            str(rca_path),
            "--prediction-key",
            PREDICTION_KEY,
            "--include-random-graph",
            "--save-csv",
            str(out_csv),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        outputs[f"{scope}_static"] = out_csv
    return outputs


def read_mean_row(path: Path) -> dict:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    rows = df.loc[df[first_col].astype(str) == "MEAN"]
    if rows.empty:
        rows = df.tail(1)
    row = rows.iloc[0].to_dict()
    return row


def read_label_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    result: dict[str, float | str] = {}
    if "affiliation_f" in df.columns:
        idx = df["affiliation_f"].astype(float).idxmax()
        best = df.loc[idx]
        result.update(
            {
                "best_affiliation_f": float(best.get("affiliation_f", 0.0)),
                "best_affiliation_precision": float(best.get("affiliation_precision", 0.0)),
                "best_affiliation_recall": float(best.get("affiliation_recall", 0.0)),
                "best_raw_f1_at_best_aff": float(best.get("f_score", 0.0)),
            }
        )
    if len(df) > 0:
        last = df.iloc[-1]
        result.update(
            {
                "fit_time": float(last.get("fit_time", 0.0)),
                "inference_time": float(last.get("inference_time", 0.0)),
            }
        )
    return result


def append_summary(row: dict) -> None:
    csv_path = ANALYSIS_DIR / "journal_validation_summary.csv"
    exists = csv_path.exists()
    fieldnames = [
        "exp_id",
        "dataset",
        "profile",
        "epochs",
        "status",
        "rca_json",
        "label_csv",
        "channel_MRR",
        "channel_Hit@1",
        "channel_Hit@3",
        "channel_Hit@5",
        "group_MRR",
        "group_Hit@1",
        "group_Hit@3",
        "group_Hit@5",
        "conditional_variable_MRR",
        "conditional_variable_Hit@1",
        "conditional_variable_Hit@3",
        "conditional_variable_Hit@5",
        "best_affiliation_f",
        "best_raw_f1_at_best_aff",
        "fit_time",
        "inference_time",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown_summary() -> None:
    csv_path = ANALYSIS_DIR / "journal_validation_summary.csv"
    md_path = ANALYSIS_DIR / "journal_validation_summary.md"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    lines = [
        "# Journal Validation Experiment Summary",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Training Runs",
        "",
        "| exp_id | dataset | profile | epochs | status | channel MRR | channel Hit@1 | group MRR | group Hit@1 | conditional MRR | best Aff-F1 | fit time |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| {exp_id} | {dataset} | {profile} | {epochs} | {status} | {channel_MRR:.4f} | "
            "{channel_Hit1:.4f} | {group_MRR:.4f} | {group_Hit1:.4f} | "
            "{conditional_variable_MRR:.4f} | {best_affiliation_f:.4f} | {fit_time:.1f}s |".format(
                exp_id=row.get("exp_id", ""),
                dataset=row.get("dataset", ""),
                profile=row.get("profile", ""),
                epochs=int(row.get("epochs", 0)),
                status=row.get("status", ""),
                channel_MRR=float(row.get("channel_MRR", 0.0) or 0.0),
                channel_Hit1=float(row.get("channel_Hit@1", 0.0) or 0.0),
                group_MRR=float(row.get("group_MRR", 0.0) or 0.0),
                group_Hit1=float(row.get("group_Hit@1", 0.0) or 0.0),
                conditional_variable_MRR=float(row.get("conditional_variable_MRR", 0.0) or 0.0),
                best_affiliation_f=float(row.get("best_affiliation_f", 0.0) or 0.0),
                fit_time=float(row.get("fit_time", 0.0) or 0.0),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(exp: Experiment) -> dict:
    marker = completed_marker(exp.exp_id)
    start_time = time.time()
    before_rca = list_files(RCA_ROOT / exp.dataset.replace(".csv", ""), "*_rca.json")
    before_labels = list_files(LABEL_ROOT / "journal_validation" / exp.exp_id, "*.csv")
    command = training_command(exp)
    stdout_path = LOG_DIR / f"{exp.exp_id}.out.log"
    stderr_path = LOG_DIR / f"{exp.exp_id}.err.log"
    meta_path = LOG_DIR / f"{exp.exp_id}.meta.json"
    meta = {
        "exp_id": exp.exp_id,
        "dataset": exp.dataset,
        "profile": exp.profile,
        "epochs": exp.epochs,
        "purpose": exp.purpose,
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RUN] {exp.exp_id}: {exp.dataset} {exp.profile} epochs={exp.epochs}", flush=True)
    returncode = run_command(command, stdout_path, stderr_path)
    duration = time.time() - start_time

    rca_files = newest_after(before_rca, RCA_ROOT / exp.dataset.replace(".csv", ""), "*_rca.json")
    label_files = newest_after(before_labels, LABEL_ROOT / "journal_validation" / exp.exp_id, "*.csv")
    rca_path = rca_files[-1] if rca_files else None
    label_path = label_files[-1] if label_files else None

    row = {
        "exp_id": exp.exp_id,
        "dataset": exp.dataset,
        "profile": exp.profile,
        "epochs": exp.epochs,
        "status": "ok" if returncode == 0 else f"failed:{returncode}",
        "rca_json": str(rca_path) if rca_path else "",
        "label_csv": str(label_path) if label_path else "",
    }
    if returncode == 0 and rca_path is not None:
        outputs = eval_rca(rca_path, exp.exp_id)
        if exp.run_static_baselines:
            outputs.update(eval_static_baselines(rca_path, exp))
        channel = read_mean_row(outputs["channel"])
        group = read_mean_row(outputs["group"])
        hier = read_mean_row(outputs["hierarchical"])
        row.update(
            {
                "channel_MRR": float(channel.get("MRR", 0.0) or 0.0),
                "channel_Hit@1": float(channel.get("Hit@1", 0.0) or 0.0),
                "channel_Hit@3": float(channel.get("Hit@3", 0.0) or 0.0),
                "channel_Hit@5": float(channel.get("Hit@5", 0.0) or 0.0),
                "group_MRR": float(group.get("MRR", 0.0) or 0.0),
                "group_Hit@1": float(group.get("Hit@1", 0.0) or 0.0),
                "group_Hit@3": float(group.get("Hit@3", 0.0) or 0.0),
                "group_Hit@5": float(group.get("Hit@5", 0.0) or 0.0),
                "conditional_variable_MRR": float(hier.get("conditional_variable_MRR", 0.0) or 0.0),
                "conditional_variable_Hit@1": float(hier.get("conditional_variable_Hit@1", 0.0) or 0.0),
                "conditional_variable_Hit@3": float(hier.get("conditional_variable_Hit@3", 0.0) or 0.0),
                "conditional_variable_Hit@5": float(hier.get("conditional_variable_Hit@5", 0.0) or 0.0),
            }
        )
        row.update(read_label_summary(label_path))

    meta.update(
        {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": duration,
            "returncode": returncode,
            "rca_json": str(rca_path) if rca_path else None,
            "label_csv": str(label_path) if label_path else None,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    append_summary(row)
    write_markdown_summary()
    marker.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] {exp.exp_id}: {row['status']} duration={duration/60:.1f}min", flush=True)
    return row


def main() -> int:
    args = parse_args()
    ensure_dirs()
    experiments = list(SUITES[args.suite])
    if args.only:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp.exp_id in wanted]
    if args.max_runs and args.max_runs > 0:
        experiments = experiments[: args.max_runs]
    ran = 0
    for exp in experiments:
        if args.skip_completed and completed_marker(exp.exp_id).exists():
            print(f"[SKIP] {exp.exp_id}: completed marker exists", flush=True)
            continue
        run_experiment(exp)
        ran += 1
    print(f"[QUEUE DONE] ran={ran}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
