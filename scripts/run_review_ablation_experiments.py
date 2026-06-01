"""Run reviewer-facing RCA ablations from existing LaGraph RCA exports.

The goal is to avoid repeating training when the question is only about RCA
evidence fusion. Training-dependent ablations such as no-normal-prior are left
as pending unless explicitly run through run_single.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"D:\Anaconda3\envs\lagraph5070\python.exe")
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis"
OUT_DIR = ANALYSIS_DIR / "review_ablation"
LOG_DIR = PROJECT_ROOT / "logs" / "review_ablation"


@dataclass(frozen=True)
class SeriesRun:
    key: str
    dataset: str
    rca: Path


def latest_main_rcas() -> list[SeriesRun]:
    candidates: dict[str, SeriesRun] = {}
    wanted = {
        "Q01_wadi_soft_allkeys_e8": "wadi",
        "Q02_swat_soft_allkeys_e8": "swat",
    }
    for path in sorted(ANALYSIS_DIR.glob("paper_experiment_queue_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            exp = record.get("experiment", {})
            exp_id = exp.get("exp_id")
            if exp_id not in wanted:
                continue
            rcas = record.get("rcas") or []
            if not rcas:
                continue
            rca_path = Path(rcas[-1])
            if rca_path.exists():
                candidates[wanted[exp_id]] = SeriesRun(wanted[exp_id], exp["dataset"], rca_path)
    return [candidates[k] for k in ("wadi", "swat") if k in candidates]


def run(cmd: list[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def rescore(src: Path, dst: Path, params: dict[str, float], log: Path) -> int:
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "rescore_rca_report.py"),
        "--rca",
        str(src),
        "--output",
        str(dst),
        "--base-weight",
        str(params["base"]),
        "--mechanism-weight",
        str(params["mechanism"]),
        "--onset-weight",
        str(params["onset"]),
        "--source-interaction-weight",
        str(params["interaction"]),
        "--graph-penalty-weight",
        str(params["graph_penalty"]),
        "--hierarchy-boost",
        str(params["hierarchy"]),
    ]
    return run(cmd, log)


def evaluate_lagraph(rca: Path, method: str, prefix: Path, log: Path) -> dict:
    hier_csv = prefix.with_name(prefix.name + "_hierarchical.csv")
    flat_csv = prefix.with_name(prefix.name + "_flat.csv")
    commands = {
        "hierarchical": [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "evaluate_hierarchical_rca.py"),
            "--rca",
            str(rca),
            "--event-source",
            "predicted",
            "--prediction-key",
            "15",
            "--method-name",
            method,
            "--summary-only",
            "--save-csv",
            str(hier_csv),
        ],
        "flat": [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "evaluate_rca.py"),
            "--rca",
            str(rca),
            "--scope",
            "channel",
            "--event-source",
            "predicted",
            "--prediction-key",
            "15",
            "--method-name",
            method,
            "--save-csv",
            str(flat_csv),
        ],
    }
    rc = {name: run(cmd, log) for name, cmd in commands.items()}
    return {"hierarchical": hier_csv, "flat": flat_csv, "returncodes": rc}


def evaluate_static(series: SeriesRun, prefix: Path, log: Path) -> Path:
    csv_path = prefix.with_name(prefix.name + "_static.csv")
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "evaluate_static_rca_baselines.py"),
        "--series",
        series.dataset,
        "--rca",
        str(series.rca),
        "--scope",
        "channel",
        "--event-source",
        "predicted",
        "--prediction-key",
        "15",
        "--include-random-graph",
        "--save-csv",
        str(csv_path),
    ]
    run(cmd, log)
    return csv_path


def mean_row(path: Path, method: str | None = None) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if method is not None and "method" in df.columns:
        df = df[df["method"] == method]
    if df.empty:
        return {}
    if "series_name" in df.columns and (df["series_name"] == "MEAN").any():
        row = df[df["series_name"] == "MEAN"].iloc[-1]
    else:
        row = df.select_dtypes("number").mean()
    keys = [
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
        "MRR",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Precision@1",
        "Precision@3",
        "Precision@5",
        "MAP@3",
        "MAP@5",
    ]
    return {key: float(row[key]) for key in keys if key in row and pd.notna(row[key])}


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = LOG_DIR / f"review_ablation_{run_id}.log"

    series_runs = latest_main_rcas()
    if not series_runs:
        raise SystemExit("No Q01/Q02 RCA exports found. Run event-quality suite first.")

    score_configs = {
        "base_residual_only": {"base": 1.0, "mechanism": 0.0, "onset": 0.0, "interaction": 0.0, "graph_penalty": 0.0, "hierarchy": 0.0},
        "no_hierarchy": {"base": 0.50, "mechanism": 0.0, "onset": 0.75, "interaction": 2.0, "graph_penalty": 0.10, "hierarchy": 0.0},
        "no_source_interaction": {"base": 0.50, "mechanism": 0.0, "onset": 0.75, "interaction": 0.0, "graph_penalty": 0.10, "hierarchy": 0.20},
    }

    rows = []
    audit_rows = []
    for series in series_runs:
        audit_rows.append({"dataset": series.dataset, "experiment": "main soft-hierarchical-rca e8", "status": "already_done", "source": str(series.rca)})
        main_outputs = evaluate_lagraph(
            series.rca,
            "soft_hierarchical_main",
            OUT_DIR / f"{run_id}_{series.key}_soft_hierarchical_main",
            log,
        )
        main_row = {"dataset": series.dataset, "experiment": "soft_hierarchical_main"}
        main_row.update(mean_row(main_outputs["hierarchical"]))
        rows.append(main_row)

        for name, params in score_configs.items():
            ablated_rca = OUT_DIR / f"{run_id}_{series.key}_{name}.json"
            prefix = OUT_DIR / f"{run_id}_{series.key}_{name}"
            if not (args.skip_existing and ablated_rca.exists()):
                rescore(series.rca, ablated_rca, params, log)
            outputs = evaluate_lagraph(ablated_rca, name, prefix, log)
            row = {"dataset": series.dataset, "experiment": name}
            row.update(mean_row(outputs["hierarchical"]))
            rows.append(row)

        static_path = evaluate_static(series, OUT_DIR / f"{run_id}_{series.key}_static_graph_controls", log)
        for method in ["zscore", "correlation_prior", "random_graph_prior"]:
            row = {"dataset": series.dataset, "experiment": method}
            row.update(mean_row(static_path, method))
            rows.append(row)

    summary_csv = OUT_DIR / f"review_ablation_summary_{run_id}.csv"
    summary_md = OUT_DIR / f"review_ablation_summary_{run_id}.md"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    cols = [
        "dataset",
        "experiment",
        "matched",
        "subsystem_MRR",
        "subsystem_Hit@1",
        "flat_variable_MRR",
        "flat_variable_Hit@1",
        "conditional_variable_MRR",
        "MRR",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "MAP@3",
        "MAP@5",
    ]
    md = [
        "# Reviewer-Facing Ablation Summary",
        "",
        f"- Run id: `{run_id}`",
        f"- Log: `{log}`",
        "- Training-dependent `no normal prior` and P2 stability/long-epoch confirmation are not rerun here.",
        "",
        "## Coverage Audit",
        "",
        markdown_table(audit_rows, ["dataset", "experiment", "status", "source"]),
        "",
        "## Results",
        "",
        markdown_table(rows, [c for c in cols if any(c in r for r in rows)]),
        "",
        "## Interpretation",
        "",
        "- `base_residual_only` tests whether residual magnitude alone explains RCA.",
        "- `no_hierarchy` tests whether the soft subsystem hierarchy contributes beyond flat variable ranking.",
        "- `no_source_interaction` tests whether source-oriented interaction is useful.",
        "- `zscore`, `correlation_prior`, and `random_graph_prior` test whether simple statistics or random graph propagation explain the result.",
    ]
    summary_md.write_text("\n".join(md), encoding="utf-8")
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
