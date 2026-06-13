import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\la_v12")
PYTHON = Path(r"D:\Anaconda3\envs\lagraph5070\python.exe")
DATASET = "SWAT_A1A2_Physical_v1.csv"
RCA_DIR = ROOT / "result" / "rca" / "SWAT_A1A2_Physical_v1"
ANALYSIS_DIR = ROOT / "result" / "analysis" / "journal_validation"
LOG_DIR = ROOT / "result" / "logs"
SUMMARY_CSV = ANALYSIS_DIR / "root_response_swat_overnight_summary.csv"
SUMMARY_MD = ANALYSIS_DIR / "root_response_swat_overnight_summary.md"


EXPERIMENTS = [
    {
        "id": "root_response_swat_e8_confidence_gate_prob",
        "note": "Current root-response profile: confidence-gated response penalty, RCA uses final root probability.",
        "extra": [],
    },
    {
        "id": "root_response_swat_e8_no_conf_gate",
        "note": "Ablation: disables source-confidence discount to isolate whether the new gate helps SWAT.",
        "extra": ["--root-response-confidence-discount", "0.0"],
    },
    {
        "id": "root_response_swat_e8_penalty05_gate",
        "note": "Tests lower response penalty initialization, which slightly improved WADI variable MRR but weakened group guardrail.",
        "extra": ["--root-response-penalty-init", "0.5"],
    },
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text)


def run_streamed(cmd: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MPLBACKEND"] = "Agg"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{now()}] CMD: {' '.join(cmd)}\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        rc = process.wait()
        log.write(f"\n[{now()}] EXIT={rc}\n")
        return rc


def latest_rca_after(start_ts: float) -> Path | None:
    if not RCA_DIR.exists():
        return None
    files = [p for p in RCA_DIR.glob("*_rca.json") if p.stat().st_mtime >= start_ts]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read_metric_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    metric_names = ["MRR", "Hit@1", "Hit@3", "Hit@5", "MAP@3", "NDCG@3"]
    result: dict[str, float] = {}
    for name in metric_names:
        vals: list[float] = []
        for row in rows:
            value = row.get(name)
            if value in (None, ""):
                continue
            try:
                vals.append(float(value))
            except ValueError:
                continue
        if vals:
            result[name] = sum(vals) / len(vals)
    return result


def evaluate(exp_id: str, rca_path: Path) -> tuple[Path, Path, dict[str, float], dict[str, float]]:
    channel_csv = ANALYSIS_DIR / f"{exp_id}_channel_pred_key15.csv"
    group_csv = ANALYSIS_DIR / f"{exp_id}_group_pred_key15.csv"
    commands = [
        [
            str(PYTHON),
            "scripts/evaluate_rca.py",
            "--rca",
            str(rca_path.relative_to(ROOT)),
            "--scope",
            "channel",
            "--event-source",
            "predicted",
            "--prediction-key",
            "15",
            "--method-name",
            exp_id,
            "--save-csv",
            str(channel_csv.relative_to(ROOT)),
        ],
        [
            str(PYTHON),
            "scripts/evaluate_rca.py",
            "--rca",
            str(rca_path.relative_to(ROOT)),
            "--scope",
            "group",
            "--event-source",
            "predicted",
            "--prediction-key",
            "15",
            "--method-name",
            exp_id,
            "--save-csv",
            str(group_csv.relative_to(ROOT)),
        ],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    return channel_csv, group_csv, read_metric_csv(channel_csv), read_metric_csv(group_csv)


def write_summary(rows: list[dict[str, object]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "exp_id",
        "status",
        "return_code",
        "started_at",
        "finished_at",
        "rca_json",
        "channel_csv",
        "group_csv",
        "channel_MRR",
        "channel_Hit@1",
        "channel_Hit@3",
        "channel_MAP@3",
        "channel_NDCG@3",
        "group_MRR",
        "group_Hit@1",
        "group_Hit@3",
        "group_MAP@3",
        "group_NDCG@3",
        "note",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Root-response SWAT overnight summary",
        "",
        f"Updated: {now()}",
        "",
        "| experiment | status | ch MRR | ch Hit@3 | ch MAP@3 | group MRR | group Hit@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("exp_id", ""),
                row.get("status", ""),
                row.get("channel_MRR", ""),
                row.get("channel_Hit@3", ""),
                row.get("channel_MAP@3", ""),
                row.get("group_MRR", ""),
                row.get("group_Hit@3", ""),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> int:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        commit = "unknown"

    rows: list[dict[str, object]] = []
    append_text(SUMMARY_MD, f"\n\nStarted root-response SWAT overnight queue at {now()}, commit={commit}\n")

    for exp in EXPERIMENTS:
        exp_id = str(exp["id"])
        started_at = now()
        start_ts = time.time()
        log_path = LOG_DIR / f"{exp_id}.log"
        cmd = [
            str(PYTHON),
            "-u",
            "ts_benchmark/run_single.py",
            "--epochs",
            "8",
            "--checkpoint-policy",
            "best-val",
            "--datasets",
            DATASET,
            "--arch-profile",
            "source-bottleneck-root-response-rca",
            "--export-rca",
            "--rca-export-lite",
            "--rca-export-top-k",
            "20",
            "--rca-prediction-key",
            "15",
            "--num-workers",
            "0",
            "--inference-num-workers",
            "0",
            "--batch-size",
            "256",
            "--save-dir",
            f"label/journal_validation/{exp_id}",
        ] + list(exp.get("extra", []))
        append_text(SUMMARY_MD, f"\n## {exp_id}\n\nStarted: {started_at}\n\n")
        rc = run_streamed(cmd, log_path)
        finished_at = now()
        row: dict[str, object] = {
            "exp_id": exp_id,
            "status": "train_failed" if rc != 0 else "trained",
            "return_code": rc,
            "started_at": started_at,
            "finished_at": finished_at,
            "note": exp["note"],
        }
        if rc == 0:
            rca_path = latest_rca_after(start_ts)
            if rca_path is None:
                row["status"] = "missing_rca"
            else:
                row["rca_json"] = str(rca_path)
                try:
                    channel_csv, group_csv, channel_metrics, group_metrics = evaluate(exp_id, rca_path)
                    row["channel_csv"] = str(channel_csv)
                    row["group_csv"] = str(group_csv)
                    for metric, value in channel_metrics.items():
                        row[f"channel_{metric}"] = fmt(value)
                    for metric, value in group_metrics.items():
                        row[f"group_{metric}"] = fmt(value)
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "eval_failed"
                    row["note"] = f"{exp['note']} | eval_error={exc}"
        rows.append(row)
        write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
