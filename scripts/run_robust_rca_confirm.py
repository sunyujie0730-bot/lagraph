import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\la_v12")
PYTHON = Path(r"D:\Anaconda3\envs\lagraph5070\python.exe")
DATASET = "SWAT_A1A2_Physical_v1.csv"
RCA_DIR = ROOT / "result" / "rca" / "SWAT_A1A2_Physical_v1"
LABEL_DIR = ROOT / "result" / "label" / "LaGraph"
LOG_DIR = ROOT / "logs" / "robust_rca_confirm"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT = Path.home() / "Desktop" / f"LaGraph_robust_RCA_confirm_{RUN_ID}.md"


FAST_EXPERIMENTS = [
    {
        "id": "R01_robust_fast8",
        "question": "Robust RCA profile without synthetic RCA loss.",
        "args": [
            "--epochs", "8",
            "--datasets", DATASET,
            "--arch-profile", "mechanism-predictive-robust-rca",
            "--vq-cooldown-epochs", "2",
            "--num-workers", "2",
            "--prefetch-factor", "2",
            "--export-rca",
            "--rca-export-lite",
            "--rca-export-top-k", "20",
            "--rca-prediction-key", "15",
        ],
    },
    {
        "id": "R02_robust_synth010_fast8",
        "question": "Robust RCA profile with synthetic RCA loss 0.10, interval 4.",
        "args": [
            "--epochs", "8",
            "--datasets", DATASET,
            "--arch-profile", "mechanism-predictive-robust-rca",
            "--vq-cooldown-epochs", "2",
            "--lambda-synthetic-rca", "0.10",
            "--synthetic-aux-interval", "4",
            "--num-workers", "2",
            "--prefetch-factor", "2",
            "--export-rca",
            "--rca-export-lite",
            "--rca-export-top-k", "20",
            "--rca-prediction-key", "15",
        ],
    },
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_report(text):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def run_streamed(cmd, log_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{now()}] CMD: {' '.join(map(str, cmd))}\n\n")
        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        return proc.wait()


def latest_new_file(directory, pattern, timestamp):
    if not directory.exists():
        return None
    files = [p for p in directory.glob(pattern) if p.stat().st_mtime >= timestamp]
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1] if files else None


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def summarize_label(path):
    if not path or not path.exists():
        return {}
    rows = read_csv_rows(path)
    if not rows:
        return {}
    best_aff = max(rows, key=lambda r: as_float(r, "affiliation_f", -1.0))
    best_adjust = max(rows, key=lambda r: as_float(r, "adjust_f_score", -1.0))
    best_point = max(rows, key=lambda r: as_float(r, "f_score", -1.0))
    return {
        "label_csv": str(path),
        "best_aff_ratio": best_aff.get("typical_anomaly_ratio", ""),
        "best_aff_f": as_float(best_aff, "affiliation_f"),
        "best_adjust_ratio": best_adjust.get("typical_anomaly_ratio", ""),
        "best_adjust_f": as_float(best_adjust, "adjust_f_score"),
        "best_point_ratio": best_point.get("typical_anomaly_ratio", ""),
        "best_point_f": as_float(best_point, "f_score"),
        "fit_time": as_float(rows[0], "fit_time"),
        "inference_time": as_float(rows[0], "inference_time"),
    }


def summarize_rca_csv(path):
    rows = read_csv_rows(path)
    out = {"csv": str(path), "n_events": len(rows)}
    for metric in ["MRR", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5"]:
        vals = [as_float(row, metric) for row in rows]
        out[metric] = sum(vals) / max(len(vals), 1)
    return out


def evaluate_rca(rca_json, exp_id):
    jobs = [
        ("pred_channel", ["--scope", "channel", "--event-source", "predicted", "--prediction-key", "15"]),
        ("pred_group", ["--scope", "group", "--event-source", "predicted", "--prediction-key", "15"]),
        ("true_channel", ["--scope", "channel", "--event-source", "true"]),
        ("true_group", ["--scope", "group", "--event-source", "true"]),
    ]
    results = {}
    for name, args in jobs:
        out_csv = RCA_DIR / f"{rca_json.stem}_{exp_id}_{name}.csv"
        log_path = LOG_DIR / f"{RUN_ID}_{exp_id}_{name}_eval.log"
        cmd = [
            PYTHON,
            ROOT / "scripts" / "evaluate_rca.py",
            "--rca", rca_json,
            *args,
            "--save-csv", out_csv,
        ]
        code = run_streamed(cmd, log_path)
        results[name] = {"returncode": code, **summarize_rca_csv(out_csv)}
    return results


def metric(summary, *path, default=0.0):
    cur = summary
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return float(cur) if isinstance(cur, (int, float)) else default


def format_row(summary):
    return (
        f"| {summary['id']} | {summary['status']} | "
        f"{metric(summary, 'detection', 'best_aff_f'):.4f} | "
        f"{metric(summary, 'detection', 'best_adjust_f'):.4f} | "
        f"{metric(summary, 'detection', 'best_point_f'):.4f} | "
        f"{metric(summary, 'rca', 'pred_channel', 'MRR'):.4f} | "
        f"{metric(summary, 'rca', 'pred_channel', 'Hit@1'):.4f} | "
        f"{metric(summary, 'rca', 'pred_channel', 'Hit@3'):.4f} | "
        f"{metric(summary, 'rca', 'pred_group', 'MRR'):.4f} | "
        f"{metric(summary, 'rca', 'true_channel', 'MRR'):.4f} | "
        f"{summary.get('elapsed_human', '')} |"
    )


def run_experiment(exp):
    start = time.time()
    exp_id = exp["id"]
    append_report(f"## {exp_id}\n\n- Start: {now()}\n- Question: {exp['question']}")
    train_log = LOG_DIR / f"{RUN_ID}_{exp_id}_train.log"
    cmd = [PYTHON, ROOT / "ts_benchmark" / "run_single.py", *exp["args"]]
    code = run_streamed(cmd, train_log)
    elapsed = time.time() - start
    rca_json = latest_new_file(RCA_DIR, "*_rca.json", start)
    label_csv = latest_new_file(LABEL_DIR, "LaGraph_*.csv", start)
    summary = {
        "id": exp_id,
        "status": "ok" if code == 0 else f"failed:{code}",
        "returncode": code,
        "elapsed_human": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "train_log": str(train_log),
        "rca_json": str(rca_json) if rca_json else "",
        "detection": summarize_label(label_csv),
        "rca": evaluate_rca(rca_json, exp_id) if code == 0 and rca_json else {},
    }
    append_report(format_row(summary))
    append_report(f"- Train log: `{train_log}`\n- RCA json: `{summary['rca_json']}`")
    (LOG_DIR / f"{RUN_ID}_{exp_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RCA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("", encoding="utf-8")
    append_report(
        "# LaGraph Robust RCA Confirmation\n\n"
        f"- Start time: {now()}\n"
        f"- Dataset: `{DATASET}`\n"
        "- Goal: confirm whether robust RCA export improves predicted-event variable-level RCA.\n"
        "- Robust RCA profile uses mechanism learning internally but exports RCA with `base + graph`, `mechanism=0`.\n"
    )
    append_report(
        "| Experiment | Status | Aff-F | Adjust-F | Point-F1 | Pred Ch MRR | Hit@1 | Hit@3 | Pred Group MRR | True Ch MRR | Time |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    summaries = [run_experiment(exp) for exp in FAST_EXPERIMENTS]
    ok = [s for s in summaries if s["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda s: metric(s, "rca", "pred_channel", "MRR"))
        best_mrr = metric(best, "rca", "pred_channel", "MRR")
        best_hit1 = metric(best, "rca", "pred_channel", "Hit@1")
        append_report(
            "## Fast-stage decision\n\n"
            f"- Best fast experiment: `{best['id']}` with Pred Channel MRR={best_mrr:.4f}, Hit@1={best_hit1:.4f}.\n"
        )
        if best_mrr >= 0.45 and best_hit1 >= 0.30:
            args = [
                "--epochs", "15",
                "--datasets", DATASET,
                "--arch-profile", "mechanism-predictive-robust-rca",
                "--vq-cooldown-epochs", "3",
                "--num-workers", "2",
                "--prefetch-factor", "2",
                "--export-rca",
                "--rca-export-lite",
                "--rca-export-top-k", "20",
                "--rca-prediction-key", "15",
            ]
            if "synth010" in best["id"]:
                args.extend(["--lambda-synthetic-rca", "0.10", "--synthetic-aux-interval", "4"])
            confirm = {
                "id": f"R03_confirm15_from_{best['id']}",
                "question": "15-epoch confirmation of the best robust RCA fast-stage setting.",
                "args": args,
            }
            summaries.append(run_experiment(confirm))
        else:
            append_report("- Decision: skip 15-epoch confirmation because the fast-stage gain is below threshold.")

    append_report("## Final table\n")
    append_report(
        "| Experiment | Status | Aff-F | Adjust-F | Point-F1 | Pred Ch MRR | Hit@1 | Hit@3 | Pred Group MRR | True Ch MRR | Time |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(format_row(s) for s in summaries)
    )
    append_report(f"Finished at {now()}.")
    print(f"REPORT={REPORT}")


if __name__ == "__main__":
    main()
