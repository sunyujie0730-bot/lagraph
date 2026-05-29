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
LABEL_DIR = ROOT / "result" / "label" / "LaGraph"
EXPERIMENT_DIR = ROOT / "result" / "experiments"
LOG_DIR = ROOT / "logs" / "overnight_rca"
DESKTOP = Path.home() / "Desktop"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT = DESKTOP / f"LaGraph_overnight_RCA_experiments_{RUN_ID}.md"


EXPERIMENTS = [
    {
        "id": "E01_baseline_mechanism_predictive_fast8",
        "question": "同等训练预算下，当前 mechanism-predictive 基线的检测与 RCA 水平。",
        "args": [
            "--epochs", "8",
            "--datasets", DATASET,
            "--arch-profile", "mechanism-predictive",
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
        "id": "E02_scheme1_rca_interval4_fast8",
        "question": "方案1：加入变量级 synthetic RCA ranking loss，周期为每 4 个 batch 一次。",
        "args": [
            "--epochs", "8",
            "--datasets", DATASET,
            "--arch-profile", "mechanism-predictive-rca",
            "--vq-cooldown-epochs", "2",
            "--synthetic-aux-interval", "4",
            "--num-workers", "2",
            "--prefetch-factor", "2",
            "--export-rca",
            "--rca-export-lite",
            "--rca-export-top-k", "20",
            "--rca-prediction-key", "15",
        ],
    },
    {
        "id": "E03_scheme1_rca_interval4_lambda010_fast8",
        "question": "方案1强监督版：保持 interval=4，将 synthetic RCA loss 从 0.05 提到 0.10。",
        "args": [
            "--epochs", "8",
            "--datasets", DATASET,
            "--arch-profile", "mechanism-predictive-rca",
            "--vq-cooldown-epochs", "2",
            "--synthetic-aux-interval", "4",
            "--lambda-synthetic-rca", "0.10",
            "--num-workers", "2",
            "--prefetch-factor", "2",
            "--export-rca",
            "--rca-export-lite",
            "--rca-export-top-k", "20",
            "--rca-prediction-key", "15",
        ],
    },
    {
        "id": "E04_scheme1_rca_interval8_fast8",
        "question": "方案1低频版：每 8 个 batch 一次，判断辅助监督频率降低后是否仍保留收益。",
        "args": [
            "--epochs", "8",
            "--datasets", DATASET,
            "--arch-profile", "mechanism-predictive-rca",
            "--vq-cooldown-epochs", "2",
            "--synthetic-aux-interval", "8",
            "--num-workers", "2",
            "--prefetch-factor", "2",
            "--export-rca",
            "--rca-export-lite",
            "--rca-export-top-k", "20",
            "--rca-prediction-key", "15",
        ],
    },
]


def append_report(text: str) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def files_newer_than(directory: Path, pattern: str, timestamp: float):
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.glob(pattern) if p.stat().st_mtime >= timestamp],
        key=lambda p: p.stat().st_mtime,
    )


def latest_new_file(directory: Path, pattern: str, timestamp: float):
    files = files_newer_than(directory, pattern, timestamp)
    return files[-1] if files else None


def run_streamed(cmd, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{now()}] CMD: {' '.join(map(str, cmd))}\n\n")
        process = subprocess.Popen(
            list(map(str, cmd)),
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
        return process.wait()


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def summarize_label(path: Path):
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
        "best_aff_precision": as_float(best_aff, "affiliation_precision"),
        "best_aff_recall": as_float(best_aff, "affiliation_recall"),
        "best_adjust_ratio": best_adjust.get("typical_anomaly_ratio", ""),
        "best_adjust_f": as_float(best_adjust, "adjust_f_score"),
        "best_point_ratio": best_point.get("typical_anomaly_ratio", ""),
        "best_point_f": as_float(best_point, "f_score"),
        "fit_time": as_float(rows[0], "fit_time"),
        "inference_time": as_float(rows[0], "inference_time"),
    }


def summarize_train_history(timestamp: float):
    dirs = files_newer_than(EXPERIMENT_DIR, "Unknown_*", timestamp)
    if not dirs:
        return {}
    history = dirs[-1] / "train_history.json"
    if not history.exists():
        return {}
    try:
        data = json.loads(history.read_text(encoding="utf-8"))
    except Exception:
        return {"train_history": str(history)}
    return {
        "train_history": str(history),
        "best_val_loss": data.get("best_val_loss"),
        "best_epoch": data.get("best_epoch"),
        "total_time_human": data.get("total_time_human"),
        "early_stopped": data.get("early_stopped"),
    }


def summarize_rca_csv(path: Path):
    if not path or not path.exists():
        return {}
    rows = read_csv_rows(path)
    if not rows:
        return {}
    metrics = ["MRR", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5"]
    summary = {"csv": str(path), "n_events": len(rows)}
    for metric in metrics:
        values = [as_float(r, metric) for r in rows]
        summary[metric] = sum(values) / max(len(values), 1)
    return summary


def evaluate_rca(rca_json: Path, exp_id: str):
    outputs = {}
    jobs = [
        ("pred_channel", ["--scope", "channel", "--event-source", "predicted", "--prediction-key", "15"]),
        ("pred_group", ["--scope", "group", "--event-source", "predicted", "--prediction-key", "15"]),
        ("true_channel", ["--scope", "channel", "--event-source", "true"]),
        ("true_group", ["--scope", "group", "--event-source", "true"]),
    ]
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
        outputs[name] = {
            "returncode": code,
            "log": str(log_path),
            **summarize_rca_csv(out_csv),
        }
    return outputs


def metric(summary, path, default=0.0):
    current = summary
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return default


def experiment_analysis(summary, baseline=None):
    pred_mrr = metric(summary, ["rca", "pred_channel", "MRR"])
    pred_h1 = metric(summary, ["rca", "pred_channel", "Hit@1"])
    true_mrr = metric(summary, ["rca", "true_channel", "MRR"])
    group_mrr = metric(summary, ["rca", "pred_group", "MRR"])
    aff = metric(summary, ["detection", "best_aff_f"])
    adjust = metric(summary, ["detection", "best_adjust_f"])
    if baseline:
        d_pred = pred_mrr - metric(baseline, ["rca", "pred_channel", "MRR"])
        d_h1 = pred_h1 - metric(baseline, ["rca", "pred_channel", "Hit@1"])
        d_true = true_mrr - metric(baseline, ["rca", "true_channel", "MRR"])
        d_aff = aff - metric(baseline, ["detection", "best_aff_f"])
        verdict = []
        if d_pred >= 0.03 or d_h1 >= 0.05:
            verdict.append("变量级 predicted-event RCA 有实质提升，方案值得继续。")
        elif d_true >= 0.03 and d_pred < 0.03:
            verdict.append("true-event RCA 有提升但 predicted-event 不明显，主要瓶颈可能在检测边界或预测事件匹配。")
        elif group_mrr > pred_mrr + 0.10:
            verdict.append("子系统级强于变量级，说明图/机制信号偏粗粒度，变量级定位仍需更强约束。")
        else:
            verdict.append("相对基线没有明显提升，不应直接作为论文主结果。")
        if d_aff < -0.03:
            verdict.append("Aff-F 下降较多，需要警惕 RCA 辅助损失伤害检测。")
        elif d_aff > 0.03:
            verdict.append("Aff-F 同时提升，说明辅助目标没有明显破坏检测。")
        return " ".join(verdict)
    if pred_mrr >= 0.45 and pred_h1 >= 0.25:
        return "基线变量级 RCA 已经有一定强度，后续提升需要看 MRR/Hit@1 是否继续增加。"
    return "基线变量级 RCA 不高，方案1需要重点观察 predicted-event channel MRR 与 Hit@1。"


def format_summary(summary):
    det = summary.get("detection", {})
    rca = summary.get("rca", {})
    return (
        f"| {summary['id']} | {summary.get('status', '')} | "
        f"{det.get('best_aff_f', 0):.4f} | {det.get('best_adjust_f', 0):.4f} | {det.get('best_point_f', 0):.4f} | "
        f"{metric(summary, ['rca', 'pred_channel', 'MRR']):.4f} | {metric(summary, ['rca', 'pred_channel', 'Hit@1']):.4f} | "
        f"{metric(summary, ['rca', 'pred_group', 'MRR']):.4f} | {metric(summary, ['rca', 'true_channel', 'MRR']):.4f} | "
        f"{summary.get('elapsed_human', '')} |"
    )


def write_initial_report():
    REPORT.write_text("", encoding="utf-8")
    append_report(
        f"# LaGraph Overnight RCA Experiments\n\n"
        f"- Start time: {now()}\n"
        f"- Project: `{ROOT}`\n"
        f"- Python: `{PYTHON}`\n"
        f"- Dataset: `{DATASET}`\n"
        f"- Goal: 快速判断方案1 `mechanism-predictive-rca` 是否比 `mechanism-predictive` 更适合变量级 RCA。\n"
        f"- Primary metrics: predicted-event channel-level `MRR` / `Hit@1`，其次看 group-level RCA 和 Aff-F。\n"
    )
    append_report("## Planned Queue\n\n" + "\n".join(
        f"{i + 1}. `{exp['id']}`: {exp['question']}" for i, exp in enumerate(EXPERIMENTS)
    ))
    append_report("## Running Log\n\nEach experiment appends its own metrics and analysis below. A clean comparison table is written at the end.")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RCA_DIR.mkdir(parents=True, exist_ok=True)
    write_initial_report()
    summaries = []
    baseline = None

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        start_ts = time.time()
        append_report(f"## {exp_id}\n\n- Start: {now()}\n- Question: {exp['question']}")
        cmd = [PYTHON, ROOT / "ts_benchmark" / "run_single.py", *exp["args"]]
        run_log = LOG_DIR / f"{RUN_ID}_{exp_id}_train.log"
        code = run_streamed(cmd, run_log)
        end_ts = time.time()
        elapsed = end_ts - start_ts
        elapsed_human = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        rca_json = latest_new_file(RCA_DIR, "*_rca.json", start_ts)
        label_csv = latest_new_file(LABEL_DIR, "LaGraph_*.csv", start_ts)
        summary = {
            "id": exp_id,
            "question": exp["question"],
            "returncode": code,
            "status": "ok" if code == 0 else f"failed:{code}",
            "elapsed_seconds": elapsed,
            "elapsed_human": elapsed_human,
            "train_log": str(run_log),
            "rca_json": str(rca_json) if rca_json else "",
            "detection": summarize_label(label_csv) if label_csv else {},
            "train": summarize_train_history(start_ts),
            "rca": {},
        }

        if code == 0 and rca_json:
            summary["rca"] = evaluate_rca(rca_json, exp_id)
        elif not rca_json:
            append_report(f"- Warning: no new RCA json found for `{exp_id}`.")

        if baseline is None and exp_id.startswith("E01_"):
            baseline = summary
        summary["analysis"] = experiment_analysis(summary, baseline if baseline is not summary else None)
        summaries.append(summary)

        append_report(
            f"### Analysis: {exp_id}\n\n"
            f"{format_summary(summary)}\n\n"
            f"- Train log: `{run_log}`\n"
            f"- RCA json: `{summary.get('rca_json', '')}`\n"
            f"- Train history: `{summary.get('train', {}).get('train_history', '')}`\n"
            f"- Conclusion: {summary['analysis']}"
        )

        summary_json = LOG_DIR / f"{RUN_ID}_{exp_id}_summary.json"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_report("## Final Comparison\n")
    append_report(
        "| Experiment | Status | Best Aff-F | Best Adjust-F | Best Point-F1 | Pred Channel MRR | Pred Hit@1 | Pred Group MRR | True Channel MRR | Time |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(format_summary(s) for s in summaries)
    )

    if baseline and len(summaries) > 1:
        best = max(summaries[1:], key=lambda s: metric(s, ["rca", "pred_channel", "MRR"]))
        append_report(
            "## Overall Judgment\n\n"
            f"- Best non-baseline experiment by predicted channel MRR: `{best['id']}`.\n"
            f"- Baseline predicted channel MRR: {metric(baseline, ['rca', 'pred_channel', 'MRR']):.4f}; "
            f"best scheme MRR: {metric(best, ['rca', 'pred_channel', 'MRR']):.4f}.\n"
            f"- Baseline Hit@1: {metric(baseline, ['rca', 'pred_channel', 'Hit@1']):.4f}; "
            f"best scheme Hit@1: {metric(best, ['rca', 'pred_channel', 'Hit@1']):.4f}.\n"
            f"- Recommendation: {experiment_analysis(best, baseline)}"
        )

    append_report(f"Finished at {now()}.")
    print(f"REPORT={REPORT}")


if __name__ == "__main__":
    main()
