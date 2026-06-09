#!/usr/bin/env python3
"""Run post-hoc ablations for unified event-aware source RCA.

This runner uses fixed exported RCA JSON files and does not retrain models. It
produces four comparable settings per dataset:

1. original: existing RCA export.
2. event_only: event refinement without source reranking.
3. rerank_only: conservative source reranking without event refinement.
4. full: event refinement plus conservative source reranking.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
ANALYSIS_DIR = PROJECT_ROOT / "result" / "analysis" / "journal_validation"
REFINED_DIR = PROJECT_ROOT / "result" / "rca_refined" / "unified_event_source_ablation"
PREDICTION_KEY = "15"


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    rca: Path


@dataclass(frozen=True)
class Variant:
    name: str
    event_mode: str
    rerank_mode: str
    source_rca: bool


VARIANTS = [
    Variant("event_only", "refine", "none", False),
    Variant("rerank_only", "rerank-only", "source", False),
    Variant("full", "refine", "source", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wadi-rca",
        type=Path,
        default=PROJECT_ROOT / "result" / "rca" / "WADI_A1_2017_ds10" / "20260609_024331_rca.json",
    )
    parser.add_argument(
        "--swat-rca",
        type=Path,
        default=PROJECT_ROOT / "result" / "rca" / "SWAT_A1A2_Physical_v1" / "20260609_045835_rca.json",
    )
    parser.add_argument("--merge-gap", type=int, default=30)
    parser.add_argument("--max-merge-span", type=int, default=1500)
    parser.add_argument("--expand-left", type=int, default=20)
    parser.add_argument("--expand-right", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--compact-json", action="store_true", default=True)
    return parser.parse_args()


def run(command: list[str | Path]) -> None:
    subprocess.run([str(part) for part in command], cwd=PROJECT_ROOT, check=True)


def output_rca(dataset: str, variant: str) -> Path:
    return REFINED_DIR / f"{dataset}_{variant}.json"


def apply_variant(dataset: DatasetSpec, variant: Variant, args: argparse.Namespace) -> Path:
    out = output_rca(dataset.label, variant.name)
    command: list[str | Path] = [
        PYTHON,
        "scripts/apply_event_source_rca_refinement.py",
        "--rca",
        dataset.rca,
        "--output",
        out,
        "--prediction-key",
        PREDICTION_KEY,
        "--event-mode",
        variant.event_mode,
        "--rerank-mode",
        variant.rerank_mode,
        "--merge-gap",
        str(args.merge_gap),
        "--max-merge-span",
        str(args.max_merge_span),
        "--min-event-len",
        "5",
        "--expand-left",
        str(args.expand_left),
        "--expand-right",
        str(args.expand_right),
        "--top-k",
        str(args.top_k),
        "--exported-weight",
        "1.0",
        "--base-weight",
        "0.10",
        "--root-weight",
        "0.10",
        "--source-gate-weight",
        "0.10",
        "--onset-weight",
        "0.15",
        "--source-interaction-weight",
        "0.20",
        "--mechanism-weight",
        "0.0",
        "--mechanism-residual-weight",
        "0.05",
        "--graph-penalty-weight",
        "0.02",
        "--propagation-penalty-weight",
        "0.02",
        "--hierarchy-boost",
        "0.05",
    ]
    if args.compact_json:
        command.append("--compact-json")
    run(command)
    return out


def eval_paths(dataset: str, variant: str) -> dict[str, Path]:
    stem = f"unified_ablation_{dataset}_{variant}"
    return {
        "channel": ANALYSIS_DIR / f"{stem}_channel_pred_key15.csv",
        "group": ANALYSIS_DIR / f"{stem}_group_pred_key15.csv",
        "hierarchical": ANALYSIS_DIR / f"{stem}_hierarchical_pred_key15.csv",
        "diagnosis_csv": ANALYSIS_DIR / f"{stem}_detection_vs_ranking.csv",
        "diagnosis_md": ANALYSIS_DIR / f"{stem}_detection_vs_ranking.md",
    }


def evaluate(dataset: str, variant: str, rca: Path) -> dict[str, Path]:
    paths = eval_paths(dataset, variant)
    run(
        [
            PYTHON,
            "scripts/evaluate_rca.py",
            "--rca",
            rca,
            "--scope",
            "channel",
            "--event-source",
            "predicted",
            "--prediction-key",
            PREDICTION_KEY,
            "--method-name",
            f"{dataset}_{variant}",
            "--save-csv",
            paths["channel"],
        ]
    )
    run(
        [
            PYTHON,
            "scripts/evaluate_rca.py",
            "--rca",
            rca,
            "--scope",
            "group",
            "--event-source",
            "predicted",
            "--prediction-key",
            PREDICTION_KEY,
            "--method-name",
            f"{dataset}_{variant}",
            "--save-csv",
            paths["group"],
        ]
    )
    run(
        [
            PYTHON,
            "scripts/evaluate_hierarchical_rca.py",
            "--rca",
            rca,
            "--event-source",
            "predicted",
            "--prediction-key",
            PREDICTION_KEY,
            "--method-name",
            f"{dataset}_{variant}",
            "--summary-only",
            "--save-csv",
            paths["hierarchical"],
        ]
    )
    run(
        [
            PYTHON,
            "scripts/analyze_rca_detection_vs_ranking.py",
            "--hierarchical-csv",
            paths["hierarchical"],
            "--method-name",
            f"{dataset}_{variant}",
            "--save-md",
            paths["diagnosis_md"],
            "--save-csv",
            paths["diagnosis_csv"],
        ]
    )
    return paths


def mean_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    rows = df[df["series_name"].astype(str).str.upper().eq("MEAN")]
    if rows.empty:
        return df.iloc[-1].to_dict()
    return rows.iloc[-1].to_dict()


def diagnosis_summary(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {
        "diagnosis_detection_problem": float(df["diagnosis"].isin(["A_event_not_detected", "B_event_weak_overlap"]).mean()),
        "diagnosis_ranking_problem": float(
            df["diagnosis"]
            .isin(["D_rank_near_miss_top3", "E_subsystem_wrong", "F_subsystem_ok_global_rank_wrong", "G_subsystem_ok_variable_missing"])
            .mean()
        ),
        "diagnosis_top1_ok": float(df["diagnosis"].eq("C_rank_ok_top1").mean()),
    }


def collect_row(dataset: str, variant: str, paths: dict[str, Path]) -> dict[str, Any]:
    channel = mean_row(paths["channel"])
    group = mean_row(paths["group"])
    hier = mean_row(paths["hierarchical"])
    row = {
        "dataset": dataset,
        "variant": variant,
        "matched": hier.get("matched"),
        "true_coverage": hier.get("true_coverage"),
        "event_iou": hier.get("event_iou"),
        "var_MRR": channel.get("MRR"),
        "var_Hit@1": channel.get("Hit@1"),
        "var_Hit@3": channel.get("Hit@3"),
        "var_Hit@5": channel.get("Hit@5"),
        "subsys_MRR": group.get("MRR"),
        "subsys_Hit@1": group.get("Hit@1"),
        "subsys_Hit@3": group.get("Hit@3"),
        "flat_variable_MRR": hier.get("flat_variable_MRR"),
        "conditional_variable_MRR": hier.get("conditional_variable_MRR"),
        "hierarchical_variable_MRR": hier.get("hierarchical_variable_MRR"),
    }
    row.update(diagnosis_summary(paths["diagnosis_csv"]))
    return row


def original_paths(dataset: str, original_stem: str) -> dict[str, Path]:
    paths = {
        "channel": ANALYSIS_DIR / f"{original_stem}_channel_pred_key15.csv",
        "group": ANALYSIS_DIR / f"{original_stem}_group_pred_key15.csv",
        "hierarchical": ANALYSIS_DIR / f"{original_stem}_hierarchical_pred_key15.csv",
        "diagnosis_csv": ANALYSIS_DIR / f"{original_stem}_detection_vs_ranking.csv",
        "diagnosis_md": ANALYSIS_DIR / f"{original_stem}_detection_vs_ranking.md",
    }
    if not paths["diagnosis_csv"].exists():
        run(
            [
                PYTHON,
                "scripts/analyze_rca_detection_vs_ranking.py",
                "--hierarchical-csv",
                paths["hierarchical"],
                "--method-name",
                f"{dataset}_original",
                "--save-md",
                paths["diagnosis_md"],
                "--save-csv",
                paths["diagnosis_csv"],
            ]
        )
    return paths


def write_summary(rows: list[dict[str, Any]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "unified_rca_posthoc_ablation_summary.csv"
    md_path = ANALYSIS_DIR / "unified_rca_posthoc_ablation_summary.md"
    columns = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    display = [
        "dataset",
        "variant",
        "matched",
        "true_coverage",
        "event_iou",
        "var_MRR",
        "var_Hit@1",
        "var_Hit@3",
        "var_Hit@5",
        "subsys_MRR",
        "subsys_Hit@1",
        "conditional_variable_MRR",
        "hierarchical_variable_MRR",
        "diagnosis_detection_problem",
        "diagnosis_ranking_problem",
    ]
    lines = [
        "# Unified RCA Post-hoc Ablation Summary",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| " + " | ".join(display) + " |",
        "| " + " | ".join("---" for _ in display) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in display) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(csv_path)


def main() -> None:
    args = parse_args()
    datasets = [
        DatasetSpec("wadi", args.wadi_rca),
        DatasetSpec("swat", args.swat_rca),
    ]
    original_stems = {
        "wadi": "weight_search_wadi_onset_r20_g35_o60_e8",
        "swat": "weight_search_swat_center_confirm_r30_g35_o60_e8",
    }
    rows = []
    for dataset in datasets:
        rows.append(collect_row(dataset.label, "original", original_paths(dataset.label, original_stems[dataset.label])))
        for variant in VARIANTS:
            rca = apply_variant(dataset, variant, args)
            paths = evaluate(dataset.label, variant.name, rca)
            rows.append(collect_row(dataset.label, variant.name, paths))
    write_summary(rows)


if __name__ == "__main__":
    main()
