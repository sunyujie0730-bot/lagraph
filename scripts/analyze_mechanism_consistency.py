#!/usr/bin/env python3
"""Audit whether LaGraph RCA rankings are supported by mechanism evidence.

This is not another RCA metric. It answers a reviewer-facing question:
does the exported top root cause agree with interpretable evidence such as
normal-mechanism deviation, onset evidence, source score, and subsystem rank?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_hierarchical_rca import (  # noqa: E402
    DEFAULT_REGISTRY,
    load_json,
    metric_row,
    root_cause_group_name,
    selected_events,
)


COMPONENTS = [
    "score",
    "base_score",
    "mechanism_score",
    "source_score",
    "source_gate_score",
    "onset_score",
    "propagation_score",
    "graph_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rca", type=Path, nargs="+", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--event-source", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--prediction-key", type=str, default="15")
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--support-k", type=int, default=3)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def ranking_by_component(event: dict, component: str, scope: str) -> list[str]:
    items = list(event.get("channel_ranking", []))
    if not items:
        return []
    if scope == "channel":
        ranked = sorted(
            items,
            key=lambda item: float(item.get(component, item.get("score", 0.0))),
            reverse=True,
        )
        return [str(item.get("name", "")) for item in ranked]

    group_scores: dict[str, float] = {}
    for item in items:
        name = str(item.get("name", ""))
        group = root_cause_group_name(item.get("group") or name)
        value = float(item.get(component, item.get("score", 0.0)))
        group_scores[group] = max(group_scores.get(group, float("-inf")), value)
    return [name for name, _ in sorted(group_scores.items(), key=lambda pair: pair[1], reverse=True)]


def exported_ranking(event: dict, scope: str) -> list[str]:
    if scope == "channel":
        return [str(item.get("name", "")) for item in event.get("channel_ranking", [])]
    names = [str(item.get("name", "")) for item in event.get("group_ranking", [])]
    if names:
        return list(dict.fromkeys(root_cause_group_name(name) for name in names))
    return ranking_by_component(event, "score", "group")


def first_rank(ranking: list[str], roots: set[str]) -> int | None:
    for idx, name in enumerate(ranking, start=1):
        if name in roots:
            return idx
    return None


def top_name(ranking: list[str]) -> str:
    return ranking[0] if ranking else ""


def component_nonzero(event: dict, component: str) -> float:
    values = [
        abs(float(item.get(component, 0.0)))
        for item in event.get("channel_ranking", [])
    ]
    if not values:
        return 0.0
    return 1.0 if max(values) > 1e-12 else 0.0


def analyze_one(path: Path, args: argparse.Namespace) -> list[dict]:
    rca = load_json(path)
    pairs, prediction_key = selected_events(rca, args)
    method = args.method_name or path.stem
    rows = []

    for meta_event, event, best_overlap in pairs:
        root_variables = set(meta_event.get("root_variables", []))
        root_groups = set(meta_event.get("root_groups", []))
        if not event or best_overlap <= 0:
            row = {
                "method": method,
                "rca_path": str(path),
                "series_name": rca.get("series_name", ""),
                "event_id": meta_event.get("event_id"),
                "prediction_key": prediction_key,
                "matched": 0.0,
                "top1_variable_correct": 0.0,
                "top1_group_correct": 0.0,
            }
            if root_variables:
                row.update({f"exported_variable_{k}": v for k, v in metric_row([], root_variables, args.k, "x").items()})
                for component in COMPONENTS:
                    row.update({
                        f"{component}_variable_{k}": v
                        for k, v in metric_row([], root_variables, args.k, "x").items()
                    })
                    row[f"{component}_root_variable_top{max(1, int(args.support_k))}"] = 0.0
            if root_groups:
                row.update({f"exported_group_{k}": v for k, v in metric_row([], root_groups, args.k, "x").items()})
                for component in COMPONENTS:
                    row.update({
                        f"{component}_group_{k}": v
                        for k, v in metric_row([], root_groups, args.k, "x").items()
                    })
                    row[f"{component}_root_group_top{max(1, int(args.support_k))}"] = 0.0
            row[f"top1_supported_by_mechanism_top{max(1, int(args.support_k))}"] = 0.0
            row[f"top1_supported_by_onset_top{max(1, int(args.support_k))}"] = 0.0
            row[f"top1_supported_by_source_top{max(1, int(args.support_k))}"] = 0.0
            rows.append(row)
            continue

        exported_channel_rank = exported_ranking(event, "channel")
        exported_group_rank = exported_ranking(event, "group")
        top_variable = top_name(exported_channel_rank)
        top_group = root_cause_group_name(top_variable) if top_variable else ""
        root_variable_rank = first_rank(exported_channel_rank, root_variables)
        root_group_rank = first_rank(exported_group_rank, root_groups)

        row = {
            "method": method,
            "rca_path": str(path),
            "series_name": rca.get("series_name", ""),
            "event_id": meta_event.get("event_id"),
            "event_start": meta_event.get("start"),
            "event_end": meta_event.get("end"),
            "pred_event_id": event.get("event_id"),
            "pred_start": event.get("start"),
            "pred_end": event.get("end"),
            "overlap": best_overlap,
            "prediction_key": prediction_key,
            "matched": 1.0,
            "root_variables": ";".join(sorted(root_variables)),
            "root_groups": ";".join(sorted(root_groups)),
            "top_variable": top_variable,
            "top_group": top_group,
            "top1_variable_correct": 1.0 if top_variable in root_variables else 0.0,
            "top1_group_correct": 1.0 if top_group in root_groups else 0.0,
            "exported_variable_rank": root_variable_rank or 0,
            "exported_group_rank": root_group_rank or 0,
        }

        if root_variables:
            row.update({f"exported_variable_{k}": v for k, v in metric_row(exported_channel_rank, root_variables, args.k, "x").items()})
        if root_groups:
            row.update({f"exported_group_{k}": v for k, v in metric_row(exported_group_rank, root_groups, args.k, "x").items()})

        for component in COMPONENTS:
            channel_rank = ranking_by_component(event, component, "channel")
            group_rank = ranking_by_component(event, component, "group")
            support_k = max(1, int(args.support_k))
            row[f"{component}_nonzero"] = component_nonzero(event, component)
            row[f"{component}_top_variable"] = top_name(channel_rank)
            row[f"{component}_top_group"] = top_name(group_rank)
            if root_variables:
                rank = first_rank(channel_rank, root_variables)
                row[f"{component}_root_variable_rank"] = rank or 0
                row[f"{component}_root_variable_top{support_k}"] = (
                    1.0 if rank is not None and rank <= support_k else 0.0
                )
                row.update({
                    f"{component}_variable_{k}": v
                    for k, v in metric_row(channel_rank, root_variables, args.k, "x").items()
                })
            if root_groups:
                rank = first_rank(group_rank, root_groups)
                row[f"{component}_root_group_rank"] = rank or 0
                row[f"{component}_root_group_top{support_k}"] = (
                    1.0 if rank is not None and rank <= support_k else 0.0
                )
                row.update({
                    f"{component}_group_{k}": v
                    for k, v in metric_row(group_rank, root_groups, args.k, "x").items()
                })

        mechanism_rank = ranking_by_component(event, "mechanism_score", "channel")
        onset_rank = ranking_by_component(event, "onset_score", "channel")
        source_rank = ranking_by_component(event, "source_score", "channel")
        support_k = max(1, int(args.support_k))
        row[f"top1_supported_by_mechanism_top{support_k}"] = (
            1.0 if top_variable and top_variable in mechanism_rank[:support_k] else 0.0
        )
        row[f"top1_supported_by_onset_top{support_k}"] = (
            1.0 if top_variable and top_variable in onset_rank[:support_k] else 0.0
        )
        row[f"top1_supported_by_source_top{support_k}"] = (
            1.0 if top_variable and top_variable in source_rank[:support_k] else 0.0
        )
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not col.endswith("_id")
        and col not in {"event_start", "event_end", "pred_start", "pred_end"}
    ]
    summary = df[numeric_cols].mean().to_frame("mean").T
    return summary


def main() -> None:
    args = parse_args()
    all_rows = []
    for path in args.rca:
        all_rows.extend(analyze_one(path, args))
    if not all_rows:
        raise SystemExit("No RCA events found.")

    df = pd.DataFrame(all_rows)
    summary = summarize(df)
    if not args.summary_only:
        display_cols = [
            "method",
            "series_name",
            "event_id",
            "matched",
            "root_variables",
            "top_variable",
            "top1_variable_correct",
            "top1_group_correct",
            "exported_variable_x_MRR",
            "mechanism_score_variable_x_MRR",
            "onset_score_variable_x_MRR",
            "source_score_variable_x_MRR",
            f"top1_supported_by_mechanism_top{args.support_k}",
        ]
        display_cols = [col for col in display_cols if col in df.columns]
        print("\nPer-event mechanism consistency:")
        print(df[display_cols].to_string(index=False))

    print("\nSummary:")
    key_cols = [
        "matched",
        "top1_variable_correct",
        "top1_group_correct",
        "exported_variable_x_MRR",
        "exported_group_x_MRR",
        "mechanism_score_variable_x_MRR",
        "mechanism_score_group_x_MRR",
        "onset_score_variable_x_MRR",
        "source_score_variable_x_MRR",
        f"top1_supported_by_mechanism_top{args.support_k}",
        f"top1_supported_by_onset_top{args.support_k}",
        f"top1_supported_by_source_top{args.support_k}",
    ]
    key_cols = [col for col in key_cols if col in summary.columns]
    print(summary[key_cols].to_string(index=False))

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([df, summary.assign(method="MEAN")], ignore_index=True)
        combined.to_csv(args.save_csv, index=False)
        print(f"\nSaved {args.save_csv}")


if __name__ == "__main__":
    main()
