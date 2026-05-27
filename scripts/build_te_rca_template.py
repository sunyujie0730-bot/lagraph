#!/usr/bin/env python3
"""Build TE-MM RCA annotation templates without fabricating ground truth.

TE-MM labels require a verified mapping from file fault number to disturbance
IDV and from disturbance IDV to process unit or variable roots. The dataset
README suggests that d01-d28 may correspond to IDV(28)-IDV(01), so this script
records both candidate mappings for manual verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECT_META = PROJECT_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "anomaly_detect" / "label_sources"


TE_IDV_DESCRIPTIONS = {
    1: "A/C feed ratio, B composition constant (stream 4)",
    2: "B composition, A/C ratio constant (stream 4)",
    3: "D feed temperature (stream 2)",
    4: "Reactor cooling water inlet temperature",
    5: "Condenser cooling water inlet temperature",
    6: "A feed loss (stream 1)",
    7: "C header pressure loss-reduced availability (stream 4)",
    8: "A, B, C feed composition (stream 4)",
    9: "D feed temperature (stream 2)",
    10: "C feed temperature (stream 4)",
    11: "Reactor cooling water inlet temperature",
    12: "Condenser cooling water inlet temperature",
    13: "Reaction kinetics",
    14: "Reactor cooling water valve",
    15: "Condenser cooling water valve",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Condenser cooling water flow",
    22: "E feed temperature (stream 3)",
    23: "A feed pressure (stream 1)",
    24: "D feed pressure (stream 2)",
    25: "E feed pressure (stream 3)",
    26: "A and C feed pressure (stream 4)",
    27: "Pressure fluctuation in reactor cooling water recirculating unit",
    28: "Pressure fluctuation in condenser cooling water recirculating unit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect-meta", type=Path, default=DEFAULT_DETECT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def subsystem_candidates(description: str) -> str:
    text = description.lower()
    stages = []
    if any(key in text for key in ["feed", "stream 1", "stream 2", "stream 3", "stream 4"]):
        stages.append("feed")
    if "reactor" in text or "reaction" in text:
        stages.append("reactor")
    if "condenser" in text:
        stages.append("condenser")
    if "cooling water" in text:
        stages.append("cooling_water")
    return ";".join(stages)


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.detect_meta)
    te_rows = meta.loc[meta["file_name"].astype(str).str.startswith("TE_MM_")].copy()
    if te_rows.empty:
        raise SystemExit("No TE_MM rows found in DETECT_META.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "te_mm_fault_mapping_template.csv"

    rows = []
    for _, row in te_rows.iterrows():
        fault_file_id = int(row["fault_id"])
        direct_idv = fault_file_id
        reversed_idv = 29 - fault_file_id
        train_len = int(row["train_lens"])
        length = int(row["length"])
        # Converted TE-MM files concatenate normal data and faulty data. With the
        # current conversion ratio, the test segment starts at train_len.
        normal_len = 7201
        event_start = max(0, normal_len - train_len)
        event_end = length - train_len
        rows.append(
            {
                "file": row["file_name"],
                "mode": int(row["mode"]),
                "fault_file_id": fault_file_id,
                "candidate_idv_direct": direct_idv,
                "candidate_idv_reversed": reversed_idv,
                "direct_description": TE_IDV_DESCRIPTIONS.get(direct_idv, ""),
                "reversed_description": TE_IDV_DESCRIPTIONS.get(reversed_idv, ""),
                "candidate_subsystem_direct": subsystem_candidates(TE_IDV_DESCRIPTIONS.get(direct_idv, "")),
                "candidate_subsystem_reversed": subsystem_candidates(TE_IDV_DESCRIPTIONS.get(reversed_idv, "")),
                "event_start": event_start,
                "event_end": event_end,
                "selected_idv": "",
                "subsystem_root": "",
                "variable_roots": "",
                "source": "TODO: verify dXX-to-IDV mapping from TE-MM generation code or paper",
                "label_status": "needs_manual_source",
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(rows)} TE-MM files")


if __name__ == "__main__":
    main()
