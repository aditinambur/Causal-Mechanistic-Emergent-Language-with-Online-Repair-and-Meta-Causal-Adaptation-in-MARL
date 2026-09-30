#!/usr/bin/env python3
"""
Utility script to parse all existing experiment runs in results/ and export them into a structured Excel workbook.
"""

import os
import sys
import json
import csv
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE_ROOT / "experimentation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    json_path = RESULTS_DIR / "benchmark_results.json"
    excel_path = RESULTS_DIR / "benchmark_results.xlsx"
    csv_path = RESULTS_DIR / "benchmark_results.csv"

    if not json_path.exists() and not csv_path.exists():
        print(f"[ERROR] No benchmark data found at {json_path} or {csv_path}.")
        return

    if json_path.exists():
        with open(json_path, "r") as f:
            records = json.load(f)
    else:
        records = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            records = list(reader)

    if not HAS_PANDAS:
        print("[WARNING] pandas is not installed. Data remains preserved in CSV/JSON format.")
        return

    df = pd.DataFrame(records)
    with pd.ExcelWriter(excel_path, engine="openpyxl" if HAS_OPENPYXL else None) as writer:
        df.to_excel(writer, sheet_name="Full Benchmark Records", index=False)
        if "config_name" in df.columns and "controller_type" in df.columns:
            summary_cols = [
                "baseline_reward", "baseline_comm_effect", "baseline_kl",
                "degraded_reward", "degraded_comm_effect", "repaired_reward", "repaired_comm_effect"
            ]
            valid_cols = [c for c in summary_cols if c in df.columns]
            summary_df = df.groupby(["config_name", "controller_type"])[valid_cols].mean().reset_index()
            summary_df.to_excel(writer, sheet_name="Aggregated Means", index=False)

    print(f"[EXPORT] Excel workbook created successfully at: {excel_path}")


if __name__ == "__main__":
    main()
