#!/usr/bin/env python3
"""
Research Paper Metrics Aggregator.
Consolidates raw benchmark logs into statistical summaries across the 4 paper pillars:
  1. Task-Level Utility
  2. Causal Mechanistic Validity (CIC)
  3. Degradation Detection Specificity
  4. Online Repair Plasticity & Parameter Efficiency
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

BENCHMARKING_ROOT = WORKSPACE_ROOT / "benchmarking"
DATA_RESULTS_DIR = BENCHMARKING_ROOT / "data" / "results"


def load_benchmark_records(json_path: Path) -> List[Dict[str, Any]]:
    """Load benchmark results JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def calculate_mean_sem(values: List[float]) -> str:
    """Format as mean ± standard error."""
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if not clean_vals:
        return "N/A"
    mean = np.mean(clean_vals)
    sem = np.std(clean_vals, ddof=1) / np.sqrt(len(clean_vals)) if len(clean_vals) > 1 else 0.0
    return f"{mean:.2f} ± {sem:.2f}"


def aggregate_by_scale_and_arm(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group records by (scale, arm) and compute statistics across seeds."""
    groups = {}
    for r in records:
        scale = r.get("scale") or r.get("config_name") or f"{r.get('num_agents')}a{r.get('num_landmarks')}l"
        arm = r.get("arm") or r.get("controller_type") or "default"
        key = (scale, arm)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    summary_table = []
    for (scale, arm), arm_records in sorted(groups.items()):
        seeds = [r.get("seed") for r in arm_records]
        
        # Pillar 1: Task Utility
        baseline_rewards = [r.get("baseline_reward") for r in arm_records]
        comm_effects = [r.get("baseline_comm_effect") for r in arm_records]
        degraded_rewards = [r.get("degraded_reward") for r in arm_records]
        repaired_rewards = [r.get("repaired_reward") for r in arm_records]
        
        # Pillar 2: Causal CIC
        kl_divs = [r.get("baseline_kl") for r in arm_records]
        val_sens = [r.get("baseline_value_sens") for r in arm_records]

        # Pillar 3: Degradation
        drop_ratios = [r.get("reward_drop_ratio") for r in arm_records]
        triggers_fired = sum(1 for r in arm_records if r.get("detector_fired") is True)

        # Pillar 4: Plasticity & Recovery
        rew_recoveries = [r.get("reward_recovery_pct") for r in arm_records]
        comm_recoveries = [r.get("comm_recovery_pct") for r in arm_records]
        accepted_count = sum(1 for r in arm_records if r.get("repair_decision") == "ACCEPTED")
        confirmed_count = sum(1 for r in arm_records if r.get("heldout_validation") == "CONFIRMED")

        row = {
            "Scale": scale,
            "Arm": arm,
            "N_Seeds": len(seeds),
            "Baseline Return": calculate_mean_sem(baseline_rewards),
            "Comm Effect (ΔR)": calculate_mean_sem(comm_effects),
            "CIC KL Divergence": calculate_mean_sem(kl_divs),
            "Value Sensitivity": calculate_mean_sem(val_sens),
            "Degraded Return": calculate_mean_sem(degraded_rewards),
            "Reward Drop %": calculate_mean_sem([d * 100 if d is not None else None for d in drop_ratios]),
            "Trigger Rate": f"{triggers_fired}/{len(seeds)}",
            "Repaired Return": calculate_mean_sem(repaired_rewards),
            "Reward Recovery %": calculate_mean_sem(rew_recoveries),
            "Comm Recovery %": calculate_mean_sem(comm_recoveries),
            "Acceptance Rate": f"{accepted_count}/{len(seeds)}",
            "Held-Out Confirms": f"{confirmed_count}/{len(seeds)}"
        }
        summary_table.append(row)

    return summary_table


def generate_markdown_report(summary: List[Dict[str, Any]], output_path: Path):
    """Write Markdown summary table."""
    if not summary:
        return
    headers = list(summary[0].keys())
    lines = []
    lines.append("# Benchmark Experimental Metrics Summary\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Saved Markdown summary to: {output_path}")


def main():
    json_path = DATA_RESULTS_DIR / "benchmark_master.json"
    if not json_path.exists():
        # Fallback to experimentation/results if benchmark_master.json is not yet populated
        fallback = WORKSPACE_ROOT / "experimentation" / "results" / "benchmark_results.json"
        if fallback.exists():
            print(f"[INFO] Using existing fallback results from {fallback}")
            json_path = fallback
        else:
            print(f"[ERROR] No results found at {json_path} or {fallback}.")
            sys.exit(1)

    records = load_benchmark_records(json_path)
    summary = aggregate_by_scale_and_arm(records)
    
    md_output = DATA_RESULTS_DIR / "metrics_summary.md"
    generate_markdown_report(summary, md_output)

    print("\n" + "="*60)
    print("AGGREGATED BENCHMARK METRICS:")
    print("="*60)
    for row in summary:
        print(f"[{row['Scale']} | {row['Arm']}]")
        print(f"   Reward: Baseline {row['Baseline Return']} -> Degraded {row['Degraded Return']} -> Repaired {row['Repaired Return']}")
        print(f"   Comm Gain: {row['Comm Effect (ΔR)']} | CIC KL: {row['CIC KL Divergence']}")
        print(f"   Recovery: Rew {row['Reward Recovery %']} | Comm {row['Comm Recovery %']} | Accepted: {row['Acceptance Rate']}")
        print()


if __name__ == "__main__":
    main()
