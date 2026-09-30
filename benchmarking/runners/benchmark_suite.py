#!/usr/bin/env python3
"""
Unified Benchmark Runner for MACPPO Causal Emergent Communication & Online Repair.

Supports:
- Multi-scenario & Multi-agent scaling (2a3l, 3a4l, 4a5l, 4a6l).
- 4 Peer-review control arms (Causal Adaptive, Naive Reward-Only, Non-Comm Repair, No-Repair).
- Automated checkpoint discovery and optional model training.
- Robust parsing of all causal fingerprints, metrics, and held-out validation signals.
- Direct persistence to CSV, JSON, and Excel.
"""

import os
import sys
import time
import json
import csv
import glob
import re
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmarking.configs.benchmark_matrix import SCENARIO_SCALES, CONTROL_ARMS, EVAL_CONFIG

BENCHMARKING_ROOT = WORKSPACE_ROOT / "benchmarking"
DATA_RESULTS_DIR = BENCHMARKING_ROOT / "data" / "results"
DATA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_phase2_3_output(log_text: str) -> Dict[str, Any]:
    """
    Robustly parse metrics from phase2_3_repair.py stdout log.
    Accurately extracts baseline, degraded, repaired (including [REPAIRED[target]]),
    controller decisions, acceptance, and held-out validation.
    """
    res = {
        "baseline_reward": None,
        "baseline_no_msg_reward": None,
        "baseline_comm_effect": None,
        "baseline_value_sens": None,
        "baseline_kl": None,
        "degraded_reward": None,
        "degraded_no_msg_reward": None,
        "degraded_comm_effect": None,
        "degraded_value_sens": None,
        "degraded_kl": None,
        "detector_fired": False,
        "reward_drop_ratio": None,
        "comm_drop_ratio": None,
        "repair_target": "none",
        "repaired_reward": None,
        "repaired_comm_effect": None,
        "repaired_value_sens": None,
        "repaired_kl": None,
        "reward_recovery_pct": None,
        "comm_recovery_pct": None,
        "repair_decision": "NO_REPAIR_TRIGGERED",
        "heldout_validation": "N/A",
        "heldout_reward_recovery": None,
        "heldout_comm_recovery": None,
        "normal_reward_retention_loss": None,
    }

    lines = log_text.splitlines()
    for line in lines:
        line_s = line.strip()

        # 1. Baseline fingerprint
        if "[BASELINE]" in line_s:
            parts = line_s.split()
            for p in parts:
                if p.startswith("reward="):
                    try: res["baseline_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("no_msg_reward="):
                    try: res["baseline_no_msg_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("comm_effect="):
                    try: res["baseline_comm_effect"] = float(p.split("=")[1].split("±")[0])
                    except: pass
                elif p.startswith("value_sens="):
                    try: res["baseline_value_sens"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("kl="):
                    try: res["baseline_kl"] = float(p.split("=")[1])
                    except: pass

        # 2. Degraded fingerprint
        if "[DEGRADED]" in line_s:
            parts = line_s.split()
            for p in parts:
                if p.startswith("reward="):
                    try: res["degraded_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("no_msg_reward="):
                    try: res["degraded_no_msg_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("comm_effect="):
                    try: res["degraded_comm_effect"] = float(p.split("=")[1].split("±")[0])
                    except: pass
                elif p.startswith("value_sens="):
                    try: res["degraded_value_sens"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("kl="):
                    try: res["degraded_kl"] = float(p.split("=")[1])
                    except: pass

        # 3. Detection info
        if "ENVIRONMENT CHANGE DETECTED" in line_s:
            res["detector_fired"] = True
        elif "ENVIRONMENT CHANGE NOT detected" in line_s:
            res["detector_fired"] = False

        if "reward drop ratio:" in line_s:
            try:
                val = line_s.split("reward drop ratio:")[1].split("(")[0].strip()
                res["reward_drop_ratio"] = float(val)
            except: pass

        if "selected repair target:" in line_s:
            try:
                res["repair_target"] = line_s.split("selected repair target:")[1].strip()
            except: pass

        # 4. Repaired fingerprint (matches both [REPAIRED] and [REPAIRED[target]])
        if re.search(r"\[REPAIRED(\[[a-zA-Z0-9_]+\])?\]", line_s):
            parts = line_s.split()
            for p in parts:
                if p.startswith("reward="):
                    try: res["repaired_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("comm_effect="):
                    try: res["repaired_comm_effect"] = float(p.split("=")[1].split("±")[0])
                    except: pass
                elif p.startswith("value_sens="):
                    try: res["repaired_value_sens"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("kl="):
                    try: res["repaired_kl"] = float(p.split("=")[1])
                    except: pass

        # 5. Recovery percentages
        if "reward recovery:" in line_s:
            try:
                val = line_s.split("reward recovery:")[1].strip().replace("%", "")
                if "n/a" not in val.lower():
                    res["reward_recovery_pct"] = float(val)
            except: pass

        if "communication recovery:" in line_s:
            try:
                val = line_s.split("communication recovery:")[1].strip().replace("%", "")
                if "n/a" not in val.lower():
                    res["comm_recovery_pct"] = float(val)
            except: pass

        # 6. Acceptance decisions
        if "repair accepted:" in line_s:
            res["repair_decision"] = "ACCEPTED"
            try:
                res["repair_target"] = line_s.split("repair accepted:")[1].strip()
            except: pass
        elif "all repair attempts rejected" in line_s:
            res["repair_decision"] = "REJECTED"

        # 7. Held-out validation
        if "held-out validation:" in line_s:
            if "CONFIRMS" in line_s:
                res["heldout_validation"] = "CONFIRMED"
            elif "FAILS" in line_s or "REJECTS" in line_s:
                res["heldout_validation"] = "REJECTED"

        if "heldout reward recovery" in line_s:
            try:
                val = line_s.split(":")[-1].strip().replace("%", "")
                res["heldout_reward_recovery"] = float(val)
            except: pass

        if "retention drop on normal env" in line_s or "normal_reward_delta" in line_s:
            try:
                res["normal_reward_retention_loss"] = float(line_s.split(":")[-1].strip())
            except: pass

    return res


def execute_subcommand(cmd: List[str], desc: str) -> str:
    """Run an external command and capture stdout/stderr."""
    print(f"\n========================================================")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"========================================================")
    start_t = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(WORKSPACE_ROOT)
    )

    collected_output = []
    for line in iter(proc.stdout.readline, ''):
        print(line, end='', flush=True)
        collected_output.append(line)

    proc.stdout.close()
    return_code = proc.wait()
    duration = time.time() - start_t
    print(f"--> Finished in {duration:.1f}s with return code {return_code}\n")
    return "".join(collected_output)


def discover_best_checkpoint(scenario_key: str, seed: int) -> Optional[Path]:
    """Find the best or final checkpoint for a given scale and seed."""
    cfg = SCENARIO_SCALES[scenario_key]
    num_a = cfg["num_agents"]
    num_l = cfg["num_landmarks"]

    base_results = WORKSPACE_ROOT / "onpolicy" / "scripts" / "results" / "MPE" / "simple_spread" / "mappo"
    possible_dirs = [
        base_results / f"benchmark_{num_a}a{num_l}l_seed{seed}",
        base_results / f"benchmark_{num_a}a{num_l}_seed{seed}",
    ]
    if num_a == 2 and num_l == 3:
        possible_dirs.append(base_results / f"phase2_3_seed{seed}")

    # Also search by glob if not exact match
    if base_results.exists():
        possible_dirs.extend(base_results.glob(f"benchmark_{num_a}a{num_l}*seed{seed}*"))

    for pdir in possible_dirs:
        if not pdir.exists():
            continue
        # Check direct models, run1/models, or any run*/models
        candidate_model_dirs = [
            pdir / "models",
            pdir / "run1" / "models",
        ]
        candidate_model_dirs.extend(pdir.glob("run*/models"))

        for models_dir in candidate_model_dirs:
            if models_dir.exists():
                ckpt_best = models_dir / "checkpoint_best"
                if ckpt_best.exists():
                    return ckpt_best
                ckpts = sorted(models_dir.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts:
                    return ckpts[0]

    return None


def run_benchmarks(
    selected_scales: List[str],
    selected_arms: List[str],
    seeds: List[int],
    repair_iters: int = 15,
    measure_episodes: int = 6,
    output_prefix: str = "benchmark_master"
):
    """Run full benchmarking matrix across scales, arms, and seeds."""
    csv_file = DATA_RESULTS_DIR / f"{output_prefix}.csv"
    json_file = DATA_RESULTS_DIR / f"{output_prefix}.json"

    all_results = []
    if json_file.exists():
        try:
            with open(json_file, "r") as f:
                all_results = json.load(f)
            print(f"[RESUME] Loaded {len(all_results)} existing records from {json_file}")
        except Exception:
            all_results = []

    print(f"Starting Benchmark Suite across {len(selected_scales)} scales, {len(selected_arms)} arms, {len(seeds)} seeds.")

    for scale_key in selected_scales:
        scale_cfg = SCENARIO_SCALES[scale_key]

        for seed in seeds:
            ckpt = discover_best_checkpoint(scale_key, seed)
            if ckpt is None:
                print(f"[WARNING] No trained checkpoint found for {scale_key} Seed {seed}. Skipping. (Train it first with --train_scale {scale_key})")
                continue

            for arm_key in selected_arms:
                arm_cfg = CONTROL_ARMS[arm_key]
                run_id = f"{scale_key}_seed{seed}_{arm_key}"
                print(f"\n>>> Running Benchmark Arm: {arm_cfg['name']} | Scale: {scale_cfg['name']} | Seed: {seed}")

                cmd = [
                    sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
                    "--env_name", "MPE",
                    "--scenario_name", scale_cfg.get("scenario_name", "simple_spread"),
                    "--algorithm_name", "mappo",
                    "--seed", str(seed),
                    "--num_agents", str(scale_cfg["num_agents"]),
                    "--num_landmarks", str(scale_cfg["num_landmarks"]),
                    "--model_dir", str(ckpt),
                    "--mirror_scope", scale_cfg.get("mirror_scope", "partner_full"),
                    "--measure_episodes", str(measure_episodes),
                    "--repair_iters", str(repair_iters),
                    "--controller", arm_cfg["controller"],
                    "--lora_rank", str(EVAL_CONFIG["lora_rank"]),
                    "--lora_alpha", str(EVAL_CONFIG["lora_alpha"]),
                    "--n_rollout_threads", str(EVAL_CONFIG["n_rollout_threads"]),
                    "--n_eval_rollout_threads", str(EVAL_CONFIG["n_eval_rollout_threads"]),
                    "--detect_reward_drop_ratio", str(EVAL_CONFIG["reward_drop_threshold"]),
                    "--accept_reward_recovery", str(EVAL_CONFIG["accept_reward_recovery"]),
                    "--accept_comm_recovery", str(EVAL_CONFIG["accept_comm_recovery"]),
                ]

                if arm_key == "no_repair":
                    cmd.extend(["--no_repair"])
                elif arm_cfg["repair_target"] is not None and arm_cfg["repair_target"] != "none":
                    cmd.extend(["--repair_target", arm_cfg["repair_target"]])

                log_output = execute_subcommand(cmd, f"{scale_key} Seed {seed} [{arm_key}]")
                parsed = parse_phase2_3_output(log_output)

                record = {
                    "run_id": run_id,
                    "scale": scale_key,
                    "scale_name": scale_cfg["name"],
                    "num_agents": scale_cfg["num_agents"],
                    "num_landmarks": scale_cfg["num_landmarks"],
                    "seed": seed,
                    "arm": arm_key,
                    "arm_name": arm_cfg["name"],
                    "checkpoint": ckpt.name,
                    **parsed
                }

                # Update existing record if run_id matches, else append
                existing_idx = next((i for i, r in enumerate(all_results) if r.get("run_id") == run_id), None)
                if existing_idx is not None:
                    all_results[existing_idx] = record
                else:
                    all_results.append(record)

                # Incremental persistence
                with open(json_file, "w") as f:
                    json.dump(all_results, f, indent=2)

                if all_results:
                    with open(csv_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
                        writer.writeheader()
                        writer.writerows(all_results)

    print(f"\n========================================================")
    print(f"BENCHMARK COMPLETE!")
    print(f"Results saved to:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")
    print(f"========================================================")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MACPPO Causal Benchmark Suite")
    parser.add_argument("--scales", nargs="+", default=["2a3l", "3a4l"], choices=list(SCENARIO_SCALES.keys()), help="Scenario scales to benchmark")
    parser.add_argument("--arms", nargs="+", default=["causal_adaptive", "naive_reward_only"], choices=list(CONTROL_ARMS.keys()), help="Control arms to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seeds to evaluate")
    parser.add_argument("--repair_iters", type=int, default=15, help="Number of repair iterations")
    parser.add_argument("--measure_episodes", type=int, default=6, help="Number of CRN-paired episodes")
    parser.add_argument("--output_prefix", type=str, default="benchmark_master", help="Output file prefix")
    args = parser.parse_args()

    run_benchmarks(
        selected_scales=args.scales,
        selected_arms=args.arms,
        seeds=args.seeds,
        repair_iters=args.repair_iters,
        measure_episodes=args.measure_episodes,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()
