#!/usr/bin/env python3
"""
Full Autonomous Benchmark Automation Suite for Multi-Agent Emergent Communication & Meta-Causal Repair.

Executes:
  1. Set A: 2 Agents, 3 Landmarks (500,000 steps) across 5 seeds (1, 2, 3, 4, 5).
  2. Set B: 3 Agents, 4 Landmarks (500,000 steps) across 5 seeds (1, 2, 3, 4, 5).
  3. Automatic Checkpoint Discovery: Finds best communication checkpoint for each seed.
  4. Dual-Controller Evaluation:
     - Smart Causal Trigger (Reward + Comm Drop AND-gate)
     - Naive Trigger (Reward-only >= 30%)
  5. Automatic Excel (.xlsx), CSV, and JSON persistence of all table metrics.
"""

import os
import sys
import time
import json
import csv
import glob
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

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


def parse_phase2_3_output(log_text: str) -> Dict[str, Any]:
    """Parse key metrics from phase2_3_repair.py stdout log."""
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
        "repair_target": "none",
        "repaired_reward": None,
        "repaired_comm_effect": None,
        "repaired_value_sens": None,
        "repaired_kl": None,
        "reward_recovery_pct": None,
        "comm_recovery_pct": None,
        "repair_decision": "NO_REPAIR_TRIGGERED",
        "heldout_validation": "N/A",
    }

    lines = log_text.splitlines()
    for line in lines:
        line_s = line.strip()

        # Baseline
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

        # Degraded
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

        # Detection
        if "ENVIRONMENT CHANGE DETECTED" in line_s:
            res["detector_fired"] = True
        elif "ENVIRONMENT CHANGE NOT detected" in line_s:
            res["detector_fired"] = False

        if "reward drop ratio" in line_s:
            try:
                val = line_s.split(":")[1].split("(")[0].strip()
                res["reward_drop_ratio"] = float(val)
            except: pass

        # Repair Execution
        if "[REPAIRED]" in line_s:
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

        if "reward recovery" in line_s and ":" in line_s:
            try:
                val = line_s.split(":")[1].strip().replace("%", "")
                if val != "n/a (nothing lost)":
                    res["reward_recovery_pct"] = float(val)
            except: pass

        if "communication recovery" in line_s and ":" in line_s:
            try:
                val = line_s.split(":")[1].strip().replace("%", "")
                if val != "n/a (nothing lost)":
                    res["comm_recovery_pct"] = float(val)
            except: pass

        if "decision: ACCEPT" in line_s:
            res["repair_decision"] = "ACCEPTED"
        elif "decision: REJECT" in line_s:
            res["repair_decision"] = "REJECTED"

        if "repair accepted:" in line_s:
            res["repair_target"] = line_s.split("repair accepted:")[1].strip()
            res["repair_decision"] = "ACCEPTED"
        elif "repair REJECTED (rolled back):" in line_s:
            res["repair_target"] = line_s.split("repair REJECTED (rolled back):")[1].strip()
            res["repair_decision"] = "REJECTED"

        # Held-out validation
        if "HELD-OUT CONFIRMS" in line_s:
            res["heldout_validation"] = "CONFIRMED"
        elif "HELD-OUT DOES NOT CONFIRM" in line_s:
            res["heldout_validation"] = "FAILED"

    return res


def export_results_to_excel(results_list: List[Dict[str, Any]], excel_path: Path):
    """Write benchmark results to Excel with styled summary tabs, with robust CSV fallback."""
    csv_path = excel_path.with_suffix(".csv")
    if results_list:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_list[0].keys()))
            writer.writeheader()
            writer.writerows(results_list)
        print(f"[EXPORT] Updated CSV log at: {csv_path}")

    if not HAS_PANDAS:
        return

    try:
        import openpyxl
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl", "-q"], check=False)
            import openpyxl
        except Exception:
            print("[INFO] openpyxl not installed. CSV report is saved cleanly.")
            return

    try:
        df = pd.DataFrame(results_list)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Full Benchmark Records", index=False)

            if "config_name" in df.columns and "controller_type" in df.columns:
                summary_cols = [
                    "baseline_reward", "baseline_comm_effect", "baseline_kl",
                    "degraded_reward", "degraded_comm_effect", "repaired_reward", "repaired_comm_effect"
                ]
                valid_cols = [c for c in summary_cols if c in df.columns]
                if valid_cols:
                    summary_df = df.groupby(["config_name", "controller_type"])[valid_cols].mean().reset_index()
                    summary_df.to_excel(writer, sheet_name="Aggregated Means", index=False)

        print(f"[EXPORT] Successfully saved formatted Excel workbook to: {excel_path}")
    except Exception as e:
        print(f"[EXPORT NOTE] Excel workbook generation: {e}. CSV report is preserved at {csv_path}")


def find_best_checkpoint(run_dir: Path) -> Path:
    """Find checkpoint_best or fallback to the latest checkpoint."""
    models_dir = run_dir / "models"
    best_dir = models_dir / "checkpoint_best"
    if best_dir.exists() and (best_dir / "actor.pt").exists():
        return best_dir

    # Fallback to highest step numbered checkpoint
    checkpoints = sorted(models_dir.glob("checkpoint_*"))
    for ckpt in reversed(checkpoints):
        if ckpt.is_dir() and (ckpt / "actor.pt").exists() and ckpt.name != "checkpoint_best":
            return ckpt
            
    return models_dir


def run_command(cmd: List[str], desc: str) -> str:
    """Run a subprocess command and stream output while capturing."""
    print(f"\n=======================================================")
    print(f"[EXEC] {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================\n")
    
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(WORKSPACE_ROOT)
    )
    
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
        
    proc.wait()
    if proc.returncode != 0:
        print(f"[WARNING] Process finished with exit code {proc.returncode}")
        
    return "".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="Automated Benchmark Suite for Multi-Agent Communication & Repair")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of random seeds to evaluate (default: 1 2 3 4 5)")
    parser.add_argument("--num_env_steps", type=int, default=500000,
                        help="Number of training environment steps per run (default: 500,000)")
    parser.add_argument("--skip_training", action="store_true", default=False,
                        help="Skip Phase 1 training if runs already exist and run Phase 2/3 repairs directly")
    parser.add_argument("--repair_iters", type=int, default=15,
                        help="Online PPO repair iterations (default: 15)")
    parser.add_argument("--measure_episodes", type=int, default=8,
                        help="CRN evaluation episodes for fingerprinting (default: 8)")
    parser.add_argument("--mirror_scope", type=str, default="partner_full",
                        help="Perturbation modality: partner_full, partner, or all (default: partner_full)")
    args = parser.parse_args()

    # Configurations to test:
    # 1. 2 Agents, 3 Landmarks
    # 2. 3 Agents, 4 Landmarks
    benchmark_configs = [
        {"name": "2Agents_3Landmarks", "num_agents": 2, "num_landmarks": 3, "exp_prefix": "benchmark_2a3l"},
        {"name": "3Agents_4Landmarks", "num_agents": 3, "num_landmarks": 4, "exp_prefix": "benchmark_3a4l"}
    ]

    all_records = []
    json_path = RESULTS_DIR / "benchmark_results.json"
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    excel_path = RESULTS_DIR / "benchmark_results.xlsx"

    # Load existing records if resuming
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                all_records = json.load(f)
            print(f"[RESUME] Loaded {len(all_records)} existing records from {json_path}")
        except Exception:
            all_records = []

    print(f"\n=======================================================")
    print(f"Starting Multi-Agent Automated Benchmark Suite")
    print(f"Configurations: {[c['name'] for c in benchmark_configs]}")
    print(f"Seeds: {args.seeds}")
    print(f"Training Steps per Run: {args.num_env_steps}")
    print(f"Output Directory: {RESULTS_DIR}")
    print(f"=======================================================\n")

    for cfg in benchmark_configs:
        for seed in args.seeds:
            exp_name = f"{cfg['exp_prefix']}_seed{seed}"
            run_dir = WORKSPACE_ROOT / "onpolicy" / "scripts" / "results" / "MPE" / "simple_spread" / "mappo" / exp_name / "run1"

            # Check if this seed was already fully evaluated
            existing_run_ids = {r.get("run_id") for r in all_records}
            if f"{exp_name}_causal" in existing_run_ids and f"{exp_name}_naive" in existing_run_ids:
                print(f"[RESUME] {exp_name} already completed and logged. Skipping.")
                continue

            # ---------------------------------------------------------
            # Phase 1: Train Model (skip if checkpoint already exists or skip_training)
            # ---------------------------------------------------------
            ckpt_exists = run_dir.exists() and ((run_dir / "models" / "checkpoint_best").exists() or list((run_dir / "models").glob("checkpoint_*")))
            if not args.skip_training and not ckpt_exists:
                train_cmd = [
                    sys.executable, "-u", "onpolicy/scripts/train/train_mpe.py",
                    "--env_name", "MPE",
                    "--scenario_name", "simple_spread",
                    "--algorithm_name", "mappo",
                    "--seed", str(seed),
                    "--num_agents", str(cfg["num_agents"]),
                    "--num_landmarks", str(cfg["num_landmarks"]),
                    "--num_env_steps", str(args.num_env_steps),
                    "--use_eval",
                    "--eval_interval", "5",
                    "--eval_disable_messages",
                    "--eval_noise_std", "0.25",
                    "--experiment_name", exp_name,
                    "--use_wandb"
                ]
                run_command(train_cmd, f"Phase 1 Training: {cfg['name']} | Seed {seed} ({args.num_env_steps} steps)")
            elif ckpt_exists:
                print(f"[RESUME] Checkpoint already exists for {exp_name}. Proceeding directly to Phase 2/3 evaluation.")

            # Verify checkpoint existence
            if not run_dir.exists():
                print(f"[ERROR] Run directory not found: {run_dir}. Skipping.")
                continue

            selected_ckpt = find_best_checkpoint(run_dir)
            print(f"[BENCHMARK] Using checkpoint: {selected_ckpt}")

            # ---------------------------------------------------------
            # Phase 2 & 3: Run with Smart Causal Trigger (Controller: causal)
            # ---------------------------------------------------------
            causal_cmd = [
                sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
                "--env_name", "MPE",
                "--scenario_name", "simple_spread",
                "--algorithm_name", "mappo",
                "--seed", str(seed),
                "--num_agents", str(cfg["num_agents"]),
                "--num_landmarks", str(cfg["num_landmarks"]),
                "--model_dir", str(selected_ckpt),
                "--mirror_scope", args.mirror_scope,
                "--measure_episodes", str(args.measure_episodes),
                "--repair_iters", str(args.repair_iters),
                "--controller", "causal",
                "--lora_rank", "4",
                "--lora_alpha", "8.0",
                "--n_rollout_threads", "32",
                "--n_eval_rollout_threads", "1"
            ]
            causal_log = run_command(causal_cmd, f"Phase 2/3 (Smart Causal Trigger): {cfg['name']} | Seed {seed}")
            causal_parsed = parse_phase2_3_output(causal_log)

            record_causal = {
                "run_id": f"{exp_name}_causal",
                "config_name": cfg["name"],
                "seed": seed,
                "num_agents": cfg["num_agents"],
                "num_landmarks": cfg["num_landmarks"],
                "selected_checkpoint": selected_ckpt.name,
                "controller_type": "Smart Causal Trigger (Reward + Comm Drop)",
                **causal_parsed
            }
            all_records.append(record_causal)

            # ---------------------------------------------------------
            # Phase 2 & 3: Run with Naive Trigger (Controller: reward_only)
            # ---------------------------------------------------------
            naive_cmd = [
                sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
                "--env_name", "MPE",
                "--scenario_name", "simple_spread",
                "--algorithm_name", "mappo",
                "--seed", str(seed),
                "--num_agents", str(cfg["num_agents"]),
                "--num_landmarks", str(cfg["num_landmarks"]),
                "--model_dir", str(selected_ckpt),
                "--mirror_scope", args.mirror_scope,
                "--measure_episodes", str(args.measure_episodes),
                "--repair_iters", str(args.repair_iters),
                "--controller", "reward_only",
                "--repair_target", "lora",
                "--lora_rank", "4",
                "--lora_alpha", "8.0",
                "--n_rollout_threads", "32",
                "--n_eval_rollout_threads", "1"
            ]
            naive_log = run_command(naive_cmd, f"Phase 2/3 (Naive Trigger Control): {cfg['name']} | Seed {seed}")
            naive_parsed = parse_phase2_3_output(naive_log)

            record_naive = {
                "run_id": f"{exp_name}_naive",
                "config_name": cfg["name"],
                "seed": seed,
                "num_agents": cfg["num_agents"],
                "num_landmarks": cfg["num_landmarks"],
                "selected_checkpoint": selected_ckpt.name,
                "controller_type": "Naive Trigger (Reward Drop Only >= 30%)",
                **naive_parsed
            }
            all_records.append(record_naive)

            # Continuously save checkpoints to disk
            with open(json_path, "w") as f:
                json.dump(all_records, f, indent=2)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record_causal.keys()))
                writer.writeheader()
                writer.writerows(all_records)

            export_results_to_excel(all_records, excel_path)

    print("\n=======================================================")
    print("All Benchmark Runs Completed Successfully!")
    print(f"Excel Report : {excel_path}")
    print(f"CSV Report   : {csv_path}")
    print(f"JSON Report  : {json_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
