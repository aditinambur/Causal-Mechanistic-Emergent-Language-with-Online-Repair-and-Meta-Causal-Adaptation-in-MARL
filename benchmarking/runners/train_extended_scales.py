#!/usr/bin/env python3
"""
Training Launcher for Extended Multi-Agent Scales (e.g., 4 Agents 5 Landmarks, 4 Agents 6 Landmarks).
Trains MACPPO policies with emergent discrete-token communication on MPE simple_spread.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmarking.configs.benchmark_matrix import SCENARIO_SCALES


def train_scale(scale_key: str, seeds: list, num_env_steps: int = 500_000, n_training_threads: int = 4):
    cfg = SCENARIO_SCALES.get(scale_key)
    if not cfg:
        raise ValueError(f"Unknown scale: {scale_key}. Available: {list(SCENARIO_SCALES.keys())}")

    print(f"\n=======================================================")
    print(f"Training Extended Scale: {cfg['name']}")
    print(f"Agents: {cfg['num_agents']} | Landmarks: {cfg['num_landmarks']}")
    print(f"Seeds: {seeds} | Steps per seed: {num_env_steps}")
    print(f"=======================================================\n")

    for seed in seeds:
        exp_name = f"benchmark_{cfg['num_agents']}a{cfg['num_landmarks']}_seed{seed}"
        cmd = [
            sys.executable, "-u", "onpolicy/scripts/train/train_mpe.py",
            "--env_name", "MPE",
            "--scenario_name", cfg.get("scenario_name", "simple_spread"),
            "--algorithm_name", "mappo",
            "--seed", str(seed),
            "--num_agents", str(cfg["num_agents"]),
            "--num_landmarks", str(cfg["num_landmarks"]),
            "--num_env_steps", str(num_env_steps),
            "--n_training_threads", str(n_training_threads),
            "--n_rollout_threads", "4",
            "--use_eval",
            "--eval_interval", "10",
            "--eval_disable_messages",
            "--eval_noise_std", "0.25",
            "--experiment_name", exp_name,
            "--cuda"
        ]

        print(f"Launching training for {exp_name}...")
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, check=True)
        print(f"Training completed for {exp_name}!\n")


def main():
    parser = argparse.ArgumentParser(description="Train extended multi-agent scales")
    parser.add_argument("--scale", type=str, default="4a5l", choices=list(SCENARIO_SCALES.keys()), help="Scale to train")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Seeds to train")
    parser.add_argument("--num_env_steps", type=int, default=500_000, help="Training environment steps")
    parser.add_argument("--n_training_threads", type=int, default=4, help="Threads for training")
    args = parser.parse_args()

    train_scale(args.scale, args.seeds, args.num_env_steps, args.n_training_threads)


if __name__ == "__main__":
    main()
