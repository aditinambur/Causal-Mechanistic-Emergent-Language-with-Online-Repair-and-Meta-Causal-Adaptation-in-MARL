#!/usr/bin/env python3
"""
Task E3 Runner: LoRA Rank Sweep (r in {1, 2, 4, 8, 16}, alpha = 2*r) on 3a4l Seeds 1 & 5.

Forces --repair_target lora across each (rank, seed) pair using identical CRN layouts,
logs raw records to results/e3_lora_rank_sweep.jsonl, and aggregates JSON/CSV/Markdown
tables in benchmarking/data/results/.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_JSONL = WORKSPACE_ROOT / "results" / "e3_lora_rank_sweep.jsonl"
DATA_RESULTS_DIR = WORKSPACE_ROOT / "benchmarking" / "data" / "results"


def _fmt_mean_sem(values: List[float], suffix: str = "%") -> str:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return "N/A"
    mean = float(np.mean(clean))
    sem = float(np.std(clean, ddof=1) / np.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return f"{mean:.1f}{suffix} ± {sem:.1f}{suffix}" if len(clean) > 1 else f"{mean:.1f}{suffix}"


def run_sweep(ranks: List[int], seeds: List[int], resume: bool = False) -> None:
    """Execute phase2_3_repair.py across all (rank, seed) combinations."""
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if resume and RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((int(row.get("lora_rank", 0)), int(row.get("seed", 0))))
        print(f"[RESUME] Found {len(completed)} completed runs in {RESULTS_JSONL}")
    elif RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    for r in ranks:
        alpha = float(2 * r)
        for seed in seeds:
            if (r, seed) in completed:
                print(f"[SKIP] rank={r}, seed={seed} already logged.")
                continue
            ckpt_rel = (
                f"onpolicy/scripts/results/MPE/simple_spread/mappo/"
                f"benchmark_3a4l_seed{seed}/run1/models/checkpoint_best"
            )
            cmd = [
                sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
                "--env_name", "MPE",
                "--scenario_name", "simple_spread",
                "--algorithm_name", "mappo",
                "--seed", str(seed),
                "--num_agents", "3",
                "--num_landmarks", "4",
                "--model_dir", ckpt_rel,
                "--mirror_scope", "partner_full",
                "--measure_episodes", "6",
                "--holdout_episodes", "6",
                "--repair_iters", "15",
                "--controller", "causal",
                "--repair_target", "lora",
                "--lora_rank", str(r),
                "--lora_alpha", str(alpha),
                "--n_rollout_threads", "4",
                "--n_eval_rollout_threads", "1",
                "--detect_reward_drop_ratio", "0.10",
                "--accept_reward_recovery", "0.30",
                "--accept_comm_recovery", "0.15",
                "--results_log", "results/e3_lora_rank_sweep.jsonl",
            ]
            print(f"\n=== E3 Sweep: r={r}, alpha={alpha:.1f}, 3a4l seed={seed} ===")
            t0 = time.time()
            subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), check=True)
            print(f"=== Completed r={r}, seed={seed} in {time.time() - t0:.1f}s ===")


def aggregate_sweep() -> List[Dict[str, Any]]:
    """Aggregate results/e3_lora_rank_sweep.jsonl into JSON, CSV, and Markdown summary."""
    if not RESULTS_JSONL.exists():
        raise FileNotFoundError(f"Missing log file: {RESULTS_JSONL}")

    records: List[Dict[str, Any]] = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        fs = raw.get("final_status") or {}
        pc = fs.get("param_counts") or {}
        base = raw.get("baseline") or {}
        deg = raw.get("degraded") or {}
        rep = raw.get("repaired") or {}

        base_rew = base.get("reward")
        norm_delta = fs.get("normal_reward_delta_vs_baseline")
        retention_loss_pct = None
        retention_pct = None
        if base_rew is not None and norm_delta is not None:
            retention_loss_pct = (float(norm_delta) / (abs(float(base_rew)) + 1e-6)) * 100.0
            retention_pct = 100.0 + retention_loss_pct

        rew_rec = fs.get("reward_recovery")
        comm_rec = fs.get("comm_recovery")
        ho_rew_rec = fs.get("holdout_reward_recovery")
        ho_comm_rec = fs.get("holdout_comm_recovery")

        records.append({
            "lora_rank": int(raw.get("lora_rank", 0)),
            "lora_alpha": float(raw.get("lora_alpha", 0.0)),
            "seed": int(raw.get("seed", 0)),
            "model_dir": raw.get("model_dir"),
            "trainable_actor_params": pc.get("trainable_actor_params"),
            "trainable_lora_adapter_params": pc.get("trainable_lora_adapter_params"),
            "trainable_comm_layer_params": pc.get("trainable_comm_layer_params"),
            "trainable_attn_params": pc.get("trainable_attn_params"),
            "trainable_total_params": pc.get("trainable_total_params"),
            "baseline_reward": base_rew,
            "degraded_reward": deg.get("reward"),
            "repaired_reward": rep.get("reward"),
            "local_reward_recovery_pct": round(rew_rec * 100.0, 2) if rew_rec is not None else None,
            "local_comm_recovery_pct": round(comm_rec * 100.0, 2) if comm_rec is not None else None,
            "heldout_reward_recovery_pct": round(ho_rew_rec * 100.0, 2) if ho_rew_rec is not None else None,
            "heldout_comm_recovery_pct": round(ho_comm_rec * 100.0, 2) if ho_comm_rec is not None else None,
            "clean_retention_loss_pct": round(retention_loss_pct, 2) if retention_loss_pct is not None else None,
            "clean_retention_pct": round(retention_pct, 2) if retention_pct is not None else None,
            "accepted": bool(fs.get("accepted")),
            "holdout_confirms": bool(fs.get("holdout_confirms")),
        })

    DATA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DATA_RESULTS_DIR / "e3_lora_rank_sweep.json"
    out_csv = DATA_RESULTS_DIR / "e3_lora_rank_sweep.csv"
    out_md = DATA_RESULTS_DIR / "e3_lora_rank_sweep_summary.md"

    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if records:
        fieldnames = list(records[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    by_rank: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        by_rank.setdefault(rec["lora_rank"], []).append(rec)

    md_lines = [
        "# E3 LoRA Rank Sweep Summary (`3a4l`, alpha = 2r)",
        "",
        "## Aggregate by Rank (Mean ± SEM across Seeds)",
        "",
        "| Rank (r) | Alpha | Actor Params (LoRA / Total) | Local Rew Recov % | Local Comm Recov % | Held-Out Rew Recov % | Held-Out Comm Recov % | Clean Retention % | Accepted | Held-Out Confirmed |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in sorted(by_rank.keys()):
        grp = by_rank[r]
        alpha = grp[0]["lora_alpha"]
        lora_p = grp[0]["trainable_lora_adapter_params"]
        actor_p = grp[0]["trainable_actor_params"]
        tot_p = grp[0]["trainable_total_params"]
        acc_cnt = sum(1 for x in grp if x["accepted"])
        ho_cnt = sum(1 for x in grp if x["holdout_confirms"])
        md_lines.append(
            f"| {r} | {alpha:.0f} | {actor_p} ({lora_p} LoRA / {tot_p} w/ attn) | "
            f"{_fmt_mean_sem([x['local_reward_recovery_pct'] for x in grp])} | "
            f"{_fmt_mean_sem([x['local_comm_recovery_pct'] for x in grp])} | "
            f"{_fmt_mean_sem([x['heldout_reward_recovery_pct'] for x in grp])} | "
            f"{_fmt_mean_sem([x['heldout_comm_recovery_pct'] for x in grp])} | "
            f"{_fmt_mean_sem([x['clean_retention_pct'] for x in grp])} | "
            f"{acc_cnt}/{len(grp)} | {ho_cnt}/{len(grp)} |"
        )

    md_lines.extend([
        "",
        "## Per-Seed Breakdown",
        "",
        "| Rank (r) | Seed | Total Params (Actor / w/ Attn) | Local Rew % | Local Comm % | Held-Out Rew % | Held-Out Comm % | Clean Retention % | Accepted | Held-Out |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for rec in sorted(records, key=lambda x: (x["lora_rank"], x["seed"])):
        md_lines.append(
            f"| {rec['lora_rank']} | {rec['seed']} | "
            f"{rec['trainable_actor_params']} / {rec['trainable_total_params']} | "
            f"{rec['local_reward_recovery_pct']}% | {rec['local_comm_recovery_pct']}% | "
            f"{rec['heldout_reward_recovery_pct']}% | {rec['heldout_comm_recovery_pct']}% | "
            f"{rec['clean_retention_pct']}% ({rec['clean_retention_loss_pct']:+.2f}%) | "
            f"{rec['accepted']} | {rec['holdout_confirms']} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[REPORT] Wrote {out_json}, {out_csv}, and {out_md}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E3 LoRA rank sweep on 3a4l.")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if not args.aggregate_only:
        run_sweep(args.ranks, args.seeds, resume=args.resume)
    aggregate_sweep()


if __name__ == "__main__":
    main()
