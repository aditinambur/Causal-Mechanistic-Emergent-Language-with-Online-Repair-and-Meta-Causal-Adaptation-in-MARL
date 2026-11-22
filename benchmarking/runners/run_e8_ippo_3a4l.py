#!/usr/bin/env python3
"""
Task E8 Runner: IPPO Portability on 3a4l (Seeds 1-5, 500k Steps + Online Repair).

Trains MAPPO with --use_centralized_V (setting use_centralized_V=False, i.e., IPPO),
selects the top checkpoint per seed via select_best_checkpoint.py, evaluates the
causal online repair pipeline unchanged, and aggregates summary statistics.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

RESULTS_BASE = WORKSPACE_ROOT / "onpolicy" / "scripts" / "results" / "MPE" / "simple_spread" / "mappo"
RESULTS_JSONL = WORKSPACE_ROOT / "results" / "e8_ippo_3a4l.jsonl"
CKPT_BEST_JSONL = WORKSPACE_ROOT / "results" / "e8_ippo_3a4l_ckpt_best.jsonl"
DATA_RESULTS_DIR = WORKSPACE_ROOT / "benchmarking" / "data" / "results"


def _fmt_mean_sem(values: List[float], suffix: str = "%") -> str:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return "N/A"
    mean = float(np.mean(clean))
    sem = float(np.std(clean, ddof=1) / np.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return f"{mean:.1f}{suffix} ± {sem:.1f}{suffix}" if len(clean) > 1 else f"{mean:.1f}{suffix}"


def _find_run_dir(seed: int) -> Optional[Path]:
    exp_dir = RESULTS_BASE / f"benchmark_3a4l_ippo_seed{seed}"
    if not exp_dir.exists():
        return None
    runs = sorted(exp_dir.glob("run*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "models").exists():
            return r
    return None


def _is_training_complete(seed: int) -> bool:
    rdir = _find_run_dir(seed)
    if rdir is None:
        return False
    return (rdir / "models" / "checkpoint_486400" / "actor.pt").is_file()


def train_ippo_seed(seed: int, num_env_steps: int = 500_000) -> Path:
    if _is_training_complete(seed):
        rdir = _find_run_dir(seed)
        print(f"[SKIP TRAIN] IPPO 3a4l seed={seed} already complete at {rdir}")
        return rdir

    exp_name = f"benchmark_3a4l_ippo_seed{seed}"
    cmd = [
        sys.executable, "-u", "onpolicy/scripts/train/train_mpe.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--use_centralized_V",
        "--seed", str(seed),
        "--num_agents", "3",
        "--num_landmarks", "4",
        "--num_env_steps", str(num_env_steps),
        "--episode_length", "200",
        "--n_rollout_threads", "32",
        "--n_eval_rollout_threads", "1",
        "--use_eval",
        "--eval_interval", "5",
        "--eval_disable_messages",
        "--eval_noise_std", "0.25",
        "--experiment_name", exp_name,
        "--use_wandb",
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(f"\n=== E8 Train IPPO: {exp_name} ({num_env_steps} steps) ===")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, check=True)
    print(f"=== Finished training {exp_name} in {time.time() - t0:.1f}s ===")
    rdir = _find_run_dir(seed)
    if rdir is None:
        raise RuntimeError(f"Training completed for {exp_name} but no run directory found.")
    return rdir


def _select_checkpoint(run_dir: Path) -> Path:
    from onpolicy.scripts.select_best_checkpoint import _load_rows, score_checkpoints

    models_dir = run_dir / "models"
    csv_path = run_dir / "causal_influence.csv"
    if csv_path.is_file():
        rows = _load_rows(str(csv_path))
        ranked, _ = score_checkpoints(rows)
        for r in ranked:
            step = int(r["total_num_steps"])
            cand = models_dir / f"checkpoint_{step}"
            if (cand / "actor.pt").is_file():
                return cand
    best = models_dir / "checkpoint_best"
    if (best / "actor.pt").is_file():
        return best
    raise FileNotFoundError(f"No usable checkpoint in {models_dir}")


def _same_actor_weights(ckpt_a: Path, ckpt_b: Path) -> bool:
    fa = ckpt_a / "actor.pt"
    fb = ckpt_b / "actor.pt"
    if not fa.is_file() or not fb.is_file():
        return False
    return hashlib.sha256(fa.read_bytes()).hexdigest() == hashlib.sha256(fb.read_bytes()).hexdigest()


def run_repair_eval(seed: int, ckpt_dir: Path, results_jsonl: Path) -> None:
    rel_ckpt = ckpt_dir.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    rel_log = results_jsonl.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    cmd = [
        sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--seed", str(seed),
        "--num_agents", "3",
        "--num_landmarks", "4",
        "--model_dir", rel_ckpt,
        "--mirror_scope", "partner_full",
        "--measure_episodes", "6",
        "--holdout_episodes", "6",
        "--repair_iters", "15",
        "--controller", "causal",
        "--lora_rank", "4",
        "--lora_alpha", "8.0",
        "--n_rollout_threads", "4",
        "--n_eval_rollout_threads", "1",
        "--detect_reward_drop_ratio", "0.10",
        "--accept_reward_recovery", "0.30",
        "--accept_comm_recovery", "0.15",
        "--results_log", rel_log,
    ]
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(f"\n=== E8 Repair Eval: seed={seed}, model_dir={rel_ckpt} -> {rel_log} ===")
    subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, check=True)


def aggregate_results() -> List[Dict[str, Any]]:
    if not RESULTS_JSONL.exists():
        print(f"[INFO] No JSONL log at {RESULTS_JSONL} yet.")
        return []

    records: List[Dict[str, Any]] = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        fs = raw.get("final_status") or {}
        base = raw.get("baseline") or {}
        deg = raw.get("degraded") or {}
        rep = raw.get("repaired") or {}
        det = raw.get("detect_info") or {}

        mode = fs.get("mode")
        triggered = mode != "not_degraded"
        accepted = bool(fs.get("accepted"))

        rew_rec = None
        comm_rec = None
        if triggered:
            esc = fs.get("escalation_log") or []
            if accepted and esc:
                for item in reversed(esc):
                    if item.get("accepted"):
                        rew_rec = item.get("reward_recovery")
                        comm_rec = item.get("comm_recovery")
                        break
            elif esc:
                rew_rec = esc[-1].get("reward_recovery")
                comm_rec = esc[-1].get("comm_recovery")

        ho_rew_rec = fs.get("holdout_reward_recovery")
        ho_comm_rec = fs.get("holdout_comm_recovery")

        base_rew = base.get("reward")
        norm_delta = fs.get("normal_reward_delta_vs_baseline")
        retention_loss_pct = None
        retention_pct = None
        if base_rew is not None and norm_delta is not None:
            retention_loss_pct = (float(norm_delta) / (abs(float(base_rew)) + 1e-6)) * 100.0
            retention_pct = 100.0 + retention_loss_pct

        records.append({
            "seed": int(raw.get("seed", 0)),
            "model_dir": raw.get("model_dir"),
            "baseline_reward": round(base_rew, 2) if base_rew is not None else None,
            "baseline_comm_effect": round(base.get("comm_effect"), 2) if base.get("comm_effect") is not None else None,
            "baseline_kl": round(base.get("kl"), 4) if base.get("kl") is not None else None,
            "degraded_reward": round(deg.get("reward"), 2) if deg.get("reward") is not None else None,
            "degraded_comm_effect": round(deg.get("comm_effect"), 2) if deg.get("comm_effect") is not None else None,
            "reward_drop_ratio": round(det.get("reward_drop_ratio"), 4) if det.get("reward_drop_ratio") is not None else None,
            "causal_evidence_score": round(det.get("causal_evidence_score"), 3) if det.get("causal_evidence_score") is not None else None,
            "triggered": triggered,
            "accepted": accepted,
            "accepted_target": fs.get("accepted_target"),
            "repaired_reward": round(rep.get("reward"), 2) if rep and rep.get("reward") is not None else None,
            "local_reward_recovery_pct": round(rew_rec * 100.0, 2) if rew_rec is not None else None,
            "local_comm_recovery_pct": round(comm_rec * 100.0, 2) if comm_rec is not None else None,
            "heldout_reward_recovery_pct": round(ho_rew_rec * 100.0, 2) if ho_rew_rec is not None else None,
            "heldout_comm_recovery_pct": round(ho_comm_rec * 100.0, 2) if ho_comm_rec is not None else None,
            "holdout_confirms": fs.get("holdout_confirms"),
            "clean_retention_loss_pct": round(retention_loss_pct, 2) if retention_loss_pct is not None else None,
            "clean_retention_pct": round(retention_pct, 2) if retention_pct is not None else None,
        })

    DATA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DATA_RESULTS_DIR / "e8_ippo_3a4l.json"
    out_csv = DATA_RESULTS_DIR / "e8_ippo_3a4l.csv"
    out_md = DATA_RESULTS_DIR / "e8_ippo_3a4l_summary.md"

    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if records:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    triggered_recs = [r for r in records if r["triggered"]]
    accepted_recs = [r for r in records if r["accepted"]]
    confirmed_recs = [r for r in records if r.get("holdout_confirms") is True]

    md_lines = [
        "# E8 IPPO Portability Summary (`3a4l`, `--use_centralized_V False`, 500k steps)",
        "",
        "## Aggregate Across Seeds",
        "",
        f"- **Seeds evaluated:** {len(records)}",
        f"- **Triggered:** {len(triggered_recs)}/{len(records)}",
        f"- **Accepted:** {len(accepted_recs)}/{len(records)}",
        f"- **Held-Out Confirmed:** {len(confirmed_recs)}/{len(accepted_recs) if accepted_recs else 0}",
        f"- **Baseline Return:** {_fmt_mean_sem([r['baseline_reward'] for r in records], '')}",
        f"- **Baseline Comm Effect:** {_fmt_mean_sem([r['baseline_comm_effect'] for r in records], '')}",
        f"- **Local Reward Recovery (Accepted):** {_fmt_mean_sem([r['local_reward_recovery_pct'] for r in accepted_recs])}",
        f"- **Local Comm Recovery (Accepted):** {_fmt_mean_sem([r['local_comm_recovery_pct'] for r in accepted_recs])}",
        f"- **Held-Out Reward Recovery (Accepted):** {_fmt_mean_sem([r['heldout_reward_recovery_pct'] for r in accepted_recs])}",
        f"- **Clean Retention % (Accepted):** {_fmt_mean_sem([r['clean_retention_pct'] for r in accepted_recs])}",
        "",
        "## Per-Seed Breakdown",
        "",
        "| Seed | Checkpoint | Base Return | Base Comm | Drop Ratio | Triggered | Accepted (Target) | Local Rew % | Local Comm % | Held-Out Rew % | Clean Retention % | Held-Out |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in sorted(records, key=lambda x: x["seed"]):
        ckpt_name = Path(r["model_dir"]).name if r.get("model_dir") else "N/A"
        md_lines.append(
            f"| {r['seed']} | `{ckpt_name}` | {r['baseline_reward']} | {r['baseline_comm_effect']} | "
            f"{r['reward_drop_ratio']} | {r['triggered']} | {r['accepted']} ({r['accepted_target']}) | "
            f"{r['local_reward_recovery_pct']} | {r['local_comm_recovery_pct']} | "
            f"{r['heldout_reward_recovery_pct']} | {r['clean_retention_pct']} | {r['holdout_confirms']} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[REPORT] Wrote {out_json}, {out_csv}, and {out_md}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate IPPO on 3a4l (Task E8).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num_env_steps", type=int, default=500_000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    completed_seeds = set()
    if args.resume and RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_seeds.add(int(json.loads(line).get("seed", 0)))
    elif not args.eval_only and RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()
        if CKPT_BEST_JSONL.exists():
            CKPT_BEST_JSONL.unlink()

    for seed in args.seeds:
        if args.eval_only:
            rdir = _find_run_dir(seed)
            if rdir is None:
                print(f"[WARNING] No IPPO run found for seed {seed}, skipping.")
                continue
        else:
            rdir = train_ippo_seed(seed, num_env_steps=args.num_env_steps)

        if seed in completed_seeds:
            print(f"[SKIP EVAL] seed={seed} already in {RESULTS_JSONL}")
            continue

        selected_ckpt = _select_checkpoint(rdir)
        run_repair_eval(seed, selected_ckpt, RESULTS_JSONL)

        ckpt_best = rdir / "models" / "checkpoint_best"
        if ckpt_best.exists() and not _same_actor_weights(selected_ckpt, ckpt_best):
            run_repair_eval(seed, ckpt_best, CKPT_BEST_JSONL)

        aggregate_results()


if __name__ == "__main__":
    main()
