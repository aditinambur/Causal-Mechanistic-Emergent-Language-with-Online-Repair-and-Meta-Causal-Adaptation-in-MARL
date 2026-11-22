#!/usr/bin/env python3
"""
Task T1 Runner: Train 3a4l Seeds 6-10 (500k steps), Confirm Checkpoints via
select_best_checkpoint.py + --no_repair, Run Benchmark Arms, Update Master Results,
and Compute the Exact Two-Sided Permutation Test (2a3l N=5 vs 3a4l N=10).
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmarking.configs.benchmark_matrix import CONTROL_ARMS, EVAL_CONFIG, SCENARIO_SCALES
from benchmarking.runners.benchmark_suite import (
    _resolve_checkpoint_provenance,
    parse_phase2_3_output,
)

RESULTS_BASE = WORKSPACE_ROOT / "onpolicy" / "scripts" / "results" / "MPE" / "simple_spread" / "mappo"
T1_JSONL = WORKSPACE_ROOT / "results" / "t1_3a4l_seeds6_10.jsonl"
DATA_RESULTS_DIR = WORKSPACE_ROOT / "benchmarking" / "data" / "results"
MASTER_JSON = DATA_RESULTS_DIR / "benchmark_master.json"
MASTER_CSV = DATA_RESULTS_DIR / "benchmark_master.csv"


def _fmt_mean_sem(values: List[float], decimals: int = 2) -> str:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return "N/A"
    mean = float(np.mean(clean))
    sem = float(np.std(clean, ddof=1) / np.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {sem:.{decimals}f}" if len(clean) > 1 else f"{mean:.{decimals}f}"


def _find_run_dir(seed: int) -> Optional[Path]:
    exp_dir = RESULTS_BASE / f"benchmark_3a4l_seed{seed}"
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


def train_seed(seed: int, num_env_steps: int = 500_000) -> Path:
    if _is_training_complete(seed):
        rdir = _find_run_dir(seed)
        print(f"[SKIP TRAIN] 3a4l seed={seed} already complete at {rdir}")
        return rdir

    exp_name = f"benchmark_3a4l_seed{seed}"
    cmd = [
        sys.executable, "-u", "onpolicy/scripts/train/train_mpe.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
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
    print(f"\n=== T1 Train MAPPO 3a4l: {exp_name} ({num_env_steps} steps) ===")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, check=True)
    print(f"=== Finished training {exp_name} in {time.time() - t0:.1f}s ===")
    rdir = _find_run_dir(seed)
    if rdir is None:
        raise RuntimeError(f"Training completed for {exp_name} but no run directory found.")
    return rdir


def _run_capture(cmd: List[str]) -> str:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(WORKSPACE_ROOT),
        env=env,
    )
    out = []
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
        out.append(line)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return "".join(out)


def confirm_checkpoint(seed: int, run_dir: Path, top_k: int = 4, min_comm_effect: float = 150.0) -> Tuple[Path, List[Dict[str, Any]]]:
    """Rank checkpoints via select_best_checkpoint.py and confirm via --no_repair."""
    from onpolicy.scripts.select_best_checkpoint import _load_rows, score_checkpoints

    csv_path = run_dir / "causal_influence.csv"
    models_dir = run_dir / "models"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing causal_influence.csv at {csv_path}")

    rows = _load_rows(str(csv_path))
    ranked, source = score_checkpoints(rows)
    candidates: List[Tuple[int, Path, float]] = []
    for r in ranked:
        step = int(r["total_num_steps"])
        cand = models_dir / f"checkpoint_{step}"
        if (cand / "actor.pt").is_file():
            candidates.append((step, cand, float(r["score"])))
        if len(candidates) >= top_k:
            break

    if not candidates:
        raise FileNotFoundError(f"No ranked checkpoint_* directories found in {models_dir}")

    screened: List[Dict[str, Any]] = []
    chosen: Optional[Path] = None

    for idx, (step, cand, score) in enumerate(candidates, start=1):
        rel_ckpt = cand.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
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
            "--measure_episodes", str(EVAL_CONFIG["measure_episodes"]),
            "--n_eval_rollout_threads", "1",
            "--no_repair",
        ]
        print(f"\n=== T1 Checkpoint Screen: seed={seed} rank=#{idx} step={step} (score={score:.1f}, src={source}) ===")
        log_txt = _run_capture(cmd)
        parsed = parse_phase2_3_output(log_txt)
        comm_eff = parsed.get("baseline_comm_effect")
        drop_ratio = parsed.get("reward_drop_ratio")
        screened.append({
            "rank": idx,
            "step": step,
            "score": round(score, 2),
            "score_source": source,
            "checkpoint_path": rel_ckpt,
            "baseline_reward": parsed.get("baseline_reward"),
            "baseline_comm_effect": comm_eff,
            "reward_drop_ratio": drop_ratio,
        })
        if chosen is None and comm_eff is not None and comm_eff >= min_comm_effect:
            chosen = cand
            print(f"[SELECTED] seed={seed}: checkpoint_{step} cleared comm_effect={comm_eff:.1f} >= {min_comm_effect}")
            break

    if chosen is None:
        # Fallback to highest measured comm_effect among screened top_k
        best_item = max(
            screened,
            key=lambda x: x["baseline_comm_effect"] if x["baseline_comm_effect"] is not None else float("-inf"),
        )
        chosen = WORKSPACE_ROOT / best_item["checkpoint_path"]
        print(f"[SELECTED FALLBACK] seed={seed}: {chosen.name} (highest comm_effect={best_item['baseline_comm_effect']})")

    return chosen, screened


def evaluate_seed_arms(seed: int, ckpt: Path) -> List[Dict[str, Any]]:
    scale_cfg = SCENARIO_SCALES["3a4l"]
    rel_ckpt = ckpt.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    rel_log = T1_JSONL.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    provenance = _resolve_checkpoint_provenance(ckpt)
    records = []

    for arm_key in ("causal_adaptive", "naive_reward_only"):
        arm_cfg = CONTROL_ARMS[arm_key]
        run_id = f"3a4l_seed{seed}_{arm_key}"
        cmd = [
            sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
            "--env_name", "MPE",
            "--scenario_name", "simple_spread",
            "--algorithm_name", "mappo",
            "--seed", str(seed),
            "--num_agents", "3",
            "--num_landmarks", "4",
            "--model_dir", rel_ckpt,
            "--mirror_scope", scale_cfg["mirror_scope"],
            "--measure_episodes", str(EVAL_CONFIG["measure_episodes"]),
            "--holdout_episodes", str(EVAL_CONFIG["measure_episodes"]),
            "--repair_iters", str(EVAL_CONFIG["repair_iters"]),
            "--controller", arm_cfg["controller"],
            "--lora_rank", str(EVAL_CONFIG["lora_rank"]),
            "--lora_alpha", str(EVAL_CONFIG["lora_alpha"]),
            "--n_rollout_threads", str(EVAL_CONFIG["n_rollout_threads"]),
            "--n_eval_rollout_threads", str(EVAL_CONFIG["n_eval_rollout_threads"]),
            "--detect_reward_drop_ratio", str(EVAL_CONFIG["reward_drop_threshold"]),
            "--accept_reward_recovery", str(EVAL_CONFIG["accept_reward_recovery"]),
            "--accept_comm_recovery", str(EVAL_CONFIG["accept_comm_recovery"]),
            "--results_log", rel_log,
        ]
        if arm_cfg["repair_target"] is not None and arm_cfg["repair_target"] != "none":
            cmd.extend(["--repair_target", arm_cfg["repair_target"]])

        print(f"\n=== T1 Benchmark Eval: {run_id} ({ckpt.name}) ===")
        log_txt = _run_capture(cmd)
        parsed = parse_phase2_3_output(log_txt)
        records.append({
            "run_id": run_id,
            "scale": "3a4l",
            "scale_name": scale_cfg["name"],
            "num_agents": scale_cfg["num_agents"],
            "num_landmarks": scale_cfg["num_landmarks"],
            "seed": seed,
            "arm": arm_key,
            "arm_name": arm_cfg["name"],
            "checkpoint": ckpt.name,
            **provenance,
            **parsed,
        })
    return records


def upsert_master_records(new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    master = json.loads(MASTER_JSON.read_text(encoding="utf-8")) if MASTER_JSON.exists() else []
    for rec in new_records:
        idx = next((i for i, r in enumerate(master) if r.get("run_id") == rec["run_id"]), None)
        if idx is not None:
            master[idx] = rec
        else:
            master.append(rec)

    MASTER_JSON.write_text(json.dumps(master, indent=2), encoding="utf-8")
    fieldnames = list(dict.fromkeys([k for r in master for k in r.keys()]))
    with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(master)
    return master


def exact_two_sided_permutation_test(group_a: List[float], group_b: List[float]) -> Dict[str, Any]:
    combined = list(group_a) + list(group_b)
    n_a = len(group_a)
    n_b = len(group_b)
    n_total = len(combined)
    total_sum = sum(combined)
    obs_diff = (sum(group_b) / n_b) - (sum(group_a) / n_a)
    obs_abs = abs(obs_diff)
    extreme = 0
    total = 0
    for idxs in itertools.combinations(range(n_total), n_a):
        sum_a = sum(combined[i] for i in idxs)
        sum_b = total_sum - sum_a
        diff = (sum_b / n_b) - (sum_a / n_a)
        if abs(diff) >= obs_abs - 1e-12:
            extreme += 1
        total += 1
    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": round(sum(group_a) / n_a, 2),
        "mean_b": round(sum(group_b) / n_b, 2),
        "obs_diff_b_minus_a": round(obs_diff, 2),
        "extreme_count": extreme,
        "total_permutations": total,
        "p_value_two_sided": extreme / total,
    }


def write_summary(master: List[Dict[str, Any]], screening_by_seed: Dict[int, List[Dict[str, Any]]]) -> None:
    causal_2a3l = sorted(
        [r for r in master if r.get("scale") == "2a3l" and r.get("arm") == "causal_adaptive"],
        key=lambda x: int(x["seed"]),
    )
    causal_3a4l = sorted(
        [r for r in master if r.get("scale") == "3a4l" and r.get("arm") == "causal_adaptive"],
        key=lambda x: int(x["seed"]),
    )
    naive_3a4l = sorted(
        [r for r in master if r.get("scale") == "3a4l" and r.get("arm") == "naive_reward_only"],
        key=lambda x: int(x["seed"]),
    )

    comm_2a3l = [float(r["baseline_comm_effect"]) for r in causal_2a3l if r.get("baseline_comm_effect") is not None]
    comm_3a4l_first5 = [float(r["baseline_comm_effect"]) for r in causal_3a4l if int(r["seed"]) <= 5 and r.get("baseline_comm_effect") is not None]
    comm_3a4l_all = [float(r["baseline_comm_effect"]) for r in causal_3a4l if r.get("baseline_comm_effect") is not None]

    perm_5v5 = exact_two_sided_permutation_test(comm_2a3l, comm_3a4l_first5) if comm_3a4l_first5 else None
    perm_5v10 = exact_two_sided_permutation_test(comm_2a3l, comm_3a4l_all) if comm_3a4l_all else None

    seeds_6_10 = [r for r in causal_3a4l if int(r["seed"]) >= 6]
    triggered_all = [r for r in causal_3a4l if r.get("detector_fired")]
    accepted_all = [r for r in causal_3a4l if r.get("repair_decision") == "ACCEPTED"]
    confirmed_all = [r for r in causal_3a4l if r.get("heldout_validation") == "CONFIRMED"]

    naive_triggered = [r for r in naive_3a4l if r.get("detector_fired")]

    out_payload = {
        "n_2a3l": len(comm_2a3l),
        "n_3a4l": len(comm_3a4l_all),
        "comm_2a3l": comm_2a3l,
        "comm_3a4l": comm_3a4l_all,
        "perm_test_5v5": perm_5v5,
        "perm_test_5v10": perm_5v10,
        "screening_by_seed": screening_by_seed,
        "3a4l_seeds6_10_causal": seeds_6_10,
    }
    (DATA_RESULTS_DIR / "t1_3a4l_seeds6_10.json").write_text(
        json.dumps(out_payload, indent=2), encoding="utf-8"
    )

    with (DATA_RESULTS_DIR / "t1_3a4l_seeds6_10.csv").open("w", newline="", encoding="utf-8") as f:
        if seeds_6_10:
            w = csv.DictWriter(f, fieldnames=list(seeds_6_10[0].keys()))
            w.writeheader()
            w.writerows(seeds_6_10)

    lines = [
        "# T1 Summary: `3a4l` Seeds 6–10 & Exact Two-Sided Permutation Test",
        "",
        "## Cross-Scale Communication Benefit (`2a3l` vs `3a4l`, Exact Two-Sided Permutation Test)",
        "",
        f"- **`2a3l` (N={len(comm_2a3l)}):** {_fmt_mean_sem(comm_2a3l)} (values: {[round(v, 1) for v in comm_2a3l]})",
        f"- **`3a4l` Seeds 1–5 (N={len(comm_3a4l_first5)}):** {_fmt_mean_sem(comm_3a4l_first5)}",
        f"- **`3a4l` All Seeds (N={len(comm_3a4l_all)}):** {_fmt_mean_sem(comm_3a4l_all)} (values: {[round(v, 1) for v in comm_3a4l_all]})",
    ]
    if perm_5v5:
        lines.append(
            f"- **Exact two-sided permutation test (`2a3l` N={perm_5v5['n_a']} vs `3a4l` N={perm_5v5['n_b']}):** "
            f"p = {perm_5v5['p_value_two_sided']:.5f} ({perm_5v5['extreme_count']}/{perm_5v5['total_permutations']}, diff = {perm_5v5['obs_diff_b_minus_a']:+.2f})"
        )
    if perm_5v10:
        lines.append(
            f"- **Exact two-sided permutation test (`2a3l` N={perm_5v10['n_a']} vs `3a4l` N={perm_5v10['n_b']}):** "
            f"p = {perm_5v10['p_value_two_sided']:.5f} ({perm_5v10['extreme_count']}/{perm_5v10['total_permutations']}, diff = {perm_5v10['obs_diff_b_minus_a']:+.2f})"
        )

    lines.extend([
        "",
        f"## `3a4l` Aggregate Across All {len(causal_3a4l)} Seeds (`causal_adaptive` vs `naive_reward_only`)",
        "",
        f"- **Baseline Return:** {_fmt_mean_sem([r['baseline_reward'] for r in causal_3a4l if r.get('baseline_reward') is not None])}",
        f"- **Baseline Comm Gain:** {_fmt_mean_sem(comm_3a4l_all)}",
        f"- **Baseline CIC KL:** {_fmt_mean_sem([r['baseline_kl'] for r in causal_3a4l if r.get('baseline_kl') is not None], 3)}",
        f"- **Causal Trigger Rate:** {len(triggered_all)}/{len(causal_3a4l)} | **Naive Trigger Rate:** {len(naive_triggered)}/{len(naive_3a4l)}",
        f"- **Causal Accepted:** {len(accepted_all)}/{len(causal_3a4l)} | **Held-Out Confirmed:** {len(confirmed_all)}/{len(accepted_all) if accepted_all else 0}",
        f"- **Causal Reward Recovery % (Accepted):** {_fmt_mean_sem([r['reward_recovery_pct'] for r in accepted_all if r.get('reward_recovery_pct') is not None])}",
        f"- **Causal Comm Recovery % (Accepted):** {_fmt_mean_sem([r['comm_recovery_pct'] for r in accepted_all if r.get('comm_recovery_pct') is not None])}",
        f"- **Causal Held-Out Reward Recovery % (Accepted):** {_fmt_mean_sem([r['heldout_reward_recovery'] for r in accepted_all if r.get('heldout_reward_recovery') is not None])}",
        "",
        "## Seeds 6–10 Per-Seed Results (`causal_adaptive`)",
        "",
        "| Seed | Selected Step | Base Return | Base Comm | Drop Ratio | Triggered | Decision | Local Rew % | Local Comm % | Held-Out Rew % | Retention Loss | Held-Out |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in seeds_6_10:
        lines.append(
            f"| {r['seed']} | {r.get('checkpoint_step')} | {r.get('baseline_reward')} | {r.get('baseline_comm_effect')} | "
            f"{r.get('reward_drop_ratio')} | {r.get('detector_fired')} | {r.get('repair_decision')} ({r.get('repair_target')}) | "
            f"{r.get('reward_recovery_pct')} | {r.get('comm_recovery_pct')} | "
            f"{r.get('heldout_reward_recovery')} | {r.get('normal_reward_retention_loss')} | {r.get('heldout_validation')} |"
        )

    lines.extend(["", "## Top-4 `select_best_checkpoint.py` Screening Log (`--no_repair`)", ""])
    for s in sorted(screening_by_seed.keys()):
        lines.append(f"- **Seed {s}:** " + json.dumps(screening_by_seed[s]))

    (DATA_RESULTS_DIR / "t1_3a4l_seeds6_10_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[REPORT] Wrote T1 summary to {DATA_RESULTS_DIR / 't1_3a4l_seeds6_10_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate 3a4l seeds 6-10 (Task T1).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8, 9, 10])
    parser.add_argument("--num_env_steps", type=int, default=500_000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    screening_path = DATA_RESULTS_DIR / "t1_checkpoint_screening.json"
    screening_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    if screening_path.exists():
        raw_scr = json.loads(screening_path.read_text(encoding="utf-8"))
        screening_by_seed = {int(k): v for k, v in raw_scr.items()}

    if not args.summary_only:
        T1_JSONL.parent.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            rdir = _find_run_dir(seed) if args.eval_only else train_seed(seed, num_env_steps=args.num_env_steps)
            if rdir is None:
                print(f"[WARNING] Run dir not found for 3a4l seed={seed}, skipping.")
                continue
            ckpt, screened = confirm_checkpoint(seed, rdir)
            screening_by_seed[seed] = screened
            screening_path.write_text(json.dumps(screening_by_seed, indent=2), encoding="utf-8")

            records = evaluate_seed_arms(seed, ckpt)
            master = upsert_master_records(records)
            write_summary(master, screening_by_seed)
    else:
        master = json.loads(MASTER_JSON.read_text(encoding="utf-8"))
        write_summary(master, screening_by_seed)


if __name__ == "__main__":
    main()
