"""
Batch runner for the AAMAS rebuttal experiments (E1, E2, E2b, E10, E11).

Each experiment is a set of phase2_3_repair.py invocations per checkpoint; every run appends
one JSON row to results/rebuttal/<EXP>.jsonl (exact checkpoint path, shift, arm, all metrics).
Summarise with benchmarking/evaluators/summarize_rebuttal.py.

Checkpoints are given as SEED:PATH so the CRN seeds match the paper's benchmark runs
(phase2_3_repair uses base_seed = seed * 100003):

    python benchmarking/runners/rebuttal_experiments.py --exp E1 \
        --ckpt 1:onpolicy/scripts/results/.../benchmark_3a4l_seed1/run1/models/checkpoint_XXXX \
        --ckpt 2:... --parallel 2

    --dry_run prints the commands without running them.

Defaults match benchmarking/configs/benchmark_matrix.py EVAL_CONFIG except measure/holdout
episodes, which default to 8 (the number stated in the paper; the suite used 6 -> see D5).
"""

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "onpolicy" / "scripts" / "phase2_3_repair.py"

ARMS_E1 = ["lora", "actor_lora", "comm", "noncomm"]


def experiment_runs(exp, accel_scales, rot_degs):
    """Yield (label, extra_args) for one checkpoint."""
    forced = ["--force_repair", "--always_evaluate"]
    if exp == "E1":       # matched repair-scope arms on the paper's partner_full shift
        for arm in ARMS_E1:
            yield "E1_partner_full_{}".format(arm), ["--repair_target", arm] + forced
    elif exp == "E2b":    # decoupled 2x2: the 'both' cell is E1
        for cell in ["msg_only", "actor_only"]:
            for arm in ["lora", "actor_lora", "comm"]:
                yield "E2b_{}_{}".format(cell, arm), ["--decouple", cell, "--repair_target", arm] + forced
    elif exp == "E2":     # trigger discrimination: comm-only vs non-comm shift, both controllers
        for ctrl in ["causal", "reward_only"]:
            yield "E2_comm_perm_{}".format(ctrl), ["--shift", "comm_perm", "--controller", ctrl]
            for s in accel_scales:
                yield "E2_dynamics_a{}_{}".format(s, ctrl), [
                    "--shift", "dynamics", "--dyn_accel_scale", str(s), "--controller", ctrl]
            for cell in ["msg_only", "actor_only"]:
                yield "E2_obs_{}_{}".format(cell, ctrl), ["--decouple", cell, "--controller", ctrl]
    elif exp == "E10":    # severity sweep, both controllers
        for ctrl in ["causal", "reward_only"]:
            yield "E10_partner_x_{}".format(ctrl), ["--mirror_scope", "partner", "--controller", ctrl]
            for d in rot_degs:
                yield "E10_rot{}_{}".format(int(d), ctrl), [
                    "--perturb_angle", str(d), "--controller", ctrl]
    elif exp == "E11":    # zero vs mean vs resampled ablation, no repair
        for shift in [[], ["--shift", "comm_perm"], ["--decouple", "msg_only"], ["--decouple", "actor_only"]]:
            tag = "_".join(a.strip("-") for a in shift) or "partner_full"
            yield "E11_{}".format(tag), shift + ["--no_repair", "--extra_ablations"]
    else:
        raise ValueError(exp)


def build_cmd(seed, path, label, extra, args, out_dir):
    cmd = [sys.executable, "-u", str(SCRIPT),
           "--env_name", "MPE", "--algorithm_name", "mappo",
           "--seed", str(seed), "--model_dir", path,
           "--measure_episodes", str(args.measure_episodes),
           "--holdout_episodes", str(args.holdout_episodes),
           "--repair_iters", str(args.repair_iters),
           "--lora_rank", str(args.lora_rank), "--lora_alpha", str(args.lora_alpha),
           "--n_rollout_threads", str(args.n_rollout_threads),
           "--n_eval_rollout_threads", "1",
           "--detect_reward_drop_ratio", "0.10",
           "--accept_reward_recovery", "0.30", "--accept_comm_recovery", "0.15",
           "--results_log", str(out_dir / "{}.jsonl".format(args.exp)),
           "--run_label", label,
           "--run_dir", str(out_dir / "runs" / "{}_seed{}".format(label, seed))] + extra
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["E1", "E2", "E2b", "E10", "E11"])
    ap.add_argument("--ckpt", action="append", required=True, help="SEED:PATH (repeatable)")
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "rebuttal"))
    ap.add_argument("--measure_episodes", type=int, default=8)
    ap.add_argument("--holdout_episodes", type=int, default=8)
    ap.add_argument("--repair_iters", type=int, default=15)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--n_rollout_threads", type=int, default=4)
    ap.add_argument("--accel_scales", type=float, nargs="+", default=[0.5, 0.3])
    ap.add_argument("--rot_degs", type=float, nargs="+", default=[45, 90, 135, 180])
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in args.ckpt:
        seed, path = spec.split(":", 1)
        if not os.path.exists(os.path.join(path, "actor.pt")):
            sys.exit("no actor.pt under {}".format(path))
        for label, extra in experiment_runs(args.exp, args.accel_scales, args.rot_degs):
            jobs.append((label, seed, build_cmd(seed, path, label, extra, args, out_dir)))

    print("{} runs for {}".format(len(jobs), args.exp))
    if args.dry_run:
        for _, _, cmd in jobs:
            print(" ".join(shlex.quote(c) for c in cmd))
        return

    env = dict(os.environ, WANDB_MODE="disabled", PYTHONPATH=str(ROOT))

    def run(job):
        label, seed, cmd = job
        log = out_dir / "logs" / "{}_seed{}.log".format(label, seed)
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        print("[{}] {} seed {} -> {}".format("ok" if rc == 0 else "FAIL rc={}".format(rc), label, seed, log),
              flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        rcs = list(ex.map(run, jobs))
    print("done: {} ok, {} failed".format(rcs.count(0), len(rcs) - rcs.count(0)))


if __name__ == "__main__":
    main()
