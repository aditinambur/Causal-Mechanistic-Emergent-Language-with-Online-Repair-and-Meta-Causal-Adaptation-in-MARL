#!/usr/bin/env python
"""
TEAM_HANDOFF item 7 -- are the eight cutoff numbers reasonable, or just convenient?

phase2_3_repair.py has eight tunable thresholds (--detect_k_sigma, --detect_min_ratio,
--detect_reward_drop_ratio, --reward_only_drop_ratio, --select_severe_reward_ratio,
--select_sharp_value_sens_ratio, --accept_reward_recovery, --accept_comm_recovery). Every
one was picked as a round number and never tested. TEAM_HANDOFF item 7 asks: would the
recorded outcomes (detect / target-select / accept) have come out differently with slightly
different cutoffs?

This is a PURE POST-HOC re-computation. It re-imports the actual decision functions from
phase2_3_repair.py (detect_degradation, detect_degradation_reward_only, select_repair_target,
accept_repair) and re-runs them on the baseline/degraded/repaired fingerprints already sitting
in your --results_log rows, sweeping each threshold across a grid around its shipped default.
It never retrains and never re-runs an episode -- it can only tell you whether a DIFFERENT
cutoff would have changed the DECISION made on data you already collected. It cannot tell you
what would happen on data you haven't collected (that's what items 2-5 are for).

Usage
-----
    python onpolicy/scripts/threshold_sensitivity.py results/*.jsonl --csv_out results/sensitivity.csv

For each row and each threshold, prints the fraction of the sweep grid that would have
flipped that row's decision, and flags any row that sits within one grid step of flipping at
the DEFAULT value -- exactly the "47%, 3 points from the 50% cutoff" case TEAM_HANDOFF item 7
calls out.
"""

import argparse
import csv
import glob
import json
import sys

# NOTE on why these four functions are copied here instead of imported from
# phase2_3_repair.py: that module does `import torch` and `from onpolicy.config import
# get_config` at module scope, which pulls in the full training stack (and its conda env)
# just to reach four pure, stateless functions. Duplicating them keeps this script runnable
# from a plain analysis environment (e.g. a laptop with only pandas/scipy, no torch/MPE
# installed) with no behavior change.
#
# CORRECTNESS CONTRACT: these four functions must stay byte-identical to
# onpolicy/scripts/phase2_3_repair.py's detect_degradation / detect_degradation_reward_only /
# select_repair_target / accept_repair. If you edit the thresholds or logic in
# phase2_3_repair.py, copy the change here too, or this script will silently analyze a
# decision rule that is no longer the one actually running. A CI/test step that diffs the
# two copies (or that imports phase2_3_repair.py in an environment that does have torch and
# asserts identical output for these four functions on fixed fixtures) is worth adding
# before this script is relied on for the paper's write-up.


def detect_degradation(baseline, current, k_sigma, min_ratio, reward_drop_ratio_threshold):
    baseline_reward_mag = abs(baseline['reward']) + 1e-6
    reward_drop_ratio = (baseline['reward'] - current['reward']) / baseline_reward_mag
    reward_degraded = reward_drop_ratio >= reward_drop_ratio_threshold

    sigma_band = baseline['comm_effect'] - k_sigma * baseline['comm_effect_std']
    ratio_band = min_ratio * baseline['comm_effect']
    comm_threshold = max(sigma_band, ratio_band)

    comm_degraded = current['comm_effect'] < comm_threshold
    value_degraded = current['value_sensitivity'] < 0.5 * baseline['value_sensitivity']
    comm_related = bool(comm_degraded or value_degraded)

    degraded = bool(reward_degraded and comm_related)
    return degraded, {}


def detect_degradation_reward_only(baseline, current, reward_drop_ratio_threshold):
    baseline_mag = abs(baseline['reward']) + 1e-6
    reward_drop = baseline['reward'] - current['reward']
    reward_drop_ratio = reward_drop / baseline_mag
    degraded = bool(reward_drop_ratio >= reward_drop_ratio_threshold)
    return degraded, {}


def select_repair_target(baseline, degraded, severe_reward_ratio, sharp_value_sens_ratio,
                          order=('embedding', 'comm', 'lora', 'full')):
    baseline_reward_mag = abs(baseline['reward']) + 1e-6
    reward_drop_ratio = (baseline['reward'] - degraded['reward']) / baseline_reward_mag
    value_sens_ratio = degraded['value_sensitivity'] / max(1e-6, baseline['value_sensitivity'])

    if reward_drop_ratio >= severe_reward_ratio:
        return ('lora' if 'lora' in order else 'full'), None
    if value_sens_ratio <= sharp_value_sens_ratio:
        return 'embedding', None
    return 'comm', None


def accept_repair(baseline, degraded, repaired, reward_thresh, comm_thresh, epsilon=1e-6):
    def _recovery_ratio(key):
        lost = baseline[key] - degraded[key]
        if lost <= epsilon:
            return None
        return (repaired[key] - degraded[key]) / lost

    reward_recovery = _recovery_ratio('reward')
    comm_recovery = _recovery_ratio('comm_effect')

    gates = []
    for value, thresh in ((reward_recovery, reward_thresh), (comm_recovery, comm_thresh)):
        if value is not None:
            gates.append(value >= thresh)

    accepted = bool(gates) and all(gates)
    return accepted, {}

# name -> (default, grid of values to sweep, which decision function it feeds)
THRESHOLDS = {
    "detect_k_sigma":              (2.00, [1.0, 1.5, 2.0, 2.5, 3.0], "detect"),
    "detect_min_ratio":            (0.50, [0.3, 0.4, 0.5, 0.6, 0.7], "detect"),
    "detect_reward_drop_ratio":    (0.15, [0.10, 0.125, 0.15, 0.175, 0.20], "detect"),
    "reward_only_drop_ratio":      (0.30, [0.20, 0.25, 0.30, 0.35, 0.40], "reward_only"),
    "select_severe_reward_ratio":  (0.50, [0.35, 0.425, 0.50, 0.575, 0.65], "select"),
    "select_sharp_value_sens_ratio": (0.50, [0.35, 0.425, 0.50, 0.575, 0.65], "select"),
    "accept_reward_recovery":      (0.30, [0.20, 0.25, 0.30, 0.35, 0.40], "accept"),
    "accept_comm_recovery":        (0.30, [0.20, 0.25, 0.30, 0.35, 0.40], "accept"),
}


def _load_rows(paths):
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            except FileNotFoundError:
                print("WARNING: no such file: {}".format(path), file=sys.stderr)
    return rows


def _decision_detect(row, value, threshold_name):
    baseline, degraded = row.get("baseline"), row.get("degraded")
    if not baseline or not degraded:
        return None
    kwargs = dict(k_sigma=2.0, min_ratio=0.5, reward_drop_ratio_threshold=0.15)
    kwargs[{"detect_k_sigma": "k_sigma", "detect_min_ratio": "min_ratio",
            "detect_reward_drop_ratio": "reward_drop_ratio_threshold"}[threshold_name]] = value
    degraded_flag, _ = detect_degradation(baseline, degraded, **kwargs)
    return degraded_flag


def _decision_reward_only(row, value, threshold_name):
    baseline, degraded = row.get("baseline"), row.get("degraded")
    if not baseline or not degraded:
        return None
    degraded_flag, _ = detect_degradation_reward_only(baseline, degraded, value)
    return degraded_flag


def _decision_select(row, value, threshold_name):
    baseline, degraded = row.get("baseline"), row.get("degraded")
    if not baseline or not degraded:
        return None
    kwargs = dict(severe_reward_ratio=0.50, sharp_value_sens_ratio=0.50)
    kwargs[{"select_severe_reward_ratio": "severe_reward_ratio",
            "select_sharp_value_sens_ratio": "sharp_value_sens_ratio"}[threshold_name]] = value
    target, _ = select_repair_target(baseline, degraded, order=('embedding', 'comm', 'lora', 'full'), **kwargs)
    return target


def _decision_accept(row, value, threshold_name):
    baseline = row.get("baseline")
    fs = row.get("final_status") or {}
    log = fs.get("escalation_log") or []
    if not baseline or not log:
        return None
    last = log[-1]
    degraded, repaired = row.get("degraded"), row.get("repaired")
    if not degraded or not repaired:
        return None
    kwargs = dict(reward_thresh=0.30, comm_thresh=0.30)
    kwargs[{"accept_reward_recovery": "reward_thresh",
            "accept_comm_recovery": "comm_thresh"}[threshold_name]] = value
    accepted, _ = accept_repair(baseline, degraded, repaired, **kwargs)
    return accepted


_DECISION_FN = {
    "detect": _decision_detect,
    "reward_only": _decision_reward_only,
    "select": _decision_select,
    "accept": _decision_accept,
}


def analyze(rows, threshold_name, default, grid):
    default_source_fn = _DECISION_FN[THRESHOLDS[threshold_name][2]]
    results = []
    for row in rows:
        agent = row.get("model_dir", "?")
        default_decision = default_source_fn(row, default, threshold_name)
        if default_decision is None:
            continue
        decisions = [default_source_fn(row, v, threshold_name) for v in grid]
        n_diff = sum(1 for d in decisions if d != default_decision)
        # Distance (in grid steps) from the default to the nearest value that flips it, if any.
        default_idx = grid.index(default) if default in grid else None
        flip_distance = None
        if default_idx is not None:
            for step in range(1, len(grid)):
                lo, hi = default_idx - step, default_idx + step
                candidates = [i for i in (lo, hi) if 0 <= i < len(grid)]
                if any(decisions[i] != default_decision for i in candidates):
                    flip_distance = step
                    break
        results.append({
            "agent": agent, "threshold": threshold_name, "default_value": default,
            "default_decision": default_decision, "n_grid_points_that_flip": n_diff,
            "n_grid_points_total": len(grid), "flip_distance_in_steps": flip_distance,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_log", nargs="+", help="--results_log JSONL path(s)/glob(s).")
    ap.add_argument("--csv_out", type=str, default=None)
    ap.add_argument("--flag_within_steps", type=int, default=1,
                     help="flag a row as fragile if a decision flips within this many grid "
                          "steps of the shipped default (default 1 -- one grid step away).")
    args = ap.parse_args()

    rows = _load_rows(args.results_log)
    print("Loaded {} row(s).".format(len(rows)))
    if not rows:
        return

    all_results = []
    for name, (default, grid, _kind) in THRESHOLDS.items():
        all_results.extend(analyze(rows, name, default, grid))

    print("\n{:<40} {:<20} {:>8} {:>10} {:>10}".format(
        "agent", "threshold", "default", "%flip", "flip@steps"))
    print("-" * 92)
    fragile = []
    for r in all_results:
        pct = 100.0 * r["n_grid_points_that_flip"] / max(1, r["n_grid_points_total"])
        flag = ""
        if r["flip_distance_in_steps"] is not None and r["flip_distance_in_steps"] <= args.flag_within_steps:
            flag = "  <-- FRAGILE"
            fragile.append(r)
        print("{:<40} {:<20} {:>8} {:>9.0f}% {:>10}{}".format(
            r["agent"][-40:], r["threshold"], r["default_value"], pct,
            r["flip_distance_in_steps"] if r["flip_distance_in_steps"] is not None else "-", flag))

    print("\n{} of {} (agent, threshold) pairs are FRAGILE (decision flips within "
          "{} grid step(s) of the shipped default).".format(
              len(fragile), len(all_results), args.flag_within_steps))
    if fragile:
        print("These are the cutoffs to caveat explicitly in the paper -- e.g. TEAM_HANDOFF's "
              "own example: a 47% drop sitting 3 points from the 50% escalate-to-full cutoff.")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()) if all_results else [])
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print("\nWrote {}".format(args.csv_out))


if __name__ == "__main__":
    main()
