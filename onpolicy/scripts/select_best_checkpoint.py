#!/usr/bin/env python
"""
TEAM_HANDOFF item 1 -- replacement for the broken checkpoint_best picker.

The training-time picker (onpolicy/runner/shared/mpe_runner.py) scores every eval as
    score = reward + max(0, comm_effect) + 100 * value_sensitivity
and keeps a running max. Two failure modes this caused (see TEAM_HANDOFF.md item 1):
  - one early, noisy eval freakishly wins the 100x-weighted value_sensitivity term and is
    never beaten again, even though 98% of training is still to come;
  - a checkpoint with comm_effect=+139 out-scored one with comm_effect=+535 because reward
    and value_sensitivity happened to compensate.

This script does NOT change training. It is an offline, read-only re-ranking of the
checkpoints a finished run already produced, using only its causal_influence.csv. It fixes
the three things TEAM_HANDOFF diagnosed:
  1. no 100x weight on value_sensitivity (comm_effect, when logged, is used directly instead
     of a weighted blend -- it is literally the quantity item 2/3's usefulness screen and the
     repair experiments care about);
  2. a rolling-window average instead of a single noisy eval, so one freak reading can't win
     outright;
  3. the first --warmup_frac of training is ignored outright (nothing that early is a serious
     candidate).

Usage
-----
Rank checkpoints for one run and print the top candidates to hand-verify with --no_repair:

    python onpolicy/scripts/select_best_checkpoint.py \\
        onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed3/run1/causal_influence.csv \\
        --top_k 4

Validate the new rule against a run where you already know (by the item-1 manual method, or
because it produced one of the 3 current agents) which checkpoint is the right one:

    python onpolicy/scripts/select_best_checkpoint.py <csv> --known_good_step 2918400

This prints PASS/FAIL: did the new rule's #1 pick match the known-good step?

Notes on data availability
---------------------------
Older causal_influence.csv files (the 3 checkpoints already in this repo) were written before
eval_reward/comm_effect were added to the per-eval CSV row and only have the KL and
value_sensitivity columns. When comm_effect/eval_reward are missing, this script falls back
to a value_sensitivity + KL composite (still unweighted, still windowed, still warmup-trimmed)
and prints a warning -- do not silently trust that fallback the way the old 100x formula was
trusted; always confirm the top pick with --no_repair (see item 2 of TEAM_HANDOFF.md) before
training on top of it or pointing a repair run at it. Runs launched after this change should
have comm_effect in the CSV because mpe_runner.py's eval() already writes it when the
eval_comm_effect_vs_no_message/eval_comm_effect_vs_noisy env info exists.
"""

import argparse
import csv
import sys


def _load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                if k is None:
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            rows.append(row)
    if not rows:
        raise SystemExit("no rows found in {}".format(csv_path))
    return rows


def _windowed(values, idx, half_window):
    lo = max(0, idx - half_window)
    hi = min(len(values), idx + half_window + 1)
    window = values[lo:hi]
    return sum(window) / len(window)


def score_checkpoints(rows, warmup_frac=0.20, half_window=1):
    """
    Returns rows sorted best-first, each annotated with:
      - 'score': the windowed metric used to rank
      - 'score_source': 'comm_effect' or 'kl_value_sensitivity_fallback'
      - 'in_warmup': True if this step falls inside the trimmed warmup region (kept in the
        output for transparency, but never ranked first)
    """
    steps = [r["total_num_steps"] for r in rows]
    max_step = max(steps)
    warmup_cutoff = warmup_frac * max_step

    has_comm_effect = all("comm_effect" in r for r in rows)
    if has_comm_effect:
        raw = [r["comm_effect"] for r in rows]
        source = "comm_effect"
    else:
        # Fallback: unweighted, still meaningful without the training-time 100x term.
        # value_sensitivity is O(1), kl_mean is O(0.01-1) in the sample data -- normalize kl by
        # its own run-max so it can't be silently swamped or silently dominate.
        kl = [r.get("causal_influence_kl_mean", 0.0) for r in rows]
        vs = [r.get("causal_influence_value_sensitivity_mean", 0.0) for r in rows]
        kl_max = max(kl) or 1.0
        raw = [(k / kl_max) + v for k, v in zip(kl, vs)]
        source = "kl_value_sensitivity_fallback"

    windowed = [_windowed(raw, i, half_window) for i in range(len(raw))]

    out = []
    for r, w, in_warmup in zip(rows, windowed, [s < warmup_cutoff for s in steps]):
        rr = dict(r)
        rr["score"] = w
        rr["score_source"] = source
        rr["in_warmup"] = in_warmup
        out.append(rr)

    ranked = sorted(
        (r for r in out if not r["in_warmup"]),
        key=lambda r: r["score"], reverse=True,
    )
    if not ranked:
        raise SystemExit(
            "every checkpoint fell inside the warmup window (--warmup_frac {:.2f}) -- "
            "lower --warmup_frac or check the CSV covers a full run.".format(warmup_frac))
    return ranked, source


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("causal_influence_csv")
    ap.add_argument("--top_k", type=int, default=4,
                     help="how many ranked candidates to print (item 1's manual method "
                          "budgets ~4 min/candidate with --no_repair, ~16 min/agent for 4).")
    ap.add_argument("--warmup_frac", type=float, default=0.20,
                     help="ignore this fraction of total training steps outright.")
    ap.add_argument("--half_window", type=int, default=1,
                     help="rolling-window half-width in EVAL POINTS (not steps): 1 means "
                          "average each candidate with its immediate neighbour on each side.")
    ap.add_argument("--known_good_step", type=int, default=None,
                     help="if given, checks whether the #1 ranked step matches this "
                          "already-known-good checkpoint step and prints PASS/FAIL. Use this "
                          "to validate the rule against the 3 existing hand-picked agents "
                          "before trusting it on new runs (see TEAM_HANDOFF.md item 1).")
    args = ap.parse_args()

    rows = _load_rows(args.causal_influence_csv)
    ranked, source = score_checkpoints(rows, args.warmup_frac, args.half_window)

    print("Scored {} eval points from {} (warmup < {:.0%} of training excluded; "
          "score source: {})".format(len(rows), args.causal_influence_csv, args.warmup_frac, source))
    if source != "comm_effect":
        print("WARNING: comm_effect not found in this CSV -- ranking with the "
              "KL/value_sensitivity fallback. Confirm the top pick(s) with --no_repair "
              "before trusting them; do not point --model_dir at one unverified.")
    print()
    print("{:>12} {:>12} {:>10}".format("step", "score", "source"))
    for r in ranked[: args.top_k]:
        print("{:>12.0f} {:>12.4f} {:>10}".format(r["total_num_steps"], r["score"], r["score_source"]))

    print("\nNext step (per TEAM_HANDOFF.md item 1's manual method): for each step above, run")
    print("  --model_dir <run>/models/checkpoint_<step> --no_repair")
    print("and read comm_effect + reward_drop off the printed fingerprint. Keep the first one")
    print("that clears comm_effect >= ~+150 (see item 2's usability screen); that is your")
    print("--model_dir for the item 3-5 repair runs. Do NOT point --model_dir at checkpoint_best.")

    if args.known_good_step is not None:
        top_step = ranked[0]["total_num_steps"]
        ok = int(top_step) == int(args.known_good_step)
        print("\nValidation against known-good step {}: {}".format(
            args.known_good_step, "PASS" if ok else "FAIL"))
        if not ok:
            rank_of_known = next(
                (i + 1 for i, r in enumerate(ranked) if int(r["total_num_steps"]) == int(args.known_good_step)),
                None)
            print("  known-good step ranked #{} instead of #1 -- consider a wider "
                  "--half_window or different --warmup_frac before trusting this rule "
                  "elsewhere.".format(rank_of_known if rank_of_known else "not found in ranking"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
