#!/usr/bin/env python
"""
TEAM_HANDOFF item 6 -- turn repeated single runs into a statistic.

This is a companion to onpolicy/scripts/aggregate_repair_results.py, not a replacement.
aggregate_repair_results.py answers "are the mean recovery numbers different" (Welch's
t-test on continuous recovery %). This script answers the question: on how many AGENTS did config A beat
config B, and is that record better than chance? That is a sign test (exact binomial test,
p=0.5 null), not a t-test -- see TEAM_HANDOFF.md item 6 / "5 is the floor, p=0.031".

It does NOT run any experiments. Point it at the --results_log JSONL files produced by
onpolicy/scripts/phase2_3_repair.py for two configs you want to compare on the SAME agents
(same --model_dir), and it will:
  1. match rows across the two files by model_dir (an agent must appear in both to count --
     a comparison needs both arms run on it),
  2. score each matched agent as a win for A, a win for B, or a tie, using the success rule
     for the comparison type you pick,
  3. run an exact two-sided binomial (sign) test on the win/loss record,
  4. optionally restrict to the "informative window" for the
     trigger comparison (reward drop strictly between 0.15 and 0.30 -- outside that window
     both a causal and a naive trigger react the same way and the comparison is uninformative
     by construction, not a loss for either side).

Usage
-----
Item 4/C5 claim (communication-specific repair beats a same-sized generic repair):

    python onpolicy/scripts/statistical_analysis.py \\
        --comparison target --log_a comm.jsonl --log_b noncomm.jsonl

Item 5/C6 claim (causal trigger beats reward-only trigger):

    python onpolicy/scripts/statistical_analysis.py \\
        --comparison trigger --log_a causal.jsonl --log_b reward_only.jsonl \\
        --require_informative_window

"comm.jsonl" etc. are whatever paths you passed to phase2_3_repair.py's --results_log for
each arm -- run all 8 agents through each arm with a distinct --results_log path per arm
(or one shared path per arm across all 8 --model_dir runs; rows are matched by model_dir
regardless of how many files they're spread across, so --log_a/--log_b also accept globs).
"""

import argparse
import glob
import json
import math
import sys


def _load_rows(path_or_glob):
    rows = []
    for path in sorted(glob.glob(path_or_glob)) or [path_or_glob]:
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except FileNotFoundError:
            print("WARNING: no such file: {}".format(path), file=sys.stderr)
    return rows


def _latest_by_agent(rows):
    """One row per model_dir (agent). If a file has several rows for the same agent
    (re-runs), keep the most recent by 'timestamp' -- older attempts are superseded."""
    by_agent = {}
    for r in rows:
        agent = r.get("model_dir")
        if agent is None:
            continue
        prev = by_agent.get(agent)
        if prev is None or r.get("timestamp", 0) >= prev.get("timestamp", 0):
            by_agent[agent] = r
    return by_agent


def _succeeded(row):
    """A row counts as a successful repair if it was accepted AND, when a held-out
    re-check ran (TEAM_HANDOFF item 8), that re-check confirmed it. A DOES NOT CONFIRM
    is a failure, per TEAM_HANDOFF.md: 'a DOES NOT CONFIRM means that run's recovery
    number does not get written up as a success.'"""
    fs = row.get("final_status") or {}
    if not fs.get("accepted"):
        return False
    holdout_confirms = fs.get("holdout_confirms")
    if holdout_confirms is False:
        return False
    return True


def _reward_drop_ratio(row):
    di = row.get("detect_info") or {}
    return di.get("reward_drop_ratio")


def _informative(row_a, row_b, lo=0.15, hi=0.30):
    """TEAM_HANDOFF item 5: the trigger comparison only means something when the drop
    lands strictly between 0.15 and 0.30 -- above that both triggers fire, below that
    neither does, and the pair is uninformative either way (not a tie -- excluded)."""
    ratios = [r for r in (_reward_drop_ratio(row_a), _reward_drop_ratio(row_b)) if r is not None]
    if not ratios:
        return False
    return all(lo < r < hi for r in ratios)


def compare(rows_a, rows_b, comparison, require_informative_window):
    by_a = _latest_by_agent(rows_a)
    by_b = _latest_by_agent(rows_b)
    common_agents = sorted(set(by_a) & set(by_b))

    if not common_agents:
        print("ERROR: no model_dir appears in both logs -- nothing to compare. "
              "Make sure both arms were run on the SAME --model_dir values.", file=sys.stderr)
        sys.exit(1)

    records = []
    for agent in common_agents:
        ra, rb = by_a[agent], by_b[agent]

        if comparison == "trigger" and require_informative_window:
            if not _informative(ra, rb):
                records.append((agent, "excluded (outside 0.15-0.30 drop-ratio window)", None))
                continue

        a_ok, b_ok = _succeeded(ra), _succeeded(rb)
        if a_ok and not b_ok:
            outcome = "win_a"
        elif b_ok and not a_ok:
            outcome = "win_b"
        else:
            outcome = "tie"
        records.append((agent, outcome, (a_ok, b_ok)))

    return records


def sign_test(wins_a, wins_b):
    """Exact two-sided binomial (sign) test against p=0.5, matching TEAM_HANDOFF's own
    numbers (5-for-5 -> p=0.031, 6-for-6 -> p=0.016, 3-for-3 -> p=0.125)."""
    n = wins_a + wins_b
    if n == 0:
        return None
    k = max(wins_a, wins_b)
    try:
        from scipy import stats
        # binomtest is exact two-sided; matches TEAM_HANDOFF's cited p-values.
        result = stats.binomtest(k, n, 0.5, alternative="two-sided")
        return result.pvalue
    except ImportError:
        # Fallback: exact two-sided binomial p-value with no scipy dependency.
        def _binom_pmf(k_, n_, p_):
            return math.comb(n_, k_) * (p_ ** k_) * ((1 - p_) ** (n_ - k_))
        p_obs = _binom_pmf(k, n, 0.5)
        p_value = sum(_binom_pmf(i, n, 0.5) for i in range(n + 1) if _binom_pmf(i, n, 0.5) <= p_obs + 1e-12)
        return min(1.0, p_value)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison", choices=["target", "trigger"], required=True,
                     help="'target': item 4, comm-repair (log_a) vs generic noncomm-repair "
                          "(log_b). 'trigger': item 5, causal controller (log_a) vs "
                          "reward_only controller (log_b).")
    ap.add_argument("--log_a", required=True, help="--results_log path/glob for config A.")
    ap.add_argument("--log_b", required=True, help="--results_log path/glob for config B.")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--require_informative_window", action="store_true", default=False,
                     help="for --comparison trigger: drop agents outside the 0.15-0.30 "
                          "reward-drop-ratio band per TEAM_HANDOFF item 5.")
    args = ap.parse_args()

    rows_a = _load_rows(args.log_a)
    rows_b = _load_rows(args.log_b)
    records = compare(rows_a, rows_b, args.comparison, args.require_informative_window)

    print("{:<45} {:<45} {}".format("agent (model_dir)", "outcome", "(A_ok, B_ok)"))
    print("-" * 110)
    wins_a = wins_b = ties = excluded = 0
    for agent, outcome, oks in records:
        print("{:<45} {:<45} {}".format(agent[-45:], outcome, oks))
        if outcome == "win_a":
            wins_a += 1
        elif outcome == "win_b":
            wins_b += 1
        elif outcome == "tie":
            ties += 1
        else:
            excluded += 1

    n_used = wins_a + wins_b
    print("\n=== Summary: {} ({}) vs {} ({}) ===".format(
        args.label_a, args.log_a, args.label_b, args.log_b))
    print("agents compared         : {}".format(len(records)))
    print("excluded (uninformative): {}".format(excluded))
    print("ties                    : {}".format(ties))
    print("{} wins             : {}".format(args.label_a, wins_a))
    print("{} wins             : {}".format(args.label_b, wins_b))

    if n_used == 0:
        print("\nNo decisive (non-tie, non-excluded) agents -- nothing to sign-test yet.")
        return

    p = sign_test(wins_a, wins_b)
    leader = args.label_a if wins_a >= wins_b else args.label_b
    print("\nSign test (exact two-sided binomial, p=0.5 null): "
          "{} of {} to {}, p = {:.4f}{}".format(
              max(wins_a, wins_b), n_used, leader, p,
              "  <-- p < 0.05, statistically significant" if p < 0.05 else
              "  <-- NOT significant at p < 0.05 (need more agents -- TEAM_HANDOFF item 2)"))
    print("(reference points from TEAM_HANDOFF.md: 3-for-3 -> p=0.125, 5-for-5 -> p=0.031, "
          "6-for-6 -> p=0.016)")

    if ties:
        print("\nNote: {} tie(s) excluded from the sign test -- both configs succeeded or "
              "both failed on those agents, which the sign test cannot use as evidence "
              "either way.".format(ties))


if __name__ == "__main__":
    main()
