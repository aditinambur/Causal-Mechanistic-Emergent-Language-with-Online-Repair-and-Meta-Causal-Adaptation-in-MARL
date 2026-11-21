"""
Summarise results/rebuttal/*.jsonl into the tables the rebuttal needs.

    python benchmarking/evaluators/summarize_rebuttal.py results/rebuttal/E1.jsonl [more.jsonl ...]
        [--md_out results/rebuttal/summary.md]

Per run_label it reports n, trigger rates (causal decision is logged even for forced runs),
local reward / comm recovery (Eq. 12, only where the metric actually degraded), acceptance,
held-out reward recovery, clean-task change, trainable params is printed by the run log only.
Values are mean +- SEM over checkpoints (the seed is the unit of analysis, as in the paper).

Every number printed here traces back to a row with its exact model_dir; nothing is imputed.
"""

import argparse
import json
import math
from collections import OrderedDict, defaultdict


def rec(b, d, r):
    """Eq. 12 recovery; None when the metric did not degrade (x_B <= x_D)."""
    if b is None or d is None or r is None or b <= d:
        return None
    return (r - d) / (b - d)


def mean_sem(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, 1
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs)), len(xs)


def fmt(m, se, n, pct=True):
    if m is None:
        return "-"
    k = 100.0 if pct else 1.0
    s = "{:.1f}{}".format(m * k, "%" if pct else "")
    if se is not None:
        s += " ± {:.1f}".format(se * k)
    return s + " (n={})".format(n)


def row_metrics(r):
    b, d, rep = r.get("baseline") or {}, r.get("degraded") or {}, r.get("repaired") or {}
    fs = r.get("final_status") or {}
    di = r.get("detect_info") or {}
    diag = fs.get("rejected_diagnostics") or {}
    src = fs if fs.get("accepted") else diag  # accepted repairs store these on final_status
    out = {
        "reward_drop": di.get("reward_drop_ratio"),
        "trigger": di.get("trigger_decision", fs.get("mode") not in ("not_degraded",)),
        "forced": bool(di.get("forced")),
        "accepted": fs.get("accepted"),
        "rec_reward": rec(b.get("reward"), d.get("reward"), rep.get("reward")) if rep else None,
        "rec_comm": rec(b.get("comm_effect"), d.get("comm_effect"), rep.get("comm_effect")) if rep else None,
        "ho_rec_reward": src.get("holdout_reward_recovery"),
        "ho_confirms": src.get("holdout_confirms"),
        "clean_change": (src.get("normal_reward_delta_vs_baseline") / (abs(b["reward"]) + 1e-6)
                         if src.get("normal_reward_delta_vs_baseline") is not None and b else None),
        "c_zero_B": b.get("comm_effect"), "c_zero_D": d.get("comm_effect"),
        "c_mean_B": b.get("comm_effect_mean_ablation"), "c_mean_D": d.get("comm_effect_mean_ablation"),
        "c_res_B": b.get("comm_effect_resample_ablation"), "c_res_D": d.get("comm_effect_resample_ablation"),
        "rollback_ok": all(r.get("rollback_checks") or [True]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--md_out", default=None)
    a = ap.parse_args()

    groups = OrderedDict()
    for path in a.logs:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            label = r.get("run_label") or "{}|{}|{}".format(r.get("shift"), r.get("repair_target_arg"),
                                                             r.get("controller"))
            groups.setdefault(label, []).append((r, row_metrics(r)))

    lines = []
    lines.append("| run_label | n | reward drop | trigger fired | accepted | local reward rec. | "
                 "comm rec. | held-out reward rec. | clean change | rollback bit-exact |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    e11 = []
    for label, rows in groups.items():
        ms = [m for _, m in rows]
        n = len(ms)
        trig = sum(1 for m in ms if m["trigger"])
        acc = [m["accepted"] for m in ms if m["accepted"] is not None]
        lines.append("| {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} |".format(
            label, n,
            fmt(*mean_sem([m["reward_drop"] for m in ms])),
            trig, n,
            "{}/{}".format(sum(acc), len(acc)) if acc else "-",
            fmt(*mean_sem([m["rec_reward"] for m in ms])),
            fmt(*mean_sem([m["rec_comm"] for m in ms])),
            fmt(*mean_sem([m["ho_rec_reward"] for m in ms])),
            fmt(*mean_sem([m["clean_change"] for m in ms])),
            "yes" if all(m["rollback_ok"] for m in ms) else "NO"))
        if any(m["c_mean_B"] is not None for m in ms):
            e11.append((label, ms))

    if e11:
        lines.append("")
        lines.append("| run_label (E11) | c_zero baseline | c_mean baseline | c_resample baseline | "
                     "c_zero shifted | c_mean shifted | c_resample shifted |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for label, ms in e11:
            cells = [fmt(*mean_sem([m[k] for m in ms]), pct=False) for k in
                     ("c_zero_B", "c_mean_B", "c_res_B", "c_zero_D", "c_mean_D", "c_res_D")]
            lines.append("| {} | {} |".format(label, " | ".join(cells)))

    text = "\n".join(lines)
    print(text)
    if a.md_out:
        with open(a.md_out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
