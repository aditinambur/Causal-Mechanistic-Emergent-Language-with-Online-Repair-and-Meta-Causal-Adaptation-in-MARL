# Items 1, 3–7 — checkpoint selection, repair pipeline, statistics, threshold sensitivity

This covers item 1 (fix the checkpoint picker), items 3–5 (run the pipeline / generic-fix control / naive-trigger control), item 6 (statistics across
runs), item 7 (threshold sensitivity). 

## What's new here

Four files changed/added, all in `onpolicy/scripts/`:

| file | item | what it does |
|---|---|---|
| `select_best_checkpoint.py` | 1 | Re-ranks a run's checkpoints from `causal_influence.csv` without the training-time picker's broken 100×-weighted, single-eval, no-warmup-trim scoring. Read-only, no retraining. |
| `phase2_3_repair.py` | (bugfix, all of 3–5) | One line changed: `--env_name` now defaults to `MPE` instead of `StarCraft2`. The script only ever supported MPE (`_make_mpe_vec_env` hardcodes it), so omitting `--env_name` used to crash every rollout worker with `NotImplementedError`. No behavior change beyond that. |
| `statistical_analysis.py` | 6 | Exact sign test (binomial, p=0.5 null) comparing two `--results_log` files agent-by-agent, matched by `--model_dir`. Companion to `aggregate_repair_results.py` (which does the continuous-metric t-test) — this answers "on how many agents did A beat B," which is the number TEAM_HANDOFF's C5/C6 claims are actually framed around. |
| `threshold_sensitivity.py` | 7 | Re-runs the real decision functions (`detect_degradation`, `detect_degradation_reward_only`, `select_repair_target`, `accept_repair` — copied verbatim from `phase2_3_repair.py`, not reimplemented) across a grid around each of the 8 shipped threshold defaults, using fingerprints already sitting in your `--results_log` rows. Purely post-hoc — never retrains. |

## How to run it, start to finish

### 0. Environment setup (once)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install "numpy<2" scipy pandas matplotlib seaborn absl-py tensorboard tensorboardX \
  "wandb<0.17" setproctitle imageio Pillow pygame "protobuf<=3.20.3" gym==0.19.0
pip install -e . --no-deps
export WANDB_MODE=disabled
```
**Mac only:** `gym==0.19.0`'s own `setup.py` has a typo (`opencv-python>=3.`) that breaks on
modern pip. If the `gym==0.19.0` line above fails, download+patch it manually:
```bash
cd /tmp && curl -sL -o gym-0.19.0.tar.gz https://files.pythonhosted.org/packages/source/g/gym/gym-0.19.0.tar.gz
tar xzf gym-0.19.0.tar.gz && cd gym-0.19.0
sed -i.bak 's/opencv-python>=3\./opencv-python>=3.0/' setup.py
pip install --no-build-isolation .
```
This is an environment quirk, not a repo bug — nothing in the codebase needed to change for it,
and it likely won't happen on Linux/conda.

### 1. Pick a verified checkpoint (item 1)
```bash
python3 onpolicy/scripts/select_best_checkpoint.py <run_dir>/causal_influence.csv --top_k 4
```
For each ranked step, confirm it's actually usable before trusting it:
```bash
python3 onpolicy/scripts/phase2_3_repair.py --model_dir <run_dir>/models/checkpoint_<step> --no_repair
```
Read `comm_effect` off the printed `[BASELINE]` line. Keep the first candidate with
`comm_effect >= ~150`. **Do not use `checkpoint_best`** — that's the one item 1 exists to
route around.

### 2. Run items 3–5 on each verified checkpoint
Use one shared `--results_log` file per arm across *all* agents (not one file per agent) —
`statistical_analysis.py` matches agents by `model_dir` within a file, so everyone should log
into the same 3 files:
```bash
# item 3: real system
python3 onpolicy/scripts/phase2_3_repair.py --model_dir <ckpt> --results_log results/comm.jsonl
# item 4: generic-fix control
python3 onpolicy/scripts/phase2_3_repair.py --model_dir <ckpt> --repair_target noncomm --results_log results/noncomm.jsonl
# item 5: naive-trigger control
python3 onpolicy/scripts/phase2_3_repair.py --model_dir <ckpt> --controller reward_only --results_log results/reward_only.jsonl
```
Read the `[ACCEPTANCE]` / `[FINAL STATUS]` block at the end of each run

### 3. Statistics (item 6)
```bash
python3 onpolicy/scripts/statistical_analysis.py --comparison target --log_a results/comm.jsonl --log_b results/noncomm.jsonl
python3 onpolicy/scripts/statistical_analysis.py --comparison trigger --log_a results/comm.jsonl --log_b results/reward_only.jsonl --require_informative_window
```
Reads as: how many agents did comm-target beat generic-fix on (target), and how many did the
causal trigger catch that reward-only missed (trigger, restricted to the 0.15–0.30
reward-drop-ratio band where the comparison is actually informative).

### 4. Threshold sensitivity (item 7)
```bash
python3 onpolicy/scripts/threshold_sensitivity.py results/comm.jsonl results/noncomm.jsonl results/reward_only.jsonl --csv_out results/sensitivity.csv
```
**Pass the 3 files explicitly, not a glob (`results/*.jsonl`)** — the `results/` folder also
has other logs from earlier exploratory runs with a different naming scheme (`*_forced`,
`*_sweep`, etc.); a glob will silently mix them into your fragility counts.

## Known limitation: the fallback scorer disagrees with the manual pick on older runs

`select_best_checkpoint.py` was validated against 2 checkpoints already trusted as good
(picked by the original manual `--no_repair` method): `seed3/checkpoint_2982400` and
`seed1/checkpoint_1958400`.

```bash
python3 onpolicy/scripts/select_best_checkpoint.py <seed3 run>/causal_influence.csv --known_good_step 2982400   # PASS (#1)
python3 onpolicy/scripts/select_best_checkpoint.py <seed1 run>/causal_influence.csv --known_good_step 1958400   # FAIL (#14)
```

Result: **1 of 2 PASS** (ranked #1), the other ranked **#14**. Both of these CSVs predate
`comm_effect`/`eval_reward` being logged per-checkpoint (see the script's own docstring), so
both validations ran on the weaker KL/value-sensitivity fallback scorer, not the real
`comm_effect`-based ranking the tool is designed around — the fallback is a rough shortlist
generator, not a trusted ranking, until runs start logging `comm_effect` natively (which new
runs should, per `mpe_runner.py`'s `eval()`).

**Practical implication:** on any CSV lacking `comm_effect` (the script prints a `WARNING` when
this is the case), treat the ranked list as a shortlist of candidates to manually confirm with
`--no_repair`, not as a final answer — check more than just the #1 pick, the way item 1's
original manual method already recommends. On CSVs that do have `comm_effect` logged (any run
going forward), this limitation shouldn't apply, but it hasn't been validated on one yet since
none of the existing runs have that column — worth re-running this same `--known_good_step`
check the first time a `comm_effect`-logged CSV exists.

## How to verify this is working (for whoever reviews the PR)

1. Run steps 1–4 on any checkpoint under `onpolicy/scripts/results/MPE/simple_spread/mappo/`.
2. Confirm `select_best_checkpoint.py --known_good_step <a step you already trust>` prints
   `PASS` or a sensible `FAIL` (it's allowed to be honest about disagreeing with the old manual
   pick — that's not itself a bug, just log it).
3. Confirm the causal (`comm`) arm's `[ACCEPTANCE]` numbers beat or tie the `noncomm` and
   `reward_only` arms on the same checkpoint. A clean loss for comm on a real (non-degenerate)
   agent — flag it here if you see one.
4. Confirm `statistical_analysis.py`'s printed win/loss table matches what you'd conclude by
   reading the raw `[ACCEPTANCE]` blocks yourself. It should always agree — if it doesn't,
   that's a bug in the script, not a real finding.
5. Confirm `threshold_sensitivity.py` runs without needing torch (it embeds copies of the 4
   decision functions specifically so a torch-less analysis environment can still use it —
   see the big comment at the top of the file for why, and the correctness contract that comes
   with duplicating them).

## When item 2 lands (more agents)

No code changes needed. For each new checkpoint:
1. Run step 1 (`select_best_checkpoint.py` + `--no_repair` confirm) on it.
2. Run step 2's three commands, appending to the *same* `results/comm.jsonl` /
   `results/noncomm.jsonl` / `results/reward_only.jsonl` files.
3. Rerun step 3 and step 4 on the now-larger log files — `statistical_analysis.py` and
   `threshold_sensitivity.py` both just process whatever rows are in the files, so the sign
   test's `n` (and hopefully its p-value) grows automatically.

Reference points from TEAM_HANDOFF.md for the sign test: 3-for-3 → p=0.125 (not significant),
5-for-5 → p=0.031, 6-for-6 → p=0.016. You need somewhere around 5+ agents winning cleanly
before the paper can claim significance on the target/trigger comparisons.
