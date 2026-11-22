# T1 Summary: `3a4l` Seeds 6–10 & Exact Two-Sided Permutation Test

## Cross-Scale Communication Benefit (`2a3l` vs `3a4l`, Exact Two-Sided Permutation Test)

- **`2a3l` (N=5):** 488.42 ± 238.49 (values: [1410.3, 394.1, 395.5, 109.0, 133.2])
- **`3a4l` Seeds 1–5 (N=5):** 925.36 ± 265.50
- **`3a4l` All Seeds (N=6):** 2064.93 ± 1160.01 (values: [1268.3, -73.4, 948.6, 1458.5, 1024.8, 7762.8])
- **Exact two-sided permutation test (`2a3l` N=5 vs `3a4l` N=5):** p = 0.24603 (62/252, diff = +436.94)
- **Exact two-sided permutation test (`2a3l` N=5 vs `3a4l` N=6):** p = 0.19913 (92/462, diff = +1576.51)

## `3a4l` Aggregate Across All 6 Seeds (`causal_adaptive` vs `naive_reward_only`)

- **Baseline Return:** -6140.45 ± 576.43
- **Baseline Comm Gain:** 2064.93 ± 1160.01
- **Baseline CIC KL:** 0.110 ± 0.027
- **Causal Trigger Rate:** 2/6 | **Naive Trigger Rate:** 2/6
- **Causal Accepted:** 2/6 | **Held-Out Confirmed:** 2/2
- **Causal Reward Recovery % (Accepted):** 78.55 ± 1.85
- **Causal Comm Recovery % (Accepted):** 291.20 ± 224.80
- **Causal Held-Out Reward Recovery % (Accepted):** 66.20 ± 2.00

## Seeds 6–10 Per-Seed Results (`causal_adaptive`)

| Seed | Selected Step | Base Return | Base Comm | Drop Ratio | Triggered | Decision | Local Rew % | Local Comm % | Held-Out Rew % | Retention Loss | Held-Out |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 326400 | -8490.1 | 7762.8 | -0.22 | False | NO_REPAIR_TRIGGERED (none) | None | None | None | None | N/A |

## Top-4 `select_best_checkpoint.py` Screening Log (`--no_repair`)

- **Seed 6:** [{"rank": 1, "step": 102400, "score": 2076.12, "score_source": "comm_effect", "checkpoint_path": "onpolicy/scripts/results/MPE/simple_spread/mappo/benchmark_3a4l_seed6/run1/models/checkpoint_102400", "baseline_reward": -7914.9, "baseline_comm_effect": -1803.5, "reward_drop_ratio": 0.34}, {"rank": 2, "step": 326400, "score": 1040.66, "score_source": "comm_effect", "checkpoint_path": "onpolicy/scripts/results/MPE/simple_spread/mappo/benchmark_3a4l_seed6/run1/models/checkpoint_326400", "baseline_reward": -8490.1, "baseline_comm_effect": 7762.8, "reward_drop_ratio": -0.22}]
