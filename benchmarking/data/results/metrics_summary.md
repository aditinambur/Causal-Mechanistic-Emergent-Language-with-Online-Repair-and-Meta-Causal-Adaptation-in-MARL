# Benchmark Experimental Metrics Summary

| Scale | Arm | N_Seeds | Baseline Return | Comm Effect (ΔR) | CIC KL Divergence | Value Sensitivity | Degraded Return | Reward Drop % | Trigger Rate | Repaired Return | Reward Recovery % | Comm Recovery % | Acceptance Rate | Held-Out Confirms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2a3l | causal_adaptive | 5 | -2202.52 ± 87.60 | 488.42 ± 238.49 | 0.05 ± 0.01 | 0.71 ± 0.10 | -3382.50 ± 1237.90 | 49.80 ± 49.61 | 1/5 | -3868.70 ± 0.00 | 75.20 ± 0.00 | N/A | 1/5 | 1/5 |
| 2a3l | naive_reward_only | 5 | -2202.52 ± 87.60 | 488.42 ± 238.49 | 0.05 ± 0.01 | 0.71 ± 0.10 | -3382.50 ± 1237.90 | 49.80 ± 49.61 | 1/5 | -3868.70 ± 0.00 | 75.20 ± 0.00 | N/A | 1/5 | 1/5 |
| 3a4l | causal_adaptive | 5 | -5670.52 ± 408.84 | 925.36 ± 265.50 | 0.10 ± 0.03 | 0.70 ± 0.11 | -6497.20 ± 525.17 | 16.40 ± 11.14 | 2/5 | -5581.45 ± 264.35 | 78.55 ± 1.85 | 291.20 ± 224.80 | 2/5 | 2/5 |
| 3a4l | naive_reward_only | 5 | -5670.52 ± 408.84 | 925.36 ± 265.50 | 0.10 ± 0.03 | 0.70 ± 0.11 | -6497.20 ± 525.17 | 16.40 ± 11.14 | 2/5 | -5581.45 ± 264.35 | 78.55 ± 1.85 | 291.20 ± 224.80 | 2/5 | 2/5 |