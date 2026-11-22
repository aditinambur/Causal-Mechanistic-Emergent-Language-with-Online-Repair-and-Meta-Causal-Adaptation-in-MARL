# E8 IPPO Portability Summary (`3a4l`, `--use_centralized_V False`, 500k steps)

## Aggregate Across Seeds

- **Seeds evaluated:** 5
- **Triggered:** 3/5
- **Accepted:** 3/5
- **Held-Out Confirmed:** 3/3
- **Baseline Return:** -9070.4 ± 1786.9
- **Baseline Comm Effect:** 603.6 ± 693.2
- **Local Reward Recovery (Accepted):** 140.7% ± 16.9%
- **Local Comm Recovery (Accepted):** 73.9% ± 6.2%
- **Held-Out Reward Recovery (Accepted):** 157.9% ± 42.1%
- **Clean Retention % (Accepted):** 122.4% ± 15.0%

## Per-Seed Breakdown

| Seed | Checkpoint | Base Return | Base Comm | Drop Ratio | Triggered | Accepted (Target) | Local Rew % | Local Comm % | Held-Out Rew % | Clean Retention % | Held-Out |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `checkpoint_102400` | -5471.52 | 1374.67 | 1.0735 | True | True (lora) | 115.57 | 80.11 | 109.9 | 93.23 | True |
| 2 | `checkpoint_166400` | -8875.31 | -1402.34 | 1.1405 | True | True (lora) | 133.56 | None | 121.88 | 130.85 | True |
| 3 | `checkpoint_230400` | -5605.5 | 564.03 | 0.1059 | False | False (None) | None | None | None | None | None |
| 4 | `checkpoint_294400` | -10198.08 | 2686.12 | 0.645 | True | True (full) | 172.89 | 67.75 | 241.88 | 143.03 | True |
| 5 | `checkpoint_486400` | -15201.77 | -204.38 | -0.5435 | False | False (None) | None | None | None | None | None |
