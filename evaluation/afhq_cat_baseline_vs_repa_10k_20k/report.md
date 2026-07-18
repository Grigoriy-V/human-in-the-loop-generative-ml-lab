# AFHQ Cats Baseline vs REPA Quick Comparison

Fixed protocol: held-out official AFHQ Cats test split, 200 images, seeds 1000-1199, class 0, Heun-50, CFG 1.0, shared VAE and Inception-v3 features.

| Variant | Step | Weights | FID | KID | Precision | Recall | Failures | Duplicates | img/s | Peak VRAM (GB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_raw_10k | 10000 | raw | 55.644 | 0.02740 | 0.290 | 0.802 | 0 | 0 | 15.03 | 2.39 |
| baseline_raw_20k | 20000 | raw | 48.051 | 0.02052 | 0.340 | 0.754 | 0 | 0 | 16.88 | 2.39 |
| repa_raw_10k | 10000 | raw | 62.305 | 0.03000 | 0.210 | 0.862 | 0 | 0 | 16.96 | 2.39 |
| repa_raw_20k | 20000 | raw | 52.384 | 0.02531 | 0.310 | 0.722 | 0 | 0 | 17.05 | 2.39 |
| repa_ema_20k | 20000 | ema | 144.834 | 0.10749 | 0.030 | 0.564 | 0 | 0 | 16.06 | 2.88 |

## Matched Changes

### baseline_10k_vs_20k
- fid: -7.592698 (-13.65%)
- kid: -0.006878 (-25.10%)
- precision: +0.050000 (+17.24%)
- recall: -0.048000 (-5.99%)

### repa_10k_vs_20k
- fid: -9.921470 (-15.92%)
- kid: -0.004696 (-15.65%)
- precision: +0.100000 (+47.62%)
- recall: -0.140000 (-16.24%)

### baseline_vs_repa_10k
- fid: +6.661504 (+11.97%)
- kid: +0.002605 (+9.51%)
- precision: -0.080000 (-27.59%)
- recall: +0.060000 (+7.48%)

### baseline_vs_repa_20k
- fid: +4.332733 (+9.02%)
- kid: +0.004787 (+23.33%)
- precision: -0.030000 (-8.82%)
- recall: -0.032000 (-4.24%)

## Interpretation

Loss values are intentionally excluded: baseline and REPA optimize different objectives. Read matched checkpoint comparisons together with FID, KID, precision, recall, failures, duplicate counts, and paired fixed-seed grids; do not select a winner from FID alone.

- Baseline dynamics: compare `baseline_10k_vs_20k`.
- REPA dynamics and current raw checkpoint: compare `repa_10k_vs_20k`.
- Equal-budget REPA evidence: compare `baseline_vs_repa_10k` and `baseline_vs_repa_20k` only.
- EMA is diagnostic only: compare `repa_raw_vs_ema_20k`.

All checkpoint SHA-256 values were unchanged after evaluation: `True`. Repeated fixed-seed one-image probes were bitwise identical for every variant: `True`.
Full-1000 evaluation, sampler ablation, CFG sweep, and training were not run.
