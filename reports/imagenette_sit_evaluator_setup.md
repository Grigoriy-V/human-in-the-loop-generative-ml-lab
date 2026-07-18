# Imagenette SiT Model Evaluator Setup

Date: 2026-07-18

Git base before this evaluator work: `d1dc1bf`.

## Scope

This setup adds a reproducible evaluator for Imagenette SiT checkpoints. It does not continue the non-REPA baseline beyond 100k, does not modify any source checkpoint, and does not use DINO as the primary quality criterion. DINO is deliberately reserved for future supplemental nearest-neighbour and diversity analysis because REPA directly optimizes a DINO-alignment objective.

## Checkpoint Inventory

| Run | Path | global_step | SHA-256 |
| --- | --- | ---: | --- |
| baseline | `outputs/imagenette_sit_s_128/checkpoints/baseline_0100000.pt` | 100000 | `21F79D6AF2C3ACBFB50F8B2739ACB9BA27C4A77265EF35432B383318F7A87197` |
| baseline | `outputs/imagenette_sit_s_128/checkpoints/step_0150000.pt` | 150000 | `7481227DE527755394CAD0AD407C646FE5F0BD1662BAC8EEBA16F47940BDC346` |
| baseline | `outputs/imagenette_sit_s_128/checkpoints/step_0200000.pt` | 200000 | `2AD9FE30DFF4ECD2BAD0DDF2FA29EDD54C150A86718FE1392BF5134288E6A394` |
| REPA | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0100000.pt` | 100000 | `42C06F22CC99E83640FA205A1388C204F8A4089451EF16F9EA1F52865D2423A5` |
| REPA | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0150000.pt` | 150000 | `78F0E4141777B0A0048BDB5F12DF84E28829E152860A9C3F59DF4CA35B405BDC` |
| REPA | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0200000.pt` | 200000 | `B3F0655D22951A98784EA60DF0242490361F6E65BBAEBAC8D4F11F822A039186` |
| REPA | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0250000.pt` | 250000 | `E1512EF7BB0120BAFF55556156E5DB5297D0FEF43299C5BF6CF401250BC77F41` |
| REPA | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0300000.pt` | 300000 | `BE7EAEEFFA24DC702EE44000F1705731B874B4FC248CC0396E3320158127D180` |
| REPA | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0350000.pt` | 350000 | `3A7E08868C9E7CE49B56DD98B30D535C44804452E18BCAC9BE69E62947AB8B9A` |
| REPA immutable snapshot | `outputs/imagenette_sit_s_128_repa/checkpoints/step_0365000.pt` | 365000 | `1E1CB3E8422CED7E6BB6051BCFF91B972D2CD7370C9FBDCF267CB7A2E7199E77` |

The active REPA process was stopped by request. `latest.pt` was at step 365000 and was copied byte-for-byte to `step_0365000.pt` before further evaluation. The source checkpoint was not changed.

## Pipeline Audit

No mathematical defect was found, so the training or sampler objective was not rewritten.

- `mini_diffusion/vae.py`: cached latents are `VAE.encode(...).sample() * scaling_factor`; decode divides by the same `scaling_factor` before VAE decode.
- `mini_diffusion/sit/interpolant.py`: `x_t = (1 - t) * x0 + t * noise`, with velocity target `noise - x0`.
- `mini_diffusion/sit/sampling.py`: samples start at `t=1` and integrate to `t=0`; `dt` is negative. Euler uses `x + dt*v`; Heun uses the average of the current and predicted next velocity with the same negative step.
- Training and sampling share the same linear-interpolant velocity definition.
- Checkpoint loading restores raw model weights, EMA shadow weights, optimizer and RNG. Evaluation selects raw or EMA weights explicitly.
- The evaluator creates an immutable `(class_id, seed)` protocol and precomputes per-sample initial noise. All checkpoints use all ten Imagenette classes and the same seeds.
- CFG is `unconditional + scale * (conditional - unconditional)` using the model null class.

## Evaluator

`mini_diffusion/evaluate_sit.py` accepts multiple checkpoints, raw/EMA weights, Euler/Heun, sampler steps, CFG, samples per class, seed range, reference split, quick/full modes, and an optional shared reference cache.

- `quick`: 20 samples per class, 200 total, Heun-25.
- `full`: 100 samples per class, 1000 total, Heun-50.
- Reference Inception-v3 features are cached with dataset fingerprint, split, class mapping, preprocessing, extractor revision, sample count and feature dimension.
- Metrics: Inception KID and FID (with sample counts), ImageNet ResNet-50 target accuracy/confidence/confusion matrix, feature-manifold precision/recall, and pixel diagnostics. Generated grids, nearest-real and outlier pairs, JSON, and CSV are saved below `evaluation/` and intentionally ignored by Git.

## Quick Evaluation

Protocol: EMA, Heun-25, CFG 1.5, seed start 1000, 20 samples for each of all ten classes, Imagenette validation reference.

| Checkpoint | KID | FID | ResNet target accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline 100k | 0.06125 | 211.10 | 27.0% | 22.0% | 59.5% |
| REPA 100k | 0.06623 | 215.85 | 27.0% | 28.0% | 56.0% |
| REPA 150k | 0.07475 | 224.99 | 17.0% | 20.0% | 57.4% |
| REPA 200k | 0.06890 | 220.19 | 22.0% | 19.5% | 70.5% |
| REPA 350k | 0.05528 | 194.02 | 29.0% | 24.0% | 87.8% |

Outputs: `evaluation/imagenette_sit_quick/`. With this small protocol, REPA 350k is the preliminary best REPA checkpoint. At the time of this initial REPA evaluation, a baseline 150k checkpoint did not yet exist; the subsequent baseline continuation is recorded below.

## Baseline SiT Continuation: 100k To 200k

The non-REPA baseline was resumed from immutable `baseline_0100000.pt` and completed at 200k. Immutable milestones `step_0150000.pt` and `step_0200000.pt` were written; model and EMA tensors at 200k are finite and the checkpoint contains no REPA projector or REPA metadata.

Baseline-only test command:

```powershell
.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q -k "not repa"
```

Result: `27 passed, 6 deselected`. During this run, `test_reference_cache_key_changes_with_files` exposed a Windows stale-cache risk: a same-size file rewrite could retain the same timestamp. `reference_cache_key` now includes each file content SHA-256, so reference Inception features are rebuilt when dataset contents change.

Fixed protocol: all ten Imagenette classes, 20 samples per class (200 total), seed start 1000, validation reference, CFG 1.5, Heun-25. This is a quick diagnostic only; the sample count is too small for a final generative-model ranking.

| Weights | Step | KID | FID | ResNet target accuracy | Precision | Recall | Failed images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMA | 100k | 0.06125 | 211.10 | 27.0% | 22.0% | 59.5% | 0 |
| EMA | 200k | 0.07429 | 221.43 | 18.5% | 22.5% | 64.3% | 0 |
| raw | 100k | 0.06537 | 217.81 | 20.5% | 23.5% | 55.7% | 0 |
| raw | 150k | 0.07335 | 227.04 | 22.5% | 20.0% | 48.9% | 0 |
| raw | 200k | 0.07890 | 226.77 | 16.0% | 13.0% | 71.5% | 0 |

The fixed grids show no obvious qualitative gain at 200k. KID/FID and target accuracy also do not improve over the 100k baseline. This is observed for both raw and EMA weights, so EMA lag alone does not explain the result. Keep `baseline_0100000.pt` as the current best quick-protocol baseline; do not use this result alone to change the objective or sampler.

Outputs: `evaluation/imagenette_sit_baseline_ema_heun25/` and `evaluation/imagenette_sit_baseline_raw_heun25/`.

## Equal-Step Baseline Vs REPA: 150k

Protocol: EMA weights, Heun-50, CFG 1.5, 100 fixed samples for each of ten Imagenette classes (1,000 total), seed start 1000, validation reference. The same class/seed/noise protocol, sampler, VAE, latent cache, and reference features were used for both checkpoints. No checkpoints were modified.

| Run | Checkpoint | KID | FID | ResNet target accuracy | Target confidence | Precision | Recall | Sampling time | Failed images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `step_0150000.pt` | 0.07269 | 165.94 | 20.1% | 6.65% | 20.9% | 48.1% | 66.87 s | 0 |
| REPA | `step_0150000.pt` | 0.07409 | 165.28 | 20.0% | 6.48% | 22.9% | 62.3% | 66.44 s | 0 |

Interpretation: FID differs by only `0.66` and target accuracy is effectively identical. KID slightly favors baseline, but REPA has materially higher feature-manifold precision (`+2.0` percentage points) and recall (`+14.2` points). The results are therefore not close on diversity/coverage, so the conditional 200k comparison was intentionally not run. This is evidence that REPA improves coverage at the matched 150k point, not proof of a broad quality improvement: the metrics remain mixed and the samples are still limited to the shared 1,000-image protocol.

Outputs: `evaluation/baseline150_vs_repa150_full_ema_heun50/`.

## Full Equal-Step Curve: Baseline Vs REPA

All points below use the same canonical full protocol: EMA, Heun-50, CFG 1.5, 100 fixed samples for each of ten classes (1,000 total), seed start 1000, Imagenette validation reference, and the shared `evaluation/reference_cache/` reference features.

| Step | Run | KID | FID | ResNet target accuracy | Precision | Recall | Failed images |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | baseline | 0.06103 | 149.95 | 26.9% | 24.6% | 51.0% | 0 |
| 100k | REPA | 0.06462 | 152.73 | 22.3% | 26.4% | 49.5% | 0 |
| 150k | baseline | 0.07269 | 165.94 | 20.1% | 20.9% | 48.1% | 0 |
| 150k | REPA | 0.07409 | 165.28 | 20.0% | 22.9% | 62.3% | 0 |
| 200k | baseline | 0.07634 | 171.78 | 17.3% | 20.9% | 63.5% | 0 |
| 200k | REPA | 0.07204 | 162.10 | 21.7% | 21.8% | 71.3% | 0 |

Curve interpretation: at 100k, baseline has better FID, KID, and class accuracy. At 150k, FID and accuracy are effectively tied while REPA increases coverage. At 200k, REPA is better on every reported metric except that neither run has high absolute quality. Across this matched curve, the non-REPA baseline degrades after 100k, while REPA becomes favorable relative to baseline by 150k and clearly favorable by 200k. This supports continued REPA investigation; it does not establish that REPA 200k is better than baseline 100k in absolute terms.

Outputs: `evaluation/baseline100_vs_repa100_full_ema_heun50/`, `evaluation/baseline150_vs_repa150_full_ema_heun50/`, and `evaluation/baseline200_vs_repa200_full_ema_heun50/`.

## Full Cross-Step Comparison: Baseline 100k Vs REPA 350k

Protocol: the same full EMA Heun-50 evaluation, CFG 1.5, 1,000 fixed class-balanced samples, seed start 1000, and validation reference.

| Run | Step | KID | FID | ResNet target accuracy | Precision | Recall | Failed images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100k | 0.06103 | 149.95 | 26.9% | 24.6% | 51.0% | 0 |
| REPA | 350k | 0.05199 | 130.96 | 29.5% | 23.9% | 72.1% | 0 |

REPA 350k improves FID by `18.99`, KID by `0.00903`, target accuracy by `2.6` percentage points, and recall by `21.1` points. Precision is `0.7` points lower. This is the strongest evaluated REPA checkpoint so far and it exceeds the best 100k baseline on the main full-protocol distribution and class metrics. It remains a cross-step comparison, not a controlled 350k baseline-vs-REPA ablation.

Outputs: `evaluation/baseline100_vs_repa350_full_ema_heun50/`.

## Experiment Closure

Final side-by-side grid: [baseline 100k EMA vs REPA 350k EMA](../docs/assets/imagenette_sit_baseline100_vs_repa350_ema_heun50.png). It was assembled from the saved full-protocol grids without a new sampling run; positions on the left and right share the same class labels, seeds, and initial noise.

- Best baseline: `baseline_0100000.pt`, EMA, 100k steps.
- Best final model: `step_0350000.pt`, REPA EMA, 350k steps.
- The full equal-step curve shows no REPA acceleration at 100k. REPA becomes favorable relative to the same-step baseline at 150k and clearly at 200k.
- The full baseline-100k versus REPA-350k comparison shows better final distribution metrics, class accuracy, and coverage for REPA, with a small precision trade-off.

Final validation performed without new training or evaluation runs:

```powershell
.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
Get-FileHash outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt,outputs\imagenette_sit_s_128_repa\checkpoints\step_0350000.pt -Algorithm SHA256
```

Result: `33 passed`. Both checkpoint model and EMA states are finite. Their verified SHA-256 values are `21F79D6AF2C3ACBFB50F8B2739ACB9BA27C4A77265EF35432B383318F7A87197` for baseline 100k and `3A7E08868C9E7CE49B56DD98B30D535C44804452E18BCAC9BE69E62947AB8B9A` for REPA 350k.

## Next Roadmap

1. Implement deterministic `img2img`: encode an input with the frozen VAE, add controlled noise at a chosen strength, then denoise with the selected REPA 350k EMA model and class CFG.
2. Implement a `hires fix` path: generate or accept a 128x128 image, upscale it, encode with the same VAE, then perform low-strength img2img refinement. Keep the initial 128x128 model unchanged and record every comparison with fixed seed and strength.

## Raw, EMA, And Sampler Diagnostic

REPA 150k, fixed 200-sample protocol, CFG 1.5:

| Weights / sampler | KID | FID | Accuracy | Precision | Recall | Sampling seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw, Heun-25 | 0.07351 | 224.15 | 15.5% | 18.5% | 69.1% | 10.81 |
| EMA, Heun-25 | 0.07475 | 224.99 | 17.0% | 20.0% | 57.4% | 10.84 |
| EMA, Heun-50 | 0.07402 | 224.27 | 16.5% | 19.5% | 59.8% | 18.40 |

EMA slightly improved class accuracy and precision in this limited run. Heun-50 did not provide a material metric improvement over Heun-25, while sampling took about 70% longer. This does not support attributing the visible smoothing solely to EMA or to the 25-step Heun preview.

## Follow-Up Fixed Evaluation: REPA 200k And 350k

Protocol: the same 200 class-balanced samples and seeds as quick evaluation, CFG 1.5, Heun-50, Imagenette validation reference. Checkpoint 365k was intentionally excluded.

| Step | Weights | KID | FID | ResNet target accuracy | Precision | Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 200k | raw | 0.07737 | 226.06 | 19.0% | 20.0% | 67.3% |
| 200k | EMA | 0.06988 | 221.01 | 21.5% | 19.0% | 68.5% |
| 350k | raw | 0.06133 | 207.37 | 23.5% | 20.0% | 77.7% |
| 350k | EMA | 0.05574 | 193.36 | 28.0% | 23.0% | 88.1% |

Conclusion: unlike the short 20k tiny-overfit run, EMA is beneficial for this long REPA run. The blurred quality of periodic REPA previews is not explained by EMA alone. REPA 350k EMA is the selected checkpoint for visual and metric comparisons: it exceeds baseline 100k under the full cross-step protocol, while an equal-step 350k baseline remains unavailable.

## VAE Reconstruction Ceiling

Imagenette validation images were encoded and decoded with frozen `stabilityai/sd-vae-ft-mse` using scaling factor `0.18215`, then evaluated against the original validation distribution.

| KID | FID | ResNet target accuracy |
| ---: | ---: | ---: |
| 0.00245 | 13.74 | 71.3% |

The VAE reconstruction ceiling is far above the generative metrics, so the observed SiT quality plateau cannot be explained by the VAE decoder alone.

## Tiny Generative Overfit

The isolated config is `mini_diffusion/configs/imagenette_sit_s_128_tiny_overfit.yaml`: 256 cached samples of class 6 (`n03417042`, garbage truck), SiT-S/2 without REPA, BF16 CUDA, random timestep/noise training, and fixed decoded EMA Heun-25 previews. It uses `num_workers: 0` and a separate output path. Final run results are recorded after the 20k cap or an earlier successful reproduction.

The run resumed from its 5k checkpoint and completed the 20k cap in about ten minutes. Loss moved from about `1.03` early in training to a variable `0.16`--`0.35` near the end. Fixed EMA previews remained blurred, but this was an EMA-lag effect rather than a broken generative path:

| Weights, Heun-50 | KID | FID | ResNet garbage-truck top-1 | Precision | Recall | Nearest-train feature distance | Exact feature duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.00962 | 95.27 | 60.9% | 68.8% | 70.1% | 8.89 | 0 |
| EMA (`0.9999`) | 0.23196 | 277.10 | 0.0% | not meaningful | not meaningful | 15.54 | 0 |

The raw fixed grid contains recognisable trucks across diverse colors and viewpoints. The nearest-neighbour and duplicate check did not identify exact feature duplicates at the configured threshold. Therefore the full dataset-to-latent-to-random-timestep/noise-to-SiT-to-ODE-to-VAE path is functional. The test does **not** prove that the larger REPA run has adequate capacity or schedule; it does rule out the VAE, velocity target, inverse scaling, and ODE sign as the primary cause of the observed blur. A 0.9999 EMA has a long effective horizon for this short diagnostic and should not be used as the only visual judge of early or small-subset training.

## Commands Run

```powershell
.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt --output evaluation\smoke --mode quick --samples-per-class 1 --steps 2 --sample-batch-size 10
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0100000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0150000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0200000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0350000.pt --output evaluation\imagenette_sit_quick --mode quick --weights ema --sampler heun --steps 25 --guidance-scale 1.5 --samples-per-class 20 --seed-start 1000 --reference-split val --sample-batch-size 20
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt --output evaluation\imagenette_sit_vae_ceiling --reference-cache evaluation\imagenette_sit_quick\reference --reference-split val --sample-batch-size 32 --vae-ceiling
.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_tiny_overfit.yaml --max-steps 5000
.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_tiny_overfit.yaml --resume outputs\imagenette_sit_s_128_tiny_overfit\checkpoints\latest.pt
.venv\Scripts\python.exe mini_diffusion\evaluate_tiny_overfit.py --checkpoint outputs\imagenette_sit_s_128_tiny_overfit\checkpoints\latest.pt --output evaluation\imagenette_sit_tiny_overfit\raw_heun50 --reference-cache evaluation\imagenette_sit_tiny_overfit\reference --class-id 6 --samples 64 --seed-start 3000 --steps 50 --guidance-scale 1.5 --weights raw
.venv\Scripts\python.exe mini_diffusion\evaluate_tiny_overfit.py --checkpoint outputs\imagenette_sit_s_128_tiny_overfit\checkpoints\latest.pt --output evaluation\imagenette_sit_tiny_overfit\ema_heun50 --reference-cache evaluation\imagenette_sit_tiny_overfit\reference --class-id 6 --samples 64 --seed-start 3000 --steps 50 --guidance-scale 1.5 --weights ema
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128_repa\checkpoints\step_0200000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0350000.pt --output evaluation\imagenette_sit_repa_raw_heun50 --reference-cache evaluation\imagenette_sit_quick\reference --mode quick --weights raw --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 20 --seed-start 1000 --reference-split val --sample-batch-size 20
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128_repa\checkpoints\step_0200000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0350000.pt --output evaluation\imagenette_sit_repa_ema_heun50 --reference-cache evaluation\imagenette_sit_quick\reference --mode quick --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 20 --seed-start 1000 --reference-split val --sample-batch-size 20
.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q -k "not repa"
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt outputs\imagenette_sit_s_128\checkpoints\step_0200000.pt --output evaluation\imagenette_sit_baseline_ema_heun25 --weights ema --sampler heun --steps 25 --guidance-scale 1.5 --samples-per-class 20 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode quick
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt outputs\imagenette_sit_s_128\checkpoints\step_0150000.pt outputs\imagenette_sit_s_128\checkpoints\step_0200000.pt --output evaluation\imagenette_sit_baseline_raw_heun25 --weights raw --sampler heun --steps 25 --guidance-scale 1.5 --samples-per-class 20 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode quick
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\step_0150000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0150000.pt --output evaluation\baseline150_vs_repa150_full_ema_heun50 --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 100 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode full --sample-batch-size 20
```

Completed equal-step comparison commands:

```powershell
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0100000.pt --output evaluation\baseline100_vs_repa100_full_ema_heun50 --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 100 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode full --sample-batch-size 20
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\step_0150000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0150000.pt --output evaluation\baseline150_vs_repa150_full_ema_heun50 --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 100 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode full --sample-batch-size 20
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\step_0200000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0200000.pt --output evaluation\baseline200_vs_repa200_full_ema_heun50 --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 100 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode full --sample-batch-size 20
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0350000.pt --output evaluation\baseline100_vs_repa350_full_ema_heun50 --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 100 --seed-start 1000 --reference-split val --reference-cache evaluation\reference_cache --mode full --sample-batch-size 20
```

The baseline was continued from 100k to 200k; no new full baseline-from-scratch training was run. No FID/CLIP dependencies beyond already installed PyTorch, torchvision, diffusers and their cached pretrained weights were added.
