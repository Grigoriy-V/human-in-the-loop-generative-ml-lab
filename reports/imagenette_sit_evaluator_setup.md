# Imagenette SiT Model Evaluator Setup

Date: 2026-07-18

Git base before this evaluator work: `d1dc1bf`.

## Scope

This setup adds a reproducible evaluator for Imagenette SiT checkpoints. It does not continue the non-REPA baseline beyond 100k, does not modify any source checkpoint, and does not use DINO as the primary quality criterion. DINO is deliberately reserved for future supplemental nearest-neighbour and diversity analysis because REPA directly optimizes a DINO-alignment objective.

## Checkpoint Inventory

| Run | Path | global_step | SHA-256 |
| --- | --- | ---: | --- |
| baseline | `outputs/imagenette_sit_s_128/checkpoints/baseline_0100000.pt` | 100000 | `21F79D6AF2C3ACBFB50F8B2739ACB9BA27C4A77265EF35432B383318F7A87197` |
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

Outputs: `evaluation/imagenette_sit_quick/`. With this small protocol, REPA 350k is the preliminary best REPA checkpoint. This is not a final baseline-vs-REPA verdict because a baseline 150k checkpoint does not exist and was not trained in this task.

## Raw, EMA, And Sampler Diagnostic

REPA 150k, fixed 200-sample protocol, CFG 1.5:

| Weights / sampler | KID | FID | Accuracy | Precision | Recall | Sampling seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw, Heun-25 | 0.07351 | 224.15 | 15.5% | 18.5% | 69.1% | 10.81 |
| EMA, Heun-25 | 0.07475 | 224.99 | 17.0% | 20.0% | 57.4% | 10.84 |
| EMA, Heun-50 | 0.07402 | 224.27 | 16.5% | 19.5% | 59.8% | 18.40 |

EMA slightly improved class accuracy and precision in this limited run. Heun-50 did not provide a material metric improvement over Heun-25, while sampling took about 70% longer. This does not support attributing the visible smoothing solely to EMA or to the 25-step Heun preview.

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
```

Future equal-step comparison command, once the frozen baseline 150k checkpoint exists:

```powershell
.venv\Scripts\python.exe mini_diffusion\evaluate_sit.py --checkpoints outputs\imagenette_sit_s_128\checkpoints\step_0150000.pt outputs\imagenette_sit_s_128_repa\checkpoints\step_0150000.pt --output evaluation\baseline150_vs_repa150 --mode full --weights ema --sampler heun --steps 50 --guidance-scale 1.5 --samples-per-class 100 --seed-start 1000 --reference-split val --reference-cache evaluation\imagenette_sit_quick\reference
```

No full baseline training was run. No FID/CLIP dependencies beyond already installed PyTorch, torchvision, diffusers and their cached pretrained weights were added.
