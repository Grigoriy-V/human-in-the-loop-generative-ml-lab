# Imagenette SiT-S/2 128x128 Baseline At 100k

Date: 2026-07-17  
Source code commit: `c5edfb8` (`Decode periodic Imagenette previews`)  
Baseline checkpoint: `outputs/imagenette_sit_s_128/checkpoints/baseline_0100000.pt`

## Scope

This report freezes the completed 100,000-step SiT-S/2 milestone as the baseline for the future SiT + REPA stage. No REPA, FID, CLIP, or additional training was run for this evaluation. No training beyond the captured 100k milestone was started.

## Checkpoint

- `global_step`: `100000`.
- Architecture: `SiT-S/2 velocity v1`.
- Trainable parameters: `32,527,888`.
- Model tensors: finite; EMA shadow tensors: finite.
- `latest.pt` SHA-256: `21F79D6AF2C3ACBFB50F8B2739ACB9BA27C4A77265EF35432B383318F7A87197`.
- `baseline_0100000.pt` was copied from `latest.pt`; its SHA-256 is identical. The source checkpoint was not modified.

## Frozen Inputs And Environment

- Frozen config: `reports/imagenette_sit_s_128_baseline_100k.yaml`.
- VAE: `stabilityai/sd-vae-ft-mse`; resolved revision `31f26fdeee1355a5c34592e401dd41e45d25a493`; scaling factor `0.18215`.
- Train cache fingerprint: `038b4f04f3066968c8d85017b689d438ae2cf09e20a434e343abfc2b1e0cfafa`.
- Cache metadata: Imagenette-160 train, RGB deterministic resize/center-crop at 128 px, cache seed 123, FP16 `[4,16,16]` latents, 9469 items.
- GPU: NVIDIA GeForce RTX 4090; PyTorch `2.11.0+cu128`; CUDA `12.8`; training dtype BF16 autocast.

## Standard EMA Evaluation

Every run used EMA weights, Heun ODE with 50 steps, class IDs 0-9, and seeds 100-104 for each class. This is 50 samples per CFG with no curated sample selection. Each directory contains 50 `sample_*.png`, `grid.png`, `metadata.json`, and `timing.json`.

| CFG | Sampling time | Grid SHA-256 | Failed samples | Grid |
| ---: | ---: | --- | ---: | --- |
| 1.0 | 54.42 s | `69f1464c8220f21f75bfacd3d779f2e221cece1a6a134d59dce8664a2257d6dd` | 0 / 50 | `outputs/imagenette_sit_s_128/evaluation_100k/cfg_1_0/grid.png` |
| 1.5 | 90.94 s | `71dbe7d12d946dcf593519f78e011d037e452b4a74d01e724324747cc294550f` | 0 / 50 | `outputs/imagenette_sit_s_128/evaluation_100k/cfg_1_5/grid.png` |
| 2.0 | 93.95 s | `5113ddd7105bec0b88594de888853625e4bf085616a6d8bd2d4eed1e1ea15ac9` | 0 / 50 | `outputs/imagenette_sit_s_128/evaluation_100k/cfg_2_0/grid.png` |

All 150 individual PNGs are RGB 128x128. All metadata arrays contain the expected 50 class/seed pairs. Pixel tensors were finite before PNG encoding; PNG metadata reports `[0, 1]` range. No corrupted, fully black, or fully white images were detected.

## Determinism

CFG 1.5 was repeated with the same checkpoint, EMA weights, Heun-50, labels, seeds, and environment. The repeated grid SHA-256 was again `71dbe7d12d946dcf593519f78e011d037e452b4a74d01e724324747cc294550f`; the PNG bytes match exactly. Re-reading the baseline checkpoint after all sampling preserved its SHA-256.

## Qualitative CFG Comparison

- CFG 1.0: broad variety and lower contrast, but the weak conditioning leaves several categories diffuse and less readable.
- CFG 1.5: best balance for this checkpoint. Fish, dog, golf-ball, and parachute-like class signals are more consistently visible without a broad increase in distorted forms.
- CFG 2.0: raises saturation and local contrast, but it also amplifies warped object boundaries and category-specific artifacts. It did not improve the overall readability enough to justify the stronger guidance.

Canonical CFG for the next SiT + REPA comparison: **1.5**. This is a fixed evaluation choice, not a claim that the 100k model is fully converged.

## Commands Actually Run

```powershell
Get-FileHash outputs\imagenette_sit_s_128\checkpoints\latest.pt -Algorithm SHA256
Copy-Item outputs\imagenette_sit_s_128\checkpoints\latest.pt outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt
.\.venv\Scripts\python.exe mini_diffusion\sample_sit.py --checkpoint outputs\imagenette_sit_s_128\checkpoints\baseline_0100000.pt --weights ema --classes <50 fixed class IDs> --seeds <100..104 repeated per class> --guidance-scale <1.0|1.5|2.0> --sampler heun --steps 50 --save-individual --output outputs\imagenette_sit_s_128\evaluation_100k\cfg_<scale>
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
```

Checkpoints, all individual PNGs, grids, and metadata remain ignored by Git. Full training was not started during this baseline-evaluation task.
