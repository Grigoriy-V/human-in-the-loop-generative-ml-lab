# Imagenette SiT-S/2 + REPA Setup

Date: 2026-07-17  
Frozen baseline parent commit: `871987ffe00eb379fa38d117b8f1dde0f4237629`  
Status: implementation, cache preparation, smoke, and benchmark complete. Full REPA training was not started.

## Scope and implementation

This experiment is separate from the frozen Imagenette SiT-S/2 100k baseline. `train.init_from: null` is explicit, and REPA resume accepts only checkpoints containing a REPA projector. The baseline checkpoint and its output directory were not modified.

The implementation follows the representation-alignment design of the official [REPA repository](https://github.com/sihyun-yu/REPA) with the frozen [DINOv2 implementation](https://github.com/facebookresearch/dinov2). The objective is:

`total_loss = flow_loss + 0.5 * repa_loss`

where `flow_loss` is the unchanged velocity MSE and `repa_loss` is negative token-wise cosine similarity. SiT exposes its hidden state after block 8 only when training requests it. The projector is `384 -> 2048 -> 2048 -> 768` with SiLU activations. It is not called by normal sampling.

Parameters: base SiT `32,527,888`; projector `6,558,464`; total trainable `39,086,352`.

## Teacher cache

Teacher: `facebookresearch/dinov2`, `dinov2_vitb14`, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. It is evaluation-only, has `requires_grad=False`, uses `torch.inference_mode()`, and is never placed in the training optimizer.

The cache reads the exact source image selected by each latent cache relative path, verifies its label, repeats the deterministic Imagenette RGB 128px crop used for VAE cache creation, resizes to 224px with bicubic antialiasing, applies ImageNet normalization, takes DINO patch tokens without CLS, and pools `16x16 -> 8x8` adaptively to `[N,64,768]`. No augmentation or horizontal flip is active. A paired flip helper is covered by unit tests for any future augmentation work.

Full train cache: `outputs/imagenette_sit_s_128_repa/dino_features/train/` (ignored by Git), FP16 `[9469,64,768]`, `930,840,704` feature bytes, all finite, min `-20.21875`, max `19.296875`, mean `-0.02254368`, std `1.49264449`. It contains all ten classes.

- Feature cache fingerprint: `c5a508c0739931e5871d27d1ec889d0e637e25b6b24d84660fd6fbdf7e640550`
- Source latent cache fingerprint: `038b4f04f3066968c8d85017b689d438ae2cf09e20a434e343abfc2b1e0cfafa`

The `.npy` feature and label arrays are memory-mapped. Windows DataLoader workers reopen mappings lazily and never pickle a full feature array.

## Runtime checks

Hardware/software: NVIDIA GeForce RTX 4090, CUDA `12.8`, PyTorch `2.11.0+cu128`, BF16 autocast. DINO cache construction and REPA smoke used CUDA. DINOv2 emitted expected xFormers-unavailable warnings; no xFormers or third-party Flash Attention was installed.

Executed commands:

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
.\.venv\Scripts\python.exe mini_diffusion\prepare_repa_features.py --config mini_diffusion\configs\imagenette_sit_s_128_repa_debug.yaml --split train --limit 64
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_repa_debug.yaml
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_repa_debug.yaml --resume outputs\imagenette_sit_s_128_repa_debug\checkpoints\latest.pt --max-steps 15
.\.venv\Scripts\python.exe mini_diffusion\prepare_repa_features.py --config mini_diffusion\configs\imagenette_sit_s_128_repa.yaml --split train
.\.venv\Scripts\python.exe mini_diffusion\benchmark_repa.py --config mini_diffusion\configs\imagenette_sit_s_128_repa.yaml --mode baseline --warmup 10 --steps 50 --output reports\repa_benchmark_baseline.json
.\.venv\Scripts\python.exe mini_diffusion\benchmark_repa.py --config mini_diffusion\configs\imagenette_sit_s_128_repa.yaml --mode repa --warmup 10 --steps 50 --output reports\repa_benchmark_repa.json
```

The full suite result was `29 passed`. The fixed-input CUDA one-batch REPA smoke reported total loss `1.337824 -> 0.159563` and cosine similarity `0.425768 -> 0.694626`. The debug checkpoint resumed from step 10 to step 15. `sample_sit.py` sampled its EMA without DINO cache or DINOv2. Two Heun-25 runs with class 0 and seed 123 produced byte-identical PNG grids, SHA-256 `bce8af121214cf61cf3f8f74489c250cbf58edd54f55eb8d312c7d98618b31fd`. A separate ordinary sampler smoke loaded the frozen baseline checkpoint successfully. The debug PNG was valid `128x128` RGB, non-black and non-white.

## Benchmark

Protocol: batch 256, workers 4, 10 warmup plus 50 measured updates. Both modes used fused AdamW, direct PyTorch SDPA with the memory-efficient backend, BF16, and `zero_grad(set_to_none=True)`.

| Mode | images/s | ms/step | peak allocated | peak reserved | OOM |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 5932.67 | 43.15 | 3.98 GB | 4.07 GB | no |
| cached REPA | 4720.26 | 54.23 | 4.58 GB | 4.63 GB | no |

REPA overhead is `25.7%` in ms/step and `0.59 GB` peak allocated VRAM. Batch 256 remains selected: it is stable with substantial headroom, so batch 128 with accumulation 2 was not needed.

## Verification coverage

The test suite covers cache metadata, relative path/label matching, 16x16 to 8x8 pooling, paired flip, projector shape, finite cosine loss, detached teacher gradients, early SiT gradients, block-8 alignment, base initialization equality with REPA on/off, unchanged ordinary forward, sampling without projector/teacher cache, checkpoint round trip, resume, and fixed-input one-batch REPA overfit.

Not performed: full 0-to-100k REPA training, final Heun-50 comparison, FID/KID/CLIP, online DINO forwarding in training, and any modification of the baseline checkpoint. The future run command is:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_repa.yaml
```
