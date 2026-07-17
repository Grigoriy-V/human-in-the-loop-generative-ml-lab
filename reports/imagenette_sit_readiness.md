# Imagenette Latent SiT-S/2 Readiness

## Scope

This milestone adds a cache-only SiT baseline without modifying the existing CIFAR-10 or Tiny ImageNet DDPM/U-Net pipelines. Full Imagenette training was intentionally not started.

## Runtime Evidence

- Device: RTX 4090 CUDA, BF16 autocast.
- VAE: `stabilityai/sd-vae-ft-mse`, resolved revision `31f26fdeee1355a5c34592e401dd41e45d25a493`, scaling factor `0.18215` read from VAE config.
- SiT-S/2: 32,527,888 trainable parameters; 4x16x16 latents, patch 2, hidden 384, depth 12, 6 heads.
- Debug cache: 32 train and 32 val latents, float16, no NaN/Inf. Train fingerprint: `cf29087e4cdcf83e1282cbe2f1de91b1d8a6af7882660e3a23b61aa81613dcf2`.
- VAE reconstruction grid: `outputs/imagenette_sit_s_128_debug/vae_reconstruction.png`. Decoder raw range was `[-1.236380, 1.231984]`; RGB is clamped only at PNG output.
- Debug training: checkpoint at step 10, resumed to 15 and then 20. Checkpoint includes model, EMA, optimizer, config, CPU/CUDA RNG and cache fingerprint.
- One-batch CUDA overfit: first-quarter loss `1.428837`, final-quarter loss `0.346980`.
- Raw Euler and EMA Heun decoded grids: `outputs/imagenette_sit_s_128_debug/samples_raw/grid.png` and `outputs/imagenette_sit_s_128_debug/samples_ema_a/grid.png`.
- Determinism: repeated EMA Heun grid SHA-256 `37E094F3A8B3DD5B886B0E660BD156F7C4D4ADCFFF01AE8C8D64704C4409059B`.

## Batch Probe

| Batch | Images/s | Seconds/step | Peak allocated GB | Peak reserved GB | OOM |
| --- | ---: | ---: | ---: | ---: | --- |
| 64 | 1895.77 | 0.03376 | 1.40 | 1.48 | no |
| 128 | 3978.03 | 0.03218 | 2.33 | 2.50 | no |
| 256 | 5112.88 | 0.05007 | 4.12 | 4.31 | no |

Decision: use batch 256 in `imagenette_sit_s_128.yaml`; it had the highest measured throughput with substantial VRAM headroom.

## Commands Run

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
.\.venv\Scripts\python.exe mini_diffusion\prepare_latents.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml --limit 32 --force
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml --resume outputs\imagenette_sit_s_128_debug\checkpoints\latest.pt --max-steps 20
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml --overfit-smoke --overfit-updates 40
.\.venv\Scripts\python.exe mini_diffusion\benchmark_sit.py --config mini_diffusion\configs\imagenette_sit_s_128.yaml --warmup 2 --steps 5 --output reports\imagenette_sit_benchmark.json
```
