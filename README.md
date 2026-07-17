# Mini Diffusion

Educational class-conditioned DDPM in plain PyTorch. The first target is a fast CIFAR-10 32x32 debug pipeline; the same code also has Tiny ImageNet 64x64 configs.

Run every command below from:

```powershell
cd D:\ML\My_first_model
```

## Environment

Use Python 3.12 on Windows for the local virtual environment. The system Python in this workspace is 3.14, but PyTorch CUDA wheels are more reliably available for 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy Pillow PyYAML tqdm tensorboard pytest
```

If the CUDA wheel is not available, install the current wheel recommended at https://pytorch.org/get-started/locally/ and rerun the checks.

## What The Model Does

Forward diffusion adds Gaussian noise to an image at a randomly selected timestep. The U-Net receives the noisy image, timestep embedding, and optional class embedding, then predicts the exact noise that was added. Reverse diffusion starts from Gaussian noise and repeatedly removes predicted noise to form an image.

Class conditioning is done with a learned class embedding added to the timestep embedding. Classifier-free guidance uses a learned null class token during training and sampling. EMA keeps a smoothed copy of model weights, which is usually better for sampling than the raw training weights.

## Tests And Smoke Checks

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests
```

One-batch overfit smoke test on CIFAR-10:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml --overfit-smoke --overfit-updates 40
```

This reuses one batch for repeated updates and fails if the loss does not noticeably decrease. `cifar10_debug.yaml` uses `fake_data: true` by default so this runtime check does not depend on CIFAR-10 network download speed; set it to `false` after CIFAR-10 is downloaded to run the same debug path on real CIFAR-10.

The CIFAR configs use a public mirror for `cifar-10-python.tar.gz` because the default Toronto host can be very slow from this workspace. Remove `mirror_url` from the YAML files to use the torchvision default.

## CIFAR-10 Debug Pipeline

Train for a few steps:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml
```

Resume from the checkpoint:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml --resume outputs\cifar10_debug\checkpoints\latest.pt
```

To force additional debug steps after the checkpoint already reached the config default:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml --resume outputs\cifar10_debug\checkpoints\latest.pt --max-steps 6
```

Generate a PNG grid:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\cifar10_debug\checkpoints\latest.pt --classes 0 1 2 3 --seeds 10 20 30 40 --guidance-scale 1.5 --output outputs\samples\cifar10_debug
```

Sampling uses a dedicated generator for each seed, FP32 reverse steps, and EMA weights by default. DDPM remains the default final-evaluation sampler. A deterministic DDIM path is available for fast previews:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\cifar10\checkpoints\latest.pt --classes 0 1 2 3 --seeds 10 20 30 40 --guidance-scale 1.5 --sampler ddim --ddim-steps 50 --output outputs\samples\cifar10_ddim50
```

Use `--weights raw` for unsmoothed weights or `--sampling-diagnostics` to print min/max/mean/std, finite status, saturation rate, and black/white failure counts.

Outputs:

- checkpoints: `outputs\<run_name>\checkpoints\latest.pt`
- TensorBoard logs: `outputs\<run_name>\logs`
- training sample grids: `outputs\<run_name>\samples`
- CLI sample grids: `outputs\samples\...`

## CIFAR-10 Full Run

Recommended first full run on RTX 4090:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10.yaml
```

The full config uses about 27M trainable parameters, cosine noise schedule, BF16 autocast when available, EMA, batch size 128, and attention at 8x8.

## CIFAR-10 Baseline

The completed 200,000-step baseline, deterministic raw/EMA sampling comparison, isolated resume check, and pre-optimization benchmark are documented in:

- `reports/cifar10_baseline.md`
- `reports/cifar10_baseline_config.yaml`
- `reports/cifar10_baseline_benchmark.json`

Run the benchmark without sampling, checkpoints, or TensorBoard:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\cifar10.yaml --warmup-steps 20 --steps 200 --output reports\cifar10_baseline_benchmark.json
```

## CIFAR-10 Optimized Run

The baseline-semantics optimized config keeps batch size 128, learning rate, model, objective, schedule, EMA decay, and 200,000-step budget unchanged:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_optimized.yaml
```

It enables persistent Windows workers, pinned non-blocking transfer, fused AdamW with fallback, foreach EMA, cuDNN autotuning, scalar logging every 50 steps, and DDIM-50 periodic previews. Five benchmark runs reached median `2272.92 images/s`, 27.15% above the historical baseline. cuDNN autotuning adds a one-time startup cost of roughly 30-35 seconds on this machine.

Reproduce the final benchmark and regenerate its summary:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\cifar10_optimized.yaml --experiment-id FINAL_OPTIMIZED --runs 5 --warmup-steps 30 --steps 200 --log reports\performance_experiments.jsonl
.\.venv\Scripts\python.exe mini_diffusion\summarize_performance.py
```

Detailed results and accepted/rejected experiments are in `reports/cifar10_optimization_report.md`.

## Tiny ImageNet

The dataset is not downloaded automatically. Download the 248.1 MB archive from
[Zenodo](https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1),
then verify and unpack it from the repository root:

```powershell
Invoke-WebRequest -Uri "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1" -OutFile datasets\tiny-imagenet-200.zip
Get-FileHash datasets\tiny-imagenet-200.zip -Algorithm MD5
Expand-Archive -LiteralPath datasets\tiny-imagenet-200.zip -DestinationPath datasets
```

The expected MD5 is `90528d7ca1a48142e341f4ef8d21d0de`. The extracted root must be:

```text
datasets/tiny-imagenet-200/
```

The loader expects the original structure:

```text
train/<wnid>/images/
val/images/
val/val_annotations.txt
wnids.txt
words.txt
```

Validate image counts, class balance, annotations, decoding, tensor shape, and normalization before training:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\validate_tiny_imagenet.py
```

Tiny ImageNet debug:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\tiny_imagenet_debug.yaml
```

Tiny ImageNet full run:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\tiny_imagenet.yaml
```

The full Tiny ImageNet config uses about 48M trainable parameters and targets 64x64 images, 200 classes, BF16, EMA, gradient accumulation, and attention at 16x16 and 8x8.

On the RTX 4090, physical batch 128 fits in VRAM but was slightly slower in the synthetic compute probe and used roughly twice the peak memory of batch 64 with two accumulation steps. The first full-run config therefore uses `batch_size: 64` and `grad_accum_steps: 2`, preserving effective batch 128 with more memory headroom. Re-benchmark the real JPEG loader after extraction before changing workers or physical batch size. Detailed readiness results are in `reports/tiny_imagenet_readiness.md`.

Synthetic preflight is available without the archive. It verifies the full model/optimizer/EMA training path but does not validate JPEG decoding or loader throughput:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\tiny_imagenet.yaml --experiment-id TINY_SYNTHETIC_64X2 --warmup-steps 2 --steps 6 --set data.fake_data=true --set data.fake_size=512 --log reports\tiny_imagenet_experiments.jsonl
.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\tiny_imagenet.yaml --experiment-id TINY_SYNTHETIC_128X1 --warmup-steps 2 --steps 6 --set data.fake_data=true --set data.fake_size=512 --set data.batch_size=128 --set train.grad_accum_steps=1 --log reports\tiny_imagenet_experiments.jsonl
```

## Notes

- Debug configs use `num_workers: 0` for Windows-safe DataLoader startup.
- All executable scripts use `if __name__ == "__main__": main()`.
- Checkpoints, datasets, logs, and samples are ignored by git.
- DDPM and deterministic DDIM sampling are both implemented; `--ddim-steps` is used only with `--sampler ddim`.

## Imagenette Latent SiT-S/2

The latent baseline is isolated from the DDPM/U-Net path. It uses Imagenette-160 at 128x128, the frozen `stabilityai/sd-vae-ft-mse` VAE, a deterministic `[4, 16, 16]` float16 latent cache, and a 32,527,888-parameter class-conditioned SiT-S/2 velocity model. The VAE scaling factor is read from its configuration and recorded in every cache; it is not loaded by `train_sit.py`.

Prepare Imagenette (the CLI downloads the official fast.ai archive if needed), write a small debug cache, and create the validation reconstruction grid:

```powershell
.\.venv\Scripts\python.exe -m pip install diffusers huggingface_hub safetensors
.\.venv\Scripts\python.exe mini_diffusion\prepare_latents.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml --limit 32 --download-dataset
```

Run the CUDA/BF16 debug chain, then resume it:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml --resume outputs\imagenette_sit_s_128_debug\checkpoints\latest.pt --max-steps 20
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\imagenette_sit_s_128_debug.yaml --overfit-smoke --overfit-updates 40
```

Generate decoded samples. `euler` and `heun` are both live ODE samplers; `heun` is the default.

```powershell
.\.venv\Scripts\python.exe mini_diffusion\sample_sit.py --checkpoint outputs\imagenette_sit_s_128_debug\checkpoints\latest.pt --weights raw --classes 0 1 2 3 --seeds 10 20 30 40 --sampler euler --steps 5 --output outputs\imagenette_sit_s_128_debug\samples_raw
.\.venv\Scripts\python.exe mini_diffusion\sample_sit.py --checkpoint outputs\imagenette_sit_s_128_debug\checkpoints\latest.pt --weights ema --classes 0 1 2 3 --seeds 10 20 30 40 --sampler heun --steps 5 --output outputs\imagenette_sit_s_128_debug\samples_ema
```

Benchmark the full SiT configuration before a future full training run:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\benchmark_sit.py --config mini_diffusion\configs\imagenette_sit_s_128.yaml --output reports\imagenette_sit_benchmark.json
```

The tested candidates selected physical batch `256` (no accumulation) on this RTX 4090. `sampling.preview_decode: true` writes an EMA-decoded PNG alongside each periodic latent preview; the frozen VAE is loaded only during this preview step and is never optimized. Do not interpret the debug training as a trained generative model: full Imagenette training has intentionally not been started.
