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

Sampling uses a dedicated generator for each seed, FP32 DDPM reverse steps, and EMA weights by default. Use `--weights raw` for the unsmoothed training weights or `--sampling-diagnostics` to print min/max/mean/std, finite status, saturation rate, and black/white failure counts.

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

## Tiny ImageNet

Download Tiny ImageNet manually and unpack it as:

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

Tiny ImageNet debug:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\tiny_imagenet_debug.yaml
```

Tiny ImageNet full run:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\tiny_imagenet.yaml
```

The full Tiny ImageNet config uses about 48M trainable parameters and targets 64x64 images, 200 classes, BF16, EMA, gradient accumulation, and attention at 16x16 and 8x8.

## Notes

- Debug configs use `num_workers: 0` for Windows-safe DataLoader startup.
- All executable scripts use `if __name__ == "__main__": main()`.
- Checkpoints, datasets, logs, and samples are ignored by git.
- The sampler CLI currently implements DDPM only; there are no inactive DDIM flags.
