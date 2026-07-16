# CIFAR-10 DDPM Baseline

## Status

The full CIFAR-10 training run completed at 200,000 optimizer steps. No new full training was started for baseline closure. All sampling, resume, and benchmark checks used the existing final checkpoint:

`outputs/cifar10/checkpoints/latest.pt`

- SHA-256: `97089682B162CB31F3C2D6A35731778553355AA7DA468AF2657B9141A86944B4`
- Global step: 200,000
- Final logged loss: 0.0492586
- First logged loss: 0.6052054 at step 6
- Approximate processed images: 25,600,000
- Equivalent CIFAR-10 epochs: 512.0
- Full-run wall time: approximately 4:38:12
- Training device: NVIDIA GeForce RTX 4090 with CUDA and BF16 autocast

The source checkpoint hash and timestamp were unchanged after the isolated resume test.

## Configuration

The checkpoint configuration and derived runtime values are frozen in `reports/cifar10_baseline_config.yaml`. The checkpoint config exactly matched `mini_diffusion/configs/cifar10.yaml` before the resume test.

- Trainable parameters: 27,285,107
- Batch/effective batch: 128/128
- Optimizer: AdamW, learning rate 0.0002, weight decay 0.01
- Diffusion: epsilon prediction, cosine schedule, 1,000 timesteps
- EMA decay: 0.9999
- CFG conditioning dropout: 0.1
- Attention resolution: 8x8
- Checkpoint and preview frequency: every 2,000 steps
- Dataset normalization: `(x - 0.5) / 0.5`, mapping `[0,1]` to `[-1,1]`

## Sampling Closure

The DDPM reverse sampler now uses a dedicated seeded `torch.Generator` for the initial noise and every reverse-step noise draw. Sampling runs in FP32 under `torch.inference_mode()`, checks finite output, clips predicted `x0` before the posterior mean, and denormalizes to `[0,1]` before PNG encoding. Periodic training previews use fixed labels, EMA weights, and a fixed preview seed without changing the training RNG, then restore the prior model training mode.

The historical PNG files under `outputs/cifar10/samples` were produced by the old sampler and intentionally remain unchanged. Some contain black/white failures. The verified baseline outputs use the corrected sampler.

The actual periodic `write_samples()` path was also executed twice on the final checkpoint with 16 fixed labels, preview seed 123, EMA weights, and guidance scale 1.5. The PNG files were byte-identical; CPU and CUDA global RNG states were unchanged; `model.training` was restored; output was finite with saturation rate 0 and zero black/white failures. The checked grid is `outputs/cifar10_baseline/periodic_check/samples/step_0200000.png`.

| Weights | Grid | Min | Max | Mean | Std | Saturation | Black | White |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | `outputs/cifar10_baseline/comparison/raw.png` | -0.999472 | 0.993478 | -0.156328 | 0.538692 | 0.0 | 0 | 0 |
| EMA | `outputs/cifar10_baseline/comparison/ema.png` | -0.999472 | 0.999472 | 0.055138 | 0.535918 | 0.0 | 0 | 0 |

Both grids use labels `[0,1,8,9]`, seeds `[100,101,108,109]`, DDPM 1,000 steps, guidance scale 1.5, and FP32 reverse sampling. Metadata is stored in `raw.json` and `ema.json` beside the grids. Periodic sampling uses EMA by default. EMA is retained as the default preview choice based on its intended smoothing behavior and the visual comparison, not training loss alone.

## Reproducibility

Two independent EMA sampling processes produced byte-identical PNG files:

- SHA-256 run 1: `FD835970A95F31B326BF420E002CFF9D8B9CCB2B0ADF591D189D8CB3107774C3`
- SHA-256 run 2: `FD835970A95F31B326BF420E002CFF9D8B9CCB2B0ADF591D189D8CB3107774C3`
- Direct pixel difference: none
- NaN/Inf: none
- Fully black/white samples: 0/0

## Checkpoint And Resume

The final checkpoint restored all 253 model tensors, all 253 EMA tensors, 253 optimizer state entries, global step 200,000, and Python/NumPy/PyTorch/CUDA RNG state fields. Five isolated CUDA training steps were run in `outputs/cifar10_baseline/resume_test`.

- Resume steps: 200001 through 200005
- Losses: 0.046226, 0.060660, 0.047389, 0.063680, 0.059623
- All losses and resulting model tensors: finite
- Test checkpoint global step: 200005
- Original checkpoint after test: unchanged SHA-256 and timestamp

## Baseline Benchmark

The benchmark used the same U-Net, CIFAR-10 loader, batch size, AdamW optimizer, BF16 autocast, gradient clipping, and EMA update as training. It performed 20 warmup steps and 200 measured steps without sampling, checkpoint saving, validation, or TensorBoard.

| Metric | Result |
| --- | ---: |
| Average step time | 0.071605 s |
| Iterations/s | 13.9654 |
| Images/s | 1787.57 |
| Average data loading time | 0.000091 s |
| CUDA peak allocated | 3.1061 GB |
| CUDA peak reserved | 3.4219 GB |
| Batch/effective batch | 128/128 |
| Device | NVIDIA GeForce RTX 4090 |
| Dtype | BF16 autocast |

Machine-readable output: `reports/cifar10_baseline_benchmark.json`.

## Tests

The fast suite completed with 10 passing tests. It covers schedule behavior, clipped `pred_x0`, U-Net forward/backward, finite sampling, deterministic sampling, independence from global training RNG, parameter preservation, restoration of training mode after periodic sampling, PNG denormalization, and checkpoint round-trip including model/EMA/optimizer/RNG restoration.

## Commands Actually Run

Commands were run from `D:\ML\My_first_model`.

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q

.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\cifar10\checkpoints\latest.pt --output outputs\cifar10_baseline\comparison --num-images 4 --classes 0 1 8 9 --seeds 100 101 108 109 --guidance-scale 1.5 --grid-size 2 --weights ema --grid-filename ema.png --metadata-filename ema.json --sampling-diagnostics

.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\cifar10\checkpoints\latest.pt --output outputs\cifar10_baseline\repro_run2 --num-images 4 --classes 0 1 8 9 --seeds 100 101 108 109 --guidance-scale 1.5 --grid-size 2 --weights ema --grid-filename ema.png --metadata-filename ema.json --sampling-diagnostics

.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\cifar10\checkpoints\latest.pt --output outputs\cifar10_baseline\comparison --num-images 4 --classes 0 1 8 9 --seeds 100 101 108 109 --guidance-scale 1.5 --grid-size 2 --weights raw --grid-filename raw.png --metadata-filename raw.json --sampling-diagnostics

.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10.yaml --resume outputs\cifar10\checkpoints\latest.pt --max-steps 200005 --output-dir outputs\cifar10_baseline\resume_test

.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\cifar10.yaml --warmup-steps 20 --steps 200 --output reports\cifar10_baseline_benchmark.json
```

An additional PowerShell here-string was piped to `.\.venv\Scripts\python.exe -` to load the final checkpoint and call `write_samples()` twice at steps 200000 and 200001 with output redirected to `outputs/cifar10_baseline/periodic_check`. It asserted identical PNG bytes, restored training mode, and unchanged CPU/CUDA global RNG states.

The original 200,000-step run was completed before this closure pass with `mini_diffusion/train.py --config mini_diffusion/configs/cifar10.yaml`. Only the five-step isolated resume test performed training during baseline closure.

## Limitations And Readiness

- Sampling validation used four fixed classes/seeds, including both seeds previously known to collapse. It is not an exhaustive quality metric.
- No FID or Inception Score was calculated; raw/EMA quality comparison is visual plus technical diagnostics.
- Historical periodic samples are not regenerated and still document the previous sampler failure.
- Git commit and tag were not created because `D:\ML\My_first_model` is not a Git repository.

The CIFAR-10 DDPM baseline is technically reproducible and ready to serve as the pre-optimization reference. Performance optimization should remain a separate stage and compare against `reports/cifar10_baseline_benchmark.json`.
