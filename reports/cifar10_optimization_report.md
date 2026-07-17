# CIFAR-10 Performance Optimization Report

## Scope And Environment

This stage optimized the existing CIFAR-10 DDPM training implementation without changing the U-Net architecture, parameter count, diffusion objective, noise schedule, loss, learning rate, EMA decay, CFG dropout, normalization, checkpoint schema, batch semantics, or 200,000-step budget.

- Baseline tag: `cifar10-ddpm-baseline`
- Baseline commit: `b82b61b2cd3d3bf3bb835fb5ae819663c33ce0db`
- Experiment harness commit: `d7b13678a35ccc9ad510032095e1c2c05057a374`
- Python: 3.12.4
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090
- Dtype: BF16 autocast
- Trainable parameters: 27,285,107

The original checkpoint remained unchanged at step 200,000 with SHA-256 `97089682B162CB31F3C2D6A35731778553355AA7DA468AF2657B9141A86944B4`.

## Protocol

Each significant variant used an isolated process, seed 123, 30 warmup steps, at least 200 measured steps, sampling/checkpoints/TensorBoard disabled, CUDA synchronization around the measured window, and reset peak memory statistics. Decisions use median images/s from at least three runs. Changes below 3% are noise; 3-5% candidates require additional confirmation.

The benchmark records wall throughput, CUDA Event step/transfer/optimizer/EMA timings, data wait, mean/median/p95, allocated/reserved VRAM, GPU utilization, finite loss, environment, commit, overrides, and run index. Raw data is in `performance_experiments.jsonl`; the resolved keep/reject table is `performance_experiments.csv`.

## Baseline And Final Result

| Metric | Historical | Repeated A0 | Optimized |
| --- | ---: | ---: | ---: |
| Runs | 1 | 3 | 5 |
| Images/s | 1787.57 | 1812.39 median | 2272.92 median |
| Iterations/s | 13.965 | 14.159 median | 17.757 median |
| Mean wall step | 0.071605 s | 0.070625 s median | 0.056315 s median |
| Peak allocated VRAM | 3.106 GB | 3.106 GB | 3.118 GB |
| Peak reserved VRAM | 3.422 GB | 3.422 GB | 3.566 GB |
| Batch/effective batch | 128/128 | 128/128 | 128/128 |

Final improvement is 27.15% versus the historical baseline and 25.41% versus repeated A0. All measured losses were finite.

## Experiment Results

| Experiment | Median images/s | Versus reference | Decision |
| --- | ---: | ---: | --- |
| A0 repeated baseline | 1812.39 | +1.4% vs historical | Control |
| Batch 256, workers restart | 1455.75 | -19.7% vs A0 | Reject |
| Batch 256, persistent workers | 2100.31 | +15.9% vs A0 | Keep finding |
| Batch 512, persistent workers | 2192.14 | +4.4% vs batch 256 | Throughput-only winner |
| Batch 768, persistent workers | 2197.76 | +0.3% vs batch 512 | Reject |
| Foreach EMA | 1866.26 | +3.0% vs A0 | Keep |
| Fused AdamW alone | 1817.11 | +0.3% vs A0 | Reject alone |
| Scalar sync every 50 steps | 1921.16 | +6.0% vs A0 | Keep |
| Channels-last | 1680.08 | -7.3% vs A0 | Reject |
| cuDNN benchmark | 1881.96 | +3.8% vs A0 | Keep after combined check |
| Channels-last + cuDNN | 1812.98 | +0.0% vs A0 | Reject |
| SDPA attention | 1801.37 | -0.6% vs A0 | Reject for CIFAR-10 |
| Workers 0, batch 256 | 1732.81 | -18.8% vs workers 2 | Reject for full run |
| Workers 2, batch 256 | 2134.59 | Reference loader winner | Keep |
| Workers 8, batch 256 | 2135.37 | +0.0% vs workers 2 | Reject extra processes |
| Unpinned blocking transfer | 2044.29 | -4.2% vs loader winner | Reject |
| Pinned blocking transfer | 2085.35 | -2.3% vs loader winner | Reject |
| Prefetch factor 4 | 2100.20 | -1.6% vs factor 2 | Reject |
| Log50 + foreach EMA | 2004.14 | +10.6% vs A0 | Keep |
| Log50 + EMA + cuDNN | 2204.73 | +21.6% vs A0 | Keep |
| Final + fused AdamW | 2275.18, five runs | +3.2% vs non-fused combination | Keep |

Batch 512 is the practical maximum-throughput result, with 10.94 GB allocated and 12.67 GB reserved. Batch 768 used 16.16/18.85 GB and provided no meaningful gain. The default optimized config retains batch 128 because changing it would change optimizer updates and processed images for the fixed 200,000-step budget.

## Profiler Findings

The short PyTorch Profiler run identified convolution forward/backward and GroupNorm as dominant kernels. Manual EMA was a visible separate region. CIFAR-10 data loading is not a steady-state bottleneck with multiprocessing, but Windows worker restart at every epoch was severe when `persistent_workers` was false.

The old `itertools.cycle(DataLoader)` also cached batches after the first epoch. It was replaced by an iterator that starts a fresh DataLoader epoch without retaining all prior tensors.

## Preview Overhead

| Operation | Time | Amortized at interval 2000 |
| --- | ---: | ---: |
| DDPM, 1000 steps, 16 images | 19.887 s | about 15% of optimized training time |
| DDIM, 50 steps, 16 images | 0.966 s | about 0.86% |
| DDIM, 25 steps, 16 images | 0.810 s | about 0.72% |
| 437 MB checkpoint save | 0.341 s | about 0.30% |
| TensorBoard images | 0 s | not enabled |

DDIM-50 is the periodic preview default. DDPM remains the default CLI sampler and final-evaluation path. DDIM-25/50/DDPM outputs were finite with no black/white failures. Two independent EMA DDIM-50 grids were byte-identical with SHA-256 `6F41836C873D72E0365C1DE8EB6CE0A55660001F49A61C88E1522F1A6140C8AA`.

## Accepted Configuration

`mini_diffusion/configs/cifar10_optimized.yaml` enables:

- `num_workers: 2`, pinned memory, persistent workers, prefetch factor 2, non-blocking transfer;
- scalar loss/TensorBoard synchronization every 50 steps;
- foreach EMA update while preserving the checkpoint schema;
- fused AdamW with a standard AdamW fallback;
- `torch.backends.cudnn.benchmark = True`;
- deterministic EMA DDIM-50 periodic previews.

Channels-last, SDPA for CIFAR-10, larger prefetch, eight workers, and `torch.compile` are not enabled. Native Windows `torch.compile` reported `TritonMissing`; no unofficial compiler packages were installed, and the training path now falls back explicitly to eager.

## Correctness Checks

- Pytest: 13 passed.
- One-batch overfit: first-quarter average 0.678398, last-quarter average 0.174257.
- Baseline short loss at steps 50/100: 0.1281/0.1391.
- Optimized short loss at steps 50/100: 0.1275/0.1393.
- Optimized checkpoint resume: step 100 to 105.
- Legacy standard-AdamW baseline checkpoint resumed with fused AdamW: step 200,000 to 200,001.
- Resumed model and EMA tensors: all finite.
- Optimizer state entries after resume: 253.
- EMA and raw DDIM-50: finite, no black/white failures.
- Periodic DDIM test: model mode restored and global training RNG unchanged.

Only smoke training and benchmarks were run during this optimization stage. No new full 200,000-step training was run. CUDA on the RTX 4090 was used for benchmarks, smoke training, resume, profiler, and sampling.

## Tiny ImageNet Recommendations

Tiny ImageNet was not present at `datasets/tiny-imagenet-200`, so no full training or loader benchmark was run. Start with BF16, fused AdamW fallback, foreach EMA, cuDNN benchmark, pinned non-blocking transfer, persistent workers, and DDIM-50 previews every 5,000 steps. Retune batch size and workers at 64x64 rather than copying CIFAR-10 values. Re-test SDPA because Tiny ImageNet uses attention at 16x16 and 8x8.

If a measured Tiny ImageNet profiler run shows JPEG decoding above 10% of step time after worker tuning, add one decoded `uint8` NumPy memmap cache under `datasets/`. Keep random horizontal flip and normalization as runtime transforms. Do not add multiple cache formats before measuring this bottleneck.

## Reproduction Commands

All commands run from `D:\ML\My_first_model`.

```powershell
.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\cifar10.yaml --experiment-id A0 --runs 3 --warmup-steps 30 --steps 200 --log reports\performance_experiments.jsonl
.\.venv\Scripts\python.exe mini_diffusion\profiler_run.py --config mini_diffusion\configs\cifar10.yaml --output outputs\perf\profiler_A1 --warmup-steps 5
.\.venv\Scripts\python.exe mini_diffusion\benchmark.py --config mini_diffusion\configs\cifar10_optimized.yaml --experiment-id FINAL_OPTIMIZED --runs 5 --warmup-steps 30 --steps 200 --log reports\performance_experiments.jsonl
.\.venv\Scripts\python.exe mini_diffusion\preview_benchmark.py --checkpoint outputs\cifar10\checkpoints\latest.pt --output reports\cifar10_preview_overhead.json --work-dir outputs\perf\preview_overhead --sample-count 16
.\.venv\Scripts\python.exe mini_diffusion\summarize_performance.py
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_optimized.yaml --overfit-smoke --overfit-updates 40
```
