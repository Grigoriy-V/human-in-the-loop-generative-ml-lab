# Tiny ImageNet 64x64 Readiness

## Status

The Tiny ImageNet archive is not present at `datasets/tiny-imagenet-200/`. No network download was attempted and no real Tiny ImageNet training was run. The real loader path was checked for fast, clear missing-data behavior and was exercised with a temporary miniature copy of the original directory structure.

The full model training path was measured on synthetic 64x64 inputs to decide whether physical batch 128 is suitable for the first run.

## Environment

- Python: 3.12.4
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 4090, 23.99 GB
- Dtype: BF16 autocast
- Trainable parameters: 48,371,755
- Model: `mini_diffusion/configs/tiny_imagenet.yaml`

## Batch And Attention Probe

Each comparison used the same U-Net, diffusion loss, gradient clipping, fused AdamW, foreach EMA, cuDNN autotuning, and effective batch 128. Measurements used two warmup optimizer steps and six synchronized measured optimizer steps in isolated processes. Inputs were synthetic GPU tensors, so the results exclude JPEG decoding and DataLoader throughput.

| Physical batch | Accumulation | Attention | Images/s | Mean optimizer step | Peak allocated | Peak reserved | Result |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 64 | 2 | manual | 462.85 | 0.2765 s | 7.73 GB | 8.29 GB | Keep |
| 128 | 1 | manual | 448.26 | 0.2855 s | 14.23 GB | 16.00 GB | Fits, reject as default |
| 64 | 2 | SDPA | 442.50 | 0.2893 s | 7.81 GB | 8.64 GB | Reject |

A separate first-step batch-128 probe also completed successfully at 13.97 GB peak allocated and 14.87 GB peak reserved. Its 33-second first step included cuDNN algorithm selection and is not a steady-state throughput measurement. All measured losses were finite.

## Decision

- Keep physical `batch_size: 64` with `grad_accum_steps: 2` for effective batch 128.
- Keep manual attention. SDPA was slower at this resolution and model shape.
- Enable pinned non-blocking transfer, four persistent workers, prefetch factor two, fused AdamW fallback, foreach EMA, cuDNN autotuning, and DDIM-50 periodic previews in the initial full config.
- Treat four workers as a starting value, not a final loader result. Re-benchmark workers and physical batch size with real JPEG files before starting the full run.
- Use `mini_diffusion/validate_tiny_imagenet.py` as the required data-integrity gate after extraction.

## Dataset Gate

The validator requires the original 200 classes, 100,000 training images, 10,000 validation images, 500 training images per class, and 50 validation images per class. It also checks annotations, one decoded DataLoader batch, `[3, 64, 64]` tensors, finite values, `[-1, 1]` normalization, and class indices.

Archive source: `https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1`

Expected archive MD5: `90528d7ca1a48142e341f4ef8d21d0de`

## Remaining Checks

The archive has been extracted and the 64x64 debug gate is complete. Before the full run, only a longer repeated benchmark is optional; the first-run configuration is ready.

## Real-Data Batch Comparison

The real Tiny ImageNet loader used four persistent Windows workers, pinned non-blocking transfer, five warmup steps, and 30 synchronized measured optimizer steps. Sampling, checkpointing, and TensorBoard logging were disabled.

| Physical batch | Accumulation | Effective batch | Images/s | Mean optimizer step | Peak allocated | Peak reserved | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 2 | 128 | 458.07 | 0.2794 s | 7.73 GB | 8.29 GB | Keep |
| 128 | 2 | 256 | 453.92 | 0.5640 s | 14.41 GB | 16.60 GB | Reject as default |

Both runs had finite loss and approximately 99% GPU utilization. Data wait was below 0.4 ms per optimizer step, so JPEG decoding is not a current bottleneck. Batch `128 x 2` changes the optimization semantics to effective batch 256 while providing 0.9% lower throughput and consuming about twice the VRAM. The full config remains `batch_size: 64` with `grad_accum_steps: 2`.

## Real Debug Pipeline

The real-dataset debug configuration completed train, checkpoint, resume, and sampling on CUDA:

- trained steps 1 through 4 and wrote `outputs/tiny_imagenet_debug_real/checkpoints/latest.pt`;
- resumed from step 4 to step 6 with finite losses;
- wrote periodic PNG grids at steps 2, 4, and 6;
- generated `outputs/tiny_imagenet_debug_real/cli_samples/tiny_debug.png` from EMA weights;
- sampling diagnostics reported finite values, zero saturation, and zero black/white failures.

## Checks Actually Run

- `python -m pytest mini_diffusion/tests -q`: 16 passed.
- `python -m compileall -q mini_diffusion`: passed.
- `python mini_diffusion/validate_tiny_imagenet.py`: stopped immediately with the expected missing-dataset error and download URL.
- Full batch-128 CUDA training-step probe: passed with finite loss.
- Isolated synthetic comparisons for manual `64 x 2`, manual `128 x 1`, and SDPA `64 x 2`: passed with finite losses; results are in the table above.
- `benchmark.py` synthetic smoke with the prepared config and four persistent Windows workers: passed with finite loss and effective batch 128.
- Tiny ImageNet debug-model one-batch overfit on 64x64 FakeData: average loss decreased from 0.868714 in the first quarter to 0.362374 in the last quarter over 40 updates.
- Real-data benchmarks for `64 x 2` and `128 x 2`: both passed with finite loss; the real-data table above records the result.
- Real-dataset debug train, checkpoint, resume, periodic sampling, and EMA CLI sampling: passed.

CUDA was used for every model/batch probe. Only synthetic smoke training and performance measurements were run; no Tiny ImageNet debug or full training run was performed.
