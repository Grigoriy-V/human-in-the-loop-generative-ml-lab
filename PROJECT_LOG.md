# Project Log

This log records completed milestones and experiments for Mini Diffusion. It focuses on outcomes, root causes, and decisions. Detailed measurements and command transcripts belong in `reports/`.

## Project Conventions

- Add an entry after a milestone or a meaningful experiment is complete, not after every implementation step.
- Use repository-relative paths in documentation and commands.
- Keep datasets, checkpoints, logs, and large generated outputs out of Git.
- Small representative sample grids may be copied to `docs/assets/` when they are useful for explaining a result.
- Keep detailed metrics in `reports/`; summarize only the conclusion and resulting decision here.

## 2026-07-17: Windows-Safe DDPM Foundation

### Goal

Build a small educational diffusion project in plain PyTorch with a working CIFAR-10 path before expanding to Tiny ImageNet.

### Work Completed

- Implemented epsilon-prediction DDPM training, linear and cosine schedules, forward noising, reverse sampling, EMA, checkpoint/resume, classifier-free guidance, and a class-conditioned U-Net.
- Added CIFAR-10 and Tiny ImageNet loaders and configurations. Tiny ImageNet remains opt-in and requires a manually prepared dataset.
- Standardized executable entrypoints with `if __name__ == "__main__": main()` for Windows multiprocessing compatibility.
- Created a Python 3.12 virtual environment because the available PyTorch CUDA wheels were compatible with Python 3.12, while the system Python 3.14 interpreter was not a reliable target.
- Added Windows-safe debug configurations with `num_workers: 0` and a one-batch overfit smoke test that verifies the model can reduce loss on a fixed batch.

### Decision

CIFAR-10 is the correctness gate for the project. Tiny ImageNet work should proceed only after the CIFAR-10 train, checkpoint, resume, and sample pipeline remains healthy.

## 2026-07-17: CIFAR-10 Debug Pipeline

### Goal

Verify the complete runtime path before committing GPU time to a full training run.

### Work Completed

- Confirmed that the manually downloaded CIFAR-10 archive could be used from `datasets/`.
- Ran debug training, created a checkpoint, resumed from it, and generated a PNG sample grid.
- Added unit coverage for diffusion schedules, U-Net shape and conditioning behavior, finite backward passes, checkpoint round trips, and short sampling.
- Verified the higher-priority one-batch overfit path with several dozen updates and a clear loss reduction.

### Decision

The debug pipeline passed and became the required preflight check for later training and performance changes.

## 2026-07-17: CIFAR-10 Version 0 Training

### Goal

Train the first complete CIFAR-10 baseline without changing the validated objective or model architecture.

### Outcome

The CUDA/BF16 run completed its full 200,000-step budget on an NVIDIA GeForce RTX 4090. The final checkpoint, configuration snapshot, resume validation, and baseline benchmark were preserved as the reference for later work.

Detailed results: [`reports/cifar10_baseline.md`](reports/cifar10_baseline.md)

Frozen configuration: [`reports/cifar10_baseline_config.yaml`](reports/cifar10_baseline_config.yaml)

### Decision

Tag the implementation as `cifar10-ddpm-baseline` and require later optimizations to preserve the model, objective, schedule, batch semantics, checkpoint compatibility, and training budget unless an experiment explicitly targets model quality.

## 2026-07-17: Sampler Failure Investigation

### Problem

Periodic training grids contained randomly collapsed black or white images. The failures did not align consistently with particular CIFAR-10 classes, so the class-conditioning path was not treated as the primary cause.

### Resolution

- Corrected the DDPM reverse path to clip predicted `x0` before computing the posterior mean.
- Kept reverse sampling in FP32 under inference mode and added finite-output diagnostics.
- Used a dedicated seeded generator for initial and per-step noise so sampling does not disturb training RNG state.
- Standardized conversion from model-space `[-1, 1]` values to PNG-space `[0, 1]`.
- Made periodic previews deterministic with fixed labels, a fixed seed, EMA weights, and restoration of the prior model mode.

### Decision

Historical failed grids remain local evidence and are not rewritten. New sampler output must be finite, reproducible for fixed inputs, and free of fully black or white samples in the smoke cases. EMA remains the default preview weight set; DDPM remains the reference final sampler.

## 2026-07-17: Baseline Closure

### Goal

Prove that the completed checkpoint was usable as a stable pre-optimization baseline.

### Work Completed

- Verified deterministic raw and EMA sampling with the corrected sampler.
- Exercised the real periodic preview path without changing global training RNG or leaving the model in the wrong mode.
- Restored model, EMA, optimizer, step, and RNG state from the final checkpoint and completed an isolated resume run.
- Recorded a CUDA throughput and memory benchmark without sampling, checkpoint, or TensorBoard overhead.

### Decision

The baseline was accepted as technically reproducible. Checkpoints and generated outputs remain ignored by Git; only configuration, code, and compact reports are versioned.

## 2026-07-17: CIFAR-10 Training Optimization

### Goal

Increase training throughput on native Windows while preserving baseline training semantics and checkpoint compatibility.

### Experiment Method

Added an isolated benchmark harness, CUDA timing, profiler support, structured experiment logs, and a keep/reject decision table. Significant variants were measured repeatedly after warmup instead of being selected from a single run.

Detailed report: [`reports/cifar10_optimization_report.md`](reports/cifar10_optimization_report.md)

Decision table: [`reports/performance_decisions.yaml`](reports/performance_decisions.yaml)

### Findings And Changes

- Replaced `itertools.cycle(DataLoader)` because it retained the first epoch's batches and prevented normal epoch iteration.
- Kept persistent workers, two loader workers, pinned non-blocking transfers, and prefetch factor two for the full Windows run.
- Reduced scalar synchronization frequency, used foreach EMA updates, enabled cuDNN autotuning, and enabled fused AdamW with a compatible fallback.
- Rejected channels-last, CIFAR-10 SDPA, extra workers, deeper prefetching, and a larger default batch because they were slower, insignificant, or changed baseline semantics.
- Left `torch.compile` disabled on native Windows after the supported Triton backend was unavailable.
- Implemented a real deterministic DDIM path and selected DDIM-50 for low-overhead periodic previews. The CLI exposes sampler controls only because both DDPM and DDIM paths are functional.

### Outcome

The accepted configuration in `mini_diffusion/configs/cifar10_optimized.yaml` improved measured throughput materially while keeping batch size 128, the 200,000-step budget, model architecture, objective, and checkpoint schema unchanged. Correctness checks, one-batch overfit, short loss comparisons, checkpoint resume, deterministic sampling, and the full test suite passed. This optimization stage used CUDA for benchmarks and smoke checks but did not run a second full 200,000-step training job.

### Decision

Tag the optimized implementation as `cifar10-ddpm-optimized`. Use DDIM-50 for periodic previews and DDPM for reference final evaluation. Keep all raw benchmark data in `reports/` and generated profiler/sample output outside Git.

## Current State And Next Milestone

- CIFAR-10 debug and baseline pipelines are complete.
- The original full Version 0 training is complete; the optimized configuration has only benchmark, smoke, resume, and sampling validation so far.
- Tiny ImageNet support exists, but `datasets/tiny-imagenet-200/` has not been provided and no Tiny ImageNet training or loader benchmark has been run.
- The next milestone is to prepare Tiny ImageNet, repeat the debug correctness gate at 64x64, profile data loading and attention at that resolution, and only then select full-run settings.
