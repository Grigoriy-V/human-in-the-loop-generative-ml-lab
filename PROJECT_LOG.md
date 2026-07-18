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

## 2026-07-17: Tiny ImageNet 64x64 Readiness

### Goal

Prepare the 64x64 training path and determine whether physical batch 128 is a good first-run setting before the dataset is available.

### Outcome

The real loader reports the missing archive immediately and now has a standalone integrity validator. A temporary original-layout dataset verified train/validation decoding without adding data to Git. Synthetic CUDA probes confirmed that physical batch 128 fits on the RTX 4090, but `64 x accumulation 2` was slightly faster and used substantially less memory at the same effective batch. SDPA was also slower than manual attention for this model.

Detailed results: [`reports/tiny_imagenet_readiness.md`](reports/tiny_imagenet_readiness.md)

### Decision

Use physical batch 64 with two accumulation steps for the initial config. Keep manual attention and retest batch size and Windows worker count with real JPEG files after the dataset passes validation. Do not start full training before real-data overfit, debug checkpoint/resume, and sample generation pass.

## 2026-07-17: Tiny ImageNet Real-Data Gate

### Goal

Confirm that the extracted Tiny ImageNet archive, configured loader, and 64x64 training path are ready for a full run.

### Outcome

The extracted archive passed the integrity gate. Real-loader benchmarks selected `batch_size: 64` with `grad_accum_steps: 2`: `128 x 2` changed the effective batch to 256, consumed about twice the VRAM, and did not improve throughput. The real debug train, checkpoint, resume, periodic preview, and EMA CLI PNG path all passed on CUDA.

Detailed results: [`reports/tiny_imagenet_readiness.md`](reports/tiny_imagenet_readiness.md)

### Decision

Start the first full Tiny ImageNet run with `mini_diffusion/configs/tiny_imagenet.yaml`. Keep generated checkpoints, logs, and preview grids outside Git.

## 2026-07-17: Tiny ImageNet Partial 150k Snapshot

### Goal

Close a deliberately stopped Tiny ImageNet run with a verified checkpoint, reproducible EMA sample, and compact training metadata.

### Outcome

The latest checkpoint was saved at step 150,000 of 400,000, or 37.5% complete. It restored model, EMA, optimizer, and RNG state successfully with finite tensors. A fixed EMA DDIM-50 grid was generated and added as a compact documentation asset. The logged loss and recent training speed were frozen in `reports/`.

Detailed snapshot: [`reports/tiny_imagenet_partial.md`](reports/tiny_imagenet_partial.md)

### Decision

Mark this as a partial-training snapshot, not a completed model. Keep the checkpoint and large outputs ignored by Git. The code and compact report can be tagged as the reproducible 150k state.

## Current State And Next Milestone

- CIFAR-10 debug and baseline pipelines are complete.
- The original full Version 0 training is complete; the optimized configuration has only benchmark, smoke, resume, and sampling validation so far.
- Tiny ImageNet has a verified partial snapshot at 150,000 of 400,000 steps. The extracted archive, real debug pipeline, and real loader benchmark passed; `batch_size: 64` with `grad_accum_steps: 2` remains selected.
- The next milestone is either resuming this exact experiment from its saved checkpoint or beginning a separately scoped training run. Both should retain periodic DDIM-50 previews and checkpoints outside Git.

## 2026-07-17: Imagenette Latent SiT-S/2 Debug Gate

### Goal

Establish a separate Imagenette-160 latent generative baseline: frozen Stable Diffusion VAE, deterministic latent cache, SiT-S/2 velocity training, and ODE decoding checks.

### Outcome

The CUDA/BF16 debug path passed from real data through VAE reconstruction, 32-image train/val caches, SiT checkpoint/resume, CUDA one-batch overfit, raw Euler and EMA Heun decoded PNGs. Repeated fixed EMA sampling was byte-identical. A short batch probe selected 256 as the full-config batch size; no full Imagenette training was started.

Detailed evidence: [`reports/imagenette_sit_readiness.md`](reports/imagenette_sit_readiness.md)

### Decision

Keep `mini_diffusion/sit/` independent of the existing DDPM/U-Net modules. Preserve cache-only SiT training and VAE decoding as a separate CLI step. Keep caches, checkpoints, previews, and large outputs ignored by Git.

## 2026-07-17: Imagenette SiT-S/2 Performance Check

### Goal

Confirm the initial full-run configuration, CUDA attention path, and loader/optimizer settings without starting long training or changing the primary cache.

### Outcome

The direct PyTorch SDPA path selected fused memory-efficient attention. This wheel cannot use Flash Attention despite an RTX 4090 and BF16 head dimension 64, because it was not built with that kernel. Batch 256 with four workers was the fastest stable probe; batch 512 was slower and used substantially more VRAM. The full cache was absent and deliberately not created, so the isolated probes reused the existing debug cache without modifying it.

Detailed evidence: [`reports/imagenette_sit_s_128_performance_check.md`](reports/imagenette_sit_s_128_performance_check.md)

### Decision

Set the first training milestone to 100,000 steps, log every 100 steps, and run fixed eight-image Heun-25 previews every 10,000 steps. Retain batch 256, workers 4, fused AdamW, foreach EMA, and the Heun-50 explicit CLI evaluation path. Full training remains unstarted.

## 2026-07-17: Imagenette Full Latent Cache

### Goal

Prepare the full deterministic Imagenette-160 latent cache required by the approved 100k-step SiT training milestone.

### Outcome

The initial Windows four-worker run exposed a non-picklable local RGB transform. The transform was moved to module scope, after which cache preparation completed successfully: `9469` train and `3925` validation FP16 `[4,16,16]` latents, all finite and with all ten classes represented. The VAE reconstruction grid was also regenerated under the full output directory.

### Decision

Keep the Windows-safe module-level transform. Cache, grid, and VAE artifacts remain ignored; full training has not been started.

## 2026-07-17: Imagenette Decoded Periodic Previews

### Goal

Make periodic SiT previews directly inspectable as RGB PNGs during training rather than requiring a manual latent decode after every sample interval.

### Outcome

`train_sit.py` now writes the existing latent preview and an EMA-decoded PNG when `sampling.preview_decode: true`. The VAE is loaded only for the preview, stays frozen, is excluded from the optimizer, and is released after decoding. An EMA Heun-25 grid was generated from the step-10,000 checkpoint to validate the path.

### Decision

Keep decoded periodic previews enabled in the full Imagenette config. The next resumed training preview will occur at step 20,000; full training remains a user-controlled run.

## 2026-07-17: Imagenette SiT-S/2 Baseline At 100k

### Goal

Freeze the completed 100,000-step Imagenette SiT-S/2 milestone with an immutable checkpoint copy and a fixed, repeatable EMA evaluation before the future REPA stage.

### Outcome

The finite step-100k model and EMA were copied from `latest.pt` to an SHA-identical baseline checkpoint. Standard Heun-50 evaluation produced 50 fixed samples for each CFG 1.0, 1.5, and 2.0. All images were valid 128x128 RGB PNGs without black/white failures. The repeated CFG 1.5 grid was byte-identical.

Detailed evidence: [`reports/imagenette_sit_s_128_baseline_100k.md`](reports/imagenette_sit_s_128_baseline_100k.md)

### Decision

Use EMA Heun-50 with CFG 1.5 and the documented ten-class, five-seed protocol as the canonical pre-REPA baseline. Do not treat the 100k checkpoint as a completed long training run.

## 2026-07-17: Imagenette SiT-S/2 + REPA Setup

### Goal

Prepare a separate from-scratch SiT-S/2 representation-alignment experiment while preserving the frozen 100k baseline unchanged.

### Outcome

Added a frozen DINOv2-B/14 feature-cache CLI, memory-mapped FP16 train features matched by relative path and label, block-8 SiT alignment, a training-only projector, REPA checkpoint/resume metadata, and tests. The full 9469-sample cache, debug train/resume/sampling path, deterministic preview, and batch-256 benchmark completed on CUDA.

Detailed evidence: [`reports/imagenette_sit_s_128_repa_setup.md`](reports/imagenette_sit_s_128_repa_setup.md)

### Decision

Retain batch 256 with workers 4: cached REPA is stable and needs about 0.59 GB more allocated VRAM than baseline. Keep the new experiment isolated at `outputs/imagenette_sit_s_128_repa/`; full 0-to-100k REPA training is intentionally not started by setup.

## 2026-07-18: Imagenette SiT Evaluator And REPA Snapshot

### Goal

Create a reproducible quality-evaluation protocol, freeze the latest REPA state, and determine whether the observed visual plateau reflects a VAE limitation, sampler choice, or the learned generative model.

### Outcome

- Stopped the requested REPA run and copied its atomically saved step-365k checkpoint to immutable `step_0365000.pt` after SHA-256 verification.
- Added an evaluator with a shared fixed class/seed/noise protocol, cached ImageNet-Inception reference features, KID/FID, ImageNet ResNet class metrics, feature precision/recall, pixel diagnostics, nearest-real/outlier grids, and raw/EMA/Heun controls.
- Quick evaluation selected REPA 350k provisionally among the available REPA milestones; this is not a final REPA claim because no baseline 150k exists.
- VAE validation reconstruction was substantially better than all generated checkpoints, so VAE decoding is not the primary explanation for the plateau.
- A 256-image garbage-truck generative overfit run reached recognisable raw SiT samples by 20k. Its 0.9999 EMA samples were still blurred, showing that short-run periodic EMA previews can lag far behind the learned raw model.

Detailed results: [`reports/imagenette_sit_evaluator_setup.md`](reports/imagenette_sit_evaluator_setup.md)

### Decision

Use the evaluator protocol for future equal-step baseline-vs-REPA comparisons. Keep DINO metrics supplemental, never the sole quality judge. Do not change objective, scaling, or sampler mathematics based on the plateau alone; inspect raw and EMA side-by-side when evaluating short or restarted runs.

## 2026-07-18: Non-REPA SiT Baseline Continuation To 200k

### Goal

Continue the frozen Imagenette SiT-S/2 baseline from 100k to 200k without REPA, preserving the original checkpoint and writing permanent 150k and 200k milestones.

### Outcome

The baseline resumed from `baseline_0100000.pt` and produced immutable `step_0150000.pt` and `step_0200000.pt`. The 200k checkpoint is finite and has no REPA state. A fixed all-class quick evaluation compared raw 100k/150k/200k and EMA 100k/200k. Neither raw nor EMA 200k improved FID/KID or target class accuracy over the 100k checkpoint; all generated images were finite and had no black/white failures.

### Decision

Keep the 100k EMA checkpoint as the current best quick-protocol non-REPA baseline. Treat the 200-sample evaluation as a diagnostic rather than a final ranking. Use the newly available equal-step baseline checkpoints for the next baseline-vs-REPA evaluation.

## 2026-07-18: Equal-Step SiT Baseline Vs REPA At 150k

### Goal

Compare non-REPA and REPA SiT-S/2 checkpoints at the same 150k step with a larger fixed 1,000-sample evaluation before deciding whether a 200k comparison is necessary.

### Outcome

EMA Heun-50 evaluation with CFG 1.5 and identical all-class seeds produced nearly equal FID (`165.94` baseline, `165.28` REPA) and target accuracy (`20.1%`, `20.0%`). REPA improved feature-manifold precision from `20.9%` to `22.9%` and recall from `48.1%` to `62.3%`; no generated image failed finite or black/white validation.

### Decision

Do not run the conditional 200k comparison: the 14.2-point recall gain is material, so the matched 150k results are not close on diversity/coverage. Record REPA as improving coverage at 150k while retaining the mixed KID/FID outcome and avoiding a broader quality claim.

## 2026-07-18: Completed Full Baseline Vs REPA Curve

### Goal

Complete the equal-step non-REPA baseline versus REPA curve at 100k, 150k, and 200k with one shared 1,000-sample evaluation protocol.

### Outcome

All three EMA Heun-50 comparisons completed with fixed class/seed/noise inputs and no failed images. Baseline was stronger at 100k (FID `149.95` vs `152.73`, accuracy `26.9%` vs `22.3%`). At 150k FID and accuracy were tied while REPA recall rose from `48.1%` to `62.3%`. At 200k REPA surpassed baseline on KID (`0.07204` vs `0.07634`), FID (`162.10` vs `171.78`), accuracy (`21.7%` vs `17.3%`), precision (`21.8%` vs `20.9%`), and recall (`71.3%` vs `63.5%`).

### Decision

Use the completed full curve as the current equal-step evidence: REPA is not beneficial at 100k but becomes favorable relative to the same-step baseline by 150k and clearly by 200k. Do not interpret this as REPA 200k exceeding the absolute 100k baseline; choose future comparisons according to the question being tested.

## 2026-07-18: Full REPA 350k Vs Baseline 100k

### Goal

Measure the selected REPA 350k checkpoint against the strongest available 100k non-REPA baseline with the same full 1,000-image protocol.

### Outcome

REPA 350k improved FID from `149.95` to `130.96`, KID from `0.06103` to `0.05199`, class accuracy from `26.9%` to `29.5%`, and recall from `51.0%` to `72.1%`. Precision changed from `24.6%` to `23.9%`. Both sets had zero finite and black/white failures.

### Decision

Use REPA 350k EMA as the current checkpoint for visual and metric comparisons. Its improvement over baseline 100k is clear under the common protocol, but label it as a cross-step result because a non-REPA 350k control does not exist.

## 2026-07-18: Imagenette SiT-S/2 Baseline Vs REPA Closure

### Goal

Close the Imagenette 128x128 baseline versus REPA experiment with final reproducibility checks and a directly inspectable side-by-side artifact, without further training or sampling.

### Outcome

Full pytest completed with `33 passed`. Frozen baseline 100k and REPA 350k checkpoints remain finite and retain their recorded SHA-256 values. A compact full-protocol side-by-side grid was generated from the existing 1,000-sample EMA Heun-50 grids and saved as `docs/assets/imagenette_sit_baseline100_vs_repa350_ema_heun50.png`.

### Decision

Close this experiment. The best baseline is EMA 100k; the best final model is REPA EMA 350k. REPA improves final quality and coverage, but it does not demonstrate earlier convergence: it is not favorable at 100k and becomes favorable only later in the equal-step curve. The next implementation milestone is deterministic img2img followed by low-strength hires fix around the selected REPA 350k model.
