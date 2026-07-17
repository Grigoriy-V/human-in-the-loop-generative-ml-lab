# Tiny ImageNet Partial Training Snapshot

## Status

Training was intentionally stopped after the latest saved checkpoint at step 150,000 of the configured 400,000-step budget. This is **37.5% complete**. It is a partial run, not a completed Tiny ImageNet training result.

The checkpoint was produced from source commit `51bd5d54980a1b190a01a4f84a688e6d310d1f9b`. The code configuration matches the checkpoint configuration semantically; the frozen snapshot is `reports/tiny_imagenet_partial_config.yaml`.

## Checkpoint Verification

- Checkpoint: `outputs/tiny_imagenet/checkpoints/latest.pt`
- Global step: 150,000
- SHA-256: `2880B1066D337A4D1F68F6850AE7A8E20D76CE94B6EBAAE44F601DDE39A39211`
- Checkpoint load: passed
- Model tensors: finite
- EMA tensors: finite
- Optimizer state entries restored: 279

## Training Snapshot

- Loss at saved checkpoint step 150,000: `0.135029`
- Effective batch size: 128 (`64 x 2`)
- Recent 10,000-step median speed: `3.7236 optimizer steps/s`, `476.63 images/s`
- Recent 10,000-step mean speed: `3.6972 optimizer steps/s`, `473.24 images/s`
- CUDA peak recorded at checkpoint: `17.78 GB`

TensorBoard contains a loss record at step 150,100 because training continued briefly after the step-150,000 checkpoint. Those additional 100 steps were not saved and are not part of this snapshot.

Machine-readable metrics: `reports/tiny_imagenet_partial_metrics.json`.

## Fixed EMA Sample

The checkpoint was sampled with EMA weights, deterministic DDIM-50, guidance scale 1.5, classes 0 through 15, and seeds 150000 through 150015. Sampling used FP32 and produced finite output with no fully black or white images.

![Fixed EMA Tiny ImageNet grid at step 150000](../docs/assets/tiny_imagenet_step_0150000_ema_ddim50.png)

- Asset SHA-256: `8D129C0CBD1D2FC6523407A588ACDD64CA6D342A6C9FA9E5DE69CE34FB3FB29B`
- Statistics: min -1.0, max 1.0, mean 0.188003, std 0.658709, saturation 0.011826

Sampling metadata remains local beside the generated grid in `outputs/tiny_imagenet_partial/ema_fixed_grid/`.

## Commands Actually Run

```powershell
.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\tiny_imagenet\checkpoints\latest.pt --output outputs\tiny_imagenet_partial\ema_fixed_grid --num-images 16 --classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --seeds 150000 150001 150002 150003 150004 150005 150006 150007 150008 150009 150010 150011 150012 150013 150014 150015 --guidance-scale 1.5 --weights ema --sampler ddim --ddim-steps 50 --grid-size 4 --grid-filename tiny_imagenet_step_0150000_ema_ddim50.png --metadata-filename tiny_imagenet_step_0150000_ema_ddim50.json --sampling-diagnostics
```

An isolated Python check also rebuilt the model and optimizer from the saved configuration, loaded model/EMA/optimizer/RNG state, and asserted finite model and EMA tensors.

## Decision

Keep the checkpoint and this metadata as the partial 150k snapshot. Do not treat it as a final quality result. Resume from `outputs/tiny_imagenet/checkpoints/latest.pt` only when continuing the same 400,000-step experiment.
