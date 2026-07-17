# Imagenette SiT-S/2 Performance Check

Date: 2026-07-17  
Base commit: `a8aa37b` (`Add Imagenette latent SiT baseline`)

## Environment

- GPU: NVIDIA GeForce RTX 4090 24 GB.
- PyTorch: `2.11.0+cu128`; CUDA runtime: `12.8`.
- Training dtype: BF16 autocast.
- Model: SiT-S/2, 32,527,888 trainable parameters, 6 heads, head dimension 64.

## Configuration Review

All fields in `imagenette_sit_s_128.yaml` are consumed by one of the pipeline entrypoints: data preparation consumes dataset/root/cache seed/preparation batch/VAE settings; training consumes cache/model/train/performance/sampling settings; `sample_sit.py` exposes explicit final sampler and step flags. No unused evaluation-sampler YAML field was added: `sample_sit.py --sampler heun --steps 50` remains the final/evaluation path.

The full configuration changed only as requested:

| Field | Before | After |
| --- | ---: | ---: |
| `train.max_steps` | 400000 | 100000 |
| `train.log_every` | 50 | 100 |
| `train.sample_every` | 5000 | 10000 |
| `sampling.preview_steps` | 50 | 25 |
| `sampling.preview_count` | 16 | 8 |

`batch_size`, architecture, learning rate, weight decay, EMA decay, and gradient accumulation are unchanged. The configuration now explicitly sets `pin_memory: true`, `non_blocking: true`, `persistent_workers: true`, and `prefetch_factor: 2`; `persistent_workers` and `prefetch_factor` are passed only when `num_workers > 0`.

## Attention And Optimizer

- Attention now calls `torch.nn.functional.scaled_dot_product_attention` directly.
- CUDA SDPA profiler selected `mem_efficient_sdp` for BF16, 6 heads, head dimension 64.
- Flash SDP is enabled as a global preference but this PyTorch wheel reports `Torch was not compiled with flash attention`; `can_use_flash_attention=False`. No third-party `flash-attn` package was installed.
- Memory-efficient SDP is eligible and selected.
- `AdamW(..., fused=True)` was accepted; benchmark reports `fused_adamw: true` with no fallback.
- `EMA(..., foreach=True)` is configured and uses the existing `_foreach_mul_` / `_foreach_add_` code path.
- Training and benchmark both use `optimizer.zero_grad(set_to_none=True)`.

## Benchmark

The required primary cache `outputs/imagenette_sit_s_128/latents/train.pt` did not exist. Creating or overwriting it was explicitly out of scope. Benchmarks therefore use the existing immutable debug cache at `outputs/imagenette_sit_s_128_debug/latents/train.pt`, repeated by `ConcatDataset` only to supply complete batches. It exercises the real cache tensor format, DataLoader worker mode, pinned/non-blocking copies, model, fused optimizer, BF16, and SDPA; it does not measure storage/cache-scale effects of the future full cache.

Each run was a separate process with 10 warmup and 50 measured steps. No checkpoint or cache was written.

| Batch | Workers | Images/s | ms/step | Peak allocated GB | Peak reserved GB | Attention | OOM |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 256 | 4 | 5821.19 | 43.98 | 3.98 | 4.07 | mem_efficient_sdp | no |
| 256 | 0 | 5766.34 | 44.40 | 3.98 | 4.07 | mem_efficient_sdp | no |
| 512 | 4 | 5461.77 | 93.74 | 7.51 | 7.62 | mem_efficient_sdp | no |

Decision: keep batch 256 and 4 workers. It is the fastest measured stable option. Batch 512 is slower and consumes 3.55 GB more reserved VRAM, so it is rejected. The 4-worker gain over zero workers is small but positive and does not increase GPU memory.

## Runtime Checks

- `python -m pytest mini_diffusion\tests -q`: `22 passed`.
- One-step debug training smoke completed on CUDA/BF16, loss `1.5915`, peak reserved VRAM `0.72 GB`. It wrote only the ignored debug checkpoint.
- One-batch overfit: average loss `1.437188` to `0.414213` after 40 updates.
- Decoded EMA Heun preview: 25 steps, 8 images, seeds 100-107, finite RGB output. Two identical runs produced SHA-256 `FA4390316512BE9819B1155C1A045F356B9328C274DF9990C97222815FA40F7B`.

## Not Performed

- At the time of this performance check, the primary cache did not exist and was intentionally not created. It was prepared later by an explicit user command: train `9469` and val `3925` finite FP16 latents. The benchmark results above remain debug-cache measurements and are not retroactively presented as full-cache loader measurements.
- No benchmark against a full-cache population was performed.
- No full training was started. No full-training checkpoint was created or modified.
