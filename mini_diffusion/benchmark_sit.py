from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from mini_diffusion.latent_cache import load_cache
from mini_diffusion.sit import linear_interpolant, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, device_dtype, load_config


def attention_backend(model, x0: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype, autocast: bool) -> str:
    t = torch.rand(x0.shape[0], device=x0.device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with torch.autocast(device_type=x0.device.type, dtype=dtype, enabled=autocast):
            model(x0, t, labels).float().sum().backward()
    names = [event.key for event in profiler.key_averages()]
    matches = [name for name in names if "scaled_dot_product" in name]
    if not matches:
        return "not_detected"
    if any("flash" in name for name in matches):
        return "flash_sdp"
    if any("efficient" in name for name in matches):
        return "mem_efficient_sdp"
    return "math_sdp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-path", required=True, help="Existing cache only; this command never writes it.")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["batch_size"] = args.batch
    cfg["data"]["num_workers"] = args.workers
    cfg["data"]["persistent_workers"] = bool(args.workers)
    device, dtype, autocast = device_dtype(cfg)
    cache = load_cache(args.cache_path, expected_resolution=cfg["data"]["resolution"], expected_vae_model_id=cfg["vae"]["model_id"])
    base = TensorDataset(cache["latents"], cache["labels"])
    repeats = (args.batch * (args.warmup + args.steps) + len(base) - 1) // len(base)
    dataset = ConcatDataset([base] * repeats)
    loader_kwargs = {"batch_size": args.batch, "shuffle": False, "drop_last": True, "num_workers": args.workers, "pin_memory": bool(cfg["data"].get("pin_memory", False))}
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    fused_active = bool(optimizer.defaults.get("fused", False))
    iterator = iter(loader)

    def update() -> None:
        x0, labels = next(iterator)
        x0 = x0.to(device, non_blocking=bool(cfg["data"].get("non_blocking", True))).float()
        labels = labels.to(device, non_blocking=bool(cfg["data"].get("non_blocking", True)))
        optimizer.zero_grad(set_to_none=True)
        noise, t = torch.randn_like(x0), torch.rand(x0.shape[0], device=device)
        xt, target = linear_interpolant(x0, noise, t)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
            loss = velocity_loss(model(xt, t, labels), target)
        loss.backward()
        optimizer.step()

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        probe_x = torch.randn(args.batch, 4, 16, 16, device=device)
        probe_labels = torch.arange(args.batch, device=device) % cfg["data"]["num_classes"]
        backend = attention_backend(model, probe_x, probe_labels, dtype, autocast)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.warmup):
            update()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.steps):
            update()
        torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - started) / args.steps
        result = {"batch": args.batch, "workers": args.workers, "warmup_steps": args.warmup, "measured_steps": args.steps,
                  "images_per_second": args.batch / seconds, "ms_per_step": seconds * 1000,
                  "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                  "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                  "attention_backend": backend, "fused_adamw": fused_active,
                  "foreach_ema_configured": bool(cfg["performance"].get("ema_foreach", False)), "oom": False}
    except torch.OutOfMemoryError:
        result = {"batch": args.batch, "workers": args.workers, "attention_backend": "not_measured", "oom": True}
        torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
