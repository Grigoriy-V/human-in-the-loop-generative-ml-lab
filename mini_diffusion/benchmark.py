from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import cycle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from mini_diffusion.diffusion import EMA
from mini_diffusion.train import (
    build_diffusion,
    build_loader,
    build_model,
    choose_device,
    choose_dtype,
    load_config,
    set_seed,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default="reports/cifar10_baseline_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("warmup-steps and steps must be positive")

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 123))
    device = choose_device()
    dtype, use_autocast = choose_dtype(device, cfg["train"].get("dtype", "bf16"))
    model = build_model(cfg).to(device).train()
    diffusion = build_diffusion(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.9999))
    loader = cycle(build_loader(cfg))
    accum = cfg["train"].get("grad_accum_steps", 1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_step(measure_data: bool = False) -> float:
        optimizer.zero_grad(set_to_none=True)
        data_seconds = 0.0
        for _ in range(accum):
            data_start = time.perf_counter()
            images, labels = next(loader)
            data_seconds += time.perf_counter() - data_start
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                loss = diffusion.loss(model, images, labels) / accum
            loss.backward()
        if cfg["train"].get("grad_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()
        ema.update(model)
        if not torch.isfinite(loss):
            raise FloatingPointError("Benchmark loss became NaN or Inf")
        return data_seconds if measure_data else 0.0

    for _ in range(args.warmup_steps):
        run_step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_start = time.perf_counter()
    data_seconds = 0.0
    for _ in range(args.steps):
        data_seconds += run_step(measure_data=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_seconds = time.perf_counter() - measured_start

    batch_size = cfg["data"]["batch_size"]
    effective_batch_size = batch_size * accum
    result = {
        "config": str(args.config),
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "measured_seconds": measured_seconds,
        "average_step_seconds": measured_seconds / args.steps,
        "iterations_per_second": args.steps / measured_seconds,
        "images_per_second": args.steps * effective_batch_size / measured_seconds,
        "average_data_loading_seconds": data_seconds / args.steps,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "autocast": use_autocast,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "trainable_parameters": trainable_parameters(model),
        "cuda_peak_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"benchmark_written: {output}")


if __name__ == "__main__":
    main()
