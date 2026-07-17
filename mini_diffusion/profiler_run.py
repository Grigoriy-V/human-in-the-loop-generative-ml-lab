from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

from mini_diffusion.benchmark import apply_overrides
from mini_diffusion.diffusion import EMA
from mini_diffusion.train import (
    build_diffusion,
    build_loader,
    build_model,
    build_optimizer,
    choose_device,
    choose_dtype,
    configure_backends,
    infinite_loader,
    load_config,
    move_images,
    move_labels,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one short training window.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/perf/profiler")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--warmup-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.get("seed", 123))
    configure_backends(cfg)
    device = choose_device()
    dtype, use_autocast = choose_dtype(device, cfg["train"].get("dtype", "bf16"))
    performance = cfg.get("performance", {})
    model = build_model(cfg).to(device).train()
    if performance.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    diffusion = build_diffusion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ema = EMA(
        model,
        decay=cfg["train"].get("ema_decay", 0.9999),
        foreach=bool(performance.get("ema_foreach", False)),
    )
    loader = infinite_loader(build_loader(cfg))

    def run_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with record_function("data_loader"):
            images, labels = next(loader)
        with record_function("cpu_to_cuda"):
            images = move_images(images, device, cfg)
            labels = move_labels(labels, device, cfg)
        with record_function("forward_and_loss"):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                loss = diffusion.loss(model, images, labels)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            if cfg["train"].get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
        with record_function("ema"):
            ema.update(model)

    for _ in range(args.warmup_steps):
        run_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(output)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(8):
            run_step()
            profiler.step()
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = profiler.key_averages().table(sort_by=sort_key, row_limit=50)
    (output / "key_averages.txt").write_text(table, encoding="utf-8")
    print(table)
    print(f"profiler_written: {output}")


if __name__ == "__main__":
    main()
