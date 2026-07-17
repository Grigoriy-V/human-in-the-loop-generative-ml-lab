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

from mini_diffusion.diffusion import EMA
from mini_diffusion.train import (
    build_diffusion,
    build_model,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    write_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="reports/cifar10_preview_overhead.json")
    parser.add_argument("--work-dir", default="outputs/perf/preview_overhead")
    parser.add_argument("--sample-count", type=int, default=16)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    cfg["output_dir"] = args.work_dir
    cfg["train"]["sample_count"] = args.sample_count
    cfg.setdefault("sampling", {})["preview_seed"] = 123
    cfg["sampling"]["guidance_scale"] = 1.5
    cfg["sampling"]["diagnostics"] = True

    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.9999))
    _, global_step = load_checkpoint(args.checkpoint, model, optimizer, ema, map_location=device)
    timings = {}
    for name, sampler, steps in (
        ("ddim25", "ddim", 25),
        ("ddim50", "ddim", 50),
        ("ddpm1000", "ddpm", diffusion.steps),
    ):
        cfg["sampling"]["preview_sampler"] = sampler
        cfg["sampling"]["preview_steps"] = steps
        cfg["output_dir"] = str(Path(args.work_dir) / name)
        synchronize(device)
        started = time.perf_counter()
        path = write_samples(model, diffusion, ema, cfg, device, global_step)
        synchronize(device)
        timings[name] = {
            "seconds": time.perf_counter() - started,
            "sampler": sampler,
            "steps": steps,
            "sample_count": args.sample_count,
            "png": str(path),
        }

    checkpoint_path = Path(args.work_dir) / "checkpoint_save" / "latest.pt"
    synchronize(device)
    started = time.perf_counter()
    save_checkpoint(checkpoint_path, model, optimizer, ema, cfg, global_step)
    synchronize(device)
    checkpoint_seconds = time.perf_counter() - started
    timings["checkpoint_save"] = {
        "seconds": checkpoint_seconds,
        "bytes": checkpoint_path.stat().st_size,
        "path": str(checkpoint_path),
    }
    timings["tensorboard_images"] = {
        "seconds": 0.0,
        "enabled_in_training": False,
        "note": "The training loop writes scalars only; PNG preview time is included above.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(json.dumps(timings, indent=2))
    print(f"preview_benchmark_written: {output}")


if __name__ == "__main__":
    main()
