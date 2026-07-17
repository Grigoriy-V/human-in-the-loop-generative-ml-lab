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

from mini_diffusion.benchmark_sit import attention_backend
from mini_diffusion.repa import repa_loss
from mini_diffusion.sit import linear_interpolant, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, build_projector, device_dtype, load_config, make_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--mode", choices=("baseline", "repa"), required=True)
    parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--steps", type=int, default=50); parser.add_argument("--output", required=True)
    args = parser.parse_args(); cfg = load_config(args.config); repa_enabled = args.mode == "repa"
    device, dtype, autocast = device_dtype(cfg); model = build_model(cfg).to(device); projector = build_projector(cfg, device) if repa_enabled else None
    modules = torch.nn.ModuleList([model] + ([projector] if projector else [])); optimizer = build_optimizer(modules, cfg); loader, _ = make_loader(cfg, repa_enabled); iterator = iter(loader)

    def update() -> None:
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x0, labels = batch[0].to(device, non_blocking=cfg["data"].get("non_blocking", True)).float(), batch[1].to(device, non_blocking=cfg["data"].get("non_blocking", True)); features = batch[2].to(device, non_blocking=cfg["data"].get("non_blocking", True)) if projector else None
        optimizer.zero_grad(set_to_none=True); noise, t = torch.randn_like(x0), torch.rand(x0.shape[0], device=device); xt, target = linear_interpolant(x0, noise, t)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
            if projector:
                velocity, hidden = model(xt, t, labels, return_hidden_after=cfg["repa"]["alignment_depth"])
                loss = velocity_loss(velocity, target) + float(cfg["repa"]["coefficient"]) * repa_loss(projector(hidden), features)[0]
            else: loss = velocity_loss(model(xt, t, labels), target)
        loss.backward(); optimizer.step()

    result: dict[str, object] = {"mode": args.mode, "batch": cfg["data"]["batch_size"], "workers": cfg["data"]["num_workers"], "warmup_steps": args.warmup, "measured_steps": args.steps, "oom": False}
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        probe = torch.randn(cfg["data"]["batch_size"], 4, 16, 16, device=device); labels = torch.arange(cfg["data"]["batch_size"], device=device) % cfg["data"]["num_classes"]
        result["attention_backend"] = attention_backend(model, probe, labels, dtype, autocast)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.warmup): update()
        torch.cuda.synchronize(device); started = time.perf_counter()
        for _ in range(args.steps): update()
        torch.cuda.synchronize(device); seconds = (time.perf_counter() - started) / args.steps
        result.update({"images_per_second": cfg["data"]["batch_size"] / seconds, "ms_per_step": seconds * 1000, "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3, "fused_adamw": bool(optimizer.defaults.get("fused", False)), "projector_parameters": sum(p.numel() for p in projector.parameters()) if projector else 0})
    except torch.OutOfMemoryError:
        result["oom"] = True; torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
