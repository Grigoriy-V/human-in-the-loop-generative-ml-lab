from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch import nn

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
    prepare_train_model,
    set_seed,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", default="benchmark")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--log", default="reports/performance_experiments.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--disable-gpu-utilization", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value: {override}")
        dotted_key, raw_value = override.split("=", 1)
        target = cfg
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = yaml.safe_load(raw_value)
    return cfg


def spawn_runs(args: argparse.Namespace) -> bool:
    if args.runs == 1:
        return False
    if args.runs < 1:
        raise ValueError("runs must be positive")
    child_args = []
    skip_next = False
    for item in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if item == "--runs":
            skip_next = True
            continue
        if item.startswith("--runs=") or item == "--run-index" or item.startswith("--run-index="):
            if item == "--run-index":
                skip_next = True
            continue
        child_args.append(item)
    for run_index in range(args.runs):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *child_args,
            "--runs",
            "1",
            "--run-index",
            str(run_index),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    return True


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


class GpuUtilizationSampler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.samples: list[float] = []
        self.memory_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                utilization, memory = output.splitlines()[0].split(",")
                self.samples.append(float(utilization.strip()))
                self.memory_mb.append(float(memory.strip()))
            except (OSError, ValueError, subprocess.CalledProcessError, IndexError):
                self.enabled = False
                return
            self._stop.wait(0.25)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def summarize(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_mean": None, f"{prefix}_median": None, f"{prefix}_p95": None}
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def elapsed_seconds(pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[float]:
    return [start.elapsed_time(end) / 1000.0 for start, end in pairs]


def append_jsonl(path: str | Path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if spawn_runs(args):
        return
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("warmup-steps and steps must be positive")

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
    train_model, compile_status = prepare_train_model(model, cfg, device)
    loader = infinite_loader(build_loader(cfg))
    accum = int(cfg["train"].get("grad_accum_steps", 1))
    scalar_sync_every = int(performance.get("scalar_log_every", 1))

    def run_step(measure: bool, step_index: int) -> dict:
        optimizer.zero_grad(set_to_none=True)
        data_seconds = 0.0
        event_groups: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        if measure and device.type == "cuda":
            for name in ("step", "transfer", "optimizer", "ema"):
                event_groups[name] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
        for micro_step in range(accum):
            data_start = time.perf_counter()
            images, labels = next(loader)
            data_seconds += time.perf_counter() - data_start
            if measure and device.type == "cuda" and micro_step == 0:
                event_groups["step"][0].record()
                event_groups["transfer"][0].record()
            images = move_images(images, device, cfg)
            labels = move_labels(labels, device, cfg)
            if measure and device.type == "cuda" and micro_step == 0:
                event_groups["transfer"][1].record()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                loss = diffusion.loss(train_model, images, labels) / accum
            loss.backward()
        if measure and device.type == "cuda":
            event_groups["optimizer"][0].record()
        if cfg["train"].get("grad_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()
        if measure and device.type == "cuda":
            event_groups["optimizer"][1].record()
            event_groups["ema"][0].record()
        ema.update(model)
        if measure and device.type == "cuda":
            event_groups["ema"][1].record()
            event_groups["step"][1].record()
        if scalar_sync_every > 0 and (step_index + 1) % scalar_sync_every == 0:
            float(loss.detach().cpu())
        return {
            "data_seconds": data_seconds,
            "events": event_groups,
            "finite": torch.isfinite(loss.detach()),
        }

    for warmup_index in range(args.warmup_steps):
        run_step(False, warmup_index)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    sampler = GpuUtilizationSampler(device.type == "cuda" and not args.disable_gpu_utilization)
    sampler.start()
    measured_start = time.perf_counter()
    data_times: list[float] = []
    event_pairs = {name: [] for name in ("step", "transfer", "optimizer", "ema")}
    finite = torch.ones((), device=device, dtype=torch.bool)
    for step_index in range(args.steps):
        step_result = run_step(True, step_index)
        data_times.append(step_result["data_seconds"])
        finite.logical_and_(step_result["finite"])
        for name, pair in step_result["events"].items():
            event_pairs[name].append(pair)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_seconds = time.perf_counter() - measured_start
    sampler.stop()

    batch_size = int(cfg["data"]["batch_size"])
    effective_batch_size = batch_size * accum
    result = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config": str(args.config),
        "overrides": args.overrides,
        "run_index": args.run_index,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "measured_seconds": measured_seconds,
        "iterations_per_second": args.steps / measured_seconds,
        "images_per_second": args.steps * effective_batch_size / measured_seconds,
        "wall_step_seconds_mean": measured_seconds / args.steps,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(dtype),
        "autocast": use_autocast,
        "compile_status": compile_status,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "trainable_parameters": trainable_parameters(model),
        "finite_loss": bool(finite.cpu()),
        "cuda_peak_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None
        ),
        "gpu_utilization_mean": statistics.fmean(sampler.samples) if sampler.samples else None,
        "gpu_utilization_median": statistics.median(sampler.samples) if sampler.samples else None,
        "gpu_memory_used_mb_mean": statistics.fmean(sampler.memory_mb) if sampler.memory_mb else None,
        "result": "unsupported" if compile_status.startswith("eager_fallback") else "measured",
        "keep_reject": "pending",
        "notes": "",
    }
    result.update(summarize(data_times, "data_seconds"))
    if device.type == "cuda":
        for name, pairs in event_pairs.items():
            result.update(summarize(elapsed_seconds(pairs), f"{name}_seconds"))

    append_jsonl(args.log, result)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"benchmark_appended: {args.log}")


if __name__ == "__main__":
    main()
