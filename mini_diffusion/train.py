from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from mini_diffusion.data import build_cifar10, build_tiny_imagenet
from mini_diffusion.diffusion import EMA, GaussianDiffusion, UNet
from mini_diffusion.sampling import denormalize_to_unit, make_generator, sample_statistics


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict):
    data = cfg["data"]
    if data["dataset"] == "cifar10":
        return build_cifar10(
            data["root"],
            train=True,
            download=True,
            mirror_url=data.get("mirror_url"),
            fake_data=data.get("fake_data", False),
            fake_size=data.get("fake_size", 128),
        )
    if data["dataset"] == "tiny_imagenet":
        return build_tiny_imagenet(data["root"], split="train", resolution=data["resolution"])
    raise ValueError(f"Unknown dataset: {data['dataset']}")


def build_loader(cfg: dict, shuffle: bool = True) -> DataLoader:
    data = cfg["data"]
    workers = int(data.get("num_workers", 0))
    kwargs = {
        "dataset": build_dataset(cfg),
        "batch_size": data["batch_size"],
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": data.get("pin_memory", torch.cuda.is_available()),
        "drop_last": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def infinite_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def configure_backends(cfg: dict) -> None:
    performance = cfg.get("performance", {})
    torch.backends.cudnn.benchmark = bool(performance.get("cudnn_benchmark", False))


def move_images(images: torch.Tensor, device: torch.device, cfg: dict) -> torch.Tensor:
    non_blocking = bool(cfg["data"].get("non_blocking", True))
    images = images.to(device, non_blocking=non_blocking)
    if cfg.get("performance", {}).get("channels_last", False):
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def move_labels(labels: torch.Tensor, device: torch.device, cfg: dict) -> torch.Tensor:
    return labels.to(device, non_blocking=bool(cfg["data"].get("non_blocking", True)))


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    kwargs = {
        "lr": cfg["train"]["learning_rate"],
        "weight_decay": cfg["train"].get("weight_decay", 0.0),
    }
    if cfg.get("performance", {}).get("fused_adamw", False):
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except (RuntimeError, TypeError):
        if "fused" not in kwargs:
            raise
        print("warning: fused AdamW unavailable; falling back to standard AdamW")
        kwargs.pop("fused")
        return torch.optim.AdamW(model.parameters(), **kwargs)


def prepare_train_model(
    model: nn.Module, cfg: dict, device: torch.device
) -> tuple[nn.Module, str]:
    compile_mode = cfg.get("performance", {}).get("compile_mode", "none")
    if compile_mode == "none":
        return model, "eager"
    if not hasattr(torch, "compile"):
        print("warning: torch.compile unavailable; using eager model")
        return model, "eager_fallback_unavailable"
    if device.type == "cuda" and importlib.util.find_spec("triton") is None:
        print("warning: torch.compile CUDA backend requires Triton; using eager model")
        return model, "eager_fallback_missing_triton"
    try:
        return torch.compile(model, mode=compile_mode), f"compiled_{compile_mode}"
    except Exception as exc:
        print(f"warning: torch.compile setup failed; using eager model: {exc}")
        return model, "eager_fallback_setup_error"
def build_model(cfg: dict) -> UNet:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return UNet(
        image_size=data_cfg["resolution"],
        in_channels=3,
        out_channels=3,
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=tuple(model_cfg["attention_resolutions"]),
        dropout=model_cfg.get("dropout", 0.0),
        num_classes=data_cfg.get("num_classes"),
        class_cond=model_cfg.get("class_cond", True),
        cond_drop_prob=model_cfg.get("cond_drop_prob", 0.0),
        num_heads=model_cfg.get("num_heads", 1),
        attention_backend=cfg.get("performance", {}).get("attention_backend", "manual"),
    )


def build_diffusion(cfg: dict) -> GaussianDiffusion:
    return GaussianDiffusion(
        steps=cfg["diffusion"]["steps"],
        schedule=cfg["diffusion"].get("schedule", "cosine"),
    )


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(device: torch.device, requested: str) -> tuple[torch.dtype, bool]:
    if device.type == "cuda" and requested == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True
    if device.type == "cuda" and requested == "fp16":
        return torch.float16, True
    return torch.float32, False


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    cfg: dict,
    global_step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "config": cfg,
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMA | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[dict, int]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state"]])
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt:
        random.setstate(ckpt["python_rng_state"])
    return ckpt.get("config", {}), int(ckpt.get("global_step", 0))


@torch.inference_mode()
def write_samples(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    ema: EMA,
    cfg: dict,
    device: torch.device,
    step: int,
) -> Path:
    out_dir = Path(cfg["output_dir"]) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = cfg["train"].get("sample_count", 4)
    labels = torch.arange(count, device=device) % cfg["data"].get("num_classes", count)
    sampling_cfg = cfg.get("sampling", {})
    seed = int(sampling_cfg.get("preview_seed", cfg.get("seed", 123)))
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.5))
    sampler = str(sampling_cfg.get("preview_sampler", "ddpm"))
    ddim_steps = int(sampling_cfg.get("preview_steps", 50))
    generator = make_generator(device, seed)
    was_training = model.training
    try:
        with ema.average_parameters(model):
            model.eval()
            images = diffusion.sample(
                model,
                (count, 3, cfg["data"]["resolution"], cfg["data"]["resolution"]),
                labels=labels,
                guidance_scale=guidance_scale,
                device=device,
                generator=generator,
                sampler=sampler,
                ddim_steps=ddim_steps,
            )
    finally:
        model.train(was_training)
    stats = sample_statistics(images)
    if sampling_cfg.get("diagnostics", False):
        print("sample_diagnostics: " + json.dumps(stats, sort_keys=True))
    path = out_dir / f"step_{step:07d}.png"
    save_image(denormalize_to_unit(images), path, nrow=int(math.sqrt(count)) or 1)
    return path


def print_runtime_summary(cfg: dict, model: nn.Module, device: torch.device, dtype: torch.dtype, autocast: bool) -> None:
    batch = cfg["data"]["batch_size"]
    accum = cfg["train"].get("grad_accum_steps", 1)
    print(f"trainable_parameters: {trainable_parameters(model):,}")
    print(f"device: {device}")
    print(f"dtype: {dtype} autocast={autocast}")
    print(f"batch_size: {batch}")
    print(f"effective_batch_size: {batch * accum}")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"cuda_vram_total_gb: {total:.2f}")
        print(f"cuda_vram_peak_gb: {peak:.2f}")


def train(cfg: dict, resume: str | None = None) -> Path:
    set_seed(cfg.get("seed", 123))
    configure_backends(cfg)
    device = choose_device()
    dtype, use_autocast = choose_dtype(device, cfg["train"].get("dtype", "bf16"))
    model = build_model(cfg).to(device)
    performance = cfg.get("performance", {})
    if performance.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    diffusion = build_diffusion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ema = EMA(
        model,
        decay=cfg["train"].get("ema_decay", 0.9999),
        foreach=bool(performance.get("ema_foreach", False)),
    )
    global_step = 0
    if resume:
        _, global_step = load_checkpoint(resume, model, optimizer, ema, map_location=device)
        print(f"resumed_from: {resume}")
        print(f"resume_global_step: {global_step}")

    print_runtime_summary(cfg, model, device, dtype, use_autocast)
    loader = infinite_loader(build_loader(cfg))
    out_dir = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    writer = SummaryWriter(out_dir / "logs")
    max_steps = cfg["train"]["max_steps"]
    accum = cfg["train"].get("grad_accum_steps", 1)
    latest_path = ckpt_dir / "latest.pt"
    train_model, compile_status = prepare_train_model(model, cfg, device)
    print(f"compile_status: {compile_status}")

    model.train()
    pbar = tqdm(total=max_steps, initial=global_step, desc=cfg["name"])
    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for _ in range(accum):
            images, labels = next(loader)
            images = move_images(images, device, cfg)
            labels = move_labels(labels, device, cfg)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                loss = diffusion.loss(train_model, images, labels) / accum
            loss.backward()
            total_loss += loss.detach() * accum
        if cfg["train"].get("grad_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()
        ema.update(model)
        global_step += 1
        pbar.update(1)
        scalar_log_every = int(performance.get("scalar_log_every", 1))
        log_value = None
        if global_step % scalar_log_every == 0:
            log_value = float(total_loss.cpu())
            writer.add_scalar("train/loss", log_value, global_step)
            if device.type == "cuda":
                writer.add_scalar(
                    "system/cuda_peak_gb",
                    torch.cuda.max_memory_allocated(device) / 1024**3,
                    global_step,
                )
        if global_step % cfg["train"].get("log_every", 50) == 0:
            if log_value is None:
                log_value = float(total_loss.cpu())
            pbar.set_postfix(loss=f"{log_value:.4f}")
        if global_step % cfg["train"].get("sample_every", 1000) == 0:
            sample_path = write_samples(model, diffusion, ema, cfg, device, global_step)
            print(f"sample_written: {sample_path}")
        if global_step % cfg["train"].get("save_every", 1000) == 0:
            save_checkpoint(latest_path, model, optimizer, ema, cfg, global_step)
            print(f"checkpoint_written: {latest_path}")

    save_checkpoint(latest_path, model, optimizer, ema, cfg, global_step)
    writer.close()
    pbar.close()
    if device.type == "cuda":
        print(f"cuda_vram_peak_gb_final: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f}")
    print(f"checkpoint_written: {latest_path}")
    return latest_path


def overfit_smoke(cfg: dict, updates: int = 40) -> None:
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["num_workers"] = 0
    set_seed(cfg.get("seed", 123))
    configure_backends(cfg)
    device = choose_device()
    dtype, use_autocast = choose_dtype(device, cfg["train"].get("dtype", "bf16"))
    model = build_model(cfg).to(device)
    if cfg.get("performance", {}).get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    diffusion = build_diffusion(cfg).to(device)
    smoke_cfg = json.loads(json.dumps(cfg))
    smoke_cfg["train"]["learning_rate"] = max(cfg["train"]["learning_rate"], 1e-3)
    smoke_cfg["train"]["weight_decay"] = 0.0
    optimizer = build_optimizer(model, smoke_cfg)
    images, labels = next(iter(build_loader(cfg, shuffle=False)))
    images = move_images(images, device, cfg)
    labels = move_labels(labels, device, cfg)
    t = torch.randint(0, diffusion.steps, (images.shape[0],), device=device, dtype=torch.long)
    noise = torch.randn_like(images)
    xt = diffusion.q_sample(images, t, noise)
    losses = []
    model.train()
    print_runtime_summary(cfg, model, device, dtype, use_autocast)
    for _ in tqdm(range(updates), desc="one_batch_overfit"):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
            pred = model(xt, t, labels)
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    first = sum(losses[: max(1, updates // 4)]) / max(1, updates // 4)
    last = sum(losses[-max(1, updates // 4) :]) / max(1, updates // 4)
    print(f"overfit_loss_first_avg: {first:.6f}")
    print(f"overfit_loss_last_avg: {last:.6f}")
    if not math.isfinite(last) or last >= first * 0.95:
        raise RuntimeError("One-batch overfit smoke failed: loss did not noticeably decrease.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--overfit-smoke", action="store_true")
    parser.add_argument("--overfit-updates", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.overfit_smoke:
        overfit_smoke(cfg, updates=args.overfit_updates)
    else:
        train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
