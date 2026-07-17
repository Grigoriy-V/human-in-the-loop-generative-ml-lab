from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from mini_diffusion.diffusion import EMA
from mini_diffusion.latent_cache import cache_fingerprint, load_cache
from mini_diffusion.repa import LatentFeatureDataset, RepaProjector, repa_loss
from mini_diffusion.sit import SiT, linear_interpolant, sample_ode, velocity_loss
from mini_diffusion.vae import decode_latents, load_frozen_vae


def load_config(path: str | Path) -> dict: return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def device_dtype(cfg: dict) -> tuple[torch.device, torch.dtype, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and cfg["train"].get("dtype") == "bf16" and torch.cuda.is_bf16_supported()
    return device, torch.bfloat16 if use_bf16 else torch.float32, use_bf16
def build_model(cfg: dict) -> SiT:
    m, d = cfg["model"], cfg["data"]
    return SiT(input_size=d["latent_resolution"], in_channels=4, patch_size=m["patch_size"], hidden_size=m["hidden_size"], depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"], num_classes=d["num_classes"], cond_drop_prob=m["cond_drop_prob"])
def build_projector(cfg: dict, device: torch.device | None = None) -> RepaProjector | None:
    repa = cfg.get("repa", {})
    if not repa.get("enabled", False):
        return None
    # Projector construction must not perturb the RNG stream used by SiT/noise/loader.
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        projector = RepaProjector(cfg["model"]["hidden_size"], repa["projector_hidden_dim"], repa["teacher_feature_dim"])
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return projector.to(device) if device is not None else projector
def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    kwargs = {"lr": cfg["train"]["learning_rate"], "weight_decay": cfg["train"].get("weight_decay", 0.0)}
    if cfg.get("performance", {}).get("fused_adamw", False): kwargs["fused"] = True
    try: return torch.optim.AdamW(model.parameters(), **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None); print("warning: fused AdamW unavailable; using AdamW"); return torch.optim.AdamW(model.parameters(), **kwargs)
def make_loader(cfg: dict, repa_enabled: bool = False) -> tuple[DataLoader, dict]:
    payload = load_cache(Path(cfg["data"]["cache_dir"]) / "train.pt", expected_resolution=cfg["data"]["resolution"], expected_vae_model_id=cfg["vae"]["model_id"])
    cache_limit = cfg["data"].get("cache_limit")
    if cache_limit is not None:
        payload = {**payload, "latents": payload["latents"][:cache_limit], "labels": payload["labels"][:cache_limit], "relative_paths": payload["relative_paths"][:cache_limit]}
    workers = int(cfg["data"].get("num_workers", 0))
    loader_kwargs = {"batch_size": cfg["data"]["batch_size"], "shuffle": True, "drop_last": True,
                     "num_workers": workers, "pin_memory": bool(cfg["data"].get("pin_memory", torch.cuda.is_available()))}
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    dataset = LatentFeatureDataset(payload, Path(cfg["repa"]["feature_cache_dir"]) / "train") if repa_enabled else TensorDataset(payload["latents"], payload["labels"])
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, payload
def infinite(loader):
    while True: yield from loader
def save_checkpoint(path: Path, model, optimizer, ema, cfg, step: int, fingerprint: str, projector: nn.Module | None = None, feature_metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"architecture": "SiT-S/2 velocity v1", "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "global_step": step, "config": cfg, "cache_fingerprint": fingerprint, "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate()}
    if projector is not None:
        payload.update({"repa_format_version": 1, "repa_projector": projector.state_dict(), "repa": cfg["repa"], "feature_cache_metadata": feature_metadata})
    torch.save(payload, path)
def resume_checkpoint(path: str, model, optimizer, ema, device, expected_cache_fingerprint: str | None = None, projector: nn.Module | None = None, expected_feature_fingerprint: str | None = None) -> int:
    # Keep CPU RNG state on CPU; map_location=device would incorrectly move it to CUDA.
    ckpt = torch.load(path, map_location="cpu", weights_only=False); model.load_state_dict(ckpt["model"]); ema.load_state_dict(ckpt["ema"]); optimizer.load_state_dict(ckpt["optimizer"]); torch.set_rng_state(ckpt["torch_rng_state"].cpu()); np.random.set_state(ckpt["numpy_rng_state"]); random.setstate(ckpt["python_rng_state"])
    if expected_cache_fingerprint is not None and ckpt.get("cache_fingerprint") != expected_cache_fingerprint:
        raise ValueError("Checkpoint cache fingerprint does not match the current latent cache")
    if projector is not None:
        if "repa_projector" not in ckpt:
            raise ValueError("REPA resume requires a checkpoint created by REPA training; baseline init_from is forbidden")
        projector.load_state_dict(ckpt["repa_projector"])
        cache_metadata = ckpt.get("feature_cache_metadata") or {}
        if expected_feature_fingerprint is not None and cache_metadata.get("fingerprint") != expected_feature_fingerprint:
            raise ValueError("Checkpoint feature cache fingerprint does not match the current REPA cache")
    for name, parameter in model.named_parameters():
        if name in ema.shadow:
            ema.shadow[name] = ema.shadow[name].to(parameter.device, dtype=parameter.dtype)
    if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all([x.cpu() for x in ckpt["cuda_rng_state"]])
    return int(ckpt["global_step"])
def overfit_smoke(cfg: dict, updates: int) -> None:
    set_seed(cfg.get("seed", 123)); device, dtype, autocast = device_dtype(cfg); model = build_model(cfg).to(device); projector = build_projector(cfg, device); modules = list(model.parameters()) + (list(projector.parameters()) if projector else []); optimizer = torch.optim.AdamW(modules, lr=max(cfg["train"]["learning_rate"], 1e-3)); loader, _ = make_loader(cfg, projector is not None); batch = next(iter(loader)); x0, labels = batch[0].to(device).float(), batch[1].to(device); features = batch[2].to(device) if projector else None; noise, t = torch.randn_like(x0), torch.linspace(0.1, 0.9, x0.shape[0], device=device); xt, target = linear_interpolant(x0, noise, t); losses = []; cosines = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
            if projector:
                velocity, hidden = model(xt, t, labels, return_hidden_after=cfg["repa"]["alignment_depth"])
                flow_loss = velocity_loss(velocity, target); align_loss, cosine = repa_loss(projector(hidden), features)
                loss = flow_loss + float(cfg["repa"]["coefficient"]) * align_loss; cosines.append(float(cosine.detach()))
            else: loss = velocity_loss(model(xt, t, labels), target)
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    first, last = sum(losses[:updates // 4]) / (updates // 4), sum(losses[-updates // 4:]) / (updates // 4); print(f"overfit_loss_first_avg: {first:.6f}\noverfit_loss_last_avg: {last:.6f}")
    if projector: print(f"overfit_cosine_first_avg: {sum(cosines[:updates // 4]) / (updates // 4):.6f}\noverfit_cosine_last_avg: {sum(cosines[-updates // 4:]) / (updates // 4):.6f}")
    if not math.isfinite(last) or last >= first * 0.8: raise RuntimeError("One-batch overfit failed")
@torch.inference_mode()
def write_latent_preview(model, ema, cfg: dict, device: torch.device, step: int) -> Path:
    sampling = cfg.get("sampling", {}); count = int(sampling.get("preview_count", 4)); labels = torch.arange(count, device=device) % cfg["data"]["num_classes"]
    generator = torch.Generator(device=device).manual_seed(int(sampling.get("preview_seed", cfg.get("seed", 123))))
    with ema.average_parameters(model):
        latents = sample_ode(model, (count, 4, cfg["data"]["latent_resolution"], cfg["data"]["latent_resolution"]), labels, device, steps=int(sampling.get("preview_steps", 50)), sampler=sampling.get("preview_sampler", "heun"), guidance_scale=float(sampling.get("guidance_scale", 1.5)), generator=generator, diagnostics=True).cpu()
    path = Path(cfg["output_dir"]) / "latent_previews" / f"step_{step:07d}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save({"latents": latents, "labels": labels.cpu(), "seed": sampling.get("preview_seed", cfg.get("seed", 123)), "step": step}, path)
    if not sampling.get("preview_decode", False):
        return path
    vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision"))
    try:
        images = decode_latents(vae, latents.to(device)).cpu()
        pixels = ((images + 1.0) * 0.5).clamp(0, 1)
        if pixels.shape != (count, 3, cfg["data"]["resolution"], cfg["data"]["resolution"]) or not torch.isfinite(pixels).all():
            raise RuntimeError("Decoded preview is non-finite or has an invalid shape")
        image_path = Path(cfg["output_dir"]) / "samples" / f"step_{step:07d}_ema_{sampling.get('preview_sampler', 'heun')}{sampling.get('preview_steps', 50)}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(pixels, image_path, nrow=max(1, int(count**0.5)))
        print(f"decoded_preview_written: {image_path}")
    finally:
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return path
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--resume"); parser.add_argument("--max-steps", type=int); parser.add_argument("--overfit-smoke", action="store_true"); parser.add_argument("--overfit-updates", type=int, default=40); args = parser.parse_args()
    cfg = load_config(args.config)
    if args.overfit_smoke: overfit_smoke(cfg, args.overfit_updates); return
    if cfg.get("train", {}).get("init_from"):
        raise ValueError("init_from is forbidden for this SiT+REPA experiment; train from scratch or resume its own REPA checkpoint")
    max_steps = args.max_steps or cfg["train"]["max_steps"]; set_seed(cfg.get("seed", 123)); device, dtype, autocast = device_dtype(cfg)
    torch.backends.cudnn.benchmark = bool(cfg.get("performance", {}).get("cudnn_benchmark", False)); model = build_model(cfg).to(device); projector = build_projector(cfg, device); trainable = nn.ModuleList([model] + ([projector] if projector else [])); optimizer = build_optimizer(trainable, cfg); ema = EMA(model, cfg["train"]["ema_decay"], foreach=cfg.get("performance", {}).get("ema_foreach", False)); loader, payload = make_loader(cfg, projector is not None); feature_metadata = getattr(loader.dataset, "metadata", None)
    if projector is not None:
        # Open/validate cache in the parent before workers are spawned; workers reopen mmap handles lazily.
        loader.dataset._open_features(); feature_metadata = loader.dataset.metadata
    feature_fp = feature_metadata.get("fingerprint") if feature_metadata else None
    step = resume_checkpoint(args.resume, model, optimizer, ema, device, cache_fingerprint(payload), projector, feature_fp) if args.resume else 0
    params = sum(p.numel() for p in trainable.parameters() if p.requires_grad); accum = cfg["train"].get("grad_accum_steps", 1); print(f"trainable_parameters: {params:,}\nbase_sit_parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\nprojector_parameters: {sum(p.numel() for p in projector.parameters() if p.requires_grad) if projector else 0:,}\ndevice: {device}\ndtype: {dtype} autocast={autocast}\nbatch_size: {cfg['data']['batch_size']}\neffective_batch_size: {cfg['data']['batch_size'] * accum}\nvae_loaded_for_training: false\ndino_loaded_for_training: false")
    out = Path(cfg["output_dir"]); writer = SummaryWriter(out / "logs"); batches = infinite(loader); pbar = tqdm(total=max_steps, initial=step, desc=cfg["name"]); latest = out / "checkpoints" / "latest.pt"; model.train()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True); total = 0.0
        for _ in range(accum):
            batch = next(batches); x0, labels = batch[0], batch[1]; features = batch[2] if projector else None; non_blocking = bool(cfg["data"].get("non_blocking", True)); x0, labels = x0.to(device, non_blocking=non_blocking).float(), labels.to(device, non_blocking=non_blocking); features = features.to(device, non_blocking=non_blocking) if features is not None else None; noise = torch.randn_like(x0); t = torch.rand(x0.shape[0], device=device); xt, target = linear_interpolant(x0, noise, t)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                if projector:
                    velocity, hidden = model(xt, t, labels, return_hidden_after=cfg["repa"]["alignment_depth"])
                    flow_loss = velocity_loss(velocity, target); alignment_loss, cosine = repa_loss(projector(hidden), features); loss = (flow_loss + float(cfg["repa"]["coefficient"]) * alignment_loss) / accum
                else:
                    flow_loss = velocity_loss(model(xt, t, labels), target); alignment_loss = cosine = None; loss = flow_loss / accum
            loss.backward(); total += float(loss.detach()) * accum
        if cfg["train"].get("grad_clip", 0): nn.utils.clip_grad_norm_(trainable.parameters(), cfg["train"]["grad_clip"])
        optimizer.step(); ema.update(model); step += 1; pbar.update(1)
        if step % cfg["train"].get("log_every", 1) == 0:
            writer.add_scalar("train/loss", total, step); writer.add_scalar("train/flow_loss", float(flow_loss.detach()), step)
            if projector: writer.add_scalar("train/repa_loss", float(alignment_loss.detach()), step); writer.add_scalar("train/cosine_similarity", float(cosine.detach()), step); writer.add_scalar("train/repa_coefficient", float(cfg["repa"]["coefficient"]), step)
            pbar.set_postfix(loss=f"{total:.4f}")
        if step % cfg["train"].get("sample_every", 10**9) == 0: print(f"latent_preview_written: {write_latent_preview(model, ema, cfg, device, step)}")
        if step % cfg["train"].get("save_every", 100) == 0: save_checkpoint(latest, model, optimizer, ema, cfg, step, cache_fingerprint(payload), projector, feature_metadata); print(f"checkpoint_written: {latest}")
    save_checkpoint(latest, model, optimizer, ema, cfg, step, cache_fingerprint(payload), projector, feature_metadata); writer.close(); pbar.close()
    if device.type == "cuda": print(f"cuda_vram_peak_allocated_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}\ncuda_vram_peak_reserved_gb: {torch.cuda.max_memory_reserved() / 1024**3:.2f}")
    print(f"checkpoint_written: {latest}")


if __name__ == "__main__": main()
