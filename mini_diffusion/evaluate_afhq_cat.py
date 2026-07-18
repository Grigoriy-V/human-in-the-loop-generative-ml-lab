from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torchvision import models
from torchvision.utils import save_image

from mini_diffusion.data import AFHQCatDataset
from mini_diffusion.diffusion import EMA
from mini_diffusion.evaluator import _batched_features, fid, kid, pixel_diagnostics, precision_recall, protocol_noise, sha256
from mini_diffusion.sit import sample_ode
from mini_diffusion.train_sit import build_model
from mini_diffusion.vae import decode_latents, encode_latents, load_frozen_vae


def load_config_from_checkpoint(path: str | Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if cfg["data"].get("dataset") != "afhq-cat-official" or cfg["data"].get("num_classes") != 1:
        raise ValueError("Checkpoint is not an AFHQ Cats one-class SiT run")
    return checkpoint, cfg


def inception(device: torch.device):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    try:
        model = models.inception_v3(weights=weights).eval(); model.fc = torch.nn.Identity()
    except Exception as exc:
        raise RuntimeError("Could not load pretrained Inception-v3 weights for AFHQ evaluation") from exc
    return model.to(device), weights.transforms()


def reference_images(cfg: dict, split: str) -> tuple[torch.Tensor, list[str]]:
    dataset = AFHQCatDataset(cfg["data"]["root"], split, cfg["data"]["resolution"], augmentation_variants=1, seed=cfg.get("seed", 123))
    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    paths = [str(entry.path.relative_to(dataset.root)).replace("\\", "/") for entry in dataset.entries]
    return (images + 1) * 0.5, paths


def duplicate_diagnostics(features: torch.Tensor, threshold: float) -> dict[str, float | int]:
    distances = torch.cdist(features.float(), features.float())
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    pairs = torch.triu(distances <= threshold, diagonal=1).sum()
    return {"nearest_generated_feature_distance_mean": float(nearest.mean()), "nearest_generated_feature_distance_min": float(nearest.min()), "duplicate_feature_threshold": threshold, "duplicate_pairs": int(pairs)}


def write_result(output: Path, checkpoint_path: Path | None, images: torch.Tensor, generated_features: torch.Tensor, reference_images_tensor: torch.Tensor, reference_features: torch.Tensor, metadata: dict, metrics: dict) -> None:
    output.mkdir(parents=True, exist_ok=True); save_image(images, output / "grid.png", nrow=10)
    distances = torch.cdist(generated_features.float(), reference_features.float()); nearest = distances.argmin(1)
    selected = torch.arange(min(20, len(images)))
    paired = torch.stack([item for index in selected.tolist() for item in (images[index], reference_images_tensor[nearest[index]])])
    save_image(paired, output / "nearest_real.png", nrow=4)
    metadata.update({"checkpoint": str(checkpoint_path) if checkpoint_path else None, "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path else None, "grid_sha256": sha256(output / "grid.png")})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def generated_images(checkpoint: dict, cfg: dict, *, weights: str, samples: int, seed_start: int, steps: int, guidance_scale: float, device: torch.device, batch_size: int) -> torch.Tensor:
    model = build_model(cfg).to(device); model.load_state_dict(checkpoint["model"]); ema = EMA(model, cfg["train"]["ema_decay"]); ema.load_state_dict(checkpoint["ema"])
    vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision")); labels = torch.zeros(samples, dtype=torch.long); specs = [type("Spec", (), {"class_id": 0, "seed": seed_start + index})() for index in range(samples)]
    noise = protocol_noise(specs, (4, cfg["data"]["latent_resolution"], cfg["data"]["latent_resolution"])); result = []
    context = ema.average_parameters(model) if weights == "ema" else nullcontext()
    with context, torch.inference_mode():
        for start in range(0, samples, batch_size):
            latent = sample_ode(model, tuple(noise[start:start + batch_size].shape), labels[start:start + batch_size].to(device), device, steps=steps, sampler="heun", guidance_scale=guidance_scale, diagnostics=True, initial_noise=noise[start:start + batch_size])
            result.append(((decode_latents(vae, latent) + 1) * 0.5).clamp(0, 1).cpu())
    del model, vae
    if device.type == "cuda": torch.cuda.empty_cache()
    return torch.cat(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="AFHQ Cats one-class SiT evaluator; uses the official held-out test split only for metrics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--config", help="Required for --vae-ceiling before any training checkpoint exists.")
    parser.add_argument("--output", required=True); parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--samples", type=int, default=1000); parser.add_argument("--seed-start", type=int, default=1000); parser.add_argument("--steps", type=int, default=50); parser.add_argument("--guidance-scale", type=float, default=1.5); parser.add_argument("--sample-batch-size", type=int, default=20); parser.add_argument("--duplicate-threshold", type=float, default=0.1); parser.add_argument("--vae-ceiling", action="store_true")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path:
        checkpoint, cfg = load_config_from_checkpoint(checkpoint_path)
    else:
        if not args.vae_ceiling:
            parser.error("--config is valid only with --vae-ceiling; sampling requires --checkpoint")
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if cfg["data"].get("dataset") != "afhq-cat-official" or cfg["data"].get("num_classes") != 1:
            raise ValueError("Config is not an AFHQ Cats one-class SiT run")
        checkpoint = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = inception(device); test_images, test_paths = reference_images(cfg, "test"); reference_features = _batched_features(model, preprocess, test_images, device, 64)
    started = time.perf_counter()
    if args.vae_ceiling:
        vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision")); reconstructed = []
        with torch.inference_mode():
            for start in range(0, len(test_images), args.sample_batch_size):
                generator = torch.Generator(device=device).manual_seed(args.seed_start + start)
                latent = encode_latents(vae, test_images[start:start + args.sample_batch_size].to(device) * 2 - 1, generator)
                reconstructed.append(((decode_latents(vae, latent) + 1) * 0.5).clamp(0, 1).cpu())
        images = torch.cat(reconstructed); mode = "vae_ceiling"
    else:
        images = generated_images(checkpoint, cfg, weights=args.weights, samples=args.samples, seed_start=args.seed_start, steps=args.steps, guidance_scale=args.guidance_scale, device=device, batch_size=args.sample_batch_size); mode = "sampling"
    if not torch.isfinite(images).all() or tuple(images.shape[1:]) != (3, 128, 128): raise RuntimeError("AFHQ evaluation produced invalid images")
    features = _batched_features(model, preprocess, images, device, 64)
    metrics = {"kid": kid(reference_features, features), "fid": fid(reference_features, features, device), "fid_sample_count": {"real": len(reference_features), "generated": len(features)}, "generative_precision": precision_recall(reference_features, features)[0], "generative_recall": precision_recall(reference_features, features)[1], "pixel": pixel_diagnostics(images), "nearest_test_feature_distance_mean": float(torch.cdist(features.float(), reference_features.float()).min(dim=1).values.mean()), **duplicate_diagnostics(features, args.duplicate_threshold)}
    metadata = {"mode": mode, "weights": args.weights, "sampler": "heun", "steps": args.steps, "guidance_scale": args.guidance_scale, "seed_start": args.seed_start, "samples": len(images), "test_split_paths": test_paths, "sampling_seconds": time.perf_counter() - started, "device": str(device), "torch": torch.__version__, "vae_model_id": cfg["vae"]["model_id"], "latent_scaling_factor": cfg["vae"].get("scaling_factor", "from_vae_config")}
    write_result(Path(args.output), checkpoint_path, images, features, test_images, reference_features, metadata, metrics); print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
