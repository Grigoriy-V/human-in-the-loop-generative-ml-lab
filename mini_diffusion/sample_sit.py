from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.utils import save_image

from mini_diffusion.diffusion import EMA
from mini_diffusion.sit import sample_ode
from mini_diffusion.train_sit import build_model
from mini_diffusion.vae import decode_latents, load_frozen_vae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--classes", nargs="+", type=int, required=True); parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sampler", choices=("euler", "heun"), default="heun"); parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5); parser.add_argument("--save-individual", action="store_true"); args = parser.parse_args()
    if len(args.classes) != len(args.seeds): raise ValueError("--classes and --seeds must have the same length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False); cfg = ckpt["config"]
    model = build_model(cfg).to(device); model.load_state_dict(ckpt["model"]); ema = EMA(model, cfg["train"]["ema_decay"]); ema.load_state_dict(ckpt["ema"])
    vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision")); out = Path(args.output); out.mkdir(parents=True, exist_ok=True); images = []
    context = ema.average_parameters(model) if args.weights == "ema" else nullcontext()
    with context, torch.inference_mode():
        for class_id, seed in zip(args.classes, args.seeds):
            generator = torch.Generator(device=device).manual_seed(seed); labels = torch.tensor([class_id], device=device)
            latent = sample_ode(model, (1, 4, cfg["data"]["latent_resolution"], cfg["data"]["latent_resolution"]), labels, device, steps=args.steps, sampler=args.sampler, guidance_scale=args.guidance_scale, generator=generator, diagnostics=True)
            images.append(decode_latents(vae, latent).cpu())
    batch = torch.cat(images); pixels = ((batch + 1) * 0.5).clamp(0, 1)
    if not torch.isfinite(pixels).all() or pixels.shape[-2:] != (128, 128): raise RuntimeError("Invalid decoded sample")
    path = out / "grid.png"; save_image(pixels, path, nrow=max(1, int(len(images) ** 0.5)))
    if args.save_individual:
        for index, image in enumerate(pixels): save_image(image, out / f"sample_{index:04d}.png")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"checkpoint": str(args.checkpoint), "global_step": ckpt["global_step"], "weights": args.weights, "classes": args.classes, "seeds": args.seeds, "sampler": args.sampler, "steps": args.steps, "guidance_scale": args.guidance_scale, "vae_model_id": vae.model_id, "vae_revision": vae.revision, "latent_scaling_factor": vae.scaling_factor, "grid_sha256": digest, "shape": list(pixels.shape), "min": float(pixels.min()), "max": float(pixels.max()), "mean": float(pixels.mean()), "std": float(pixels.std())}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); print(f"grid_written: {path}\nmetadata_written: {out / 'metadata.json'}")


if __name__ == "__main__": main()
