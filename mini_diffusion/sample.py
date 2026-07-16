from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.utils import save_image

from mini_diffusion.diffusion import EMA
from mini_diffusion.sampling import denormalize_to_unit, make_generator, sample_statistics
from mini_diffusion.train import build_diffusion, build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/samples")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--classes", nargs="*", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=0)
    parser.add_argument("--save-individual", action="store_true")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--grid-filename", default="grid.png")
    parser.add_argument("--metadata-filename", default="metadata.json")
    parser.add_argument("--sampling-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.9999))
    _, global_step = load_checkpoint(args.checkpoint, model, ema=ema, map_location=device)

    classes = args.classes or list(range(args.num_images or cfg["train"].get("sample_count", 4)))
    num_images = args.num_images or len(classes)
    if len(classes) < num_images:
        classes = (classes * ((num_images + len(classes) - 1) // len(classes)))[:num_images]
    seeds = args.seeds or list(range(num_images))
    if len(seeds) < num_images:
        seeds = (seeds * ((num_images + len(seeds) - 1) // len(seeds)))[:num_images]

    labels = torch.tensor(classes[:num_images], device=device, dtype=torch.long)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    was_training = model.training
    weights_context = ema.average_parameters(model) if args.weights == "ema" else nullcontext()
    try:
        model.eval()
        with torch.inference_mode(), weights_context:
            for seed, label in zip(seeds[:num_images], labels):
                generator = make_generator(device, seed)
                img = diffusion.sample(
                    model,
                    (1, 3, cfg["data"]["resolution"], cfg["data"]["resolution"]),
                    labels=label[None],
                    guidance_scale=args.guidance_scale,
                    device=device,
                    generator=generator,
                )
                images.append(img.cpu())
    finally:
        model.train(was_training)

    batch = torch.cat(images, dim=0)
    stats = sample_statistics(batch)
    if not stats["isfinite"]:
        raise FloatingPointError("Sampling output contains NaN or Inf values.")
    if args.sampling_diagnostics:
        print("sample_diagnostics: " + json.dumps(stats, sort_keys=True))
    nrow = args.grid_size or max(1, int(num_images**0.5))
    grid_path = out_dir / args.grid_filename
    save_image(denormalize_to_unit(batch), grid_path, nrow=nrow)
    if args.save_individual:
        for i, img in enumerate(batch):
            save_image(denormalize_to_unit(img), out_dir / f"sample_{i:04d}.png")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "global_step": global_step,
        "weights": args.weights,
        "classes": classes[:num_images],
        "seeds": seeds[:num_images],
        "guidance_scale": args.guidance_scale,
        "sampler": "ddpm",
        "sampling_steps": diffusion.steps,
        "sampling_dtype": "float32",
        "device": str(device),
        "grid": str(grid_path),
        "statistics": stats,
    }
    metadata_path = out_dir / args.metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"grid_written: {grid_path}")
    print(f"metadata_written: {metadata_path}")


if __name__ == "__main__":
    main()
