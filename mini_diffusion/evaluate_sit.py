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

from mini_diffusion.evaluator import (
    fixed_protocol, generate_images, load_or_build_reference, make_feature_models,
    load_reference_images, reconstruct_vae, score, write_comparison_artifacts, write_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible Imagenette SiT checkpoint evaluator.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--sampler", choices=("euler", "heun"), default="heun")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--reference-split", choices=("val", "train"), default="val")
    parser.add_argument("--reference-cache", help="Shared cache directory for reference Inception features.")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample-batch-size", type=int, default=20)
    parser.add_argument("--vae-ceiling", action="store_true", help="Evaluate val images after frozen VAE encode/decode instead of sampling checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_defaults = {"quick": (20, 25), "full": (100, 50)}
    default_count, default_steps = mode_defaults[args.mode]
    samples_per_class = args.samples_per_class or default_count
    steps = args.steps or default_steps
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    cfg = first["config"]
    specs = fixed_protocol(cfg["data"]["num_classes"], samples_per_class, args.seed_start)
    inception, inception_tf, classifier, classifier_tf = make_feature_models(device)
    reference_cache = Path(args.reference_cache) if args.reference_cache else Path(args.output) / "reference"
    reference = load_or_build_reference(reference_cache, cfg["data"]["root"], args.reference_split, cfg["data"]["resolution"], inception, inception_tf, device)
    summary = []
    if args.vae_ceiling:
        images, labels, _, _, _ = load_reference_images(cfg["data"]["root"], args.reference_split, cfg["data"]["resolution"])
        specs = [type(specs[0])(int(label), args.seed_start + index) for index, label in enumerate(labels.tolist())]
        started = time.perf_counter()
        reconstructed = reconstruct_vae(images, cfg, device, args.sample_batch_size)
        metrics, features = score(reconstructed, specs, reference, inception, inception_tf, classifier, classifier_tf, device)
        out = Path(args.output) / "vae_reconstruction" / args.reference_split
        metadata = {"mode": "vae_reconstruction", "reference_split": args.reference_split, "dataset_root": cfg["data"]["root"], "sampling_seconds": time.perf_counter() - started, "device": str(device), "torch": torch.__version__, "vae_model_id": cfg["vae"]["model_id"]}
        write_result(out, Path(args.checkpoints[0]), first, specs, reconstructed, metrics, reference, features, metadata)
        summary.append({"output": str(out), "fid": metrics["fid"], "kid": metrics["kid"], "class_accuracy": metrics["imagenet_top1_accuracy"]})
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary[-1], sort_keys=True))
        return
    for checkpoint_text in args.checkpoints:
        checkpoint_path = Path(checkpoint_text)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"]["data"]["num_classes"] != 10:
            raise ValueError(f"{checkpoint_path} is not an Imagenette checkpoint")
        started = time.perf_counter()
        images = generate_images(checkpoint, specs, weights=args.weights, sampler=args.sampler, steps=steps, guidance_scale=args.guidance_scale, device=device, batch_size=args.sample_batch_size)
        metrics, features = score(images, specs, reference, inception, inception_tf, classifier, classifier_tf, device)
        elapsed = time.perf_counter() - started
        run_name = checkpoint_path.parent.parent.name
        out = Path(args.output) / run_name / f"step_{int(checkpoint['global_step']):07d}"
        metadata = {"mode": args.mode, "weights": args.weights, "sampler": args.sampler, "steps": steps, "guidance_scale": args.guidance_scale, "samples_per_class": samples_per_class, "seed_start": args.seed_start, "reference_split": args.reference_split, "dataset_root": cfg["data"]["root"], "sampling_seconds": elapsed, "device": str(device), "torch": torch.__version__}
        write_result(out, checkpoint_path, checkpoint, specs, images, metrics, reference, features, metadata)
        summary.append({"checkpoint": str(checkpoint_path), "global_step": int(checkpoint["global_step"]), "output": str(out), "sampling_seconds": elapsed, "kid": metrics["kid"], "fid": metrics["fid"], "class_accuracy": metrics["imagenet_top1_accuracy"], "precision": metrics["generative_precision"], "recall": metrics["generative_recall"]})
        print(json.dumps(summary[-1], sort_keys=True))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_comparison_artifacts(summary, output)


if __name__ == "__main__":
    main()
