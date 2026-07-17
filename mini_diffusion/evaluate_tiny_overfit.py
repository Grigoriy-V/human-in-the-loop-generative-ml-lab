from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mini_diffusion.evaluator import (
    SampleSpec, generate_images, load_or_build_reference, make_feature_models,
    score, write_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the isolated Imagenette single-class overfit experiment.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--class-id", type=int, default=6)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    args = parser.parse_args()
    if args.samples < 4:
        raise ValueError("--samples must be at least 4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.checkpoint); checkpoint = torch.load(path, map_location="cpu", weights_only=False); cfg = checkpoint["config"]
    inception, inception_tf, classifier, classifier_tf = make_feature_models(device)
    reference = load_or_build_reference(Path(args.reference_cache), cfg["data"]["root"], "train", cfg["data"]["resolution"], inception, inception_tf, device)
    indices = (reference["labels"] == args.class_id).nonzero(as_tuple=True)[0]
    reference = {**reference, "features": reference["features"][indices], "labels": reference["labels"][indices], "paths": [reference["paths"][index] for index in indices.tolist()], "image_indices": indices.tolist()}
    specs = [SampleSpec(args.class_id, args.seed_start + index) for index in range(args.samples)]
    images = generate_images(checkpoint, specs, weights=args.weights, sampler="heun", steps=args.steps, guidance_scale=args.guidance_scale, device=device, batch_size=min(32, args.samples))
    metrics, features = score(images, specs, reference, inception, inception_tf, classifier, classifier_tf, device)
    distances = torch.cdist(features.float(), reference["features"].float())
    generated_distances = torch.cdist(features.float(), features.float()); generated_distances.fill_diagonal_(float("inf"))
    metrics["memorization"] = {"nearest_train_feature_distance_mean": float(distances.min(1).values.mean()), "duplicate_pairs_distance_lt_0_1": int((generated_distances < 0.1).triu(1).sum())}
    out = Path(args.output)
    metadata = {"mode": "tiny_overfit", "dataset_root": cfg["data"]["root"], "reference_split": "train", "class_id": args.class_id, "samples": args.samples, "weights": args.weights, "sampler": "heun", "steps": args.steps, "guidance_scale": args.guidance_scale, "device": str(device)}
    write_result(out, path, checkpoint, specs, images, metrics, reference, features, metadata)
    print(json.dumps({"output": str(out), "global_step": checkpoint["global_step"], "accuracy": metrics["imagenet_top1_accuracy"], "kid": metrics["kid"], "fid": metrics["fid"], "memorization": metrics["memorization"]}, sort_keys=True))


if __name__ == "__main__":
    main()
