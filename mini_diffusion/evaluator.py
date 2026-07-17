from __future__ import annotations

import csv
import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

from mini_diffusion.data.imagenette import imagenette_root
from mini_diffusion.diffusion import EMA
from mini_diffusion.sit import sample_ode
from mini_diffusion.train_sit import build_model
from mini_diffusion.vae import decode_latents, encode_latents, load_frozen_vae


# Imagenette class directory names are ImageNet-1k WordNet IDs.  These are the
# target logits of torchvision's standard ImageNet classifiers.
IMAGENETTE_IMAGENET_INDEX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


@dataclass(frozen=True)
class SampleSpec:
    class_id: int
    seed: int


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_protocol(num_classes: int, samples_per_class: int, seed_start: int) -> list[SampleSpec]:
    if num_classes != 10:
        raise ValueError("Imagenette evaluator requires exactly 10 classes")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    return [SampleSpec(class_id, seed_start + class_id * samples_per_class + offset)
            for class_id in range(num_classes) for offset in range(samples_per_class)]


def protocol_noise(specs: list[SampleSpec], latent_shape: tuple[int, int, int]) -> torch.Tensor:
    return torch.stack([
        torch.randn(latent_shape, generator=torch.Generator(device="cpu").manual_seed(spec.seed))
        for spec in specs
    ])


def _weights_or_error(factory, weights, name: str):
    try:
        return factory(weights=weights).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pretrained {name} weights. Run evaluation with network access once; "
            "ordinary pytest never downloads evaluator weights."
        ) from exc


def make_feature_models(device: torch.device):
    inception_weights = models.Inception_V3_Weights.IMAGENET1K_V1
    inception = _weights_or_error(models.inception_v3, inception_weights, "Inception-v3")
    inception.fc = torch.nn.Identity()
    resnet_weights = models.ResNet50_Weights.IMAGENET1K_V2
    classifier = _weights_or_error(models.resnet50, resnet_weights, "ResNet-50")
    return inception.to(device), inception_weights.transforms(), classifier.to(device), resnet_weights.transforms()


def _batched_features(model: torch.nn.Module, preprocess, images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = preprocess(images[start:start + batch_size]).to(device)
            result.append(model(batch).float().cpu())
    return torch.cat(result)


def _batched_logits(model: torch.nn.Module, preprocess, images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            result.append(model(preprocess(images[start:start + batch_size]).to(device)).float().cpu())
    return torch.cat(result)


def reference_cache_key(root: Path, split: str, class_to_idx: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / split).rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
            digest.update(str(path.stat().st_size).encode())
            digest.update(str(path.stat().st_mtime_ns).encode())
    digest.update(json.dumps(class_to_idx, sort_keys=True).encode())
    return digest.hexdigest()


def load_reference_images(root: str | Path, split: str, resolution: int) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, int], str]:
    dataset_root = imagenette_root(root).resolve()
    transform = transforms.Compose([transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(resolution), transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_root / split, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    images, labels = [], []
    for batch, target in loader:
        images.append(batch); labels.append(target)
    paths = [str(Path(path).resolve().relative_to(dataset_root)) for path, _ in dataset.samples]
    return torch.cat(images), torch.cat(labels), paths, dataset.class_to_idx, reference_cache_key(dataset_root, split, dataset.class_to_idx)


def load_or_build_reference(cache_dir: Path, root: str | Path, split: str, resolution: int, inception, inception_tf, device: torch.device) -> dict[str, Any]:
    images, labels, paths, class_to_idx, fingerprint = load_reference_images(root, split, resolution)
    path = cache_dir / f"imagenette_{split}_inception_v3.pt"
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") == fingerprint and cached.get("preprocessing") == "Inception_V3_Weights.IMAGENET1K_V1":
            return cached
    features = _batched_features(inception, inception_tf, images, device, 64)
    payload = {
        "fingerprint": fingerprint, "split": split, "preprocessing": "Inception_V3_Weights.IMAGENET1K_V1",
        "extractor_model": "torchvision Inception-v3", "extractor_revision": "IMAGENET1K_V1",
        "sample_count": len(images), "class_mapping": class_to_idx, "feature_dim": int(features.shape[1]),
        "features": features, "labels": labels, "paths": paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(0)
    return centered.T @ centered / max(1, len(features) - 1)


def validate_feature_inputs(real: torch.Tensor, generated: torch.Tensor) -> None:
    if real.ndim != 2 or generated.ndim != 2:
        raise ValueError("Feature tensors must be rank-2 [N, D]")
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("At least two real and generated features are required")
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Real and generated feature dimensions must match")
    if not torch.isfinite(real).all() or not torch.isfinite(generated).all():
        raise ValueError("Feature tensors must be finite")


def fid(real: torch.Tensor, generated: torch.Tensor, device: torch.device) -> float:
    validate_feature_inputs(real, generated)
    real, generated = real.to(device, torch.float64), generated.to(device, torch.float64)
    mean_delta = (real.mean(0) - generated.mean(0)).square().sum()
    cov_real, cov_generated = _covariance(real), _covariance(generated)
    values, vectors = torch.linalg.eigh(cov_real)
    sqrt_real = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    product = sqrt_real @ cov_generated @ sqrt_real
    trace_sqrt = torch.linalg.eigvalsh(product).clamp_min(0).sqrt().sum()
    return float((mean_delta + torch.trace(cov_real) + torch.trace(cov_generated) - 2 * trace_sqrt).cpu())


def kid(real: torch.Tensor, generated: torch.Tensor, subsets: int = 20, subset_size: int = 100) -> float:
    validate_feature_inputs(real, generated)
    if torch.equal(real, generated):
        return 0.0
    size = min(subset_size, len(real), len(generated))
    if size < 2:
        return float("nan")
    values = []
    for index in range(subsets):
        generator = torch.Generator().manual_seed(9000 + index)
        a = real[torch.randperm(len(real), generator=generator)[:size]].float()
        b = generated[torch.randperm(len(generated), generator=generator)[:size]].float()
        scale = float(a.shape[1])
        kaa, kbb, kab = ((a @ a.T / scale + 1).pow(3), (b @ b.T / scale + 1).pow(3), (a @ b.T / scale + 1).pow(3))
        values.append(float((kaa.sum() - kaa.diag().sum()) / (size * (size - 1)) + (kbb.sum() - kbb.diag().sum()) / (size * (size - 1)) - 2 * kab.mean()))
    return float(np.mean(values))


def precision_recall(real: torch.Tensor, generated: torch.Tensor, k: int = 3) -> tuple[float, float]:
    validate_feature_inputs(real, generated)
    if k < 1 or min(len(real), len(generated)) <= k:
        raise ValueError("k must be smaller than both feature-set sizes")
    real, generated = real.float(), generated.float()
    real_dist = torch.cdist(real, real); real_dist.fill_diagonal_(float("inf")); real_radius = real_dist.kthvalue(k, dim=1).values
    generated_dist = torch.cdist(generated, generated); generated_dist.fill_diagonal_(float("inf")); generated_radius = generated_dist.kthvalue(k, dim=1).values
    cross = torch.cdist(generated, real)
    precision = (cross <= real_radius.unsqueeze(0)).any(1).float().mean()
    recall = (cross <= generated_radius.unsqueeze(1)).any(0).float().mean()
    return float(precision), float(recall)


def pixel_diagnostics(images: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(images).all(dim=(1, 2, 3))
    brightness = images.mean(dim=(1, 2, 3)); contrast = images.std(dim=(1, 2, 3))
    saturation = ((images <= 1 / 255) | (images >= 1 - 1 / 255)).float().mean(dim=(1, 2, 3))
    laplacian = images[:, :, 1:-1, 1:-1] * -4 + images[:, :, :-2, 1:-1] + images[:, :, 2:, 1:-1] + images[:, :, 1:-1, :-2] + images[:, :, 1:-1, 2:]
    detail = laplacian.abs().mean(dim=(1, 2, 3))
    black_white = (brightness < 0.01) | (brightness > 0.99)
    return {"finite_failures": int((~finite).sum()), "black_white_failures": int(black_white.sum()), "saturation_mean": float(saturation.mean()), "brightness_mean": float(brightness.mean()), "brightness_std": float(brightness.std()), "contrast_mean": float(contrast.mean()), "detail_laplacian_mean": float(detail.mean()), "low_detail_warnings": int((detail < 0.01).sum())}


def save_grid(images: torch.Tensor, path: Path, nrow: int = 10) -> None:
    save_image(images.clamp(0, 1), path, nrow=nrow)


def generate_images(checkpoint: dict[str, Any], specs: list[SampleSpec], *, weights: str, sampler: str, steps: int, guidance_scale: float, device: torch.device, batch_size: int) -> torch.Tensor:
    cfg = checkpoint["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    ema = EMA(model, cfg["train"]["ema_decay"]); ema.load_state_dict(checkpoint["ema"])
    vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision"))
    labels = torch.tensor([spec.class_id for spec in specs], dtype=torch.long)
    noise = protocol_noise(specs, (4, cfg["data"]["latent_resolution"], cfg["data"]["latent_resolution"]))
    images = []
    context = ema.average_parameters(model) if weights == "ema" else nullcontext()
    with context, torch.inference_mode():
        for start in range(0, len(specs), batch_size):
            end = start + batch_size
            latent = sample_ode(model, tuple(noise[start:end].shape), labels[start:end].to(device), device, steps=steps, sampler=sampler, guidance_scale=guidance_scale, diagnostics=True, initial_noise=noise[start:end])
            images.append(((decode_latents(vae, latent) + 1) * 0.5).clamp(0, 1).cpu())
    del vae, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pixels = torch.cat(images)
    if not torch.isfinite(pixels).all() or tuple(pixels.shape[1:]) != (3, 128, 128):
        raise RuntimeError("Generated image batch is non-finite or has an invalid shape")
    return pixels


def reconstruct_vae(images: torch.Tensor, cfg: dict[str, Any], device: torch.device, batch_size: int) -> torch.Tensor:
    vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision"))
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            generator = torch.Generator(device=device).manual_seed(100000 + start)
            latents = encode_latents(vae, images[start:start + batch_size].to(device) * 2 - 1, generator)
            result.append(((decode_latents(vae, latents) + 1) * 0.5).clamp(0, 1).cpu())
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(result)


def score(images: torch.Tensor, specs: list[SampleSpec], reference: dict[str, Any], inception, inception_tf, classifier, classifier_tf, device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    features = _batched_features(inception, inception_tf, images, device, 64)
    logits = _batched_logits(classifier, classifier_tf, images, device, 64)
    targets = torch.tensor([IMAGENETTE_IMAGENET_INDEX[name] for name, _ in sorted(reference["class_mapping"].items(), key=lambda pair: pair[1])])
    expected = targets[torch.tensor([spec.class_id for spec in specs])]
    probabilities = logits.softmax(1); predicted = probabilities.argmax(1)
    correct = predicted.eq(expected)
    per_class = {}
    for class_id in range(10):
        indices = [index for index, spec in enumerate(specs) if spec.class_id == class_id]
        if indices:
            per_class[str(class_id)] = {"accuracy": float(correct[indices].float().mean()), "confidence": float(probabilities[indices, expected[indices]].mean())}
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    inverse = {value: index for index, value in enumerate(targets.tolist())}
    for spec, prediction in zip(specs, predicted.tolist()):
        if prediction in inverse:
            confusion[spec.class_id, inverse[prediction]] += 1
    precision, recall = precision_recall(reference["features"], features)
    metrics = {"kid": kid(reference["features"], features), "fid": fid(reference["features"], features, device), "fid_sample_count": {"real": len(reference["features"]), "generated": len(features)}, "generative_precision": precision, "generative_recall": recall, "imagenet_top1_accuracy": float(correct.float().mean()), "imagenet_target_confidence": float(probabilities[torch.arange(len(expected)), expected].mean()), "per_class": per_class, "confusion_matrix_imagenette": confusion.tolist(), "pixel": pixel_diagnostics(images)}
    return metrics, features


def write_result(out: Path, checkpoint_path: Path, checkpoint: dict[str, Any], specs: list[SampleSpec], images: torch.Tensor, metrics: dict[str, Any], reference: dict[str, Any], features: torch.Tensor, metadata: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_grid(images, out / "grid.png")
    distances = torch.cdist(features.float(), reference["features"].float())
    nearest = distances.argmin(1)
    outlier = distances.min(1).values.argsort(descending=True)[:min(20, len(images))]
    nearest_images, _, _, _, _ = load_reference_images(metadata["dataset_root"], metadata["reference_split"], 128)
    image_indices = reference.get("image_indices")
    if image_indices is not None:
        nearest_images = nearest_images[torch.tensor(image_indices, dtype=torch.long)]
    paired = torch.stack([item for index in outlier.tolist() for item in (images[index], nearest_images[nearest[index]])])
    save_grid(paired, out / "outliers.png", nrow=4)
    selected = torch.arange(min(20, len(images)))
    paired = torch.stack([item for index in selected.tolist() for item in (images[index], nearest_images[nearest[index]])])
    save_grid(paired, out / "nearest_real.png", nrow=4)
    metadata.update({"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path), "global_step": int(checkpoint["global_step"]), "sample_protocol": [asdict(spec) for spec in specs], "reference": {key: value for key, value in reference.items() if key not in {"features", "labels", "paths"}}, "grid_sha256": sha256(out / "grid.png")})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "accuracy", "confidence"]); writer.writeheader()
        for class_id, row in metrics["per_class"].items(): writer.writerow({"class_id": class_id, **row})


def write_comparison_artifacts(summary: list[dict[str, Any]], output: Path) -> None:
    if not summary:
        return
    output.mkdir(parents=True, exist_ok=True)
    fields = ["checkpoint", "global_step", "sampling_seconds", "kid", "fid", "class_accuracy", "precision", "recall", "output"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    image = Image.new("RGB", (960, 560), "white"); draw = ImageDraw.Draw(image)
    panels = [("fid", "FID", (20, 30, 460, 260)), ("kid", "KID", (500, 30, 940, 260)), ("class_accuracy", "ResNet top-1", (20, 300, 460, 530)), ("recall", "Recall", (500, 300, 940, 530))]
    for field, title, (left, top, right, bottom) in panels:
        values = [float(item[field]) for item in summary]
        steps = [int(item["global_step"]) for item in summary]
        minimum, maximum = min(values), max(values)
        if math.isclose(minimum, maximum): maximum = minimum + 1.0
        draw.rectangle((left, top, right, bottom), outline="black")
        draw.text((left + 6, top + 6), title, fill="black")
        points = []
        for step, value in zip(steps, values):
            x = left + 30 + (right - left - 45) * ((step - min(steps)) / max(1, max(steps) - min(steps)))
            y = bottom - 25 - (bottom - top - 50) * ((value - minimum) / (maximum - minimum))
            points.append((x, y))
        if len(points) > 1: draw.line(points, fill=(24, 91, 155), width=2)
        for point in points: draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(24, 91, 155))
        draw.text((left + 6, bottom - 18), f"min {minimum:.4f}  max {maximum:.4f}", fill="black")
    image.save(output / "metrics_by_step.png")
