from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from mini_diffusion.data import AFHQCatDataset
from mini_diffusion.latent_cache import CACHE_FORMAT_VERSION, cache_statistics, load_cache, validate_cache
from mini_diffusion.vae import decode_latents, encode_latents, load_frozen_vae


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cache_path(cfg: dict, split: str) -> Path:
    return Path(cfg["data"]["cache_dir"]) / f"{split}.pt"


def build_dataset(cfg: dict, split: str) -> AFHQCatDataset:
    data = cfg["data"]
    return AFHQCatDataset(
        data["root"], split, int(data["resolution"]),
        augmentation_variants=int(data.get("augmentation_variants", 4)), seed=int(data.get("augmentation_seed", cfg.get("seed", 123))), crop_scale=tuple(float(value) for value in data.get("augmentation_scale", (0.85, 1.0))),
    )


def make_cache(cfg: dict, split: str, *, force: bool, limit: int | None) -> Path:
    path = cache_path(cfg, split)
    if path.exists() and not force:
        raise FileExistsError(f"Cache exists: {path}. Pass --force to overwrite it.")
    dataset = build_dataset(cfg, split)
    if limit is not None:
        dataset.entries = dataset.entries[:limit]
    data, vae_cfg = cfg["data"], cfg["vae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_frozen_vae(vae_cfg["model_id"], device, vae_cfg.get("revision"))
    loader = DataLoader(dataset, batch_size=int(data.get("prepare_batch_size", 16)), shuffle=False, num_workers=int(data.get("prepare_num_workers", 0)), pin_memory=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(int(data.get("latent_seed", cfg.get("seed", 123))))
    latents, labels, pixel_hashes, latent_hashes = [], [], [], []
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc=f"encode_afhq_cat_{split}"):
            encoded = encode_latents(vae, images.to(device), generator).cpu().to(torch.float16)
            latents.append(encoded)
            labels.append(batch_labels.long().cpu())
            pixel_hashes.extend(hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest() for image in images)
            latent_hashes.extend(hashlib.sha256(latent.contiguous().numpy().tobytes()).hexdigest() for latent in encoded)
    manifest = dataset.manifest()
    for item, pixel_hash, latent_hash in zip(manifest, pixel_hashes, latent_hashes):
        item.update({"pixel_sha256": pixel_hash, "latent_sha256": latent_hash})
    grouped: dict[str, set[str]] = {}
    for item in manifest:
        grouped.setdefault(str(item["source_path"]), set()).add(str(item["pixel_sha256"]))
    expected_variants = int(data.get("augmentation_variants", 4)) if split == "train" else 1
    duplicate_sources = sorted(path for path, variants in grouped.items() if len(variants) != expected_variants)
    if duplicate_sources:
        raise RuntimeError(f"AFHQ deterministic augmentation produced non-unique pixel variants for {len(duplicate_sources)} source images; first={duplicate_sources[0]}")
    payload = {
        "latents": torch.cat(latents), "labels": torch.cat(labels),
        "relative_paths": [str(item["source_path"]) for item in manifest],
        "metadata": {
            "format_version": CACHE_FORMAT_VERSION, "dataset": "afhq-cat-official", "split": split,
            "resolution": int(data["resolution"]), "num_classes": 1, "vae_model_id": vae.model_id,
            "vae_revision": vae.revision, "latent_scaling_factor": vae.scaling_factor,
            "cache_seed": int(data.get("latent_seed", cfg.get("seed", 123))),
            "preprocessing": "train: deterministic random square crop (80-100%) -> optional horizontal flip -> bicubic resize(128) -> Normalize(0.5); test: center square crop -> bicubic resize(128) -> Normalize(0.5)",
            "augmentation_variants": int(data.get("augmentation_variants", 4)) if split == "train" else 1,
            "augmentation_scale": list(data.get("augmentation_scale", (0.85, 1.0))) if split == "train" else None,
            "source_image_count": len({item["source_path"] for item in manifest}),
        },
    }
    validate_cache(payload, expected_resolution=int(data["resolution"]), expected_vae_model_id=vae.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest_path = path.parent / f"{split}_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest), encoding="utf-8")
    print("cache_statistics: " + json.dumps(cache_statistics(payload, path), sort_keys=True))
    print("augmentation_validation: " + json.dumps({"source_count": len(grouped), "cache_count": len(manifest), "expected_variants_per_source": expected_variants, "unique_pixel_variants": sum(len(values) for values in grouped.values()), "duplicate_sources": len(duplicate_sources)}, sort_keys=True))
    print(f"manifest_written: {manifest_path}")
    return path


def reconstruction_grid(cfg: dict, count: int) -> Path:
    dataset = build_dataset(cfg, "test")
    images = torch.stack([dataset[index][0] for index in range(min(count, len(dataset)))])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_frozen_vae(cfg["vae"]["model_id"], device, cfg["vae"].get("revision"))
    generator = torch.Generator(device=device).manual_seed(int(cfg["data"].get("latent_seed", cfg.get("seed", 123))))
    reconstructions = decode_latents(vae, encode_latents(vae, images.to(device), generator)).cpu()
    if reconstructions.shape != images.shape or not torch.isfinite(reconstructions).all():
        raise RuntimeError("Invalid VAE reconstruction output")
    path = Path(cfg["output_dir"]) / "vae_reconstruction_test.png"; path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(((images + 1) * 0.5, (reconstructions + 1) * 0.5)).clamp(0, 1), path, nrow=len(images))
    print(f"reconstruction_grid_written: {path}")
    return path


def write_cache_report(cfg: dict) -> Path:
    train_path, test_path = cache_path(cfg, "train"), cache_path(cfg, "test")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Both AFHQ train.pt and test.pt are required before writing cache stats")
    train, test = load_cache(train_path, expected_resolution=cfg["data"]["resolution"], expected_vae_model_id=cfg["vae"]["model_id"]), load_cache(test_path, expected_resolution=cfg["data"]["resolution"], expected_vae_model_id=cfg["vae"]["model_id"])
    path = Path("reports") / f"{cfg['name']}_cache_stats.md"; path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        "# AFHQ Cats Cache Stats", "", "Generated by `prepare_afhq_cat_latents.py` from local official AFHQ files.", "",
        "| Split | Source images | Cached latents | Augmentation variants | Cache fingerprint |", "| --- | ---: | ---: | ---: | --- |",
        f"| train | {train['metadata']['source_image_count']} | {len(train['latents'])} | {train['metadata']['augmentation_variants']} | `{cache_statistics(train)['fingerprint']}` |",
        f"| test | {test['metadata']['source_image_count']} | {len(test['latents'])} | {test['metadata']['augmentation_variants']} | `{cache_statistics(test)['fingerprint']}` |", "",
        "The test cache is held out and must never be passed to `train_sit.py`.", "",
    ])
    path.write_text(text, encoding="utf-8"); print(f"cache_report_written: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official AFHQ cat train/test latent caches.")
    parser.add_argument("--config", required=True); parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--limit", type=int); parser.add_argument("--force", action="store_true"); parser.add_argument("--reconstruction-only", action="store_true"); parser.add_argument("--reconstruction-count", type=int, default=8)
    args = parser.parse_args(); cfg = load_config(args.config)
    reconstruction_grid(cfg, args.reconstruction_count)
    if not args.reconstruction_only:
        for split in (("train", "test") if args.split == "both" else (args.split,)):
            print(f"cache_written: {make_cache(cfg, split, force=args.force, limit=args.limit)}")
        if cache_path(cfg, "train").exists() and cache_path(cfg, "test").exists():
            write_cache_report(cfg)


if __name__ == "__main__":
    main()
