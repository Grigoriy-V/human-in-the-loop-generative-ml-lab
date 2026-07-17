from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from mini_diffusion.data import build_imagenette
from mini_diffusion.latent_cache import CACHE_FORMAT_VERSION, cache_statistics, validate_cache
from mini_diffusion.vae import decode_latents, encode_latents, load_frozen_vae


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cache_path(cfg: dict, split: str) -> Path:
    return Path(cfg["data"]["cache_dir"]) / f"{split}.pt"


def image_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


def make_cache(cfg: dict, split: str, limit: int | None, force: bool, download_dataset: bool) -> Path:
    data_cfg, vae_cfg = cfg["data"], cfg["vae"]
    path = cache_path(cfg, split)
    if path.exists() and not force:
        raise FileExistsError(f"Cache exists: {path}. Pass --force to overwrite it.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_imagenette(data_cfg["root"], split, data_cfg["resolution"], download=download_dataset)
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    vae = load_frozen_vae(vae_cfg["model_id"], device, vae_cfg.get("revision"))
    generator = torch.Generator(device=device).manual_seed(int(data_cfg.get("cache_seed", 123)))
    latents, labels, paths = [], [], []
    loader = image_loader(dataset, int(data_cfg.get("prepare_batch_size", 16)), int(data_cfg.get("num_workers", 0)))
    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(tqdm(loader, desc=f"encode_{split}")):
            encoded = encode_latents(vae, images.to(device), generator).cpu().to(torch.float16)
            latents.append(encoded)
            labels.append(batch_labels.long().cpu())
            base = batch_index * loader.batch_size
            for index in range(images.shape[0]):
                source_index = dataset.indices[base + index] if isinstance(dataset, Subset) else base + index
                image_path = dataset.dataset.samples[source_index][0] if isinstance(dataset, Subset) else dataset.samples[source_index][0]
                root = dataset.dataset.root if isinstance(dataset, Subset) else dataset.root
                paths.append(str(Path(image_path).resolve().relative_to(Path(root).resolve())))
    payload = {"latents": torch.cat(latents), "labels": torch.cat(labels), "relative_paths": paths,
               "metadata": {"format_version": CACHE_FORMAT_VERSION, "dataset": "imagenette-160", "split": split,
                            "resolution": data_cfg["resolution"], "vae_model_id": vae.model_id, "vae_revision": vae.revision,
                            "latent_scaling_factor": vae.scaling_factor, "cache_seed": int(data_cfg.get("cache_seed", 123)),
                            "preprocessing": "RGB -> Resize(shorter_side=128,bicubic) -> CenterCrop(128) -> ToTensor -> Normalize(0.5,0.5)"}}
    validate_cache(payload, expected_resolution=data_cfg["resolution"], expected_vae_model_id=vae.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print("cache_statistics: " + json.dumps(cache_statistics(payload, path), sort_keys=True))
    return path


def reconstruction_grid(cfg: dict, count: int, download_dataset: bool) -> Path:
    data_cfg, vae_cfg = cfg["data"], cfg["vae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_imagenette(data_cfg["root"], "val", data_cfg["resolution"], download=download_dataset)
    images = torch.stack([dataset[index][0] for index in range(min(count, len(dataset)))])
    vae = load_frozen_vae(vae_cfg["model_id"], device, vae_cfg.get("revision"))
    generator = torch.Generator(device=device).manual_seed(int(data_cfg.get("cache_seed", 123)))
    reconstructions = decode_latents(vae, encode_latents(vae, images.to(device), generator)).cpu()
    if reconstructions.shape != images.shape or not torch.isfinite(reconstructions).all():
        raise RuntimeError("Invalid VAE reconstruction output")
    raw_min, raw_max = float(reconstructions.min()), float(reconstructions.max())
    # Stable Diffusion VAEs can slightly overshoot the normalized RGB interval.
    # Clamp only at the PNG boundary; finite/shapes are validated above.
    grid = torch.cat(((images + 1) * 0.5, (reconstructions + 1) * 0.5)).clamp(0, 1)
    output = Path(cfg["output_dir"]) / "vae_reconstruction.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output, nrow=images.shape[0])
    print(f"reconstruction_raw_range: [{raw_min:.6f}, {raw_max:.6f}]")
    print(f"reconstruction_grid_written: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--reconstruction-only", action="store_true")
    parser.add_argument("--reconstruction-count", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    reconstruction_grid(cfg, args.reconstruction_count, args.download_dataset)
    if not args.reconstruction_only:
        for split in ("train", "val"):
            print(f"cache_written: {make_cache(cfg, split, args.limit, args.force, args.download_dataset)}")


if __name__ == "__main__":
    main()
