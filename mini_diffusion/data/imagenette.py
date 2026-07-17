from __future__ import annotations

import tarfile
from pathlib import Path
from urllib.request import urlretrieve

from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGENETTE_160_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def convert_rgb(image):
    return image.convert("RGB")


def imagenette_root(root: str | Path) -> Path:
    root = Path(root)
    return root / "imagenette2-160" if (root / "imagenette2-160").is_dir() else root


def ensure_imagenette(root: str | Path, download: bool = False) -> Path:
    root = Path(root)
    dataset_root = imagenette_root(root)
    if (dataset_root / "train").is_dir() and (dataset_root / "val").is_dir():
        return dataset_root
    if not download:
        raise FileNotFoundError(f"Imagenette-160 was not found at {root}. Run prepare_latents.py with --download-dataset.")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-160.tgz"
    if not archive.exists():
        print(f"downloading_imagenette: {IMAGENETTE_160_URL}")
        urlretrieve(IMAGENETTE_160_URL, archive)
    print(f"extracting_imagenette: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")
    return ensure_imagenette(root, download=False)


def build_imagenette(root: str | Path, split: str, resolution: int, download: bool = False) -> Dataset:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    dataset_root = ensure_imagenette(root, download=download)
    transform = transforms.Compose([
        transforms.Lambda(convert_rgb),
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return datasets.ImageFolder(dataset_root / split, transform=transform)
