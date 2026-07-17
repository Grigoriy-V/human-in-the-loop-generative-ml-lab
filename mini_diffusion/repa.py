from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


REPA_CACHE_FORMAT_VERSION = 1
REPA_REQUIRED_METADATA = {
    "format_version", "teacher", "teacher_source", "teacher_revision",
    "teacher_input_resolution", "preprocessing", "latent_cache_fingerprint",
    "split", "feature_shape", "feature_dtype", "pooling", "class_mapping",
}


class RepaProjector(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 2048, output_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


def repa_loss(projected_student: torch.Tensor, teacher_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-wise cosine loss and mean cosine similarity, both accumulated in FP32."""
    student = F.normalize(projected_student.float(), dim=-1)
    teacher = F.normalize(teacher_features.detach().float(), dim=-1)
    cosine = (student * teacher).sum(dim=-1)
    return -cosine.mean(), cosine.mean()


def pool_teacher_grid(tokens: torch.Tensor) -> torch.Tensor:
    """Pool DINOv2 ViT-B/14 patch tokens [B, 256, 768] from 16x16 to SiT's 8x8 grid."""
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 768):
        raise ValueError(f"Expected DINO patch tokens [B, 256, 768], got {tuple(tokens.shape)}")
    grid = tokens.transpose(1, 2).reshape(tokens.shape[0], 768, 16, 16)
    return F.adaptive_avg_pool2d(grid, (8, 8)).flatten(2).transpose(1, 2).contiguous()


def flip_latents_and_features(latents: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3 or features.shape[1] != 64:
        raise ValueError("Teacher features must have shape [B, 64, D]")
    return latents.flip(-1), features.reshape(features.shape[0], 8, 8, features.shape[-1]).flip(2).reshape_as(features)


def feature_cache_paths(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    return {"root": root, "features": root / "features.npy", "labels": root / "labels.npy", "paths": root / "relative_paths.json", "metadata": root / "metadata.json"}


def feature_cache_fingerprint(metadata: dict[str, Any], features_path: str | Path) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with Path(features_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_cache(directory: str | Path, *, expected_latent_fingerprint: str | None = None, validate_values: bool = True) -> tuple[np.memmap, np.memmap, list[str], dict[str, Any]]:
    paths = feature_cache_paths(directory)
    if not all(path.exists() for path in (paths["features"], paths["labels"], paths["paths"], paths["metadata"])):
        raise FileNotFoundError(f"Incomplete REPA feature cache: {paths['root']}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not REPA_REQUIRED_METADATA.issubset(metadata) or metadata["format_version"] != REPA_CACHE_FORMAT_VERSION:
        raise ValueError("REPA feature cache metadata is incomplete or unsupported")
    if expected_latent_fingerprint is not None and metadata["latent_cache_fingerprint"] != expected_latent_fingerprint:
        raise ValueError("REPA feature cache does not match the current latent cache fingerprint")
    features = np.load(paths["features"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    relative_paths = json.loads(paths["paths"].read_text(encoding="utf-8"))
    if features.dtype != np.float16 or features.ndim != 3 or tuple(features.shape[1:]) != (64, 768):
        raise ValueError("REPA features must be float16 [N, 64, 768]")
    if labels.dtype != np.int64 or labels.shape != (features.shape[0],) or len(relative_paths) != features.shape[0]:
        raise ValueError("REPA labels or relative paths do not match features")
    if validate_values and not np.isfinite(features).all():
        raise ValueError("REPA feature cache contains NaN or Inf")
    return features, labels, relative_paths, metadata


class LatentFeatureDataset(Dataset):
    """Latents stay compact in RAM; the roughly 1 GB teacher array remains mmap-backed per worker."""
    def __init__(self, latent_payload: dict[str, Any], feature_cache_dir: str | Path) -> None:
        self.latents = latent_payload["latents"]
        self.labels = latent_payload["labels"]
        self.relative_paths = latent_payload["relative_paths"]
        self.feature_cache_dir = str(feature_cache_dir)
        self._features: np.memmap | None = None
        self._feature_labels: np.memmap | None = None
        self._feature_paths: list[str] | None = None
        self.metadata: dict[str, Any] | None = None
        self._validated_in_parent = False

    def _open_features(self) -> None:
        if self._features is None:
            self._features, self._feature_labels, self._feature_paths, self.metadata = load_feature_cache(self.feature_cache_dir, validate_values=not self._validated_in_parent)
            self._validated_in_parent = True
            if self._feature_paths != self.relative_paths or not np.array_equal(self._feature_labels, self.labels.numpy()):
                raise ValueError("Latent cache paths/labels do not match REPA feature cache")

    def __getstate__(self) -> dict[str, Any]:
        # Windows spawn workers must reopen the mmap, never pickle a full feature array.
        state = self.__dict__.copy()
        state["_features"] = None
        state["_feature_labels"] = None
        state["_feature_paths"] = None
        state["metadata"] = None
        return state

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._open_features()
        assert self._features is not None
        return self.latents[index], self.labels[index], torch.from_numpy(np.array(self._features[index], copy=True))


def atomic_replace_directory(source: Path, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"Feature cache exists: {destination}. Pass --force to overwrite it.")
        shutil.rmtree(destination)
    os.replace(source, destination)


def new_staging_directory(destination: Path) -> Path:
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    return staging
