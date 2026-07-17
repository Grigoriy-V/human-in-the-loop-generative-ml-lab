from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch

CACHE_FORMAT_VERSION = 1
REQUIRED_METADATA = {"format_version", "dataset", "split", "resolution", "vae_model_id", "vae_revision", "latent_scaling_factor", "cache_seed", "preprocessing"}


def validate_cache(payload: dict[str, Any], *, expected_resolution: int | None = None, expected_vae_model_id: str | None = None) -> None:
    if not isinstance(payload, dict) or set(payload) != {"latents", "labels", "relative_paths", "metadata"}:
        raise ValueError("Invalid latent cache top-level keys")
    latents, labels, paths, metadata = payload["latents"], payload["labels"], payload["relative_paths"], payload["metadata"]
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or tuple(latents.shape[1:]) != (4, 16, 16):
        raise ValueError("Latents must have shape [N, 4, 16, 16]")
    if latents.dtype not in {torch.float16, torch.bfloat16, torch.float32} or not torch.isfinite(latents).all():
        raise ValueError("Latents must use a floating dtype and be finite")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != latents.shape[0] or labels.dtype != torch.long:
        raise ValueError("Labels must be int64 [N] and match latents")
    if (labels < 0).any() or (labels >= 10).any() or not isinstance(paths, list) or len(paths) != latents.shape[0]:
        raise ValueError("Invalid Imagenette labels or relative paths")
    if not isinstance(metadata, dict) or not REQUIRED_METADATA.issubset(metadata):
        raise ValueError("Latent cache metadata is incomplete")
    if metadata["format_version"] != CACHE_FORMAT_VERSION or metadata["resolution"] != 128:
        raise ValueError("Unsupported cache version or resolution")
    if expected_resolution is not None and metadata["resolution"] != expected_resolution:
        raise ValueError("Cache resolution is incompatible with config")
    if expected_vae_model_id is not None and metadata["vae_model_id"] != expected_vae_model_id:
        raise ValueError("Cache VAE model is incompatible with config")
    if not isinstance(metadata["latent_scaling_factor"], (float, int)) or metadata["latent_scaling_factor"] <= 0:
        raise ValueError("Invalid latent scaling factor")


def load_cache(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_cache(payload, **kwargs)
    return payload


def cache_fingerprint(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(repr(sorted(payload["metadata"].items())).encode())
    digest.update(payload["latents"].contiguous().numpy().tobytes())
    digest.update(payload["labels"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_statistics(payload: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    x = payload["latents"].float()
    stats: dict[str, Any] = {"count": int(x.shape[0]), "latent_shape": list(x.shape[1:]), "dtype": str(payload["latents"].dtype), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max()), "nan": int(torch.isnan(x).sum()), "inf": int(torch.isinf(x).sum()), "label_distribution": dict(sorted(Counter(payload["labels"].tolist()).items())), "fingerprint": cache_fingerprint(payload)}
    if path is not None:
        stats["size_bytes"] = Path(path).stat().st_size
    return stats
