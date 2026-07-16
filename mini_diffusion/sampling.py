from __future__ import annotations

import torch


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def denormalize_to_unit(images: torch.Tensor) -> torch.Tensor:
    return ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def sample_statistics(images: torch.Tensor) -> dict[str, float | int | bool]:
    values = images.detach().float()
    finite = torch.isfinite(values)
    unit = denormalize_to_unit(values)
    flat_dims = tuple(range(1, unit.ndim))
    black = (unit.amax(dim=flat_dims) == 0).sum()
    white = (unit.amin(dim=flat_dims) == 1).sum()
    saturated = (values <= -1.0) | (values >= 1.0)
    return {
        "min": float(values.min().cpu()),
        "max": float(values.max().cpu()),
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
        "isfinite": bool(finite.all().cpu()),
        "saturation_rate": float(saturated.float().mean().cpu()),
        "black_failure_count": int(black.cpu()),
        "white_failure_count": int(white.cpu()),
    }
