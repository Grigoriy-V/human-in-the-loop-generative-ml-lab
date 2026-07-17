from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_interpolant(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x0.shape != noise.shape:
        raise ValueError("x0 and noise shapes must match")
    scale = t.reshape(-1, *([1] * (x0.ndim - 1))).to(dtype=x0.dtype)
    return (1.0 - scale) * x0 + scale * noise, noise - x0


def velocity_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction.float(), target.float())
