from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Continuous sinusoidal embeddings for normalized timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat((args.cos(), args.sin()), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.mlp[0].in_features))


def sincos_2d_position_embedding(embed_dim: int, grid_size: tuple[int, int]) -> torch.Tensor:
    """Fixed 2D embedding, parameterized by grid size for future resolutions."""
    if embed_dim % 4:
        raise ValueError("embed_dim must be divisible by four for 2D sin/cos embeddings")
    h, w = grid_size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = 1.0 / (10_000 ** (omega / (embed_dim // 4)))
    out_y = y.reshape(-1, 1).float() * omega[None]
    out_x = x.reshape(-1, 1).float() * omega[None]
    return torch.cat((out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()), dim=1)
