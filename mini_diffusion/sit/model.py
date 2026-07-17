from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimestepEmbedder, sincos_2d_position_embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class SiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        batch, tokens, channels = attn_input.shape
        qkv = self.qkv(attn_input).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attention = attention.transpose(1, 2).reshape(batch, tokens, channels)
        x = x + gate_msa[:, None] * self.proj(attention)
        x = x + gate_mlp[:, None] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SiT(nn.Module):
    """Class-conditioned SiT-S/2 for 4x16x16 Stable Diffusion VAE latents."""
    def __init__(self, input_size: int = 16, in_channels: int = 4, patch_size: int = 2, hidden_size: int = 384,
                 depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0, num_classes: int = 10,
                 cond_drop_prob: float = 0.1) -> None:
        super().__init__()
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.input_size, self.in_channels, self.patch_size = input_size, in_channels, patch_size
        self.num_classes, self.cond_drop_prob = num_classes, cond_drop_prob
        self.grid_size = (input_size // patch_size, input_size // patch_size)
        self.x_embedder = PatchEmbed(in_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.register_buffer("pos_embed", sincos_2d_position_embedding(hidden_size, self.grid_size)[None], persistent=False)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # adaLN-Zero and output are deliberately zero at initialization.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    @property
    def null_class(self) -> int:
        return self.num_classes

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1:] != (self.in_channels, self.input_size, self.input_size):
            raise ValueError(f"Expected [B,{self.in_channels},{self.input_size},{self.input_size}], got {tuple(x.shape)}")
        if labels is None:
            labels = torch.full((x.shape[0],), self.null_class, device=x.device, dtype=torch.long)
        labels = labels.long()
        if self.training and self.cond_drop_prob:
            drop = torch.rand(labels.shape, device=x.device) < self.cond_drop_prob
            labels = torch.where(drop, torch.full_like(labels, self.null_class), labels)
        if (labels < 0).any() or (labels > self.null_class).any():
            raise ValueError("labels must be class ids or the null class id")
        tokens = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)
        conditioning = self.t_embedder(t) + self.y_embedder(labels)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        return self.unpatchify(self.final_layer(tokens, conditioning))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, tokens, _ = x.shape
        h, w = self.grid_size
        if tokens != h * w:
            raise ValueError("Token count does not match configured grid")
        p, c = self.patch_size, self.in_channels
        return x.reshape(b, h, w, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h * p, w * p)
