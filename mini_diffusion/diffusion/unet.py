from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import AttentionBlock, Downsample, ResBlock, Upsample, group_norm, sinusoidal_embedding


@dataclass
class BlockSpec:
    block: nn.Module
    uses_skip: bool = False


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8,),
        dropout: float = 0.0,
        num_classes: int | None = None,
        class_cond: bool = True,
        cond_drop_prob: float = 0.0,
        num_heads: int = 1,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_cond = bool(class_cond and num_classes is not None)
        self.cond_drop_prob = cond_drop_prob
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.label_emb = nn.Embedding(num_classes + 1, time_dim) if self.class_cond else None
        self.null_label = num_classes if self.class_cond else None

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        ch = base_channels
        resolution = image_size
        self.downs = nn.ModuleList()
        skip_channels: list[int] = []
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout)])
                ch = out_ch
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.downs.append(layers)
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                resolution //= 2

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                layers = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, time_dim, dropout)])
                ch = out_ch
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.ups.append(layers)
            if level != 0:
                self.ups.append(nn.ModuleList([Upsample(ch)]))
                resolution *= 2

        self.out = nn.Sequential(
            group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def _embedding(self, t: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))
        if self.class_cond:
            if labels is None:
                labels = torch.full((t.shape[0],), self.null_label, device=t.device, dtype=torch.long)
            else:
                labels = labels.to(device=t.device, dtype=torch.long)
                if self.training and self.cond_drop_prob > 0:
                    drop = torch.rand(labels.shape, device=t.device) < self.cond_drop_prob
                    labels = torch.where(drop, torch.full_like(labels, self.null_label), labels)
            emb = emb + self.label_emb(labels)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._embedding(t, labels)
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []

        for layers in self.downs:
            if isinstance(layers[0], Downsample):
                h = layers[0](h)
                continue
            h = layers[0](h, emb)
            for layer in layers[1:]:
                h = layer(h)
            # skip tensor shape is [B, C, H, W] at the current U-Net scale.
            skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for layers in self.ups:
            if isinstance(layers[0], Upsample):
                h = layers[0](h)
                continue
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = layers[0](h, emb)
            for layer in layers[1:]:
                h = layer(h)
        return self.out(h)
