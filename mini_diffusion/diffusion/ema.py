from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, foreach: bool = False):
        self.decay = decay
        self.foreach = foreach
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.foreach:
            names_and_params = [
                (name, p.detach()) for name, p in model.named_parameters() if p.requires_grad
            ]
            shadows = [self.shadow[name] for name, _ in names_and_params]
            params = [p for _, p in names_and_params]
            torch._foreach_mul_(shadows, self.decay)
            torch._foreach_add_(shadows, params, alpha=1.0 - self.decay)
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.copy_(self.shadow[name].to(p.device, dtype=p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in backup:
                        p.copy_(backup[name].to(p.device, dtype=p.dtype))
