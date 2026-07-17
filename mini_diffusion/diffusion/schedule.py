from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, steps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.steps = steps
        if schedule == "linear":
            betas = linear_beta_schedule(steps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def loss(self, model: nn.Module, x0: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        t = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, labels)
        return F.mse_loss(pred.float(), noise.float())

    @torch.inference_mode()
    def _predict_eps(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        if labels is not None and guidance_scale != 1.0 and getattr(model, "class_cond", False):
            cond = model(x, t, labels)
            uncond = model(x, t, None)
            return uncond + guidance_scale * (cond - uncond)
        return model(x, t, labels)

    @torch.inference_mode()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        eps = self._predict_eps(model, x, t, labels, guidance_scale)
        pred_x0 = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * eps
        ).clamp(-1, 1)
        mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * pred_x0
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        var = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        nonzero = (t != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        sampler: str = "ddpm",
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        device = torch.device(device)
        x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        with torch.autocast(device_type=device.type, enabled=False):
            if sampler == "ddpm":
                for i in reversed(range(self.steps)):
                    t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                    x = self.p_sample(model, x, t, labels, guidance_scale, generator)
            elif sampler == "ddim":
                if not 1 <= ddim_steps <= self.steps:
                    raise ValueError(f"ddim_steps must be in [1, {self.steps}]")
                timesteps = torch.linspace(
                    self.steps - 1, 0, ddim_steps, device=device, dtype=torch.float64
                ).round().long()
                next_timesteps = torch.cat([timesteps[1:], timesteps.new_tensor([-1])])
                for current, next_step in zip(timesteps.tolist(), next_timesteps.tolist()):
                    t = torch.full((shape[0],), current, device=device, dtype=torch.long)
                    eps = self._predict_eps(model, x, t, labels, guidance_scale)
                    alpha = self.alphas_cumprod[current]
                    pred_x0 = ((x - torch.sqrt(1.0 - alpha) * eps) / torch.sqrt(alpha)).clamp(-1, 1)
                    if next_step < 0:
                        x = pred_x0
                    else:
                        alpha_next = self.alphas_cumprod[next_step]
                        x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * eps
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"{sampler.upper()} sampling produced NaN or Inf values.")
        return x.clamp(-1, 1)
