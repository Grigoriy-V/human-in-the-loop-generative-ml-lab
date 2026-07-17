from __future__ import annotations

import torch


def _velocity(model, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None, guidance_scale: float) -> torch.Tensor:
    conditional = model(x, t, labels)
    if labels is None or guidance_scale == 1.0:
        return conditional
    null_labels = torch.full_like(labels, model.null_class)
    unconditional = model(x, t, null_labels)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.inference_mode()
def sample_ode(model, shape: tuple[int, ...], labels: torch.Tensor | None, device: torch.device, *, steps: int = 50,
               sampler: str = "heun", guidance_scale: float = 1.0, generator: torch.Generator | None = None,
               diagnostics: bool = False, initial_noise: torch.Tensor | None = None) -> torch.Tensor:
    if sampler not in {"euler", "heun"}:
        raise ValueError("sampler must be 'euler' or 'heun'")
    if steps < 1:
        raise ValueError("steps must be positive")
    if initial_noise is None:
        x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError("initial_noise shape must match shape")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("initial_noise must be finite")
        x = initial_noise.to(device=device, dtype=torch.float32).clone()
    times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            t = torch.full((shape[0],), times[index], device=device)
            dt = times[index + 1] - times[index]
            velocity = _velocity(model, x, t, labels, guidance_scale).float()
            if sampler == "heun" and index < steps - 1:
                predicted = x + dt * velocity
                next_t = torch.full((shape[0],), times[index + 1], device=device)
                next_velocity = _velocity(model, predicted, next_t, labels, guidance_scale).float()
                x = x + dt * (velocity + next_velocity) * 0.5
            else:
                x = x + dt * velocity
            if diagnostics and not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite latent after ODE step {index}")
    finally:
        model.train(was_training)
    return x
