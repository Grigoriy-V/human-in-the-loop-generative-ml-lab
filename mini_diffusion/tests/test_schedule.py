import torch
from torch import nn

from mini_diffusion.diffusion import GaussianDiffusion


def test_q_sample_shape_and_noise_scale():
    diffusion = GaussianDiffusion(steps=1000, schedule="linear")
    x0 = torch.zeros(2, 3, 32, 32)
    noise = torch.ones_like(x0)
    near_zero = diffusion.q_sample(x0, torch.zeros(2, dtype=torch.long), noise)
    high_t = diffusion.q_sample(x0, torch.full((2,), 999, dtype=torch.long), noise)
    assert near_zero.shape == x0.shape
    assert high_t.shape == x0.shape
    assert near_zero.abs().mean() < 0.02
    assert high_t.abs().mean() > 0.95


class ZeroNoiseModel(nn.Module):
    def forward(self, x, timesteps, labels=None):
        return torch.zeros_like(x)


def test_p_sample_clips_predicted_x0_before_posterior_mean():
    diffusion = GaussianDiffusion(steps=10, schedule="cosine")
    x = torch.full((2, 3, 8, 8), 10.0)
    t = torch.zeros(2, dtype=torch.long)

    result = diffusion.p_sample(ZeroNoiseModel(), x, t)

    assert torch.isfinite(result).all()
    assert result.min() >= -1 - 1e-5
    assert result.max() <= 1 + 1e-5
