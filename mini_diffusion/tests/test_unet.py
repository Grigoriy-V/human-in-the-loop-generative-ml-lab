import torch
from torch.nn import functional as F

from mini_diffusion.diffusion import GaussianDiffusion, UNet
from mini_diffusion.diffusion.blocks import AttentionBlock


def small_unet(class_cond: bool = True):
    return UNet(
        image_size=32,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        num_classes=10,
        class_cond=class_cond,
    )


def test_unet_preserves_resolution_with_and_without_labels():
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 8, (2,))
    labels = torch.tensor([1, 2])
    model = small_unet(class_cond=True)
    assert model(x, t, labels).shape == x.shape
    assert model(x, t, None).shape == x.shape
    no_label_model = small_unet(class_cond=False)
    assert no_label_model(x, t, None).shape == x.shape


def test_forward_backward_is_finite():
    model = small_unet()
    diffusion = GaussianDiffusion(steps=8, schedule="linear")
    x = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    loss = diffusion.loss(model, x, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_synthetic_one_batch_overfit_decreases_loss():
    torch.manual_seed(7)
    model = small_unet()
    diffusion = GaussianDiffusion(steps=8, schedule="linear")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    t = torch.randint(0, diffusion.steps, (2,), dtype=torch.long)
    noise = torch.randn_like(x)
    xt = diffusion.q_sample(x, t, noise)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xt, t, labels), noise)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.95


def test_sdpa_attention_matches_manual_forward_and_backward():
    torch.manual_seed(17)
    manual = AttentionBlock(32, num_heads=4, backend="manual")
    sdpa = AttentionBlock(32, num_heads=4, backend="sdpa")
    sdpa.load_state_dict(manual.state_dict())
    x_manual = torch.randn(2, 32, 8, 8, requires_grad=True)
    x_sdpa = x_manual.detach().clone().requires_grad_(True)

    out_manual = manual(x_manual)
    out_sdpa = sdpa(x_sdpa)
    torch.testing.assert_close(out_sdpa, out_manual, rtol=1e-5, atol=1e-6)

    out_manual.square().mean().backward()
    out_sdpa.square().mean().backward()
    torch.testing.assert_close(x_sdpa.grad, x_manual.grad, rtol=2e-5, atol=2e-6)
