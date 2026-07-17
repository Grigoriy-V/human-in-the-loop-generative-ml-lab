from __future__ import annotations

import copy

import torch

from mini_diffusion.diffusion import EMA
from mini_diffusion.latent_cache import CACHE_FORMAT_VERSION, load_cache, validate_cache
from mini_diffusion.sit import SiT, linear_interpolant, sample_ode, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, resume_checkpoint, save_checkpoint


def tiny_sit() -> SiT:
    return SiT(input_size=16, hidden_size=48, depth=2, num_heads=6, mlp_ratio=2.0, cond_drop_prob=0.0)


def test_cache_round_trip_and_validation(tmp_path) -> None:
    payload = {"latents": torch.randn(3, 4, 16, 16).half(), "labels": torch.tensor([0, 1, 9]), "relative_paths": ["a.jpg", "b.jpg", "c.jpg"], "metadata": {"format_version": CACHE_FORMAT_VERSION, "dataset": "imagenette-160", "split": "train", "resolution": 128, "vae_model_id": "stabilityai/sd-vae-ft-mse", "vae_revision": None, "latent_scaling_factor": 0.18215, "cache_seed": 123, "preprocessing": "test"}}
    path = tmp_path / "cache.pt"; torch.save(payload, path)
    assert load_cache(path, expected_resolution=128, expected_vae_model_id="stabilityai/sd-vae-ft-mse")["latents"].shape == (3, 4, 16, 16)
    broken = copy.deepcopy(payload); broken["metadata"].pop("cache_seed")
    try: validate_cache(broken)
    except ValueError: pass
    else: raise AssertionError("invalid metadata was accepted")


def test_patchify_and_forward_labels_and_null() -> None:
    model = tiny_sit(); x = torch.randn(2, 4, 16, 16); t = torch.tensor([0.2, 0.8])
    assert model(x, t, torch.tensor([1, 2])).shape == x.shape
    assert model(x, t, torch.full((2,), model.null_class)).shape == x.shape


def test_interpolant_endpoints_and_finite_backward() -> None:
    x0, noise = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16)
    xt0, target = linear_interpolant(x0, noise, torch.zeros(2)); xt1, _ = linear_interpolant(x0, noise, torch.ones(2))
    assert torch.equal(xt0, x0) and torch.equal(xt1, noise) and target.shape == x0.shape
    model = tiny_sit(); loss = velocity_loss(model(xt0, torch.zeros(2), torch.tensor([0, 1])), target); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_euler_heun_cfg_and_determinism() -> None:
    model = tiny_sit().eval(); labels = torch.tensor([1])
    for sampler in ("euler", "heun"):
        first = sample_ode(model, (1, 4, 16, 16), labels, torch.device("cpu"), steps=3, sampler=sampler, guidance_scale=1.5, generator=torch.Generator().manual_seed(7), diagnostics=True)
        second = sample_ode(model, (1, 4, 16, 16), labels, torch.device("cpu"), steps=3, sampler=sampler, guidance_scale=1.5, generator=torch.Generator().manual_seed(7), diagnostics=True)
        assert first.shape == (1, 4, 16, 16) and torch.isfinite(first).all() and torch.equal(first, second)


def test_sit_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    cfg = {"data": {"latent_resolution": 16, "num_classes": 10}, "model": {"patch_size": 2, "hidden_size": 48, "depth": 2, "num_heads": 6, "mlp_ratio": 2.0, "cond_drop_prob": 0.0}, "train": {"learning_rate": 1e-3, "weight_decay": 0.0, "ema_decay": 0.9}}
    model = build_model(cfg); optimizer = build_optimizer(model, cfg); ema = EMA(model, 0.9); path = tmp_path / "latest.pt"; save_checkpoint(path, model, optimizer, ema, cfg, 5, "test")
    restored = build_model(cfg); restored_optimizer = build_optimizer(restored, cfg); restored_ema = EMA(restored, 0.9)
    assert resume_checkpoint(str(path), restored, restored_optimizer, restored_ema, torch.device("cpu")) == 5
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))


def test_foreach_ema_updates() -> None:
    model = tiny_sit()
    ema = EMA(model, decay=0.5, foreach=True)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    assert any(not torch.equal(before[name], value) for name, value in ema.shadow.items())


def test_one_batch_overfit_velocity() -> None:
    torch.manual_seed(0); model = SiT(input_size=16, hidden_size=48, depth=2, num_heads=6, mlp_ratio=2.0, cond_drop_prob=0.0); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x0, noise, labels, t = torch.randn(4, 4, 16, 16), torch.randn(4, 4, 16, 16), torch.tensor([0, 1, 2, 3]), torch.tensor([0.2, 0.4, 0.6, 0.8]); xt, target = linear_interpolant(x0, noise, t); losses = []
    for _ in range(40):
        optimizer.zero_grad(); loss = velocity_loss(model(xt, t, labels), target); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.75
