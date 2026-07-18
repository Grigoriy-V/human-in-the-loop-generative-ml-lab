from __future__ import annotations

import json

import numpy as np
import torch

from mini_diffusion.diffusion import EMA
from mini_diffusion.latent_cache import CACHE_FORMAT_VERSION, cache_fingerprint
from mini_diffusion.repa import (
    REPA_CACHE_FORMAT_VERSION,
    LatentFeatureDataset,
    RepaProjector,
    feature_cache_paths,
    flip_latents_and_features,
    pool_teacher_grid,
    repa_loss,
)
from mini_diffusion.sit import SiT, linear_interpolant, sample_ode, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, build_projector, resume_checkpoint, save_checkpoint


def cfg(repa: bool = True) -> dict:
    result = {"seed": 123, "data": {"latent_resolution": 16, "num_classes": 10}, "model": {"patch_size": 2, "hidden_size": 48, "depth": 2, "num_heads": 6, "mlp_ratio": 2.0, "cond_drop_prob": 0.0}, "train": {"learning_rate": 1e-3, "weight_decay": 0.0, "ema_decay": 0.9}, "performance": {}}
    if repa:
        result["repa"] = {"enabled": True, "alignment_depth": 1, "projector_hidden_dim": 64, "teacher_feature_dim": 768, "coefficient": 0.5, "feature_cache_dir": "unused"}
    return result


def payload() -> dict:
    return {"latents": torch.randn(3, 4, 16, 16).half(), "labels": torch.tensor([0, 1, 2]), "relative_paths": ["a.jpg", "b.jpg", "c.jpg"], "metadata": {"format_version": CACHE_FORMAT_VERSION, "dataset": "imagenette-160", "split": "train", "resolution": 128, "vae_model_id": "stabilityai/sd-vae-ft-mse", "vae_revision": None, "latent_scaling_factor": 0.18215, "cache_seed": 123, "preprocessing": "test"}}


def write_feature_cache(root, cached: dict) -> None:
    paths = feature_cache_paths(root); paths["root"].mkdir(parents=True)
    features = np.random.default_rng(0).normal(size=(3, 64, 768)).astype(np.float16)
    np.save(paths["features"], features); np.save(paths["labels"], cached["labels"].numpy())
    paths["paths"].write_text(json.dumps(cached["relative_paths"]), encoding="utf-8")
    metadata = {"format_version": REPA_CACHE_FORMAT_VERSION, "teacher": "dinov2_vitb14", "teacher_source": "facebookresearch/dinov2", "teacher_revision": "test", "teacher_input_resolution": 224, "preprocessing": "test", "latent_cache_fingerprint": cache_fingerprint(cached), "split": "train", "feature_shape": [3, 64, 768], "feature_dtype": "float16", "pooling": "test", "class_mapping": [0, 1, 2], "fingerprint": "test"}
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")


def test_feature_cache_path_label_matching_and_metadata(tmp_path) -> None:
    cached = payload(); root = tmp_path / "features"; write_feature_cache(root, cached)
    dataset = LatentFeatureDataset(cached, root)
    latent, label, features = dataset[1]
    assert latent.shape == (4, 16, 16) and label.item() == 1 and features.shape == (64, 768)
    assert features.dtype == torch.float16 and torch.isfinite(features).all()
    broken = payload(); broken["relative_paths"] = ["wrong.jpg", "b.jpg", "c.jpg"]
    try: LatentFeatureDataset(broken, root)[0]
    except ValueError: pass
    else: raise AssertionError("mismatched relative paths were accepted")


def test_pooling_flip_projector_and_cosine_loss() -> None:
    tokens = torch.arange(2 * 256 * 768, dtype=torch.float32).reshape(2, 256, 768)
    pooled = pool_teacher_grid(tokens)
    assert pooled.shape == (2, 64, 768)
    latents, flipped = flip_latents_and_features(torch.randn(2, 4, 16, 16), pooled)
    assert latents.shape == (2, 4, 16, 16) and flipped.shape == pooled.shape
    projector = RepaProjector(384, 64, 768); student = projector(torch.randn(2, 64, 384)); loss, cosine = repa_loss(student, pooled)
    assert torch.isfinite(loss) and torch.isfinite(cosine) and -1.0 <= float(cosine.detach()) <= 1.0


def test_teacher_no_grad_alignment_block_and_base_initialization() -> None:
    torch.manual_seed(77); baseline = build_model(cfg(False))
    torch.manual_seed(77); repa_model = build_model(cfg(True)); projector = build_projector(cfg(True))
    assert all(torch.equal(a, b) for a, b in zip(baseline.parameters(), repa_model.parameters()))
    x, t, labels, teacher = torch.randn(2, 4, 16, 16), torch.tensor([0.2, 0.8]), torch.tensor([0, 1]), torch.randn(2, 64, 768, requires_grad=True)
    velocity, hidden = repa_model(x, t, labels, return_hidden_after=1); loss = velocity_loss(velocity, torch.randn_like(velocity)) + 0.5 * repa_loss(projector(hidden), teacher)[0]; loss.backward()
    assert teacher.grad is None and projector.net[0].weight.grad is not None and repa_model.blocks[0].qkv.weight.grad is not None


def test_repa_disabled_matches_forward_and_sampling_needs_no_projector() -> None:
    model = SiT(input_size=16, hidden_size=48, depth=2, num_heads=6, mlp_ratio=2.0, cond_drop_prob=0.0).eval()
    x, t, labels = torch.randn(1, 4, 16, 16), torch.tensor([0.4]), torch.tensor([2])
    first, second = model(x, t, labels), model(x, t, labels)
    assert torch.equal(first, second)
    generated = sample_ode(model, (1, 4, 16, 16), labels, torch.device("cpu"), steps=2, sampler="heun", generator=torch.Generator().manual_seed(5))
    assert generated.shape == x.shape and torch.isfinite(generated).all()


def test_repa_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    config = cfg(True); model = build_model(config); projector = build_projector(config); combined = torch.nn.ModuleList([model, projector]); optimizer = build_optimizer(combined, config); ema = EMA(model, 0.9)
    path = tmp_path / "repa.pt"; save_checkpoint(path, model, optimizer, ema, config, 5, "latent-fp", projector, {"fingerprint": "feature-fp"})
    restored = build_model(config); restored_projector = build_projector(config); restored_optimizer = build_optimizer(torch.nn.ModuleList([restored, restored_projector]), config); restored_ema = EMA(restored, 0.9)
    assert resume_checkpoint(str(path), restored, restored_optimizer, restored_ema, torch.device("cpu"), "latent-fp", restored_projector, "feature-fp") == 5
    assert all(torch.equal(a, b) for a, b in zip(projector.parameters(), restored_projector.parameters()))


def test_one_batch_repa_overfit_fixed_inputs() -> None:
    torch.manual_seed(9); model = SiT(input_size=16, hidden_size=48, depth=2, num_heads=6, mlp_ratio=2.0, cond_drop_prob=0.0); projector = RepaProjector(48, 64, 768); optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=3e-3)
    x0, noise, labels, t, teacher = torch.randn(4, 4, 16, 16), torch.randn(4, 4, 16, 16), torch.tensor([0, 1, 2, 3]), torch.tensor([0.2, 0.4, 0.6, 0.8]), torch.randn(4, 64, 768); xt, target = linear_interpolant(x0, noise, t); total, cosine = [], []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True); velocity, hidden = model(xt, t, labels, return_hidden_after=1); flow = velocity_loss(velocity, target); align, similarity = repa_loss(projector(hidden), teacher); loss = flow + 0.5 * align; loss.backward(); optimizer.step(); total.append(float(loss.detach())); cosine.append(float(similarity.detach()))
    assert sum(total[-10:]) / 10 < sum(total[:10]) / 10 * 0.8
    assert sum(cosine[-10:]) / 10 > sum(cosine[:10]) / 10
