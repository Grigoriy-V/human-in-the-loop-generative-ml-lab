import pytest
import torch
from torchvision.utils import save_image

from mini_diffusion.diffusion import EMA, GaussianDiffusion, UNet
from mini_diffusion.sampling import denormalize_to_unit, make_generator, sample_statistics
from mini_diffusion.train import write_samples


def build_tiny_model():
    return UNet(
        image_size=16,
        base_channels=8,
        channel_mults=(1,),
        num_res_blocks=1,
        attention_resolutions=(),
        num_classes=2,
        class_cond=True,
    )


def test_short_sampling_run_creates_image(tmp_path):
    model = build_tiny_model()
    diffusion = GaussianDiffusion(steps=2, schedule="linear")
    labels = torch.tensor([0])
    images = diffusion.sample(model, (1, 3, 16, 16), labels=labels)
    path = tmp_path / "sample.png"
    save_image((images + 1) * 0.5, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_sampling_is_deterministic_and_does_not_change_global_rng_or_parameters():
    model = build_tiny_model().eval()
    diffusion = GaussianDiffusion(steps=2, schedule="linear")
    labels = torch.tensor([1])
    parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng_state = torch.get_rng_state().clone()

    first = diffusion.sample(
        model,
        (1, 3, 16, 16),
        labels=labels,
        generator=make_generator("cpu", 77),
    )
    second = diffusion.sample(
        model,
        (1, 3, 16, 16),
        labels=labels,
        generator=make_generator("cpu", 77),
    )

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name, value in model.state_dict().items():
        assert torch.equal(value, parameters[name])


def test_periodic_sampling_is_repeatable_and_restores_training_mode(tmp_path):
    model = build_tiny_model().train()
    diffusion = GaussianDiffusion(steps=2, schedule="linear")
    ema = EMA(model, decay=0.99)
    cfg = {
        "output_dir": str(tmp_path),
        "seed": 17,
        "data": {"resolution": 16, "num_classes": 2},
        "train": {"sample_count": 4},
        "sampling": {"preview_seed": 29, "guidance_scale": 1.0},
    }
    parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}

    first = write_samples(model, diffusion, ema, cfg, torch.device("cpu"), step=1)
    second = write_samples(model, diffusion, ema, cfg, torch.device("cpu"), step=2)

    assert first.read_bytes() == second.read_bytes()
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, parameters[name])


def test_denormalization_and_statistics():
    images = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    unit = denormalize_to_unit(images)
    stats = sample_statistics(images)

    assert torch.equal(unit, torch.tensor([[[[0.0, 0.5, 1.0]]]]))
    assert stats["isfinite"]
    assert stats["saturation_rate"] == pytest.approx(2 / 3)
    assert stats["black_failure_count"] == 0
    assert stats["white_failure_count"] == 0
