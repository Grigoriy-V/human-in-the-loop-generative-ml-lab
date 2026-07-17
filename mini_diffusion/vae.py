from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class FrozenVAE:
    model: torch.nn.Module
    model_id: str
    revision: str | None
    scaling_factor: float


def load_frozen_vae(model_id: str, device: torch.device, revision: str | None = None) -> FrozenVAE:
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError("VAE support requires `pip install diffusers huggingface_hub safetensors`.") from exc
    vae = AutoencoderKL.from_pretrained(model_id, revision=revision)
    resolved_revision = revision
    if resolved_revision is None:
        try:
            from huggingface_hub import model_info
            resolved_revision = model_info(model_id).sha
        except Exception:
            resolved_revision = None
    vae.requires_grad_(False).eval().to(device)
    scaling_factor = float(vae.config.scaling_factor)
    if scaling_factor <= 0:
        raise ValueError("VAE config has invalid scaling_factor")
    print(f"vae_model_id: {model_id}")
    print(f"vae_revision: {resolved_revision or 'default'}")
    print(f"latent_scaling_factor: {scaling_factor}")
    return FrozenVAE(vae, model_id, resolved_revision, scaling_factor)


@torch.inference_mode()
def encode_latents(vae: FrozenVAE, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return vae.model.encode(images).latent_dist.sample(generator=generator) * vae.scaling_factor


@torch.inference_mode()
def decode_latents(vae: FrozenVAE, latents: torch.Tensor) -> torch.Tensor:
    return vae.model.decode(latents / vae.scaling_factor).sample
