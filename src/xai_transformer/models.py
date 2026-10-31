"""Model zoo and cached model construction."""

from dataclasses import dataclass
from functools import lru_cache

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    timm_name: str
    description: str
    prefix_tokens: int = 1


MODEL_ZOO = {
    "Supervised ViT": ModelSpec(
        "supervised_vit", "Supervised ViT", "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "ViT-B/16 architecture with ImageNet fine-tuned weights.",
    ),
    "DeiT": ModelSpec(
        "deit", "DeiT", "deit_base_patch16_224.fb_in1k",
        "Data-efficient Image Transformer.",
    ),
    "DINO": ModelSpec(
        "dino", "DINO", "vit_small_patch16_224.dino",
        "Self-supervised ViT trained with DINO.",
    ),
}


def inference_device() -> str:
    """Choose the fastest supported local inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=3)
def load_model(model_label: str, device_name: str = "cpu") -> tuple[nn.Module, ModelSpec]:
    """Load and cache one pretrained model; weights are downloaded by timm on first use."""
    if model_label not in MODEL_ZOO:
        raise ValueError(f"Unknown model {model_label!r}. Choose one of: {', '.join(MODEL_ZOO)}")
    spec = MODEL_ZOO[model_label]
    device = torch.device(device_name)
    try:
        model = timm.create_model(spec.timm_name, pretrained=True).eval().to(device)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Unable to load {spec.label} ({spec.timm_name}). "
            "Check network access and the local model cache."
        ) from error
    return model, spec
