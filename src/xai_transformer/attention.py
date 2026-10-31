"""Attention extraction and rollout mathematics."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

Fusion = Literal["mean", "min", "max"]


@dataclass
class AttentionCapture:
    """Registers hooks and stores attention tensors for one forward pass."""

    model: nn.Module
    prefix_tokens: int = 1

    def __post_init__(self) -> None:
        self.attentions: list[torch.Tensor] = []
        self._handles = [
            module.register_forward_hook(self._hook)
            for module in self.model.modules()
            if hasattr(module, "qkv") and hasattr(module, "num_heads")
        ]
        if not self._handles:
            raise ValueError("No compatible attention modules with qkv and num_heads were found.")

    def _hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], _: object) -> None:
        if not inputs:
            return
        tokens = inputs[0]
        if tokens.ndim != 3:
            raise ValueError(f"Expected attention input [batch, tokens, channels], got {tokens.shape}.")
        qkv = module.qkv(tokens)
        batch, token_count, _ = qkv.shape
        heads = int(module.num_heads)
        head_dim = qkv.shape[-1] // (3 * heads)
        qkv = qkv.reshape(batch, token_count, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key = qkv[0], qkv[1]
        query = module.q_norm(query) if hasattr(module, "q_norm") else query
        key = module.k_norm(key) if hasattr(module, "k_norm") else key
        scale = getattr(module, "scale", head_dim ** -0.5)
        attention = (query * scale) @ key.transpose(-2, -1)
        self.attentions.append(attention.softmax(dim=-1).detach())

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@torch.inference_mode()
def extract_attention(model: nn.Module, image: torch.Tensor) -> list[torch.Tensor]:
    """Run a model once and return ``L`` tensors shaped ``[batch, heads, tokens, tokens]``."""
    capture = AttentionCapture(model, int(getattr(model, "num_prefix_tokens", 1)))
    try:
        model(image)
        if not capture.attentions:
            raise RuntimeError("The model forward pass produced no attention matrices.")
        return capture.attentions
    finally:
        capture.close()


class AttentionRollout:
    """Compute layer-wise rollout with residual connections and mass thresholding."""

    def __init__(self, attentions: list[torch.Tensor], prefix_tokens: int = 1) -> None:
        if not attentions:
            raise ValueError("At least one attention matrix is required.")
        if any(attention.ndim != 4 for attention in attentions):
            raise ValueError("Attention matrices must have shape [batch, heads, tokens, tokens].")
        if prefix_tokens < 1:
            raise ValueError("prefix_tokens must be at least 1.")
        batch_size = attentions[0].shape[0]
        token_count = attentions[0].shape[-1]
        if any(attention.shape[0] != batch_size for attention in attentions):
            raise ValueError("All attention matrices must use the same batch size.")
        if any(a.shape[-2:] != (token_count, token_count) for a in attentions):
            raise ValueError("All attention matrices must use the same token dimensions.")
        if prefix_tokens >= token_count:
            raise ValueError("prefix_tokens must be smaller than the token count.")
        self.attentions = attentions
        self.prefix_tokens = prefix_tokens

    @staticmethod
    def _fuse(attention: torch.Tensor, fusion: Fusion) -> torch.Tensor:
        if fusion == "mean":
            return attention.mean(dim=1)
        if fusion == "min":
            return attention.min(dim=1).values
        if fusion == "max":
            return attention.max(dim=1).values
        raise ValueError(f"Unknown fusion {fusion!r}.")

    @staticmethod
    def _threshold(matrix: torch.Tensor, keep_percent: float) -> torch.Tensor:
        if not 0 < keep_percent <= 100:
            raise ValueError("keep_percent must be in the interval (0, 100].")
        if keep_percent == 100:
            return matrix
        target_mass = matrix.sum(dim=-1, keepdim=True) * (keep_percent / 100)
        descending, indices = matrix.sort(dim=-1, descending=True)
        cumulative = descending.cumsum(dim=-1)
        selected = cumulative <= target_mass
        selected[..., 0] = True
        mask = torch.zeros_like(matrix, dtype=torch.bool).scatter(-1, indices, selected)
        return matrix.masked_fill(~mask, 0)

    def up_to(
        self, layer: int, fusion: Fusion = "mean", keep_percent: float = 100
    ) -> torch.Tensor:
        """Return joint attention through 1-based ``layer`` as ``[batch, tokens, tokens]``.

        ``keep_percent`` is the retained attention mass per row, not the number
        of retained entries. This keeps the control meaningful across heads and
        layers with different attention distributions.
        """
        if not 1 <= layer <= len(self.attentions):
            raise ValueError(f"layer must be between 1 and {len(self.attentions)}.")
        tokens = self.attentions[0].shape[-1]
        rollout = torch.eye(tokens, device=self.attentions[0].device).expand(
            self.attentions[0].shape[0], -1, -1
        )
        for attention in self.attentions[:layer]:
            fused = self._threshold(self._fuse(attention, fusion), keep_percent)
            fused = fused + torch.eye(tokens, device=fused.device).unsqueeze(0)
            rollout = fused / fused.sum(dim=-1, keepdim=True) @ rollout
        return rollout / rollout.sum(dim=-1, keepdim=True)

    def cls_map(self, layer: int, fusion: Fusion = "mean", keep_percent: float = 100) -> torch.Tensor:
        """Extract CLS-to-patch attention and resize it to 224x224."""
        rollout = self.up_to(layer, fusion, keep_percent)
        patch_count = rollout.shape[-1] - self.prefix_tokens
        side = int(patch_count**0.5)
        if side * side != patch_count:
            raise ValueError(f"Expected a square patch grid, got {patch_count} patches.")
        grid = rollout[:, 0, self.prefix_tokens:].reshape(-1, 1, side, side)
        return F.interpolate(grid, size=(224, 224), mode="bicubic", align_corners=False).squeeze(1)

    def layer_maps(
        self, layers: list[int], fusion: Fusion = "mean", keep_percent: float = 100
    ) -> list[torch.Tensor]:
        """Return CLS maps for multiple layers in one user-facing operation."""
        if not layers:
            raise ValueError("At least one layer is required.")
        if len(set(layers)) != len(layers):
            raise ValueError("Layer progression cannot contain duplicates.")
        return [self.cls_map(layer, fusion, keep_percent) for layer in layers]

    def head_map(self, layer: int, head: int) -> torch.Tensor:
        """Return one head's CLS-to-patch map for a selected layer."""
        if not 1 <= layer <= len(self.attentions):
            raise ValueError(f"layer must be between 1 and {len(self.attentions)}.")
        attention = self.attentions[layer - 1]
        if not 0 <= head < attention.shape[1]:
            raise ValueError(f"head must be between 0 and {attention.shape[1] - 1}.")
        patch_count = attention.shape[-1] - self.prefix_tokens
        side = int(patch_count**0.5)
        if side * side != patch_count:
            raise ValueError(f"Expected a square patch grid, got {patch_count} patches.")
        grid = attention[:, head, 0, self.prefix_tokens:].reshape(-1, 1, side, side)
        return F.interpolate(grid, size=(224, 224), mode="bicubic", align_corners=False).squeeze(1)
