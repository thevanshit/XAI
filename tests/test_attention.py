import pytest
import torch
from torch import nn

from xai_transformer.attention import AttentionRollout, extract_attention


def matrices(layers=3, heads=2, tokens=17):
    return [torch.softmax(torch.randn(1, heads, tokens, tokens), dim=-1) for _ in range(layers)]


def test_rollout_is_row_normalized():
    result = AttentionRollout(matrices()).up_to(3, keep_percent=25)
    assert result.shape == (1, 17, 17)
    assert torch.allclose(result.sum(dim=-1), torch.ones(1, 17), atol=1e-5)


def test_cls_map_requires_square_patch_grid():
    with pytest.raises(ValueError, match="square patch grid"):
        AttentionRollout(matrices(tokens=11)).cls_map(1)


def test_invalid_controls_are_rejected():
    rollout = AttentionRollout(matrices())
    with pytest.raises(ValueError):
        rollout.up_to(4)
    with pytest.raises(ValueError):
        rollout.up_to(1, keep_percent=0)
    with pytest.raises(ValueError):
        rollout.up_to(1, fusion="unknown")


def test_rollout_rejects_mismatched_batch_sizes():
    with pytest.raises(ValueError, match="batch size"):
        AttentionRollout([matrices()[0], matrices()[0].expand(2, -1, -1, -1)])


def test_head_map_rejects_non_square_patch_grid():
    with pytest.raises(ValueError, match="square patch grid"):
        AttentionRollout(matrices(tokens=11)).head_map(1, 0)


def test_layer_maps_reject_duplicate_layers():
    with pytest.raises(ValueError, match="duplicates"):
        AttentionRollout(matrices()).layer_maps([1, 1])


def test_rollout_supports_two_prefix_tokens():
    rollout = AttentionRollout(matrices(tokens=18), prefix_tokens=2)
    assert rollout.cls_map(1).shape == (1, 224, 224)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv = nn.Linear(8, 24)

    def forward(self, tokens):
        return tokens


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()

    def forward(self, image):
        return self.attention(image)


def test_hook_extracts_attention_from_timm_style_module():
    result = extract_attention(TinyModel(), torch.randn(1, 5, 8))
    assert len(result) == 1
    assert result[0].shape == (1, 2, 5, 5)
    assert torch.allclose(result[0].sum(dim=-1), torch.ones(1, 2, 5), atol=1e-5)


def test_threshold_retains_attention_mass():
    attention = torch.zeros(1, 1, 3, 3)
    attention[..., 0] = 0.6
    attention[..., 1] = 0.3
    attention[..., 2] = 0.1
    rollout = AttentionRollout([attention])
    result = rollout._threshold(attention.mean(dim=1), 50)
    assert torch.all(result[..., 0] > 0)
    assert torch.all(result[..., 1:] == 0)
