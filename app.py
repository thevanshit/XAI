"""Gradio entry point for the XAI Transformer Suite."""

from __future__ import annotations

import hashlib
import io
import time
from collections import OrderedDict
from threading import RLock

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_tensor

from xai_transformer.attention import AttentionRollout, extract_attention
from xai_transformer.models import MODEL_ZOO, inference_device, load_model
from xai_transformer.visuals import (
    draw_attention_boxes,
    make_head_gallery,
    make_layer_gallery,
    overlay_attention,
)

MAX_ATTENTION_CACHE = 8
_attention_cache: OrderedDict[tuple[str, str], tuple[list[torch.Tensor], int]] = OrderedDict()
_cache_lock = RLock()


def preprocess(image: Image.Image) -> torch.Tensor:
    tensor = resize(to_tensor(image.convert("RGB")), [224, 224])
    return normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)


def _image_key(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _get_attention(
    image: Image.Image, model_label: str
) -> tuple[list[torch.Tensor], int, str]:
    cache_key = (model_label, _image_key(image))
    with _cache_lock:
        if cache_key in _attention_cache:
            attentions, prefix_tokens = _attention_cache.pop(cache_key)
            _attention_cache[cache_key] = (attentions, prefix_tokens)
            return attentions, prefix_tokens, "cached"
    model, spec = load_model(model_label, inference_device())
    image_tensor = preprocess(image).to(next(model.parameters()).device)
    attentions = [attention.cpu() for attention in extract_attention(model, image_tensor)]
    with _cache_lock:
        _attention_cache[cache_key] = (attentions, spec.prefix_tokens)
        while len(_attention_cache) > MAX_ATTENTION_CACHE:
            _attention_cache.popitem(last=False)
    return attentions, spec.prefix_tokens, "inference"


def clear_attention_cache() -> str:
    """Release cached attention matrices after a model/image session."""
    with _cache_lock:
        _attention_cache.clear()
    return "Attention cache cleared."


def analyze(
    image: Image.Image | None, model_label: str, layer: int, fusion: str, keep_percent: float,
    show_boxes: bool,
) -> tuple[Image.Image | None, list[Image.Image], list[Image.Image], str]:
    if image is None:
        return None, [], [], "Upload an image to begin."
    try:
        started = time.perf_counter()
        attentions, prefix_tokens, source = _get_attention(image, model_label)
        rollout = AttentionRollout(attentions, prefix_tokens=prefix_tokens)
        layer = min(layer, len(attentions))
        heatmap = rollout.cls_map(layer, fusion=fusion, keep_percent=keep_percent)[0].cpu().numpy()
        result = overlay_attention(image, heatmap)
        if show_boxes:
            result = draw_attention_boxes(result, heatmap)
        head_maps = [
            rollout.head_map(layer, head)[0].cpu().numpy()
            for head in range(attentions[layer - 1].shape[1])
        ]
        gallery = make_head_gallery(image, head_maps)
        progression_layers = sorted({1, max(1, len(attentions) // 3), max(1, 2 * len(attentions) // 3), layer})
        progression_maps = [
            layer_map[0].cpu().numpy()
            for layer_map in rollout.layer_maps(progression_layers, fusion, keep_percent)
        ]
        progression = make_layer_gallery(image, progression_maps, progression_layers)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            result,
            gallery,
            progression,
            (f"{model_label}: {len(attentions)} layers, layer {layer}, "
             f"{attentions[0].shape[1]} heads | {source} | {elapsed_ms:.0f} ms"),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return None, [], [], f"Analysis failed: {error}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="XAI Transformer Suite") as demo:
        gr.Markdown(
            "# XAI Transformer Suite\n"
            "Compare how transformer training objectives focus on an image. "
            "Changing controls reuses cached attention tensors."
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Input image")
            output = gr.Image(type="pil", label="Attention rollout")
        with gr.Row():
            model = gr.Dropdown(list(MODEL_ZOO), value="Supervised ViT", label="Model")
            layer = gr.Slider(1, 12, value=12, step=1, label="Rollout through layer")
            fusion = gr.Dropdown(["mean", "min", "max"], value="mean", label="Head fusion")
        with gr.Row():
            keep = gr.Slider(
                1, 100, value=100, step=1,
                label="Retain attention mass per row (%)",
            )
            boxes = gr.Checkbox(label="Draw attention contours")
        status = gr.Textbox(label="Status")
        clear_cache = gr.Button("Clear attention cache", variant="secondary")
        head_gallery = gr.Gallery(
            label="Attention head inspector",
            columns=4,
            rows=3,
            height="auto",
            object_fit="contain",
        )
        layer_gallery = gr.Gallery(
            label="Layer-wise rollout progression",
            columns=4,
            rows=1,
            height="auto",
            object_fit="contain",
        )
        inputs = [image, model, layer, fusion, keep, boxes]
        outputs = [output, head_gallery, layer_gallery, status]
        image.change(analyze, inputs, outputs)
        for component in inputs[1:]:
            component.change(analyze, inputs, outputs)
        clear_cache.click(clear_attention_cache, outputs=status)
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", show_api=False)
