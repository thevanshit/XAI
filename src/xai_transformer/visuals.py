"""Attention heatmap, alpha blending, and contour helpers."""

import cv2
import numpy as np
from PIL import Image


def overlay_attention(image: Image.Image, attention: np.ndarray, threshold: float = 0.6) -> Image.Image:
    """Blend a viridis heatmap with alpha proportional to normalized attention."""
    base = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
    values = normalize_attention(attention)
    colored = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = np.where(values >= threshold, values * 0.75, 0)[..., None]
    result = base * (1 - alpha) + colored * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def draw_attention_boxes(image: Image.Image, attention: np.ndarray, threshold: float = 0.6) -> Image.Image:
    """Draw contours around connected high-attention regions."""
    output = np.asarray(image.convert("RGB").resize((224, 224))).copy()
    values = normalize_attention(attention)
    mask = ((values >= threshold) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 64, 32), 2)
    return Image.fromarray(output)


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """Normalize an attention map to a stable display range."""
    values = np.asarray(attention, dtype=np.float32)
    return (values - values.min()) / (np.ptp(values) + 1e-8)


def make_head_gallery(
    image: Image.Image, head_maps: list[np.ndarray], threshold: float = 0.6
) -> list[Image.Image]:
    """Create labeled attention overlays for each head in a layer."""
    gallery = []
    for index, attention in enumerate(head_maps):
        overlay = overlay_attention(image, attention, threshold)
        canvas = np.asarray(overlay.convert("RGB")).copy()
        cv2.putText(
            canvas, f"Head {index}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"Head {index}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (20, 20, 20), 1, cv2.LINE_AA
        )
        gallery.append(Image.fromarray(canvas))
    return gallery


def make_layer_gallery(
    image: Image.Image, layer_maps: list[np.ndarray], layers: list[int]
) -> list[Image.Image]:
    """Create labeled rollout snapshots showing progression through the model."""
    if len(layer_maps) != len(layers):
        raise ValueError("Each layer map must have a matching layer number.")
    gallery = []
    for layer, attention in zip(layers, layer_maps):
        canvas = np.asarray(overlay_attention(image, attention)).copy()
        cv2.putText(
            canvas, f"Through layer {layer}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"Through layer {layer}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (20, 20, 20), 1, cv2.LINE_AA
        )
        gallery.append(Image.fromarray(canvas))
    return gallery
