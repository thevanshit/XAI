# XAI Transformer Suite

Interactive visual debugging for Vision Transformers.

The suite runs a pretrained image transformer on an uploaded image, captures
its internal self-attention, and turns that attention into layer-wise rollout
maps, per-head visualizations, and object-region overlays.

It is designed as an interpretability project rather than a classifier demo:
the goal is to inspect **where a transformer is looking and how that focus
changes through the network**.

## What this project does

For one image, the application:

1. Resizes and ImageNet-normalizes the image to `224 x 224`.
2. Loads one pretrained model from the model zoo.
3. Captures attention from every transformer block with PyTorch forward hooks.
4. Stores attention tensors shaped like:

   ```text
   [batch, attention_heads, tokens, tokens]
   ```

5. Fuses attention heads using mean, minimum, or maximum fusion.
6. Adds residual identity attention and normalizes rows.
7. Recursively multiplies attention through the selected layer.
8. Extracts CLS-to-image-patch attention.
9. Reshapes the patch grid and upsamples it to `224 x 224`.
10. Displays:
    - a transparent attention overlay,
    - optional high-attention contours,
    - a per-head inspector,
    - a layer-progression gallery.

The model inference result is cached by model and image. Moving the layer
slider or changing fusion/threshold controls therefore reuses captured
attention instead of running the model again.

## What it is useful for

This project helps investigate questions such as:

- Does the model focus on the main object or background?
- Does attention become more semantic in deeper layers?
- Do different heads focus on different regions?
- Does a self-supervised model produce different attention patterns from a
  supervised model?
- How much does removing low-mass attention change the final explanation?

Attention is an interpretability signal, not a guaranteed causal explanation
of the model's prediction. A bright region means that the selected attention
rollout assigns it high importance under this visualization method; it does not
prove that the region alone caused the classification.

## Supported models

| UI label | Model identifier | Training style | Expected structure |
|---|---|---|---|
| Supervised ViT | `vit_base_patch16_224.augreg_in21k_ft_in1k` | ImageNet fine-tuned | 12 layers, 12 heads, 197 tokens |
| DeiT | `deit_base_patch16_224.fb_in1k` | Data-efficient supervised training | 12 layers, 12 heads, 197 tokens |
| DINO | `vit_small_patch16_224.dino` | Self-supervised learning | 12 layers, 6 heads, 197 tokens |

The supervised baseline uses the ViT-B/16 architecture associated with the
attached ViT paper. The project performs pretrained inference and XAI analysis;
it does not reproduce the paper's large-scale JFT/ImageNet training runs,
benchmark tables, or compute comparisons.

The attention extractor follows the installed `timm` attention implementation,
including query/key normalization and scaling. Patch extraction is
prefix-token-aware so models with more than one special token can be supported.

## Requirements

- macOS, Linux, or Windows
- Python `3.9` or newer
- At least several GB of disk space for PyTorch and model weights
- Network access on the first run of each model
- CPU, Apple Silicon MPS, or CUDA GPU

The first request for a model downloads its pretrained weights through `timm`.
Later requests reuse the in-process model cache and the local model-weight
cache.

## Install and run

### Recommended VS Code setup

From the project directory:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -e ".[dev]"
```

In VS Code:

1. Open the Command Palette.
2. Select **Python: Select Interpreter**.
3. Choose:

   ```text
   /Users/vanshit/Coding/XAI/.venv/bin/python
   ```

   On another machine, choose that machine's equivalent `.venv/bin/python`.

Start the dashboard:

```bash
./.venv/bin/python app.py
```

Open the printed local URL, normally:

```text
http://127.0.0.1:7860
```

Stop the server with `Ctrl+C`.

### Important command syntax

Use:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

Do not use:

```bash
./.venv/bin/python -m python3 -m pip install ...
```

`python` is already the executable; `python3` is not a module.

## How to use the dashboard

### 1. Upload an image

Use **Input image** to upload a JPEG, PNG, or another PIL-supported image.
The image is converted to RGB and resized to `224 x 224` for the model.

### 2. Choose a model

Use **Model** to compare:

- **Supervised ViT**: a supervised ImageNet ViT-B/16 baseline.
- **DeiT**: a data-efficient supervised transformer.
- **DINO**: a self-supervised ViT-S/16 model.

Changing the model invalidates the image/model attention cache entry and runs
the selected model once.

### 3. Select rollout depth

**Rollout through layer** controls how many transformer blocks participate.

- Layer `1`: early attention behavior.
- Middle layers: intermediate feature routing.
- Layer `12`: full-depth rollout for these model configurations.

The layer progression gallery automatically shows representative early,
middle, and selected layers.

### 4. Select head fusion

The **Head fusion** control determines how the individual heads are combined:

- **Mean**: stable overall attention pattern; recommended default.
- **Min**: conservative signal shared by all heads.
- **Max**: strongest signal from any head; often more selective and noisy.

### 5. Remove low-mass attention

**Retain attention mass per row (%)** applies dynamic noise filtering before
rollout multiplication.

For example, `25` retains the entries contributing the top 25% of each row's
attention mass, then adds the residual identity and renormalizes.

- `100`: no filtering.
- Lower values: sharper, more selective maps.
- Very low values: potentially unstable or overly sparse explanations.

### 6. Inspect contours

Enable **Draw attention contours** to draw OpenCV contours around normalized
high-attention areas.

These are visualization regions, not object-detection predictions. They should
not be interpreted as ground-truth bounding boxes.

### 7. Read the head inspector

The **Attention head inspector** shows one heatmap for each head in the
selected layer. Different heads may emphasize different shapes, edges,
textures, or object regions.

The number of panels depends on the model:

- ViT: 12 heads
- DeiT: 12 heads
- DINO ViT-S: 6 heads

### 8. Read the status line

The status line reports:

- selected model,
- number of captured layers,
- selected layer,
- number of heads,
- `inference` or `cached`,
- elapsed processing time.

Use **Clear attention cache** to release cached attention matrices. Model
objects remain cached separately so clearing attention does not force model
weights to download again.

## Mathematics implemented

For each layer, the extractor obtains attention:

```text
A_l ∈ R^(heads × tokens × tokens)
```

After head fusion:

```text
F_l = fuse(A_l)
```

Residual attention is added:

```text
R_l = F_l + I
```

Rows are normalized:

```text
R_l = R_l / row_sum(R_l)
```

Rollout through layer `L` is computed recursively:

```text
J_0 = I
J_l = R_l @ J_(l-1)
```

The first row of the final joint attention matrix is used as the CLS-token
attention distribution. Special prefix tokens are excluded from the patch
grid, and the remaining patch values are reshaped and bicubically resized to
`224 x 224`.

## Project structure

```text
.
├── app.py                         # Gradio application
├── pyproject.toml                 # Runtime and development dependencies
├── setup.py                       # Older-pip editable-install compatibility
├── Dockerfile                     # Container image
├── .dockerignore
├── src/
│   └── xai_transformer/
│       ├── attention.py           # Hooks, extraction, fusion, rollout
│       ├── models.py              # Model zoo and model caching
│       └── visuals.py             # Overlays, contours, galleries
├── tests/
│   └── test_attention.py          # Mathematical and extraction tests
└── research paper/
    └── 2010.11929v2.pdf           # Referenced ViT paper
```

## Python API examples

### Run rollout on synthetic attention

```python
import torch
from xai_transformer.attention import AttentionRollout

attention = [
    torch.softmax(torch.randn(1, 12, 197, 197), dim=-1)
    for _ in range(12)
]

rollout = AttentionRollout(attention, prefix_tokens=1)
heatmap = rollout.cls_map(layer=12)

print(heatmap.shape)
# torch.Size([1, 224, 224])
```

### Load a model

```python
from xai_transformer.models import inference_device, load_model

model, spec = load_model("Supervised ViT", inference_device())
print(spec.timm_name)
```

### Capture model attention

```python
import torch
from xai_transformer.attention import extract_attention

image = torch.randn(1, 3, 224, 224)
attention = extract_attention(model, image)

print(len(attention))
print(attention[0].shape)
# 12
# torch.Size([1, 12, 197, 197])
```

## Tests and lint

Run the test suite:

```bash
./.venv/bin/python -m pytest -q
```

Run lint:

```bash
./.venv/bin/python -m ruff check .
```

Compile-check the sources:

```bash
./.venv/bin/python -m compileall -q app.py src tests setup.py
```

The tests cover:

- row-normalized rollout,
- dynamic attention-mass thresholding,
- invalid layer/fusion inputs,
- batch-size validation,
- square patch-grid validation,
- timm-style QKV hook extraction,
- multiple prefix-token support.

## Docker

Build the image:

```bash
docker build -t xai-transformer-suite .
```

Run it:

```bash
docker run --rm -p 7860:7860 xai-transformer-suite
```

Then open:

```text
http://127.0.0.1:7860
```

Model weights are downloaded at runtime on first use. For production
deployment, mount or pre-populate a persistent model cache rather than
redownloading weights for every container start.

## Troubleshooting

### VS Code selects `/usr/bin/python3`

Select the project interpreter manually:

```text
<project>/XAI/.venv/bin/python
```

Then reinstall with that interpreter:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

### Editable installation says `setup.py` is missing

Upgrade packaging tools:

```bash
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

The repository includes both `pyproject.toml` and a small `setup.py`
compatibility shim for older environments.

### The first request is slow

The first request may:

1. import PyTorch and timm,
2. download model weights,
3. move the model to CPU, MPS, or CUDA,
4. run the first forward pass.

Later requests for the same model/image use cached models and attention.

### A model cannot be downloaded

Check network access and retry. The error includes the model identifier and
the local model-cache hint. The application intentionally does not generate
fake attention maps when weights are unavailable.

### Port 7860 is already in use

Stop the existing process or run the application with a custom launch wrapper.
To find the process on macOS/Linux:

```bash
lsof -i :7860
```

### Memory usage is high

- Use one model at a time.
- Click **Clear attention cache**.
- Prefer CPU if GPU memory is limited.
- Reduce the number of concurrent app workers.

The application limits the attention cache to eight model/image entries and
stores cached attention matrices on CPU.

## CI

GitHub Actions runs:

```text
ruff check .
pytest
```

Workflow configuration is in
[.github/workflows/ci.yml](.github/workflows/ci.yml).

## Scope and limitations

This project currently provides:

- pretrained ViT-family inference,
- attention extraction,
- rollout-based visualization,
- layer and head inspection,
- Docker and CI scaffolding.

It does not currently provide:

- training from scratch,
- reproduction of the ViT paper's JFT experiments,
- paper benchmark tables,
- calibrated causal explanations,
- object-detection ground truth comparison,
- ONNX inference,
- multi-user production queueing or authentication.

ONNX export is intentionally not enabled yet because the explainability path
depends on capturing intermediate attention that standard model export does not
automatically preserve.

## License and attribution

This project is released under the [MIT License](./LICENSE).

This project uses open-source packages including PyTorch, torchvision, timm,
Gradio, Pillow, NumPy, and OpenCV. Follow each dependency's license terms.

The architecture baseline is based on:

> Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image
> Recognition at Scale*, arXiv:2010.11929.
