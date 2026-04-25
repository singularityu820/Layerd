# LayeredDepth Training Networks

This folder switches the work from the synthetic generator to training networks
for multi-layer depth estimation.

It contains two tracks:

1. Paper baselines from LayeredDepth:
   - Multi-head Output
   - Layer Index Concatenation
   - Recurrent prediction
   - NeWCRFs-compatible backbone adapter
   - random target-layer supervision
   - snapped layered depth
   - SILog loss
2. A diffusion-based architecture:
   - conditional DDPM denoising network
   - RGB-conditioned multi-layer depth stack prediction
   - optional log-depth training target

The paper configs use `core.name: newcrfs` because the paper adopts NeWCRFs as
the backbone. For local smoke tests without the NeWCRFs repository installed,
switch `core.name` to `tiny`.

## Data Manifest

Training uses JSONL manifests. Each line describes one sample:

```json
{
  "image": "relative/or/absolute/image.png",
  "depth_layers": [
    "relative/or/absolute/depth_layer_01.npy",
    "relative/or/absolute/depth_layer_02.npy"
  ],
  "valid_masks": [
    "relative/or/absolute/mask_layer_01.png",
    "relative/or/absolute/mask_layer_02.png"
  ]
}
```

Depth files can be `.npy` arrays or image files. Image depth values are divided
by `data.depth_scale`. Masks are optional; if omitted, valid pixels are inferred
from `depth > 0`.

## Download LayeredDepth-Syn

The official synthetic training set is on Hugging Face:
`princeton-vl/LayeredDepth-Syn`. The dataset has `train` and `validation`
splits, 15,300 rows total, and columns named `image.png`, `depth_1.png`, ...,
`depth_8.png`.

Install the Hugging Face dependencies:

```bash
python -m pip install datasets huggingface_hub tqdm
```

Download and export it into this project's manifest format:

```bash
python scripts/download_layereddepth_hf.py --root data/layereddepth_syn
```

For a quick download test:

```bash
python scripts/download_layereddepth_hf.py --root data/layereddepth_syn_smoke --max-samples 8
python -m layereddepth_train.train --config configs/hf_smoke_tiny.yaml
```

Then train with:

```bash
python -m layereddepth_train.train --config configs/paper_index_concat.yaml
```

The official configs use `num_layers: 8` for LayeredDepth-Syn. They also use
`depth_scale: 1000.0` for exported PNG depth maps, so integer PNG depths are
interpreted as meters during training. Change it to `1.0` only if your depth
files are already stored in meters.

## Folder Manifests

For the simple folder layout below, a helper can create manifests:

```text
data/layereddepth_syn/
  images/000001.png
  depth/000001_layer_01.npy
  depth/000001_layer_02.npy
  ...
```

```bash
python scripts/build_manifest_from_folders.py --root data/layereddepth_syn --num-layers 7
```

## Train

```bash
python -m layereddepth_train.train --config configs/paper_index_concat.yaml
```

Diffusion track:

```bash
python -m layereddepth_train.train --config configs/diffusion_unet.yaml
```

## NeWCRFs

The official NeWCRFs repository is expected at:

```text
third_party/NeWCRFs
```

The adapter automatically adds that folder to `sys.path` before importing:

```python
from newcrfs.networks.NewCRFDepth import NewCRFDepth
```

If your local checkout uses a different location, update
`layereddepth_train/models/backbones.py`.
