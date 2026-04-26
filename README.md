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

## Environment Setup

Recommended training environment:

```text
Python 3.10
PyTorch 2.6.0
torchvision 0.21.0
CUDA 12.6 wheel for recent NVIDIA GPUs
```

Create and activate a Conda environment.

Windows:

```powershell
D:\software\miniconda3\Scripts\conda.exe create -y -n layereddepth python=3.10 pip
D:\software\miniconda3\Scripts\activate.bat layereddepth
```

Linux or cloud server:

```bash
conda create -y -n layereddepth python=3.10 pip
conda activate layereddepth
```

Install CUDA PyTorch. This command is a good default for RTX 40-series,
L40S, A100, and H100 machines with a recent NVIDIA driver:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 --timeout 1000 --retries 5
```

If your server image already pins a different CUDA runtime, use the official
PyTorch install selector and keep `torch` and `torchvision` versions matched.

Install this project and the non-PyTorch dependencies:

```bash
cd D:\document\code\layereddepth_train
python -m pip install -e .
python -m pip install -r requirements.txt
```

On Linux, use the project path on that machine instead of the Windows path:

```bash
cd ~/layereddepth_train
python -m pip install -e .
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "from layereddepth_train.models.backbones import NewCRFsAdapter; print('project import OK')"
python -c "import sys; sys.path.insert(0, r'third_party/NeWCRFs'); from newcrfs.networks.NewCRFDepth import NewCRFDepth; print('NeWCRFs import OK')"
```

On Windows, the first command should point to:

```text
D:\software\miniconda3\envs\layereddepth\python.exe
```

Run a small smoke test after downloading a few Hugging Face samples:

```bash
python scripts/download_layereddepth_hf.py --root data/layereddepth_syn_smoke --max-samples 8
python -m layereddepth_train.train --config configs/hf_smoke_tiny.yaml
```

Common setup notes:

- If `mmcv` import fails, install `mmcv-lite`.
- Hugging Face symlink warnings on Windows are safe to ignore; they only affect
  cache disk usage.
- A final `WinError 10038` after manifests are written is usually a Windows
  streaming-download cleanup warning, not a failed export.
- If training says `manifest not found`, run the Hugging Face download script or
  update `data.root` in the config.
- If `torch.save` fails with `unexpected pos`, check disk space with `df -h`.
  Checkpoints are now written atomically through `.tmp` files. By default
  `last.pt` includes optimizer state for resume, while periodic `epoch_XXXX.pt`
  files store model weights only. To make even `last.pt` smaller, set
  `train.save_optimizer: false`.

## TensorBoard

Training writes TensorBoard logs by default under each run directory:

```text
runs/paper_index_concat/tensorboard
runs/diffusion_unet/tensorboard
```

Start TensorBoard on a server:

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

If `tensorboard` is missing in an existing environment:

```bash
python -m pip install tensorboard
```

The logger records:

- `train/loss`, `train/lr`, and `train/grad_norm` every `train.log_every` steps;
- `epoch/train_loss` and `epoch/val_loss` every epoch;
- validation RGB images, target depth maps, and paper-baseline prediction depth
  maps every `tensorboard.image_every_epochs` epochs.

For diffusion configs, image logging currently records RGB and target depth
maps. Full reverse-diffusion samples are intentionally not generated during
training because that would add a large per-epoch cost.

You can tune logging in each config:

```yaml
tensorboard:
  enabled: true
  image_every_epochs: 1
  max_images: 2
  max_layers: 8
```

## Checkpoints

The training loop writes:

```text
runs/<experiment>/last.pt
runs/<experiment>/epoch_0010.pt
runs/<experiment>/epoch_0020.pt
```

Checkpoint writes are atomic: each file is first written as `*.tmp`, then renamed
after a complete save. Relevant config fields:

```yaml
train:
  save_every: 10
  save_optimizer: true
  save_epoch_optimizer: false
```

`save_optimizer: true` keeps `last.pt` resumable. `save_epoch_optimizer: false`
keeps periodic snapshots much smaller. On small disks, increase `save_every` or
move `output_dir` to a larger volume.

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

The downloader is resume-friendly by default: existing manifest records are
skipped, completed image/depth files are kept, and new samples are appended.
Media files are written through a temporary file and atomically renamed, so an
interrupted write will not be mistaken for a completed sample.

For a real resume after a long interrupted server download, prefer
`--no-streaming`. Hugging Face streaming can call `.skip(N)`, but that still
scans earlier parquet shards internally, so it may appear to start again from
`train-00000-of-00056.parquet`. Non-streaming mode builds/uses the local HF cache
and can select the remaining rows without stream-scanning from shard 0:

```bash
python scripts/download_layereddepth_hf.py --root data/layereddepth_syn --splits train --cache-dir /data/hf_cache --no-streaming
```

If you suspect the manifest is out of order, add `--no-fast-resume` to scan from
the beginning and skip known keys one by one.

To deliberately rebuild manifests and overwrite exported media:

```bash
python scripts/download_layereddepth_hf.py --root data/layereddepth_syn --overwrite --no-resume
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
