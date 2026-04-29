from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from layereddepth_train.models.diffusion import DiffusionScheduler
from layereddepth_train.train import _build_model, _load_config


class MetricTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = {"sum": 0.0, "count": 0.0}
            self.metrics[key]["sum"] += float(value)
            self.metrics[key]["count"] += 1.0

    def get_average(self) -> dict[str, float]:
        return {
            key: value["sum"] / max(value["count"], 1.0)
            for key, value in self.metrics.items()
        }


def _tuple_points(single_tuple: dict[str, Any]) -> list[tuple[int, int, int]]:
    return [
        tuple(map(int, single_tuple[key]))
        for key in sorted(single_tuple.keys())
        if key.startswith("p")
    ]


def _map_tuple_layer(layer: int, mode: str, num_predicted_layers: int) -> int:
    if mode == "auto":
        mode = "full" if num_predicted_layers >= 8 else "compact"
    if mode == "compact":
        return (layer - 1) // 2
    if mode == "full":
        return layer - 1
    raise ValueError(f"Unknown layer map mode: {mode}")


def get_depth(
    predicted_depth: np.ndarray,
    x: int,
    y: int,
    tuple_layer: int,
    layer_map: str,
    mask: np.ndarray | None = None,
) -> float | None:
    layer = _map_tuple_layer(tuple_layer, layer_map, predicted_depth.shape[0])
    if layer < 0 or layer >= predicted_depth.shape[0]:
        return None

    if mask is not None and not mask[layer, y, x]:
        return None

    depth = predicted_depth[layer, y, x]
    if not np.isfinite(depth) or depth <= 0:
        return None
    return float(depth)


def layereddepth_tuple_correct(
    single_tuple: dict[str, Any],
    predicted_depth: np.ndarray,
    layer_map: str,
    mask: np.ndarray | None = None,
) -> bool:
    last_depth = None
    is_fake = not bool(single_tuple["is_real"])

    for x, y, layer in _tuple_points(single_tuple):
        depth = get_depth(predicted_depth, x, y, layer, layer_map, mask)
        if not is_fake:
            if depth is None:
                return False
            if last_depth is not None and last_depth >= depth:
                return False
            last_depth = depth
        elif depth is not None:
            return False

    return True


def get_layer_name(single_tuple: dict[str, Any]) -> str:
    current_layer = None
    for _, _, layer in _tuple_points(single_tuple):
        if current_layer is None:
            current_layer = layer
        elif current_layer != layer:
            return "mixed"
    if current_layer is None:
        raise ValueError("Tuple has no points.")
    return str(int(current_layer))


def check_tuple_valid(
    predicted_depth: np.ndarray,
    single_tuple: dict[str, Any],
    layer_map: str,
) -> bool:
    height, width = predicted_depth.shape[1:]
    for x, y, layer in _tuple_points(single_tuple):
        mapped_layer = _map_tuple_layer(layer, layer_map, predicted_depth.shape[0])
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        if layer % 2 == 0 or layer > 7 or layer < 1:
            return False
        if mapped_layer < 0 or mapped_layer >= predicted_depth.shape[0]:
            return False
    return True


def _preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size[1], image_size[0]), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def _predict_depth(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler | None,
    image: Image.Image,
    config: dict[str, Any],
    device: torch.device,
    diffusion_steps: int,
) -> np.ndarray:
    size = config["data"].get("image_size", [384, 512])
    image_size = (int(size[0]), int(size[1]))
    original_width, original_height = image.size
    image_tensor = _preprocess_image(image, image_size).to(device)

    if scheduler is None:
        predicted = model.predict_all_layers(image_tensor)
    else:
        num_layers = int(config["model"].get("num_layers", 8))
        valid_masks = torch.ones(
            1,
            num_layers,
            image_size[0],
            image_size[1],
            device=device,
            dtype=image_tensor.dtype,
        )
        predicted = scheduler.sample_ddim(
            model=model,
            image=image_tensor,
            shape=(1, num_layers, image_size[0], image_size[1]),
            valid_masks=valid_masks,
            steps=diffusion_steps,
        )
        if bool(config["model"].get("diffusion", {}).get("log_depth", True)):
            predicted = torch.exp(predicted)
        predicted = predicted.clamp_min(0.0)

    predicted = F.interpolate(
        predicted,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    return predicted.squeeze(0).detach().cpu().numpy()


def _load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, DiffusionScheduler | None, dict[str, Any]]:
    config = _load_config(config_path)
    model, scheduler = _build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    if scheduler is not None:
        scheduler.to(device)
    return model, scheduler, config


def _configure_hf_downloads(args: argparse.Namespace) -> None:
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hf_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.hf_timeout))


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    _configure_hf_downloads(args)
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires datasets and tqdm: python -m pip install datasets tqdm"
        ) from exc

    device = torch.device(args.device)
    model, scheduler, config = _load_model(args.config, args.checkpoint, device)
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    metric_tracker = MetricTracker()
    iterator = tqdm(dataset, desc=f"evaluate {args.split}")
    for sample_index, item in enumerate(iterator):
        if args.max_samples is not None and sample_index >= args.max_samples:
            break

        relative_depth_tuples = item["tuples.json"][args.subset]
        predicted_depth = _predict_depth(
            model=model,
            scheduler=scheduler,
            image=item["image.png"],
            config=config,
            device=device,
            diffusion_steps=args.diffusion_steps,
        )

        for tuple_type in ["pairs", "trips", "quads"]:
            for single_tuple in relative_depth_tuples[tuple_type]:
                if not check_tuple_valid(predicted_depth, single_tuple, args.layer_map):
                    continue
                layer = get_layer_name(single_tuple)
                correctness = float(
                    layereddepth_tuple_correct(
                        single_tuple,
                        predicted_depth,
                        layer_map=args.layer_map,
                    )
                )
                metric_tracker.update({f"{args.subset}/{tuple_type}/{layer}": correctness})
                metric_tracker.update({f"{args.subset}/{tuple_type}/all": correctness})

    return metric_tracker.get_average()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset", required=True, choices=["layer_all", "layer_first"])
    parser.add_argument("--dataset", default="princeton-vl/LayeredDepth")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream samples from Hugging Face instead of downloading all parquet shards first.",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Allow Hugging Face Xet/CAS downloads. Disabled by default to avoid CAS 401 errors on some servers.",
    )
    parser.add_argument(
        "--hf-timeout",
        type=int,
        default=120,
        help="Hugging Face download and metadata timeout in seconds.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--layer-map",
        choices=["auto", "compact", "full"],
        default="auto",
        help=(
            "Mapping from paper tuple layers 1/3/5/7 to model output layers. "
            "'auto' uses full for 8-layer outputs and compact for 4-layer outputs. "
            "'compact' maps to 0/1/2/3. 'full' maps to 0/2/4/6."
        ),
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="DDIM sampling steps when evaluating a diffusion checkpoint.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate(args)
    text = json.dumps(results, indent=4, sort_keys=True)
    print(text)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
