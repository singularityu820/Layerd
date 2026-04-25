from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "toy"
    image_dir = root / "images"
    depth_dir = root / "depth"
    manifest_dir = root / "manifests"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    samples = []
    height, width, num_layers = 128, 160, 7
    yy, xx = np.mgrid[0:height, 0:width]

    for sample_idx in range(8):
        image = np.stack(
            [
                (xx / width + 0.1 * rng.random((height, width))) % 1.0,
                (yy / height + 0.1 * rng.random((height, width))) % 1.0,
                rng.random((height, width)) * 0.5 + 0.25,
            ],
            axis=-1,
        )
        image_path = image_dir / f"{sample_idx:04d}.png"
        Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

        depth_paths = []
        for layer_idx in range(num_layers):
            depth = 1.0 + 0.2 * layer_idx + xx / width + yy / height
            missing = rng.random((height, width)) < (0.02 * layer_idx)
            depth = depth.astype(np.float32)
            depth[missing] = 0.0
            depth_path = depth_dir / f"{sample_idx:04d}_layer_{layer_idx + 1:02d}.npy"
            np.save(depth_path, depth)
            depth_paths.append(str(depth_path.relative_to(root)))

        samples.append(
            {
                "image": str(image_path.relative_to(root)),
                "depth_layers": depth_paths,
            }
        )

    for name, subset in {"train": samples[:6], "val": samples[6:]}.items():
        with (manifest_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in subset:
                handle.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
