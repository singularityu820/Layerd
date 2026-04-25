from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEPTH_EXTENSIONS = {".npy", ".png", ".tif", ".tiff", ".exr"}


def _find_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_depth_layers(depth_dir: Path, stem: str, num_layers: int) -> list[Path] | None:
    layers: list[Path] = []
    for layer_idx in range(1, num_layers + 1):
        candidates = []
        for suffix in DEPTH_EXTENSIONS:
            candidates.extend(
                [
                    depth_dir / f"{stem}_layer_{layer_idx:02d}{suffix}",
                    depth_dir / f"{stem}_layer{layer_idx:02d}{suffix}",
                    depth_dir / f"{stem}_{layer_idx:02d}{suffix}",
                    depth_dir / stem / f"layer_{layer_idx:02d}{suffix}",
                    depth_dir / stem / f"{layer_idx:02d}{suffix}",
                ]
            )
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        if match is None:
            return None
        layers.append(match)
    return layers


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/layereddepth_syn")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--depth-dir", default="depth")
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / args.image_dir
    depth_dir = root / args.depth_dir
    manifest_dir = root / args.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for image_path in _find_images(image_dir):
        layers = _find_depth_layers(depth_dir, image_path.stem, args.num_layers)
        if layers is None:
            continue
        samples.append(
            {
                "image": _relative(image_path, root),
                "depth_layers": [_relative(path, root) for path in layers],
            }
        )

    if not samples:
        raise RuntimeError(
            f"No samples found under {root}. Expected images in {image_dir} and "
            f"depth layers like depth/<stem>_layer_01.npy."
        )

    random.Random(args.seed).shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_ratio)) if len(samples) > 1 else 0
    splits = {
        "train": samples[val_count:],
        "val": samples[:val_count],
    }

    for split, split_samples in splits.items():
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in split_samples:
                handle.write(json.dumps(sample) + "\n")
        print(f"Wrote {len(split_samples)} samples to {manifest_dir / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
