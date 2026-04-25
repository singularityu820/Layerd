from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def _safe_key(value: Any, fallback: int) -> str:
    if value is None:
        value = f"{fallback:06d}"
    text = str(value)
    if text.isdigit():
        text = f"{int(text):06d}"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or f"{fallback:06d}"


def _save_media(value: Any, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required: python -m pip install pillow") from exc

    if isinstance(value, Image.Image):
        value.save(output_path)
        return

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            output_path.write_bytes(value["bytes"])
            return
        if value.get("path") is not None:
            shutil.copyfile(value["path"], output_path)
            return

    if isinstance(value, (bytes, bytearray)):
        output_path.write_bytes(bytes(value))
        return

    raise TypeError(f"Unsupported media value for {output_path}: {type(value)!r}")


def _iter_split(repo_id: str, split: str, cache_dir: str | None, streaming: bool):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets is required:\n"
            "  python -m pip install datasets huggingface_hub"
        ) from exc

    return load_dataset(repo_id, split=split, cache_dir=cache_dir, streaming=streaming)


def _export_split(args: argparse.Namespace, split: str) -> None:
    root = Path(args.root)
    image_dir = root / "images" / split
    depth_dir = root / "depth" / split
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dataset = _iter_split(
        repo_id=args.repo_id,
        split=split,
        cache_dir=args.cache_dir,
        streaming=not args.no_streaming,
    )

    manifest_path = manifest_dir / f"{split}.jsonl"
    tmp_manifest_path = manifest_path.with_suffix(".jsonl.tmp")
    count = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    with tmp_manifest_path.open("w", encoding="utf-8") as handle:
        for row_index, sample in enumerate(tqdm(dataset, desc=f"export {split}")):
            if args.max_samples is not None and count >= args.max_samples:
                break

            key = _safe_key(sample.get(args.key_column), row_index)
            image_path = image_dir / f"{key}.png"
            _save_media(sample[args.image_column], image_path, overwrite=args.overwrite)

            depth_paths = []
            for layer_idx in range(1, args.num_layers + 1):
                column = args.depth_column_template.format(i=layer_idx)
                if column not in sample:
                    raise KeyError(
                        f"Column {column!r} not found in split {split!r}. "
                        f"Available columns: {sorted(sample.keys())}"
                    )
                depth_path = depth_dir / f"{key}_layer_{layer_idx:02d}.png"
                _save_media(sample[column], depth_path, overwrite=args.overwrite)
                depth_paths.append(depth_path.resolve().relative_to(root.resolve()).as_posix())

            record = {
                "image": image_path.resolve().relative_to(root.resolve()).as_posix(),
                "depth_layers": depth_paths,
                "hf_repo_id": args.repo_id,
                "hf_split": split,
                "hf_key": key,
            }
            handle.write(json.dumps(record) + "\n")
            count += 1

    tmp_manifest_path.replace(manifest_path)
    print(f"Wrote {count} samples to {manifest_path}")

    if split == "validation":
        val_alias = manifest_dir / "val.jsonl"
        shutil.copyfile(manifest_path, val_alias)
        print(f"Wrote validation alias to {val_alias}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LayeredDepth-Syn from Hugging Face and export training manifests."
    )
    parser.add_argument("--repo-id", default="princeton-vl/LayeredDepth-Syn")
    parser.add_argument("--root", default="data/layereddepth_syn")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--image-column", default="image.png")
    parser.add_argument("--depth-column-template", default="depth_{i}.png")
    parser.add_argument("--key-column", default="__key__")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download through the normal datasets cache instead of streaming rows.",
    )
    args = parser.parse_args()

    for split in args.splits:
        _export_split(args, split)


if __name__ == "__main__":
    main()
