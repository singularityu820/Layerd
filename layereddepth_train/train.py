from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from layereddepth_train.data import ManifestLayeredDepthDataset
from layereddepth_train.losses import masked_mse, silog_loss
from layereddepth_train.models.baselines import (
    RecurrentBaseline,
    build_paper_baseline,
    gather_layer,
)
from layereddepth_train.models.diffusion import DiffusionScheduler, build_diffusion_model


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["__project_dir__"] = str(config_path.parent.parent)
    return config


def _project_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["__project_dir__"]) / path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_config = config["data"]
    root = _project_path(config, data_config.get("root"))
    manifest = Path(data_config[f"{split}_manifest"])
    if not manifest.is_absolute():
        manifest = (root or Path(config["__project_dir__"])) / manifest
    size = data_config.get("image_size")
    image_size = tuple(size) if size is not None else None
    dataset = ManifestLayeredDepthDataset(
        manifest_path=manifest,
        root=root,
        num_layers=int(config["model"].get("num_layers", 7)),
        image_size=image_size,
        depth_scale=float(data_config.get("depth_scale", 1.0)),
        use_snapped_depth=bool(data_config.get("use_snapped_depth", True)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 4)),
        shuffle=(split == "train"),
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def _paper_step(model: torch.nn.Module, batch: dict[str, Tensor], config: dict[str, Any]) -> Tensor:
    image = batch["image"]
    depth = batch["depth"]
    valid = batch["valid"]
    b, num_layers, h, w = depth.shape
    layer_index = torch.randint(0, num_layers, (b,), device=image.device)

    if isinstance(model, RecurrentBaseline):
        previous = torch.zeros(b, 1, h, w, device=image.device, dtype=depth.dtype)
        has_previous = layer_index > 0
        if has_previous.any():
            previous_stack = gather_layer(depth, torch.clamp(layer_index - 1, min=0))
            previous[has_previous] = previous_stack[has_previous]
        prediction = model(image, previous)
    else:
        prediction = model(image, layer_index)

    target = gather_layer(depth, layer_index)
    target_valid = gather_layer(valid.float(), layer_index).bool()
    loss_config = config.get("loss", {})
    return silog_loss(
        prediction,
        target,
        target_valid,
        variance_focus=float(loss_config.get("variance_focus", 0.85)),
        scale=float(loss_config.get("scale", 10.0)),
    )


def _diffusion_step(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    batch: dict[str, Tensor],
    config: dict[str, Any],
) -> Tensor:
    image = batch["image"]
    depth = batch["depth"]
    valid = batch["valid"]
    diffusion_config = config["model"].get("diffusion", {})
    if bool(diffusion_config.get("log_depth", True)):
        target = torch.log(depth.clamp_min(float(diffusion_config.get("eps", 1e-3))))
    else:
        target = depth

    timestep = torch.randint(0, scheduler.timesteps, (image.shape[0],), device=image.device)
    noise = torch.randn_like(target)
    noisy = scheduler.q_sample(target, timestep, noise)
    prediction = model(image, noisy, timestep, valid_masks=valid.float())
    return masked_mse(prediction, noise, valid)


def _build_model(config: dict[str, Any]) -> tuple[torch.nn.Module, DiffusionScheduler | None]:
    model_config = config["model"]
    family = str(model_config.get("family", "paper")).lower()
    if family == "paper":
        return build_paper_baseline(model_config), None
    if family == "diffusion":
        scheduler_config = model_config.get("scheduler", {})
        scheduler = DiffusionScheduler(
            timesteps=int(scheduler_config.get("timesteps", 1000)),
            beta_start=float(scheduler_config.get("beta_start", 1e-4)),
            beta_end=float(scheduler_config.get("beta_end", 0.02)),
        )
        return build_diffusion_model(model_config), scheduler
    raise ValueError(f"Unknown model family: {family}")


def train(config: dict[str, Any]) -> None:
    _set_seed(int(config.get("seed", 7)))
    device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = _make_loader(config, "train")
    val_loader = _make_loader(config, "val") if config["data"].get("val_manifest") else None

    model, scheduler = _build_model(config)
    model.to(device)
    if scheduler is not None:
        scheduler.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-2)),
    )
    output_dir = _project_path(config, config["train"].get("output_dir", "runs/layereddepth"))
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(config["train"].get("epochs", 100))
    log_every = int(config["train"].get("log_every", 20))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if scheduler is None:
                loss = _paper_step(model, batch, config)
            else:
                loss = _diffusion_step(model, scheduler, batch, config)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["train"].get("grad_clip", 1.0))
            )
            optimizer.step()

            running += float(loss.detach())
            global_step += 1
            if (step + 1) % log_every == 0:
                avg = running / log_every
                print(f"epoch={epoch + 1} step={step + 1} global_step={global_step} loss={avg:.5f}")
                running = 0.0

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    if scheduler is None:
                        val_losses.append(float(_paper_step(model, batch, config)))
                    else:
                        val_losses.append(float(_diffusion_step(model, scheduler, batch, config)))
            print(f"epoch={epoch + 1} val_loss={np.mean(val_losses):.5f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if (epoch + 1) % int(config["train"].get("save_every", 10)) == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(_load_config(args.config))


if __name__ == "__main__":
    main()
