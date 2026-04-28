from __future__ import annotations

import argparse
import shutil
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


SummaryWriterType = Any


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


def _make_summary_writer(config: dict[str, Any], output_dir: Path) -> SummaryWriterType | None:
    tensorboard_config = config.get("tensorboard", {})
    if not bool(tensorboard_config.get("enabled", True)):
        return None

    log_dir = tensorboard_config.get("log_dir")
    log_dir = _project_path(config, log_dir) if log_dir else output_dir / "tensorboard"

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logging is enabled, but neither tensorboard nor "
                "tensorboardX is installed. Install with: python -m pip install tensorboard"
            ) from exc

    return SummaryWriter(log_dir=str(log_dir))


def _depth_to_display(depth: Tensor, valid: Tensor | None = None) -> Tensor:
    depth = depth.detach().float().cpu()
    if valid is not None:
        valid = valid.detach().bool().cpu()
        if valid.ndim == depth.ndim - 1:
            valid = valid.unsqueeze(1)
        valid = valid.expand_as(depth)
    else:
        valid = torch.isfinite(depth)

    values = depth[valid & torch.isfinite(depth)]
    if values.numel() == 0:
        return torch.zeros_like(depth)

    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    if float(hi - lo) < 1e-6:
        hi = lo + 1.0

    display = torch.clamp((depth - lo) / (hi - lo), 0.0, 1.0)
    display = torch.where(valid, display, torch.zeros_like(display))
    return display


@torch.no_grad()
def _log_tensorboard_images(
    writer: SummaryWriterType | None,
    model: torch.nn.Module,
    scheduler: DiffusionScheduler | None,
    batch: dict[str, Tensor] | None,
    config: dict[str, Any],
    step: int,
    tag_prefix: str,
) -> None:
    if writer is None or batch is None:
        return

    tensorboard_config = config.get("tensorboard", {})
    max_images = int(tensorboard_config.get("max_images", 2))
    max_layers = int(tensorboard_config.get("max_layers", config["model"].get("num_layers", 8)))

    image = batch["image"][:max_images]
    depth = batch["depth"][:max_images]
    valid = batch["valid"][:max_images]

    writer.add_images(f"{tag_prefix}/rgb", image.detach().cpu().clamp(0.0, 1.0), step)

    for layer_idx in range(min(depth.shape[1], max_layers)):
        layer_tag = f"layer_{layer_idx + 1:02d}"
        target = depth[:, layer_idx : layer_idx + 1]
        target_valid = valid[:, layer_idx : layer_idx + 1]
        writer.add_images(
            f"{tag_prefix}/target/{layer_tag}",
            _depth_to_display(target, target_valid),
            step,
        )

    if scheduler is not None:
        diffusion_sample_steps = int(tensorboard_config.get("diffusion_sample_steps", 0))
        if diffusion_sample_steps <= 0:
            return
        model.eval()
        device = next(model.parameters()).device
        sampled = scheduler.sample_ddim(
            model=model,
            image=image.to(device),
            shape=(image.shape[0], depth.shape[1], image.shape[-2], image.shape[-1]),
            valid_masks=valid.to(device).float(),
            steps=diffusion_sample_steps,
        )
        if bool(config["model"].get("diffusion", {}).get("log_depth", True)):
            sampled = torch.exp(sampled)
        sampled = sampled.clamp_min(0.0)
        for layer_idx in range(min(sampled.shape[1], max_layers)):
            layer_tag = f"layer_{layer_idx + 1:02d}"
            pred = sampled[:, layer_idx : layer_idx + 1]
            pred_valid = valid[:, layer_idx : layer_idx + 1].to(pred.device)
            writer.add_images(
                f"{tag_prefix}/prediction/{layer_tag}",
                _depth_to_display(pred, pred_valid),
                step,
            )
        return

    model.eval()
    prediction = model.predict_all_layers(image.to(next(model.parameters()).device))
    prediction = prediction[:max_images]
    for layer_idx in range(min(prediction.shape[1], max_layers)):
        layer_tag = f"layer_{layer_idx + 1:02d}"
        pred = prediction[:, layer_idx : layer_idx + 1]
        pred_valid = valid[:, layer_idx : layer_idx + 1].to(pred.device)
        writer.add_images(
            f"{tag_prefix}/prediction/{layer_tag}",
            _depth_to_display(pred, pred_valid),
            step,
        )


def _build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    include_optimizer: bool,
) -> dict[str, Any]:
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if include_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    return checkpoint


def _format_free_space(path: Path) -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return "unknown"
    return f"{usage.free / (1024 ** 3):.2f} GiB free"


def _save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to save checkpoint to {path}. "
            f"Disk state near output directory: {_format_free_space(path.parent)}. "
            "If disk space is low, set train.save_optimizer: false, "
            "increase train.save_every, or move train.output_dir to a larger disk."
        ) from exc


def _load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print(f"Checkpoint {path} has no optimizer state; resuming with a fresh optimizer.")
    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Resumed from {path}: completed_epoch={start_epoch} global_step={global_step}")
    return start_epoch, global_step


def train(config: dict[str, Any], resume: str | Path | None = None) -> None:
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
    writer = _make_summary_writer(config, output_dir)

    epochs = int(config["train"].get("epochs", 100))
    log_every = int(config["train"].get("log_every", 20))
    save_optimizer = bool(config["train"].get("save_optimizer", True))
    save_epoch_optimizer = bool(config["train"].get("save_epoch_optimizer", False))
    image_every_epochs = int(config.get("tensorboard", {}).get("image_every_epochs", 1))

    resume_path = Path(resume) if resume is not None else config["train"].get("resume")
    resume_path = _project_path(config, resume_path) if resume_path else None
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        start_epoch, global_step = _load_training_checkpoint(
            resume_path, model=model, optimizer=optimizer, device=device
        )

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running = 0.0
            epoch_losses = []
            for step, batch in enumerate(train_loader):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                if scheduler is None:
                    loss = _paper_step(model, batch, config)
                else:
                    loss = _diffusion_step(model, scheduler, batch, config)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["train"].get("grad_clip", 1.0))
                )
                optimizer.step()

                loss_value = float(loss.detach())
                running += loss_value
                epoch_losses.append(loss_value)
                global_step += 1
                if (step + 1) % log_every == 0:
                    avg = running / log_every
                    print(f"epoch={epoch + 1} step={step + 1} global_step={global_step} loss={avg:.5f}")
                    if writer is not None:
                        writer.add_scalar("train/loss", avg, global_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    running = 0.0

            epoch_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            if writer is not None:
                writer.add_scalar("epoch/train_loss", epoch_train_loss, epoch + 1)

            first_val_batch = None
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                        if first_val_batch is None:
                            first_val_batch = {key: value.detach() for key, value in batch.items()}
                        if scheduler is None:
                            val_losses.append(float(_paper_step(model, batch, config)))
                        else:
                            val_losses.append(float(_diffusion_step(model, scheduler, batch, config)))
                val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                print(f"epoch={epoch + 1} val_loss={val_loss:.5f}")
                if writer is not None:
                    writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)

            if image_every_epochs > 0 and (epoch + 1) % image_every_epochs == 0:
                _log_tensorboard_images(
                    writer,
                    model,
                    scheduler,
                    first_val_batch,
                    config,
                    global_step,
                    tag_prefix="val",
                )

            checkpoint = _build_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=config,
                include_optimizer=save_optimizer,
            )
            _save_checkpoint(checkpoint, output_dir / "last.pt")
            if (epoch + 1) % int(config["train"].get("save_every", 10)) == 0:
                epoch_checkpoint = checkpoint
                if save_optimizer != save_epoch_optimizer:
                    epoch_checkpoint = _build_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        global_step=global_step,
                        config=config,
                        include_optimizer=save_epoch_optimizer,
                    )
                _save_checkpoint(epoch_checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt")
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint such as runs/paper_index_concat/last.pt.",
    )
    args = parser.parse_args()
    train(_load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
