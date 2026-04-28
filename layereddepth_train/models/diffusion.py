from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.dim // 2
        device = timestep.device
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        args = timestep.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.time = nn.Linear(time_dim, out_channels)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(time_embedding)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h


class ConditionalLayeredDepthUNet(nn.Module):
    """RGB-conditioned DDPM denoiser for a stack of layered depth maps."""

    def __init__(
        self,
        num_layers: int = 7,
        base_channels: int = 64,
        time_dim: int = 256,
        condition_valid_masks: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.condition_valid_masks = condition_valid_masks

        in_channels = 3 + num_layers + (num_layers if condition_valid_masks else 0)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

        self.enc1 = TimeBlock(in_channels, base_channels, time_dim)
        self.enc2 = TimeBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = TimeBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid = TimeBlock(base_channels * 4, base_channels * 4, time_dim)
        self.dec3 = TimeBlock(base_channels * 8, base_channels * 2, time_dim)
        self.dec2 = TimeBlock(base_channels * 4, base_channels, time_dim)
        self.dec1 = TimeBlock(base_channels * 2, base_channels, time_dim)
        self.head = nn.Conv2d(base_channels, num_layers, 1)

    def forward(
        self,
        image: Tensor,
        noisy_depth: Tensor,
        timestep: Tensor,
        valid_masks: Tensor | None = None,
    ) -> Tensor:
        inputs = [image, noisy_depth]
        if self.condition_valid_masks:
            if valid_masks is None:
                valid_masks = torch.ones_like(noisy_depth)
            inputs.append(valid_masks.float())
        x = torch.cat(inputs, dim=1)
        t = self.time_mlp(timestep)

        e1 = self.enc1(x, t)
        e2 = self.enc2(F.avg_pool2d(e1, 2), t)
        e3 = self.enc3(F.avg_pool2d(e2, 2), t)
        mid = self.mid(F.avg_pool2d(e3, 2), t)

        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t)
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t)
        return self.head(d1)


class DiffusionScheduler(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, clean: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timestep][:, None, None, None]
        return sqrt_alpha * clean + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(
        self,
        model: ConditionalLayeredDepthUNet,
        image: Tensor,
        current: Tensor,
        timestep: Tensor,
        valid_masks: Tensor | None = None,
    ) -> Tensor:
        beta_t = self.betas[timestep][:, None, None, None]
        alpha_t = self.alphas[timestep][:, None, None, None]
        alpha_bar_t = self.alphas_cumprod[timestep][:, None, None, None]
        pred_noise = model(image, current, timestep, valid_masks=valid_masks)
        mean = (current - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
        noise = torch.randn_like(current)
        nonzero = (timestep != 0).float()[:, None, None, None]
        return mean + nonzero * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: ConditionalLayeredDepthUNet,
        image: Tensor,
        shape: tuple[int, int, int, int],
        valid_masks: Tensor | None = None,
    ) -> Tensor:
        current = torch.randn(shape, device=image.device)
        for step in reversed(range(self.timesteps)):
            timestep = torch.full((shape[0],), step, device=image.device, dtype=torch.long)
            current = self.p_sample(model, image, current, timestep, valid_masks=valid_masks)
        return current

    @torch.no_grad()
    def sample_ddim(
        self,
        model: ConditionalLayeredDepthUNet,
        image: Tensor,
        shape: tuple[int, int, int, int],
        valid_masks: Tensor | None = None,
        steps: int = 50,
    ) -> Tensor:
        """Fast deterministic DDIM-style sampler for validation visualization."""
        steps = max(1, min(int(steps), self.timesteps))
        current = torch.randn(shape, device=image.device)
        timesteps = torch.linspace(
            self.timesteps - 1,
            0,
            steps,
            device=image.device,
            dtype=torch.long,
        )

        for index, timestep_value in enumerate(timesteps):
            timestep = torch.full(
                (shape[0],),
                int(timestep_value.item()),
                device=image.device,
                dtype=torch.long,
            )
            pred_noise = model(image, current, timestep, valid_masks=valid_masks)
            alpha_bar_t = self.alphas_cumprod[timestep][:, None, None, None]
            pred_clean = (
                current - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            if index + 1 < len(timesteps):
                prev_timestep = torch.full(
                    (shape[0],),
                    int(timesteps[index + 1].item()),
                    device=image.device,
                    dtype=torch.long,
                )
                alpha_bar_prev = self.alphas_cumprod[prev_timestep][:, None, None, None]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            current = (
                torch.sqrt(alpha_bar_prev) * pred_clean
                + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            )
        return current


def build_diffusion_model(config: Mapping[str, Any]) -> ConditionalLayeredDepthUNet:
    return ConditionalLayeredDepthUNet(
        num_layers=int(config.get("num_layers", 7)),
        base_channels=int(config.get("base_channels", 64)),
        time_dim=int(config.get("time_dim", 256)),
        condition_valid_masks=bool(config.get("condition_valid_masks", False)),
    )
