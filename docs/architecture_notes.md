# Architecture Notes

## Paper Track

LayeredDepth evaluates multi-layer depth prediction by adapting a monocular
metric-depth backbone. The paper uses NeWCRFs and compares three training
interfaces:

- Multi-head Output: one RGB forward pass predicts all depth layers.
- Layer Index Concatenation: concatenate a normalized layer-index map to RGB and
  predict one requested layer.
- Recurrent: concatenate RGB with the previous layer depth and predict the next
  layer.

The training loop follows the paper recipe:

- train from scratch on the synthetic LayeredDepth-Syn data;
- select a random target layer at each step;
- use snapped layered depth so missing layer-k pixels inherit the previous
  valid layer where possible;
- optimize SILog loss.

The Hugging Face release exposes eight synthetic depth columns
(`depth_1.png` through `depth_8.png`), so the official training configs default
to eight layers for LayeredDepth-Syn.

Source: https://layereddepth.cs.princeton.edu/ and
https://arxiv.org/abs/2503.11633

## Diffusion Track

The diffusion model treats the whole layered-depth stack as the denoising
target. At training time, it adds DDPM noise to the stack and predicts the noise
conditioned on RGB. The default config uses log-depth targets because metric
depth values are positive and long-tailed.

This is intentionally separate from the paper baselines so it can later be
extended with stronger conditioning, latent diffusion, temporal consistency, or
joint RGB/depth encoders without disturbing the reproduced baselines.
