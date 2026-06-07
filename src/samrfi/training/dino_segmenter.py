"""
DINOv3 frozen-backbone RFI segmenter with a DPT-style decoder.

Pure dense segmentation: complex visibility patch -> 3-channel input ->
frozen DINOv3 ViT features (multiple depths) -> DPT reassemble + fusion
decoder -> single-channel RFI mask logits at input resolution.

No prompts, no boxes (unlike the SAM2 path). The backbone is frozen; only
the decoder trains, so capacity/overfit scale with the (small) head rather
than the (large) pretrained encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv3 ViT-S/16 pretrained on LVD-1689M (smallest tier). License-gated on HF.
DINOV3_MODEL_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def complex_to_channels(z: torch.Tensor, mode: str = "amplitude") -> torch.Tensor:
    """Pack a complex patch (B, H, W) complex into a 3-channel float tensor.

    amplitude: log10|z| replicated x3 (baseline, throws away phase).
    realimag:  [Re, Im, |z|] -- full complex information, no phase wrapping.
    Each channel is per-sample standardised (zero mean, unit std).
    """
    if not torch.is_complex(z):
        raise ValueError("expected a complex tensor")

    amp = z.abs()
    logamp = torch.log10(amp + 1e-9)

    def standardize(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        mean = flat.mean(1).view(-1, 1, 1)
        std = flat.std(1).view(-1, 1, 1) + 1e-6
        return (x - mean) / std

    if mode == "amplitude":
        c = standardize(logamp)
        return torch.stack([c, c, c], dim=1)
    elif mode == "realimag":
        re = standardize(z.real)
        im = standardize(z.imag)
        am = standardize(logamp)
        return torch.stack([re, im, am], dim=1)
    raise ValueError(f"unknown input mode: {mode!r} (use 'amplitude' or 'realimag')")


class ResidualConvUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(self.act(x))))
        return out + x


class FusionBlock(nn.Module):
    """DPT RefineNet fusion: optionally add skip, refine, upsample x2."""

    def __init__(self, dim: int):
        super().__init__()
        self.rcu1 = ResidualConvUnit(dim)
        self.rcu2 = ResidualConvUnit(dim)
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class DPTDecoder(nn.Module):
    """Reassemble four equal-resolution ViT feature maps into a pyramid,
    then fuse top-down to a dense mask. Resample factors recover sub-patch
    detail so thin RFI lines do not come out blocky."""

    def __init__(self, in_dim: int, dim: int = 128, n_features: int = 4):
        super().__init__()
        assert n_features == 4, "decoder assumes 4 feature maps"
        # 1x1 projections to common decoder dim
        self.projs = nn.ModuleList([nn.Conv2d(in_dim, dim, 1) for _ in range(4)])
        # Reassemble: shallow->upsample (more spatial), deep->downsample
        self.resample = nn.ModuleList(
            [
                nn.ConvTranspose2d(dim, dim, 4, stride=4),  # x4
                nn.ConvTranspose2d(dim, dim, 2, stride=2),  # x2
                nn.Identity(),  # x1
                nn.Conv2d(dim, dim, 3, stride=2, padding=1),  # /2
            ]
        )
        self.fusions = nn.ModuleList([FusionBlock(dim) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
        )

    def forward(self, feats, out_size):
        # feats: list of 4 tensors (B, in_dim, Hp, Wp), shallow -> deep
        res = [self.resample[i](self.projs[i](f)) for i, f in enumerate(feats)]
        # top-down fusion starting from the deepest (smallest) map
        x = self.fusions[3](res[3])
        x = self.fusions[2](x, _match(res[2], x))
        x = self.fusions[1](x, _match(res[1], x))
        x = self.fusions[0](x, _match(res[0], x))
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


def _match(skip, x):
    if skip.shape[-2:] != x.shape[-2:]:
        skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return skip


class DINOSegmenter(nn.Module):
    def __init__(
        self,
        backbone_size: str = "small",
        model_id: str | None = None,
        out_indices=(3, 6, 9, 12),
        decoder_dim: int = 128,
        pretrained: bool = True,
        config_only: bool = False,
    ):
        """
        backbone_size: 'small'/'base'/'large' -> DINOv3 (gated on HF).
        model_id: override with any HF ViT id (e.g. 'facebook/dinov2-small',
            ungated) -- the feature extraction is backbone-agnostic.
        config_only: random-init DINOv3-S config for offline smoke testing.
        """
        super().__init__()
        from transformers import AutoModel

        if config_only:
            from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel

            cfg = DINOv3ViTConfig(
                patch_size=16,
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                intermediate_size=1536,
                num_register_tokens=4,
                image_size=512,
            )
            self.backbone = DINOv3ViTModel(cfg)
        else:
            hf_id = model_id or DINOV3_MODEL_IDS[backbone_size]
            if pretrained:
                self.backbone = AutoModel.from_pretrained(hf_id)
            else:
                from transformers import AutoConfig

                self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(hf_id))

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        cfg = self.backbone.config
        self.patch_size = cfg.patch_size
        # CLS (1) + any register tokens. DINOv2-small has none; DINOv3 has 4.
        self.num_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self.out_indices = out_indices
        self.decoder = DPTDecoder(cfg.hidden_size, decoder_dim)

    @torch.no_grad()
    def _features(self, pixel_values):
        out = self.backbone(pixel_values, output_hidden_states=True)
        hs = out.hidden_states  # len = num_layers + 1 (index 0 = embeddings)
        B, _, H, W = pixel_values.shape
        hp, wp = H // self.patch_size, W // self.patch_size
        feats = []
        for idx in self.out_indices:
            tokens = hs[idx][:, self.num_prefix :, :]  # (B, hp*wp, C)
            feats.append(tokens.transpose(1, 2).reshape(B, -1, hp, wp))
        return feats

    def forward(self, pixel_values):
        feats = self._features(pixel_values)
        return self.decoder(feats, out_size=pixel_values.shape[-2:])
