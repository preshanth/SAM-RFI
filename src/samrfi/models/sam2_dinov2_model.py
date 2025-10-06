"""
SAM2+DINOv2 dual-encoder model for RFI detection.
Based on SAM2-UNeXT architecture (arxiv.org/html/2508.03566).

Key components:
- SAM2 encoder (frozen, with lightweight adapters)
- DINOv2 encoder (frozen)
- Dense glue layer (trainable)
- U-Net decoder (trainable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model, Dinov2Model


class Adapter(nn.Module):
    """
    Lightweight adapter for SAM2 blocks.
    32-channel bottleneck with residual connection.
    From SAM2-UNeXT paper.
    """
    def __init__(self, blk, bottleneck_dim=32):
        super().__init__()
        self.block = blk
        dim = blk.layer_norm1.normalized_shape[0]  # Get input dim from layer norm

        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, dim),
            nn.GELU()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Add adapter output to input (residual)
        adapted = x + self.adapter(x)
        # Forward through original block
        return self.block(adapted)


class DenseGlueLayer(nn.Module):
    """
    Dense glue layer for fusing SAM2 and DINOv2 features.

    Per stage:
    1. Align DINOv2 channels to SAM2 channels (1x1 conv)
    2. Resize to match SAM2 spatial dimensions
    3. Concatenate [SAM2 + DINOv2]
    4. Reduce to 128 channels
    """
    def __init__(self, dinov2_channels, sam2_stage_channels):
        """
        Args:
            dinov2_channels: 768 (base) or 1024 (large)
            sam2_stage_channels: [96,192,384,768] (tiny) or [144,288,576,1152] (large)
        """
        super().__init__()

        # Channel alignment layers (DINOv2 → SAM2 channels)
        self.align0 = nn.Conv2d(dinov2_channels, sam2_stage_channels[0], 1)
        self.align1 = nn.Conv2d(dinov2_channels, sam2_stage_channels[1], 1)
        self.align2 = nn.Conv2d(dinov2_channels, sam2_stage_channels[2], 1)
        self.align3 = nn.Conv2d(dinov2_channels, sam2_stage_channels[3], 1)

        # Reduction layers (Concat → 128 channels)
        self.reduce0 = nn.Conv2d(sam2_stage_channels[0] * 2, 128, 1)
        self.reduce1 = nn.Conv2d(sam2_stage_channels[1] * 2, 128, 1)
        self.reduce2 = nn.Conv2d(sam2_stage_channels[2] * 2, 128, 1)
        self.reduce3 = nn.Conv2d(sam2_stage_channels[3] * 2, 128, 1)

    def forward(self, dino_features, sam2_features):
        """
        Args:
            dino_features: [B, dinov2_channels, 32, 32]
            sam2_features: list of 4 tensors [x0, x1, x2, x3]
                x0: [B, C0, 256, 256]
                x1: [B, C1, 128, 128]
                x2: [B, C2, 64, 64]
                x3: [B, C3, 32, 32]

        Returns:
            list of 4 fused tensors, all [B, 128, Hi, Wi]
        """
        x0_s, x1_s, x2_s, x3_s = sam2_features

        # Align DINOv2 features to each SAM2 stage
        x0_d = F.interpolate(self.align0(dino_features), size=x0_s.shape[-2:], mode='bilinear', align_corners=False)
        x1_d = F.interpolate(self.align1(dino_features), size=x1_s.shape[-2:], mode='bilinear', align_corners=False)
        x2_d = F.interpolate(self.align2(dino_features), size=x2_s.shape[-2:], mode='bilinear', align_corners=False)
        x3_d = F.interpolate(self.align3(dino_features), size=x3_s.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate and reduce to 128 channels
        x0 = self.reduce0(torch.cat([x0_s, x0_d], dim=1))
        x1 = self.reduce1(torch.cat([x1_s, x1_d], dim=1))
        x2 = self.reduce2(torch.cat([x2_s, x2_d], dim=1))
        x3 = self.reduce3(torch.cat([x3_s, x3_d], dim=1))

        return [x0, x1, x2, x3]


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with optional skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: Features from deeper layer
            x2: Skip connection features (optional)
        """
        if x2 is not None:
            # Pad x2 if needed to match x1
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    Simple U-Net decoder with skip connections.
    From SAM2-UNeXT architecture.
    """
    def __init__(self):
        super().__init__()
        self.up3 = Up(128, 128)       # No skip
        self.up2 = Up(256, 128)       # With skip
        self.up1 = Up(256, 128)       # With skip
        self.up0 = Up(256, 128)       # With skip
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, fused_features):
        """
        Args:
            fused_features: list of [x0, x1, x2, x3], all 128 channels
                x0: [B, 128, 256, 256]  ← Largest
                x1: [B, 128, 128, 128]
                x2: [B, 128, 64, 64]
                x3: [B, 128, 32, 32]    ← Smallest (start here)

        Returns:
            [B, 1, 1024, 1024] mask
        """
        x0, x1, x2, x3 = fused_features

        # Start from smallest (deepest)
        x = self.up3(x3)           # [B, 128, 32, 32] → [B, 128, 64, 64]
        x = self.up2(x, x2)        # Concat → [B, 128, 128, 128]
        x = self.up1(x, x1)        # Concat → [B, 128, 256, 256]
        x = self.up0(x, x0)        # Concat → [B, 128, 512, 512]
        out = self.head(x)         # [B, 1, 512, 512]
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 1, 1024, 1024]
        return out


class SAM2DINOv2Model(nn.Module):
    """
    Dual-encoder model combining SAM2 and DINOv2 for RFI detection.

    Architecture:
    - SAM2 encoder @ 1024x1024 (frozen, with adapters)
    - DINOv2 encoder @ 448x448 (frozen)
    - Dense glue layer (trainable)
    - U-Net decoder (trainable)
    """
    def __init__(
        self,
        sam2_model="tiny",  # tiny, small, base_plus, large
        dinov2_model="base",  # base, large
        freeze_encoders=True,
        use_adapters=True,
        adapter_bottleneck=32
    ):
        super().__init__()

        # Model name mapping
        sam2_names = {
            "tiny": "facebook/sam2.1-hiera-tiny",
            "small": "facebook/sam2.1-hiera-small",
            "base_plus": "facebook/sam2.1-hiera-base-plus",
            "large": "facebook/sam2.1-hiera-large"
        }

        dinov2_names = {
            "base": "facebook/dinov2-with-registers-base",
            "large": "facebook/dinov2-with-registers-large"
        }

        # Channel configurations
        sam2_channels = {
            "tiny": [96, 192, 384, 768],
            "small": [96, 192, 384, 768],
            "base_plus": [112, 224, 448, 896],
            "large": [144, 288, 576, 1152]
        }

        dinov2_channels = {
            "base": 768,
            "large": 1024
        }

        self.sam2_model_name = sam2_model
        self.dinov2_model_name = dinov2_model
        self.use_adapters = use_adapters

        # Load SAM2 encoder
        print(f"Loading SAM2 {sam2_model}...")
        self.sam2 = Sam2Model.from_pretrained(sam2_names[sam2_model])
        self.sam2_backbone = self.sam2.vision_encoder.backbone

        # Freeze SAM2
        if freeze_encoders:
            for param in self.sam2.parameters():
                param.requires_grad = False

        # Add adapters to SAM2 blocks
        if use_adapters:
            print(f"Adding adapters (bottleneck={adapter_bottleneck})...")
            adapted_blocks = []
            for block in self.sam2_backbone.blocks:
                adapted_blocks.append(Adapter(block, adapter_bottleneck))
            self.sam2_backbone.blocks = nn.ModuleList(adapted_blocks)

        # Load DINOv2 encoder
        print(f"Loading DINOv2 {dinov2_model}...")
        self.dinov2 = Dinov2Model.from_pretrained(dinov2_names[dinov2_model])

        # Freeze DINOv2
        if freeze_encoders:
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # Dense glue layer
        print("Creating glue layer...")
        self.glue = DenseGlueLayer(
            dinov2_channels[dinov2_model],
            sam2_channels[sam2_model]
        )

        # U-Net decoder
        print("Creating decoder...")
        self.decoder = UNetDecoder()

        print(f"Model initialized: SAM2-{sam2_model} + DINOv2-{dinov2_model}")
        self._print_trainable_params()

    def _print_trainable_params(self):
        """Print trainable parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTrainable parameters:")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    def _extract_sam2_stages(self, x):
        """
        Extract multi-stage features from SAM2 backbone.

        Returns list of 4 stage features in PyTorch format [B, C, H, W]
        """
        # Patch embedding
        x = self.sam2_backbone.patch_embed(x)  # [B, H, W, C]

        stages = []
        for i, block in enumerate(self.sam2_backbone.blocks):
            x = block(x)
            # Save at stage boundaries (when shape changes or specific blocks)
            # Tiny: blocks 0,1,3,10
            # Large: blocks 1,7,43,47
            if self.sam2_model_name == "tiny" and i in [0, 1, 3, 10]:
                stages.append(x.permute(0, 3, 1, 2))  # [B, H, W, C] → [B, C, H, W]
            elif self.sam2_model_name == "large" and i in [1, 7, 43, 47]:
                stages.append(x.permute(0, 3, 1, 2))

        return stages

    def _extract_dinov2_spatial(self, x):
        """
        Extract spatial features from DINOv2.

        Returns [B, C, 32, 32]
        """
        # Resize to DINOv2 input size
        x_low = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)

        # Forward through DINOv2
        dino_out = self.dinov2(x_low)

        # Extract patch tokens (remove CLS token at position 0)
        # Shape: [B, num_tokens, hidden_size]
        patch_tokens = dino_out.last_hidden_state[:, 1:1025, :]  # [B, 1024, C]

        # Reshape to spatial (32x32 grid)
        B, N, C = patch_tokens.shape
        spatial = patch_tokens.reshape(B, 32, 32, C).permute(0, 3, 1, 2)  # [B, C, 32, 32]

        return spatial

    def forward(self, pixel_values):
        """
        Forward pass.

        Args:
            pixel_values: [B, 3, 1024, 1024] input images

        Returns:
            [B, 1, 1024, 1024] predicted masks
        """
        # Extract SAM2 features
        sam2_features = self._extract_sam2_stages(pixel_values)

        # Extract DINOv2 features
        dino_features = self._extract_dinov2_spatial(pixel_values)

        # Fuse features
        fused_features = self.glue(dino_features, sam2_features)

        # Decode to mask
        mask = self.decoder(fused_features)

        return mask
