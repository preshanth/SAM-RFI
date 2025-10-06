#!/usr/bin/env python3
"""
Check actual feature dimensions for SAM2.1 and DINOv2-with-registers.
Critical for implementing SAM2-UNeXT style architecture.
"""

import torch
from transformers import Sam2Model, Dinov2Model

print("="*80)
print("Feature Dimension Checker for SAM2.1 + DINOv2")
print("="*80)

# ==============================================================================
# 1. SAM2.1-tiny
# ==============================================================================
print("\n" + "="*80)
print("1. SAM2.1-tiny (facebook/sam2.1-hiera-tiny)")
print("="*80)

sam_tiny = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
sam_tiny.eval()

# Create dummy input
x = torch.randn(1, 3, 1024, 1024)
print(f"\nInput shape: {x.shape}")

# Check model structure
print("\nSAM2 model structure:")
print(f"  Vision encoder: {sam_tiny.vision_encoder}")
print(f"  Has backbone: {hasattr(sam_tiny.vision_encoder, 'backbone')}")

# Try to get features
with torch.no_grad():
    # Method 1: Full forward pass
    try:
        print("\nAttempting full forward pass (needs prompts)...")
        # This will likely fail without prompts, but shows what's needed
        output = sam_tiny(pixel_values=x)
        print(f"  Output keys: {output.keys() if hasattr(output, 'keys') else type(output)}")
    except Exception as e:
        print(f"  Failed (expected): {str(e)[:100]}")

    # Method 2: Try vision encoder only
    try:
        print("\nAttempting vision encoder only...")
        vision_out = sam_tiny.vision_encoder(x)
        print(f"  Vision output type: {type(vision_out)}")
        if hasattr(vision_out, 'keys'):
            print(f"  Output keys: {vision_out.keys()}")
        if hasattr(vision_out, 'shape'):
            print(f"  Output shape: {vision_out.shape}")
    except Exception as e:
        print(f"  Failed: {str(e)[:200]}")

    # Method 3: Check backbone directly
    if hasattr(sam_tiny.vision_encoder, 'backbone'):
        print("\nAttempting backbone only...")
        try:
            backbone_out = sam_tiny.vision_encoder.backbone(x)
            print(f"  Backbone output type: {type(backbone_out)}")
            if isinstance(backbone_out, (list, tuple)):
                print(f"  Number of stages: {len(backbone_out)}")
                for i, feat in enumerate(backbone_out):
                    print(f"  Stage {i}: {feat.shape}")
            elif hasattr(backbone_out, 'shape'):
                print(f"  Output shape: {backbone_out.shape}")
        except Exception as e:
            print(f"  Failed: {str(e)[:200]}")

print("\nSAM2.1-tiny config:")
print(f"  {sam_tiny.config}")

# ==============================================================================
# 2. SAM2.1-large (for comparison)
# ==============================================================================
print("\n" + "="*80)
print("2. SAM2.1-large (facebook/sam2.1-hiera-large) - for comparison")
print("="*80)

sam_large = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
sam_large.eval()

print("\nSAM2.1-large config:")
print(f"  {sam_large.config}")

# ==============================================================================
# 3. DINOv2-with-registers-base
# ==============================================================================
print("\n" + "="*80)
print("3. DINOv2-with-registers-base (facebook/dinov2-with-registers-base)")
print("="*80)

dino_base = Dinov2Model.from_pretrained("facebook/dinov2-with-registers-base")
dino_base.eval()

# DINOv2 input (paper uses 448×448)
x_dino = torch.randn(1, 3, 448, 448)
print(f"\nInput shape: {x_dino.shape}")

with torch.no_grad():
    dino_out = dino_base(x_dino)
    print(f"\nOutput type: {type(dino_out)}")
    print(f"Output attributes: {dir(dino_out)}")

    if hasattr(dino_out, 'last_hidden_state'):
        print(f"\nlast_hidden_state shape: {dino_out.last_hidden_state.shape}")
        # Shape: [batch, num_patches + cls_token + register_tokens, hidden_size]

        # Calculate number of patches
        patch_size = dino_base.config.patch_size
        num_patches = (448 // patch_size) ** 2
        print(f"\nPatch size: {patch_size}")
        print(f"Number of patches: {num_patches} ({448//patch_size} × {448//patch_size})")
        print(f"CLS token: 1")
        print(f"Register tokens: {dino_base.config.num_register_tokens}")
        print(f"Total tokens: {1 + num_patches + dino_base.config.num_register_tokens}")

        # Extract patch tokens (remove CLS + registers)
        num_special_tokens = 1 + dino_base.config.num_register_tokens
        patch_tokens = dino_out.last_hidden_state[:, num_special_tokens:, :]
        print(f"\nPatch tokens only: {patch_tokens.shape}")

        # Reshape to spatial
        h = w = 448 // patch_size
        spatial_features = patch_tokens.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        print(f"Spatial features (B×C×H×W): {spatial_features.shape}")

print("\nDINOv2-with-registers-base config:")
print(f"  Image size: {dino_base.config.image_size}")
print(f"  Hidden size: {dino_base.config.hidden_size}")
print(f"  Patch size: {dino_base.config.patch_size}")
print(f"  Register tokens: {dino_base.config.num_register_tokens}")

# ==============================================================================
# 4. DINOv2-with-registers-large (for production)
# ==============================================================================
print("\n" + "="*80)
print("4. DINOv2-with-registers-large (facebook/dinov2-with-registers-large)")
print("="*80)

dino_large = Dinov2Model.from_pretrained("facebook/dinov2-with-registers-large")
dino_large.eval()

with torch.no_grad():
    dino_large_out = dino_large(x_dino)
    if hasattr(dino_large_out, 'last_hidden_state'):
        print(f"\nlast_hidden_state shape: {dino_large_out.last_hidden_state.shape}")

print("\nDINOv2-with-registers-large config:")
print(f"  Image size: {dino_large.config.image_size}")
print(f"  Hidden size: {dino_large.config.hidden_size}")
print(f"  Patch size: {dino_large.config.patch_size}")
print(f"  Register tokens: {dino_large.config.num_register_tokens}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nFor SAM2-UNeXT style architecture, we need:")
print("  1. SAM2 multi-stage features (144, 288, 576, 1152 for Large)")
print("  2. DINOv2 spatial features (need to reshape from tokens)")
print("  3. Align + concatenate + reduce to 128 channels")

print("\nKnown dimensions:")
print(f"  DINOv2-base @ 448×448 → spatial: 32×32×768")
print(f"  DINOv2-large @ 448×448 → spatial: 32×32×1024")

print("\nUnknown (TODO):")
print("  SAM2.1-tiny stage outputs (need to extract from backbone)")
print("  SAM2.1-large stage outputs (for comparison)")

print("\n" + "="*80)
print("Next: Manually inspect SAM2 backbone to extract stage features")
print("="*80)
