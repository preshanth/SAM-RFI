#!/usr/bin/env python3
"""
Extract SAM2 backbone multi-stage features.
Shows actual spatial dimensions for glue layer design.
"""

import torch
from transformers import Sam2Model

print("="*80)
print("SAM2 Backbone Feature Extraction")
print("="*80)

# Input
x = torch.randn(1, 3, 1024, 1024)
print(f"\nInput shape: {x.shape}")

# ==============================================================================
# SAM2.1-tiny
# ==============================================================================
print("\n" + "="*80)
print("SAM2.1-tiny")
print("="*80)

model_tiny = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
model_tiny.eval()

print("\nExpected stage channels (from config):")
print("  Stage 1: 96")
print("  Stage 2: 192")
print("  Stage 3: 384")
print("  Stage 4: 768")

with torch.no_grad():
    # Try vision encoder
    vision_out = model_tiny.vision_encoder(x)

    print("\n--- Vision Encoder Output ---")
    print(f"Type: {type(vision_out)}")
    print(f"Keys: {vision_out.keys() if hasattr(vision_out, 'keys') else 'N/A'}")

    if hasattr(vision_out, 'last_hidden_state'):
        print(f"\nlast_hidden_state: {vision_out.last_hidden_state.shape}")

    if hasattr(vision_out, 'fpn_hidden_states') and vision_out.fpn_hidden_states is not None:
        print(f"\nfpn_hidden_states (after FPN neck):")
        for i, feat in enumerate(vision_out.fpn_hidden_states):
            print(f"  FPN stage {i}: {feat.shape}")

    # Try to access backbone directly
    print("\n--- Backbone Direct Access ---")
    backbone = model_tiny.vision_encoder.backbone
    print(f"Backbone type: {type(backbone)}")

    # Forward through backbone only
    try:
        backbone_out = backbone(x)
        print(f"\nBackbone output type: {type(backbone_out)}")

        if hasattr(backbone_out, 'keys'):
            print(f"Backbone output keys: {backbone_out.keys()}")

        # Check for feature_maps attribute (common in vision backbones)
        if hasattr(backbone_out, 'feature_maps'):
            print("\nBackbone feature_maps:")
            for i, feat in enumerate(backbone_out.feature_maps):
                print(f"  Stage {i}: {feat.shape}")

        # Check for hidden_states attribute
        if hasattr(backbone_out, 'hidden_states') and backbone_out.hidden_states is not None:
            print("\nBackbone hidden_states:")
            for i, feat in enumerate(backbone_out.hidden_states):
                print(f"  Stage {i}: {feat.shape}")

        # If it's just a tensor
        if hasattr(backbone_out, 'shape'):
            print(f"\nBackbone single output: {backbone_out.shape}")

    except Exception as e:
        print(f"Backbone forward failed: {e}")

    # Try to manually access blocks
    print("\n--- Manual Block Inspection ---")
    print(f"Number of blocks: {len(backbone.blocks)}")

    # Try forward through blocks manually to see intermediate outputs
    print("\nManual forward through blocks:")
    try:
        # Initial patch embedding
        x_tmp = backbone.patch_embed(x)
        print(f"  After patch_embed: {x_tmp.shape}")

        # Store intermediate outputs
        stage_outputs = []

        # Forward through blocks
        for i, block in enumerate(backbone.blocks):
            x_tmp = block(x_tmp)
            print(f"  After block {i}: {x_tmp.shape}")

            # Check if this is a stage boundary (when channels change)
            if i == 0 or (i > 0 and x_tmp.shape != stage_outputs[-1].shape):
                stage_outputs.append(x_tmp.clone())

        print(f"\nStage outputs ({len(stage_outputs)} stages):")
        for i, feat in enumerate(stage_outputs):
            print(f"  Stage {i}: {feat.shape}")

    except Exception as e:
        print(f"Manual forward failed: {e}")

# ==============================================================================
# SAM2.1-large (for comparison)
# ==============================================================================
print("\n" + "="*80)
print("SAM2.1-large")
print("="*80)

model_large = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
model_large.eval()

print("\nExpected stage channels (from config):")
print("  Stage 1: 144")
print("  Stage 2: 288")
print("  Stage 3: 576")
print("  Stage 4: 1152")

with torch.no_grad():
    vision_out_large = model_large.vision_encoder(x)

    print("\n--- Vision Encoder Output ---")
    if hasattr(vision_out_large, 'last_hidden_state'):
        print(f"last_hidden_state: {vision_out_large.last_hidden_state.shape}")

    if hasattr(vision_out_large, 'fpn_hidden_states') and vision_out_large.fpn_hidden_states is not None:
        print(f"\nfpn_hidden_states:")
        for i, feat in enumerate(vision_out_large.fpn_hidden_states):
            print(f"  FPN stage {i}: {feat.shape}")

    # Manual forward
    print("\n--- Manual Block Forward ---")
    backbone_large = model_large.vision_encoder.backbone
    print(f"Number of blocks: {len(backbone_large.blocks)}")

    try:
        x_tmp = backbone_large.patch_embed(x)
        print(f"  After patch_embed: {x_tmp.shape}")

        stage_outputs_large = []
        for i, block in enumerate(backbone_large.blocks):
            x_tmp = block(x_tmp)
            print(f"  After block {i}: {x_tmp.shape}")

            if i == 0 or (i > 0 and x_tmp.shape != stage_outputs_large[-1].shape):
                stage_outputs_large.append(x_tmp.clone())

        print(f"\nStage outputs ({len(stage_outputs_large)} stages):")
        for i, feat in enumerate(stage_outputs_large):
            print(f"  Stage {i}: {feat.shape}")

    except Exception as e:
        print(f"Manual forward failed: {e}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nFor dual-encoder architecture, we need:")
print("  1. Backbone stage features (BEFORE FPN neck)")
print("  2. 4 stages with different channels/spatial sizes")
print("  3. These will be fused with DINOv2 features")

print("\nRun this script and send output to determine:")
print("  - Actual spatial dimensions (H, W) for each stage")
print("  - How to extract features from HuggingFace SAM2 model")
print("  - Whether to use fpn_hidden_states or manual block extraction")

print("\n" + "="*80)
