#!/usr/bin/env python3
"""
Test SAM2+DINOv2 model forward pass on 1 batch.
Verifies dimensions, memory usage, and that model runs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.samrfi.models import SAM2DINOv2Model

print("="*80)
print("SAM2+DINOv2 Model Forward Pass Test")
print("="*80)

# Test both configs
configs = [
    ("tiny", "base", 2),    # Local config
    # ("large", "large", 1),  # Production config (comment out if no GPU)
]

for sam2_model, dinov2_model, batch_size in configs:
    print(f"\n{'='*80}")
    print(f"Testing: SAM2-{sam2_model} + DINOv2-{dinov2_model}, batch_size={batch_size}")
    print(f"{'='*80}")

    # Create model
    model = SAM2DINOv2Model(
        sam2_model=sam2_model,
        dinov2_model=dinov2_model,
        freeze_encoders=True,
        use_adapters=True
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = model.to(device)
    model.eval()

    # Create dummy input
    x = torch.randn(batch_size, 3, 1024, 1024, device=device)
    print(f"\nInput shape: {x.shape}")

    # Check memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory before: {mem_before:.2f} GB")

    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(x)

        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: torch.Size([{batch_size}, 1, 1024, 1024])")
        print(f"  Match: {output.shape == torch.Size([batch_size, 1, 1024, 1024])}")

        # Check memory after
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\nGPU memory after: {mem_after:.2f} GB")
            print(f"GPU memory peak: {mem_peak:.2f} GB")
            print(f"Memory used: {mem_peak - mem_before:.2f} GB")

        # Check output range
        print(f"\nOutput statistics:")
        print(f"  Min: {output.min().item():.4f}")
        print(f"  Max: {output.max().item():.4f}")
        print(f"  Mean: {output.mean().item():.4f}")

    except Exception as e:
        print(f"✗ Forward pass failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Test complete!")
print("="*80)
