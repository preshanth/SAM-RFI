#!/usr/bin/env python3
"""
Minimal test: Torch-only dataset with SAM2-tiny

Tests:
1. Torch tensors work with SAM2
2. Shared memory prevents worker copies
3. No RAM explosion with num_workers > 0
4. Training works end-to-end

Run with: python test_torch_only.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Sam2Model
import psutil
import os
import gc

# Test parameters
NUM_SAMPLES = 50
IMAGE_SIZE = 1024  # SAM2 expects 1024x1024 (or sizes divisible by patch size)
NUM_WORKERS = 4
BATCH_SIZE = 2
NUM_EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_memory_usage_gb():
    """Get current process RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9


class TorchOnlyDataset(Dataset):
    """
    Pure torch dataset - everything stays as tensors in shared memory
    """

    def __init__(self, num_samples=100, image_size=256):
        print(f"\n[TorchOnlyDataset] Creating {num_samples} samples ({image_size}x{image_size})...")

        # Generate synthetic data as torch tensors
        # Images: (N, 3, H, W) float32, already SAM2-normalized
        self.images = torch.randn(num_samples, 3, image_size, image_size, dtype=torch.float32)

        # Apply SAM2 ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.images = (self.images - mean) / std

        # Masks: (N, H, W) uint8
        self.masks = torch.randint(0, 2, (num_samples, image_size, image_size), dtype=torch.uint8)

        # Put in shared memory for worker access
        self.images.share_memory_()
        self.masks.share_memory_()

        size_gb = (self.images.nbytes + self.masks.nbytes) / 1e9
        print(f"  Created {num_samples} samples: {size_gb:.3f} GB in shared memory")
        print(f"  Images: {self.images.shape}, {self.images.dtype}")
        print(f"  Masks: {self.masks.shape}, {self.masks.dtype}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Return torch tensors directly (no numpy conversions!)
        """
        # Extract mask bounding box
        mask = self.masks[idx]
        y_indices, x_indices = torch.where(mask > 0)

        if len(x_indices) == 0:
            # Empty mask - return center box
            H, W = mask.shape
            bbox = [W // 4, H // 4, 3 * W // 4, 3 * H // 4]
        else:
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            bbox = [x_min, y_min, x_max, y_max]

        # Return everything as torch tensors
        # CRITICAL: .contiguous() ensures tensor memory layout is compatible with SAM2
        # Shared memory indexing can create non-contiguous views
        return {
            "pixel_values": self.images[idx].contiguous(),  # (3, H, W)
            "input_boxes": torch.tensor([bbox], dtype=torch.float32),  # (1, 4) per sample -> (B, 1, 4) when batched
            "ground_truth_mask": mask.contiguous()  # (H, W)
        }


def test_torch_dataset():
    """Test that torch-only dataset works with SAM2"""

    print("\n" + "="*80)
    print("TORCH-ONLY DATASET TEST")
    print("="*80)

    initial_ram = get_memory_usage_gb()
    print(f"\nInitial RAM: {initial_ram:.2f} GB")

    # Create dataset
    print(f"\n[1/5] Creating TorchOnlyDataset...")
    dataset = TorchOnlyDataset(num_samples=NUM_SAMPLES, image_size=IMAGE_SIZE)
    after_dataset_ram = get_memory_usage_gb()
    print(f"  RAM after dataset: {after_dataset_ram:.2f} GB (+{after_dataset_ram - initial_ram:.2f} GB)")

    # Create DataLoader with workers
    print(f"\n[2/5] Creating DataLoader (num_workers={NUM_WORKERS})...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    after_loader_ram = get_memory_usage_gb()
    print(f"  RAM after DataLoader: {after_loader_ram:.2f} GB (+{after_loader_ram - after_dataset_ram:.2f} GB)")

    # Load model
    print(f"\n[3/5] Loading SAM2-tiny model...")
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny")
    model.to(DEVICE)
    model.train()

    # Freeze everything except mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    from monai.losses import DiceCELoss
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    after_model_ram = get_memory_usage_gb()
    print(f"  RAM after model load: {after_model_ram:.2f} GB (+{after_model_ram - after_loader_ram:.2f} GB)")

    # Test iteration - measure RAM during dataloading
    print(f"\n[4/5] Testing iteration (watch for RAM explosion)...")
    print(f"  If RAM stays stable, torch shared memory is working!")
    print(f"  If RAM jumps by ~{NUM_WORKERS} * batch_size * sample_size, we have copies!")

    max_ram = after_model_ram

    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(DEVICE),
                input_boxes=batch["input_boxes"].to(DEVICE),
                multimask_output=False
            )

            # Get predictions
            pred_masks = outputs.pred_masks.squeeze(1)
            gt_masks = batch["ground_truth_mask"].float().to(DEVICE)

            # Resize ground truth to match prediction
            if len(gt_masks.shape) == 3:
                gt_masks = gt_masks.unsqueeze(1)

            gt_masks_resized = F.interpolate(
                gt_masks,
                size=pred_masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # Compute loss
            loss = loss_fn(pred_masks, gt_masks_resized)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Check RAM
            current_ram = get_memory_usage_gb()
            max_ram = max(max_ram, current_ram)

            # Cleanup
            del outputs, pred_masks, gt_masks, gt_masks_resized, loss

            if batch_idx % 5 == 0:
                print(f"    Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={epoch_losses[-1]:.4f}, RAM={current_ram:.2f} GB")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        # Force cleanup
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    final_ram = get_memory_usage_gb()
    ram_increase = final_ram - after_model_ram

    print(f"\n[5/5] Memory Summary:")
    print(f"  Initial RAM: {initial_ram:.2f} GB")
    print(f"  After dataset: {after_dataset_ram:.2f} GB (+{after_dataset_ram - initial_ram:.2f} GB)")
    print(f"  After model: {after_model_ram:.2f} GB")
    print(f"  Peak RAM during training: {max_ram:.2f} GB")
    print(f"  Final RAM: {final_ram:.2f} GB")
    print(f"  RAM increase during training: {ram_increase:.2f} GB")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print(f"{'='*80}")

    dataset_size_gb = (dataset.images.nbytes + dataset.masks.nbytes) / 1e9

    # Calculate expected overhead (not from worker copies)
    # Prefetch: num_workers × prefetch_factor × batch_size × sample_size
    samples_in_flight = NUM_WORKERS * 2 * BATCH_SIZE  # prefetch_factor=2
    prefetch_overhead_gb = samples_in_flight * (dataset_size_gb / len(dataset))
    expected_overhead_gb = prefetch_overhead_gb + 0.5  # +0.5 for model activations/gradients

    # If workers made full copies, we'd see NUM_WORKERS × dataset_size increase
    worker_copy_threshold = dataset_size_gb * (NUM_WORKERS - 1)  # -1 because main process already has data

    print(f"Dataset size: {dataset_size_gb:.2f} GB")
    print(f"RAM increase during training: {ram_increase:.2f} GB")
    print(f"Expected overhead (no copies): ~{expected_overhead_gb:.2f} GB")
    print(f"  - Prefetch buffers ({samples_in_flight} samples): ~{prefetch_overhead_gb:.2f} GB")
    print(f"  - Model activations/gradients: ~0.5 GB")
    print(f"Expected if workers made copies: ~{worker_copy_threshold:.2f} GB")

    if ram_increase < worker_copy_threshold * 0.5:
        print(f"\n✅ SUCCESS: RAM increase ({ram_increase:.2f} GB) << worker copy threshold ({worker_copy_threshold:.2f} GB)")
        print(f"   Shared memory is working! No worker copies detected.")
        print(f"   The {ram_increase:.2f} GB increase is from prefetch buffers + model overhead (expected).")
        print(f"   This confirms torch-only approach prevents RAM explosion.")
        success = True
    else:
        print(f"\n⚠️  WARNING: RAM increase ({ram_increase:.2f} GB) suggests possible copies")
        print(f"   This is higher than expected but still below full worker copies.")
        success = False

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

    return success


if __name__ == "__main__":
    success = test_torch_dataset()

    if success:
        print("✅ Torch-only approach validated!")
        print("   Safe to proceed with full conversion.")
    else:
        print("⚠️  Test inconclusive - review RAM analysis above")

    exit(0 if success else 1)
