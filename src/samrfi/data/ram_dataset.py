"""
RAM-Cached Dataset with GPU Transforms

Loads raw complex patches into RAM once, then applies GPU transforms on-the-fly during training.
Eliminates disk I/O bottleneck while keeping GPU busy.
"""

import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import json
import logging

from .gpu_transforms import GPUTransforms

logger = logging.getLogger(__name__)


class RAMCachedDataset(TorchDataset):
    """
    Dataset that loads raw complex patches into RAM (shared memory) and applies
    GPU transforms on-the-fly.

    Benefits:
    - Zero disk I/O during training (all data in RAM)
    - Shared memory → zero-copy worker access (like old TorchDataset)
    - GPU transforms → keep H100 busy, do augmentation on-the-fly
    - 75% less storage (no pre-saved augmentations)

    Args:
        data_dir: Path to directory with batch_*.pt files and metadata.json
        device: GPU device for transforms ('cuda', 'cuda:0', etc.)
        enable_augmentation: Enable 4-way physics-preserving augmentation (default: True)
        bbox_perturbation: Random bbox expansion in pixels (default: 20)
    """

    def __init__(
        self,
        data_dir,
        device='cuda',
        enable_augmentation=True,
        bbox_perturbation=20,
    ):
        self.data_dir = Path(data_dir)
        self.device = device
        self.enable_augmentation = enable_augmentation
        self.bbox_perturbation = bbox_perturbation

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Verify format
        if self.metadata.get('format') != 'raw':
            raise ValueError(
                f"Expected raw format, got {self.metadata.get('format')}. "
                f"Generate dataset with save_raw=true in config."
            )

        num_batches = self.metadata['num_batches']
        logger.info(f"Loading {self.metadata['num_samples']} raw samples from {num_batches} batch files into RAM...")

        # Load ALL batches into RAM
        complex_patches_list = []
        masks_list = []

        for batch_idx in range(num_batches):
            batch_file = self.data_dir / f"batch_{batch_idx:03d}.pt"

            # Load batch
            batch = torch.load(batch_file, weights_only=False)

            # Batch contains 'images' (raw complex) and 'labels' (masks)
            complex_patches_list.append(batch['images'])
            masks_list.append(batch['labels'])

            if (batch_idx + 1) % 5 == 0:
                logger.info(f"  Loaded {batch_idx + 1}/{num_batches} batches")

        # Concatenate all batches
        self.complex_patches = torch.cat(complex_patches_list, dim=0)
        self.masks = torch.cat(masks_list, dim=0)

        # Free batch lists
        del complex_patches_list, masks_list

        # Move to shared memory for zero-copy worker access
        self.complex_patches.share_memory_()
        self.masks.share_memory_()

        # Calculate memory usage
        mem_gb = (self.complex_patches.element_size() * self.complex_patches.numel() +
                  self.masks.element_size() * self.masks.numel()) / 1e9

        logger.info(f"✓ Loaded {len(self.complex_patches)} raw samples into RAM ({mem_gb:.2f} GB, shared memory)")
        logger.info(f"  Complex patches shape: {self.complex_patches.shape}")
        logger.info(f"  Masks shape: {self.masks.shape}")

        # Initialize GPU transforms
        self.gpu_transforms = GPUTransforms(
            device=device,
            enable_augmentation=enable_augmentation
        )

    def __len__(self):
        """
        Return total number of samples.

        With augmentation enabled (default), each raw sample has 4 variations:
        - Original
        - Vertical flip
        - Transpose
        - Transpose + vertical flip
        """
        if self.enable_augmentation:
            return len(self.complex_patches) * 4
        else:
            return len(self.complex_patches)

    def __getitem__(self, idx):
        """
        Get training sample with GPU transforms applied on-the-fly.

        Flow:
        1. Get raw complex patch from RAM (shared memory, zero-copy)
        2. Determine augmentation index (0-3)
        3. Transfer complex patch to GPU
        4. Apply GPU transforms (complex→RGB + augmentation)
        5. Return SAM2-format dict

        Returns:
            dict with:
                - pixel_values: Transformed image (3, H, W) on GPU
                - ground_truth_mask: Mask (H, W) on GPU
                - input_boxes: Bounding box (1, 4) on GPU
        """
        # Determine base patch index and augmentation index
        if self.enable_augmentation:
            base_idx = idx // 4  # Which raw patch
            aug_idx = idx % 4    # Which augmentation (0-3)
        else:
            base_idx = idx
            aug_idx = 0  # No augmentation

        # Get raw complex patch and mask from RAM (shared memory - no copy!)
        complex_patch = self.complex_patches[base_idx]  # (H, W) complex
        mask = self.masks[base_idx]  # (H, W) uint8

        # Transfer to GPU (this is fast with shared memory)
        complex_patch_gpu = complex_patch.to(self.device, non_blocking=True)
        mask_gpu = mask.float().to(self.device, non_blocking=True)

        # Apply full GPU transform pipeline
        pixel_values, transformed_mask = self.gpu_transforms.full_transform_pipeline(
            complex_patch=complex_patch_gpu,
            mask=mask_gpu,
            augmentation_index=aug_idx,  # Apply specific augmentation (0-3)
            stretch_type=None,  # No stretching for raw data
            normalize_before_stretch=False,
            normalize_after_stretch=False,
        )

        # Compute bounding box from transformed mask (on GPU)
        input_boxes = self._get_bounding_box_gpu(transformed_mask)

        return {
            "pixel_values": pixel_values,          # (3, H, W) on GPU
            "input_boxes": input_boxes,            # (1, 4) on GPU
            "ground_truth_mask": transformed_mask  # (H, W) on GPU
        }

    def _get_bounding_box_gpu(self, mask):
        """
        Extract bounding box from mask on GPU with random perturbation.

        Args:
            mask: Binary mask tensor (H, W) on GPU

        Returns:
            Bounding box tensor (1, 4) in [x_min, y_min, x_max, y_max] format
        """
        # Find non-zero indices (GPU operation)
        nonzero_indices = torch.nonzero(mask, as_tuple=False)

        if nonzero_indices.numel() == 0:
            # No RFI detected - return full image bbox
            H, W = mask.shape
            return torch.tensor([[0, 0, W, H]], dtype=torch.float32, device=self.device)

        # Get bounding box coordinates (GPU operations)
        y_indices = nonzero_indices[:, 0]
        x_indices = nonzero_indices[:, 1]

        y_min = y_indices.min().item()
        y_max = y_indices.max().item()
        x_min = x_indices.min().item()
        x_max = x_indices.max().item()

        # Apply random perturbation if enabled
        if self.bbox_perturbation > 0:
            H, W = mask.shape

            # Random perturbation (on CPU for simplicity)
            perturb = torch.randint(
                -self.bbox_perturbation,
                self.bbox_perturbation + 1,
                (4,),
                device='cpu'
            )

            x_min = max(0, x_min + perturb[0].item())
            y_min = max(0, y_min + perturb[1].item())
            x_max = min(W, x_max + perturb[2].item())
            y_max = min(H, y_max + perturb[3].item())

        # Return as tensor (SAM2 format: [x_min, y_min, x_max, y_max])
        bbox = torch.tensor(
            [[x_min, y_min, x_max, y_max]],
            dtype=torch.float32,
            device=self.device
        )

        return bbox

    def __repr__(self):
        mem_gb = (self.complex_patches.element_size() * self.complex_patches.numel() +
                  self.masks.element_size() * self.masks.numel()) / 1e9
        return (f"RAMCachedDataset(raw_samples={len(self.complex_patches)}, "
                f"total_samples={len(self)}, "
                f"augmentation={self.enable_augmentation}, "
                f"memory={mem_gb:.2f}GB, "
                f"device={self.device})")
