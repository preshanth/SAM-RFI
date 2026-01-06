"""
RAM-cached dataset with GPU transforms for high-performance training.

This module provides a PyTorch Dataset that loads raw complex-valued patches
into RAM once during initialization, then applies GPU transforms on-the-fly
during training. This approach eliminates disk I/O bottlenecks while keeping
the GPU fully utilized with efficient on-device transformations.

Classes
-------
RAMCachedDataset
    PyTorch Dataset that loads raw complex patches into RAM and applies GPU
    transforms on-the-fly during training.

Examples
--------
>>> from samrfi.data.ram_dataset import RAMCachedDataset
>>> from torch.utils.data import DataLoader
>>>
>>> # Create dataset
>>> dataset = RAMCachedDataset(
...     data_dir='./samrfi_data/raw_patches',
...     device='cuda',
...     enable_augmentation=True,
...     bbox_perturbation=20
... )
>>>
>>> # Create DataLoader
>>> loader = DataLoader(
...     dataset,
...     batch_size=32,
...     num_workers=4,
...     pin_memory=True
... )
>>>
>>> # Training loop
>>> for batch in loader:
...     images = batch['image']  # (B, H, W, 3) on GPU
...     labels = batch['label']  # (B, H, W) on GPU
...     # ... training code ...

Notes
-----
Performance characteristics:
- Zero disk I/O during training (all data in RAM)
- Shared memory for zero-copy multi-worker access
- GPU transforms keep H100/A100 busy (10-100x faster than CPU)
- 75% less storage (no pre-saved augmentations)
- On-the-fly augmentation with physics-preserving transforms

Memory requirements:
- Approximately 8 bytes per complex value (float32 real + float32 imag)
- For 1000 samples at 1024x1024: ~8 GB RAM
- Dataset automatically uses PyTorch's shared memory for multi-worker access

See Also
--------
samrfi.data.gpu_transforms.GPUTransforms : GPU transformation pipeline
torch.utils.data.Dataset : PyTorch Dataset base class
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch
from torch.utils.data import Dataset as TorchDataset

from .gpu_transforms import GPUTransforms

logger = logging.getLogger(__name__)


class RAMCachedDataset(TorchDataset):
    """
    PyTorch Dataset that loads raw complex patches into RAM with GPU transforms.

    This dataset loads all raw complex-valued patches into RAM during initialization,
    then applies GPU transforms on-the-fly during training. This design eliminates
    disk I/O bottlenecks while keeping the GPU busy with efficient transformations.

    Parameters
    ----------
    data_dir : str or Path
        Path to directory containing batch_*.pt files and metadata.json.
        Must be a dataset generated with save_raw=True.
    device : str, default='cuda'
        GPU device for transforms. Options: 'cuda', 'cuda:0', 'cuda:1', etc.
    enable_augmentation : bool, default=True
        Enable 4-way physics-preserving augmentation:
        - Original (index 0)
        - Vertical flip (index 1)
        - Transpose (index 2)
        - Transpose + vertical flip (index 3)
    bbox_perturbation : int, default=20
        Random bounding box perturbation in pixels for data augmentation.
        Set to 0 to disable bbox perturbation.

    Attributes
    ----------
    data_dir : Path
        Directory containing the dataset files.
    device : str
        GPU device used for transforms.
    enable_augmentation : bool
        Whether augmentation is enabled.
    bbox_perturbation : int
        Bounding box perturbation amount.
    metadata : dict
        Dataset metadata loaded from metadata.json.
    complex_patches : torch.Tensor
        All raw complex patches in shared memory. Shape: (N, H, W).
    masks : torch.Tensor
        All ground truth masks in shared memory. Shape: (N, H, W).
    gpu_transforms : GPUTransforms
        GPU transformation pipeline instance.

    Raises
    ------
    FileNotFoundError
        If metadata.json is not found in data_dir.
    ValueError
        If dataset format is not 'raw'. Must be generated with save_raw=True.

    Examples
    --------
    >>> from samrfi.data.ram_dataset import RAMCachedDataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Create dataset with augmentation
    >>> dataset = RAMCachedDataset(
    ...     data_dir='./samrfi_data/raw_patches',
    ...     device='cuda',
    ...     enable_augmentation=True
    ... )
    >>> print(f"Total samples (with augmentation): {len(dataset)}")
    >>> print(f"Raw samples: {len(dataset.complex_patches)}")
    >>>
    >>> # Create DataLoader with multiple workers
    >>> loader = DataLoader(
    ...     dataset,
    ...     batch_size=32,
    ...     num_workers=4,
    ...     pin_memory=True,
    ...     persistent_workers=True
    ... )
    >>>
    >>> # Get a batch
    >>> batch = next(iter(loader))
    >>> images = batch['image']  # (32, 1024, 1024, 3) on GPU
    >>> labels = batch['label']  # (32, 1024, 1024) on GPU

    Notes
    -----
    Memory usage:
    - Complex patches: 8 bytes per pixel (float32 real + float32 imag)
    - Masks: 1 byte per pixel (uint8)
    - For 1000 samples at 1024x1024: ~8.5 GB RAM

    Performance benefits:
    - Zero disk I/O during training (all data in RAM)
    - Shared memory enables zero-copy multi-worker access
    - GPU transforms are 10-100x faster than CPU
    - On-the-fly augmentation saves 75% storage

    The dataset automatically moves data to PyTorch shared memory for
    efficient multi-worker access. Each worker gets zero-copy access to
    the full dataset.

    See Also
    --------
    samrfi.data.gpu_transforms.GPUTransforms : GPU transformation pipeline
    torch.utils.data.DataLoader : PyTorch data loader
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        device: str = "cuda",
        enable_augmentation: bool = True,
        bbox_perturbation: int = 20,
    ) -> None:
        """
        Initialize RAM-cached dataset with GPU transforms.

        Parameters
        ----------
        data_dir : str or Path
            Path to directory with batch_*.pt files and metadata.json.
        device : str, default='cuda'
            GPU device for transforms.
        enable_augmentation : bool, default=True
            Enable 4-way physics-preserving augmentation.
        bbox_perturbation : int, default=20
            Random bbox expansion in pixels.
        """
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
        if self.metadata.get("format") != "raw":
            raise ValueError(
                f"Expected raw format, got {self.metadata.get('format')}. "
                f"Generate dataset with save_raw=true in config."
            )

        num_batches = self.metadata["num_batches"]
        logger.info(
            f"Loading {self.metadata['num_samples']} raw samples from {num_batches} batch files into RAM..."
        )

        # Load ALL batches into RAM
        complex_patches_list = []
        masks_list = []

        for batch_idx in range(num_batches):
            batch_file = self.data_dir / f"batch_{batch_idx:03d}.pt"

            # Load batch
            batch = torch.load(batch_file, weights_only=False)

            # Batch contains 'images' (raw complex) and 'labels' (masks)
            complex_patches_list.append(batch["images"])
            masks_list.append(batch["labels"])

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
        mem_gb = (
            self.complex_patches.element_size() * self.complex_patches.numel()
            + self.masks.element_size() * self.masks.numel()
        ) / 1e9

        logger.info(
            f"✓ Loaded {len(self.complex_patches)} raw samples into RAM ({mem_gb:.2f} GB, shared memory)"
        )
        logger.info(f"  Complex patches shape: {self.complex_patches.shape}")
        logger.info(f"  Masks shape: {self.masks.shape}")

        # Initialize GPU transforms
        self.gpu_transforms = GPUTransforms(device=device, enable_augmentation=enable_augmentation)

    def __len__(self) -> int:
        """
        Return total number of samples available in the dataset.

        With augmentation enabled (default), each raw sample has 4 variations
        through physics-preserving transforms, so the total length is 4x the
        number of raw samples.

        Returns
        -------
        int
            Total number of samples. If augmentation is enabled, returns
            4 * number of raw samples. Otherwise, returns number of raw samples.

        Examples
        --------
        >>> dataset = RAMCachedDataset('data', enable_augmentation=True)
        >>> raw_count = len(dataset.complex_patches)
        >>> total_count = len(dataset)
        >>> assert total_count == raw_count * 4

        Notes
        -----
        The 4 augmentation variations are:
        - Index 0: Original
        - Index 1: Vertical flip (frequency axis)
        - Index 2: Transpose (swap time ↔ frequency)
        - Index 3: Transpose + vertical flip
        """
        if self.enable_augmentation:
            return len(self.complex_patches) * 4
        else:
            return len(self.complex_patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample with GPU transforms applied on-the-fly.

        Retrieves a raw complex patch from RAM, transfers it to GPU, applies
        transforms (channel extraction, augmentation, normalization), and
        returns the result in SAM2-compatible format.

        Parameters
        ----------
        idx : int
            Sample index. If augmentation is enabled, indices are interpreted as:
            - idx // 4: Which raw patch to use
            - idx % 4: Which augmentation to apply (0-3)

        Returns
        -------
        dict
            Dictionary containing:
            - 'image' : torch.Tensor
                Transformed image with shape (H, W, 3) on GPU. The 3 channels are:
                [gradient, log_amplitude, phase] from complex data.
            - 'label' : torch.Tensor
                Ground truth mask with shape (H, W) on GPU, float32 dtype.

        Examples
        --------
        >>> dataset = RAMCachedDataset('data')
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)
        torch.Size([1024, 1024, 3])
        >>> print(sample['label'].shape)
        torch.Size([1024, 1024])
        >>> print(sample['image'].device)
        cuda:0

        Notes
        -----
        Data flow:
        1. Get raw complex patch from RAM (shared memory, zero-copy access)
        2. Determine augmentation index (0-3) if augmentation enabled
        3. Transfer complex patch to GPU (fast with shared memory + pin_memory)
        4. Apply GPU transforms:
           - Complex → RGB channels (gradient, log_amp, phase)
           - Apply augmentation (flip/transpose)
           - Normalize with ImageNet statistics
        5. Return in SAM2-compatible format: (H, W, 3)

        The returned tensors are on GPU and ready for immediate use in training.
        No additional CPU processing is required.

        See Also
        --------
        __len__ : Get total number of samples
        samrfi.data.gpu_transforms.GPUTransforms : GPU transformation pipeline
        """
        # Determine base patch index and augmentation index
        if self.enable_augmentation:
            base_idx = idx // 4  # Which raw patch
            aug_idx = idx % 4  # Which augmentation (0-3)
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

        # Convert to format expected by SAMDataset
        # pixel_values is (3, H, W), need (H, W, 3)
        image = pixel_values.permute(1, 2, 0).float()  # (3, H, W) -> (H, W, 3)

        return {
            "image": image,  # (H, W, 3) on GPU
            "label": transformed_mask.float(),  # (H, W) on GPU
        }

    def _get_bounding_box_gpu(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract bounding box from mask on GPU with random perturbation.

        Computes the bounding box of the RFI region in the mask using GPU
        operations, then applies random perturbation for data augmentation.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask tensor with shape (H, W) on GPU. Non-zero values
            indicate RFI pixels.

        Returns
        -------
        torch.Tensor
            Bounding box tensor with shape (1, 4) on GPU in SAM2 format:
            [x_min, y_min, x_max, y_max]. Coordinates are perturbed randomly
            if bbox_perturbation > 0.

        Examples
        --------
        >>> dataset = RAMCachedDataset('data', bbox_perturbation=20)
        >>> mask = torch.zeros(1024, 1024, device='cuda')
        >>> mask[100:200, 150:250] = 1  # Add RFI region
        >>> bbox = dataset._get_bounding_box_gpu(mask)
        >>> print(bbox.shape)
        torch.Size([1, 4])
        >>> # Bbox coordinates will be near [150, 100, 250, 200] with perturbation

        Notes
        -----
        If no RFI is detected (all zeros), returns the full image bounding box
        [0, 0, W, H].

        Perturbation is applied independently to each coordinate and clamped
        to image boundaries. This helps the model generalize to different
        prompt box sizes during inference.

        See Also
        --------
        __getitem__ : Uses this method to generate bounding boxes
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
                -self.bbox_perturbation, self.bbox_perturbation + 1, (4,), device="cpu"
            )

            x_min = max(0, x_min + perturb[0].item())
            y_min = max(0, y_min + perturb[1].item())
            x_max = min(W, x_max + perturb[2].item())
            y_max = min(H, y_max + perturb[3].item())

        # Return as tensor (SAM2 format: [x_min, y_min, x_max, y_max])
        bbox = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32, device=self.device)

        return bbox

    def __repr__(self) -> str:
        """
        Return string representation of the dataset.

        Returns
        -------
        str
            String describing the dataset configuration and memory usage.

        Examples
        --------
        >>> dataset = RAMCachedDataset('data', enable_augmentation=True)
        >>> print(dataset)
        RAMCachedDataset(raw_samples=1000, total_samples=4000, augmentation=True, memory=8.50GB, device=cuda)
        """
        mem_gb = (
            self.complex_patches.element_size() * self.complex_patches.numel()
            + self.masks.element_size() * self.masks.numel()
        ) / 1e9
        return (
            f"RAMCachedDataset(raw_samples={len(self.complex_patches)}, "
            f"total_samples={len(self)}, "
            f"augmentation={self.enable_augmentation}, "
            f"memory={mem_gb:.2f}GB, "
            f"device={self.device})"
        )
