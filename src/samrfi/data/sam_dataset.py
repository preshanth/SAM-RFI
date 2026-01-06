"""
SAM Dataset - PyTorch Dataset wrapper for SAM training.

This module provides PyTorch Dataset wrappers for SAM model training,
including standard dataset loading and batched streaming datasets for
efficient large-scale training.

Classes
-------
SAMDataset
    PyTorch Dataset wrapper for HuggingFace datasets with SAM preprocessing.
BatchedDataset
    Streaming dataset that loads batch files on-demand for memory efficiency.

Examples
--------
Standard dataset usage:

>>> from transformers import Sam2Processor
>>> processor = Sam2Processor.from_pretrained('facebook/sam2-hiera-large')
>>> sam_dataset = SAMDataset(hf_dataset, processor, bbox_perturbation=20)
>>> dataloader = DataLoader(sam_dataset, batch_size=4)

Batched dataset for large-scale training:

>>> batched_dataset = BatchedDataset('path/to/batch_dir')
>>> dataloader = DataLoader(batched_dataset, batch_size=4, num_workers=12)

Notes
-----
The SAMDataset assumes input images are already normalized with ImageNet
statistics during preprocessing. The processor parameter is maintained for
backward compatibility but is no longer used for normalization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class SAMDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for SAM training.

    Wraps a HuggingFace Dataset with preprocessed images and masks, providing
    batches ready for SAM model training. Handles bounding box extraction from
    masks with optional random perturbation for data augmentation.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace Dataset with 'image' and 'label' fields. Images should be
        pre-normalized tensors of shape (H, W, 3).
    processor : Optional[Any], default=None
        SAM2Processor from transformers. Deprecated - kept for backward
        compatibility. Normalization is now done during preprocessing.
    bbox_perturbation : int, default=20
        Random bounding box expansion in pixels for data augmentation.
        Set to 0 to disable perturbation.

    Attributes
    ----------
    dataset : Dataset
        Reference to the underlying HuggingFace Dataset.
    processor : Optional[Any]
        Deprecated SAM2Processor (no longer used).
    bbox_perturbation : int
        Maximum random pixel expansion for bounding boxes.

    Examples
    --------
    >>> from transformers import Sam2Processor
    >>> from torch.utils.data import DataLoader
    >>> processor = Sam2Processor.from_pretrained('facebook/sam2-hiera-large')
    >>> sam_dataset = SAMDataset(hf_dataset, processor, bbox_perturbation=20)
    >>> dataloader = DataLoader(sam_dataset, batch_size=4, shuffle=True)

    Notes
    -----
    Images are expected to be already normalized with ImageNet statistics
    (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) during the
    preprocessing stage. The processor parameter is maintained only for
    backward compatibility.
    """

    def __init__(
        self,
        dataset: Any,
        processor: Optional[Any] = None,
        bbox_perturbation: int = 20,
    ) -> None:
        """
        Initialize SAM dataset.

        Parameters
        ----------
        dataset : Any
            HuggingFace Dataset with 'image' and 'label' fields.
        processor : Optional[Any], default=None
            SAM2Processor (deprecated, no longer used).
        bbox_perturbation : int, default=20
            Random bbox expansion in pixels (0 = no perturbation).
        """
        self.dataset = dataset
        self.processor = processor  # No longer used - kept for backward compatibility
        self.bbox_perturbation = bbox_perturbation

    def __len__(self) -> int:
        """
        Get number of samples in dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample by index.

        Retrieves a preprocessed image and mask from the dataset, extracts
        a bounding box with optional perturbation, and returns tensors in
        the format expected by SAM2 models.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'pixel_values' : torch.Tensor of shape (3, H, W)
                Normalized image tensor in channels-first format.
            - 'input_boxes' : torch.Tensor of shape (1, 4)
                Bounding box prompt [x_min, y_min, x_max, y_max].
            - 'ground_truth_mask' : torch.Tensor of shape (H, W)
                Binary ground truth mask.

        Examples
        --------
        >>> dataset = SAMDataset(hf_dataset)
        >>> sample = dataset[0]
        >>> sample['pixel_values'].shape
        torch.Size([3, 256, 256])
        >>> sample['input_boxes'].shape
        torch.Size([1, 4])
        """
        item = self.dataset[idx]
        image = item["image"]  # Already normalized with ImageNet stats during generation
        ground_truth_mask = item["label"]  # Use view directly, no copy

        # Get bounding box from mask
        bbox = self._get_bounding_box(ground_truth_mask)

        # Convert to tensors (skip processor - normalization already done offline)
        # Image is already (H, W, 3) normalized, need (3, H, W) for PyTorch
        pixel_values = image.permute(2, 0, 1).contiguous()  # (H,W,3) -> (3,H,W)

        # Process bounding boxes to SAM2 format
        input_boxes = torch.tensor([bbox], dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "input_boxes": input_boxes,
            "ground_truth_mask": ground_truth_mask,
        }

    def _get_bounding_box(self, mask: torch.Tensor) -> List[int]:
        """
        Extract bounding box from mask with random perturbation.

        Finds the minimal bounding box containing all positive pixels in the mask,
        then optionally applies random perturbation for data augmentation.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask tensor of shape (H, W).

        Returns
        -------
        List[int]
            Bounding box coordinates [x_min, y_min, x_max, y_max].

        Notes
        -----
        For empty masks (all zeros), returns full image bounding box [0, 0, W, H].
        This handles inference mode where masks may be initialized as empty.

        Perturbation is applied independently to each edge of the bounding box,
        with values clipped to image boundaries.

        Examples
        --------
        >>> mask = torch.zeros(256, 256)
        >>> mask[50:150, 100:200] = 1
        >>> dataset = SAMDataset(None, bbox_perturbation=10)
        >>> bbox = dataset._get_bounding_box(mask)
        >>> # bbox will be approximately [90, 40, 210, 160] with random perturbation
        """
        # Find mask extent
        y_indices, x_indices = torch.where(mask > 0)

        if len(x_indices) == 0 or len(y_indices) == 0:
            # Empty mask (inference mode) - use full image bbox
            H, W = mask.shape
            return [0, 0, int(W), int(H)]

        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()

        # Add random perturbation (configurable)
        H, W = mask.shape
        if self.bbox_perturbation > 0:
            x_min = max(0, x_min - np.random.randint(0, self.bbox_perturbation))
            x_max = min(W, x_max + np.random.randint(0, self.bbox_perturbation))
            y_min = max(0, y_min - np.random.randint(0, self.bbox_perturbation))
            y_max = min(H, y_max + np.random.randint(0, self.bbox_perturbation))

        # Convert to native Python int (processor doesn't accept numpy.int64)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]


class BatchedDataset(TorchDataset):
    """
    Streaming dataset that loads batch files on-demand in worker processes.

    Efficient dataset for large-scale training that loads pre-batched .pt files
    on-demand in DataLoader worker processes. Uses OS filesystem cache for
    repeated access efficiency and maintains per-worker LRU cache.

    Parameters
    ----------
    data_dir : str or Path
        Path to directory containing batch_*.pt files and metadata.json.

    Attributes
    ----------
    data_dir : Path
        Directory containing batch files.
    metadata : Dict[str, Any]
        Metadata loaded from metadata.json.
    num_samples : int
        Total number of samples across all batches.
    samples_per_batch : int
        Number of samples in each batch file.
    num_batches : int
        Total number of batch files.
    _worker_cache : Dict[int, Dict[str, torch.Tensor]]
        Per-worker LRU cache for loaded batches.
    _worker_cache_max_size : int
        Maximum number of batches to cache per worker.

    Raises
    ------
    FileNotFoundError
        If metadata.json is not found in data_dir.

    Examples
    --------
    >>> dataset = BatchedDataset('path/to/batch_dir')
    >>> dataloader = DataLoader(dataset, batch_size=4, num_workers=12)
    >>> for batch in dataloader:
    ...     images = batch['image']
    ...     labels = batch['label']

    Notes
    -----
    Directory structure expected:
        data_dir/
        ├── batch_000.pt  (images, labels)
        ├── batch_001.pt
        ├── ...
        └── metadata.json

    Memory usage: Only active batches in worker memory (~2-3 batches per worker).
    No RAM budget needed - relies on OS cache + SSD speed. Each worker loads
    batches independently for parallel I/O.
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initialize batched dataset from directory.

        Parameters
        ----------
        data_dir : str
            Path to directory containing batch_*.pt files.

        Raises
        ------
        FileNotFoundError
            If metadata.json is not found in data_dir.
        """
        import json
        import logging

        self.data_dir = Path(data_dir)
        logger = logging.getLogger(__name__)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.num_samples = self.metadata["num_samples"]
        self.samples_per_batch = self.metadata["samples_per_batch"]
        self.num_batches = self.metadata["num_batches"]

        # Per-worker batch cache (initialized in each worker process)
        # This is a class attribute that will be separate in each forked worker
        self._worker_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._worker_cache_max_size = 3  # Keep last 3 batches per worker

        logger.info(
            f"BatchedDataset: {self.num_samples} samples across {self.num_batches} batch files"
        )
        logger.info(
            "  Streaming mode: Workers load batches on-demand (OS cache handles efficiency)"
        )

    def __len__(self) -> int:
        """
        Get number of samples in dataset.

        Returns
        -------
        int
            Total number of samples across all batch files.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index with on-demand batch loading.

        Loads batch file on-demand in DataLoader worker process, enabling
        parallel disk I/O across workers. Each worker maintains a small
        LRU cache for efficiency.

        Parameters
        ----------
        idx : int
            Global sample index across all batches.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys:
            - 'image' : torch.Tensor of shape (H, W, 3)
                Pre-normalized image tensor.
            - 'label' : torch.Tensor of shape (H, W)
                Binary ground truth mask.

        Notes
        -----
        This method runs in DataLoader worker processes, enabling parallel
        disk I/O. The batch file is loaded on first access and cached for
        subsequent accesses to samples in the same batch.

        Examples
        --------
        >>> dataset = BatchedDataset('path/to/batches')
        >>> sample = dataset[0]
        >>> sample['image'].shape
        torch.Size([1024, 1024, 3])
        """
        batch_num = idx // self.samples_per_batch
        local_idx = idx % self.samples_per_batch

        # Load batch (with per-worker caching)
        batch = self._load_batch_cached(batch_num)

        return {
            "image": batch["images"][local_idx].contiguous(),
            "label": batch["labels"][local_idx].contiguous(),
        }

    def _load_batch_cached(self, batch_num: int) -> Dict[str, torch.Tensor]:
        """
        Load batch with simple LRU caching per worker.

        Maintains per-worker cache of recently accessed batches to minimize
        disk I/O when DataLoader samples are accessed in batch order.

        Parameters
        ----------
        batch_num : int
            Batch file index to load.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'images' and 'labels' tensors.

        Notes
        -----
        Each worker maintains its own cache (3 batches), so with 12 workers
        we have at most 12 * 3 * batch_size memory usage across all workers.
        OS filesystem cache further improves efficiency for repeated access.
        """
        # Check cache
        if batch_num in self._worker_cache:
            return self._worker_cache[batch_num]

        # Load from disk (THIS RUNS IN WORKER PROCESS - parallel I/O!)
        batch = self._load_batch_from_disk(batch_num)

        # Simple LRU: if cache full, remove oldest
        if len(self._worker_cache) >= self._worker_cache_max_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self._worker_cache))
            del self._worker_cache[oldest_key]

        # Add to cache
        self._worker_cache[batch_num] = batch
        return batch

    def _load_batch_from_disk(self, batch_num: int) -> Dict[str, torch.Tensor]:
        """
        Load single batch from disk.

        Parameters
        ----------
        batch_num : int
            Batch file index to load.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'images' : torch.Tensor of shape (N, H, W, 3)
            - 'labels' : torch.Tensor of shape (N, H, W)

        Raises
        ------
        FileNotFoundError
            If batch file doesn't exist.
        """
        batch_file = self.data_dir / f"batch_{batch_num:03d}.pt"
        data = torch.load(batch_file, weights_only=False)
        return {"images": data["images"], "labels": data["labels"]}

    def __repr__(self) -> str:
        """
        String representation of dataset.

        Returns
        -------
        str
            Formatted string with dataset statistics.
        """
        return (
            f"BatchedDataset(samples={self.num_samples}, "
            f"batches={self.num_batches}, "
            f"samples_per_batch={self.samples_per_batch}, "
            f"streaming=True)"
        )
