"""
SAM Dataset - PyTorch Dataset wrapper for SAM training

Wraps HuggingFace Dataset to provide batches for SAM training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class SAMDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for SAM training.

    Takes a HuggingFace Dataset (from Preprocessor) and a SAM processor,
    returns batches ready for training.

    Usage:
        >>> from transformers import Sam2Processor
        >>> processor = Sam2Processor.from_pretrained('facebook/sam2-hiera-large')
        >>> sam_dataset = SAMDataset(hf_dataset, processor)
        >>> dataloader = DataLoader(sam_dataset, batch_size=4)
    """

    def __init__(self, dataset, processor=None, bbox_perturbation=20):
        """
        Initialize SAM dataset.

        Args:
            dataset: HuggingFace Dataset with 'image' and 'label' fields
            processor: SAM2Processor from transformers (deprecated - normalization done offline)
            bbox_perturbation: Random bbox expansion in pixels (0 = no perturbation)
        """
        self.dataset = dataset
        self.processor = processor  # No longer used - kept for backward compatibility
        self.bbox_perturbation = bbox_perturbation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get training sample.

        Returns:
            dict with:
                - pixel_values: Processed image tensor
                - ground_truth_mask: Ground truth mask
                - input_boxes: Bounding box prompt
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

    def _get_bounding_box(self, mask):
        """
        Extract bounding box from mask with random perturbation.

        Args:
            mask: Binary mask tensor

        Returns:
            Bounding box [x_min, y_min, x_max, y_max]
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

    Uses PyTorch multiprocessing properly: each worker loads batches independently
    when needed. OS filesystem cache handles repeated access efficiently.

    Directory structure:
        data_dir/
        ├── batch_000.pt  (images, labels)
        ├── batch_001.pt
        ├── ...
        └── metadata.json

    Memory usage: Only active batches in worker memory (~2-3 batches per worker)
    No RAM budget needed - relies on OS cache + SSD speed.

    Args:
        data_dir: Path to directory containing batch_*.pt files
    """

    def __init__(self, data_dir):
        import json
        import logging
        from pathlib import Path

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
        self._worker_cache = {}
        self._worker_cache_max_size = 3  # Keep last 3 batches per worker

        logger.info(
            f"BatchedDataset: {self.num_samples} samples across {self.num_batches} batch files"
        )
        logger.info(
            "  Streaming mode: Workers load batches on-demand (OS cache handles efficiency)"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get sample by index. Loads batch file on-demand in worker process.

        This runs in the DataLoader worker, so disk I/O is parallelized
        across workers. Each worker maintains a small LRU cache.

        Returns:
            dict with 'image' and 'label' keys (compatible with SAMDataset)
        """
        batch_num = idx // self.samples_per_batch
        local_idx = idx % self.samples_per_batch

        # Load batch (with per-worker caching)
        batch = self._load_batch_cached(batch_num)

        return {
            "image": batch["images"][local_idx].contiguous(),
            "label": batch["labels"][local_idx].contiguous(),
        }

    def _load_batch_cached(self, batch_num):
        """
        Load batch with simple LRU caching per worker.

        Each worker maintains its own cache (3 batches), so with 12 workers
        we have at most 12 * 3 * 1.36 GB = ~49 GB total across all workers.
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

    def _load_batch_from_disk(self, batch_num):
        """Load single batch from disk."""
        batch_file = self.data_dir / f"batch_{batch_num:03d}.pt"
        data = torch.load(batch_file, weights_only=False)
        return {"images": data["images"], "labels": data["labels"]}

    def __repr__(self):
        return (
            f"BatchedDataset(samples={self.num_samples}, "
            f"batches={self.num_batches}, "
            f"samples_per_batch={self.samples_per_batch}, "
            f"streaming=True)"
        )
