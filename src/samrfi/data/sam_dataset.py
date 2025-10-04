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
        ground_truth_mask = np.array(item["label"])

        # Get bounding box from mask
        bbox = self._get_bounding_box(ground_truth_mask)

        # Convert to tensors (skip processor - normalization already done offline)
        # Image is already (H, W, 3) normalized, need (3, H, W) for PyTorch
        pixel_values = torch.from_numpy(image).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)

        # Process bounding boxes to SAM2 format
        input_boxes = torch.tensor([[bbox]], dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "input_boxes": input_boxes,
            "ground_truth_mask": ground_truth_mask
        }

    def _get_bounding_box(self, mask):
        """
        Extract bounding box from mask with random perturbation.

        Args:
            mask: Binary mask array

        Returns:
            Bounding box [x_min, y_min, x_max, y_max]
        """
        # Find mask extent
        y_indices, x_indices = np.where(mask > 0)

        if len(x_indices) == 0 or len(y_indices) == 0:
            # Empty mask - return center box (as Python int)
            H, W = mask.shape
            return [int(W // 4), int(H // 4), int(3 * W // 4), int(3 * H // 4)]

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

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
    Loads data from multiple batch files with LRU caching.

    Compatible with SAMDataset wrapper - provides same __getitem__ interface.

    Directory structure:
        data_dir/
        ├── batch_000.npz  (images, labels)
        ├── batch_001.npz
        ├── ...
        └── metadata.json

    Args:
        data_dir: Path to directory containing batch_*.npz files
        cache_size: Number of batch files to keep in RAM (default: 3)
    """

    def __init__(self, data_dir, cache_size=3):
        import json
        from pathlib import Path
        from functools import lru_cache

        self.data_dir = Path(data_dir)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.num_samples = self.metadata['num_samples']
        self.samples_per_batch = self.metadata['samples_per_batch']
        self.num_batches = self.metadata['num_batches']

        # LRU cache for batch files
        self._cache_size = cache_size
        self._load_batch = lru_cache(maxsize=cache_size)(self._load_batch_uncached)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get sample by index.

        Returns:
            dict with 'image' and 'label' keys (compatible with SAMDataset)
        """
        batch_num = idx // self.samples_per_batch
        local_idx = idx % self.samples_per_batch

        # Load batch (from cache or disk)
        batch = self._load_batch(batch_num)

        return {
            'image': batch['images'][local_idx],
            'label': batch['labels'][local_idx]
        }

    def _load_batch_uncached(self, batch_num):
        """Load batch file from disk (wrapped by LRU cache)"""
        batch_file = self.data_dir / f"batch_{batch_num:03d}.npz"
        data = np.load(batch_file)
        return {
            'images': data['images'],
            'labels': data['labels']
        }

    def __repr__(self):
        return (f"BatchedDataset(samples={self.num_samples}, "
                f"batches={self.num_batches}, "
                f"samples_per_batch={self.samples_per_batch}, "
                f"cache_size={self._cache_size})")
