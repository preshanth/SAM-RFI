"""
SAM Dataset - PyTorch Dataset wrapper for SAM training

Wraps HuggingFace Dataset to provide batches for SAM training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.multiprocessing import Manager


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
            "ground_truth_mask": ground_truth_mask
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
            # Empty mask - return center box (as Python int)
            H, W = mask.shape
            return [int(W // 4), int(H // 4), int(3 * W // 4), int(3 * H // 4)]

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
    Loads data from multiple batch files with shared RAM preloading.

    Compatible with SAMDataset wrapper - provides same __getitem__ interface.

    Directory structure:
        data_dir/
        ├── batch_000.pt  (images, labels)
        ├── batch_001.pt
        ├── ...
        └── metadata.json

    Args:
        data_dir: Path to directory containing batch_*.pt files
        ram_budget_gb: RAM budget in GB for preloading batches (default: None = old LRU behavior)
        cache_size: [Deprecated] Number of batch files for LRU cache (only if ram_budget_gb=None)
    """

    def __init__(self, data_dir, ram_budget_gb=None, cache_size=3):
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

        self.num_samples = self.metadata['num_samples']
        self.samples_per_batch = self.metadata['samples_per_batch']
        self.num_batches = self.metadata['num_batches']

        # Determine batch file size by probing first file
        probe_file = self.data_dir / "batch_000.pt"
        probe_data = torch.load(probe_file)
        batch_size_bytes = (probe_data['images'].element_size() * probe_data['images'].numel() +
                            probe_data['labels'].element_size() * probe_data['labels'].numel())
        self.batch_size_gb = batch_size_bytes / 1e9
        del probe_data  # Free memory

        # Choose caching strategy
        if ram_budget_gb is not None:
            # New: Shared preload cache
            self._use_preload = True

            # Special case: ram_budget_gb = -1 means "load all data"
            if ram_budget_gb < 0:
                self.batches_to_cache = self.num_batches
                total_size_gb = self.num_batches * self.batch_size_gb
                logger.info(f"Loading ALL data: {self.num_batches} batches ({total_size_gb:.1f} GB)")
            else:
                self.batches_to_cache = int(ram_budget_gb / self.batch_size_gb)
                self.batches_to_cache = min(self.batches_to_cache, self.num_batches)
                logger.info(f"Preloading {self.batches_to_cache} batches ({self.batches_to_cache * self.batch_size_gb:.1f} GB) into RAM")

            self._preload_cache(0, self.batches_to_cache)
            logger.info(f"Cache loaded: {self.batches_to_cache}/{self.num_batches} batches ({100*self.batches_to_cache/self.num_batches:.1f}%)")
        else:
            # Old: Per-worker LRU cache (deprecated)
            from functools import lru_cache
            self._use_preload = False
            self._cache_size = cache_size
            self._load_batch = lru_cache(maxsize=cache_size)(self._load_batch_uncached)
            logger.warning(f"Using deprecated LRU cache (cache_size={cache_size}). Consider using ram_budget_gb instead.")

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

        if self._use_preload:
            # Check if batch is in preloaded cache
            cache_offset = batch_num - self._cache_start_batch
            if 0 <= cache_offset < len(self._cached_batches):
                # Hit: return from preloaded cache (shared memory tensors)
                batch = self._cached_batches[cache_offset]
                # Return torch tensors directly (no numpy conversion!)
                return {
                    'image': batch['images'][local_idx].contiguous(),
                    'label': batch['labels'][local_idx].contiguous()
                }
            else:
                # Miss: load from disk (shouldn't happen often with proper config)
                batch = self._load_batch_from_disk(batch_num)
                return {
                    'image': batch['images'][local_idx].contiguous(),
                    'label': batch['labels'][local_idx].contiguous()
                }
        else:
            # Old LRU cache behavior
            batch = self._load_batch(batch_num)
            return {
                'image': batch['images'][local_idx],
                'label': batch['labels'][local_idx]
            }

    def _preload_cache(self, start_batch, num_batches):
        """Preload multiple batch files into RAM using shared memory"""
        import logging
        logger = logging.getLogger(__name__)

        self._cache_start_batch = start_batch

        # Use shared memory for arrays that workers can access
        # Convert numpy arrays to torch tensors in shared memory
        self._cached_batches = []

        for i in range(num_batches):
            batch_num = start_batch + i
            if batch_num >= self.num_batches:
                break

            batch_file = self.data_dir / f"batch_{batch_num:03d}.pt"
            data = torch.load(batch_file)

            # Put tensors in shared memory (accessible by all workers)
            images_tensor = data['images'].share_memory_()
            labels_tensor = data['labels'].share_memory_()

            self._cached_batches.append({
                'images': images_tensor,
                'labels': labels_tensor
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Loaded {i+1}/{num_batches} batches...")

    def _load_batch_from_disk(self, batch_num):
        """Load single batch from disk (fallback for cache misses)"""
        batch_file = self.data_dir / f"batch_{batch_num:03d}.pt"
        data = torch.load(batch_file)
        return {
            'images': data['images'],
            'labels': data['labels']
        }

    def _load_batch_uncached(self, batch_num):
        """Load batch file from disk (wrapped by LRU cache) - deprecated path"""
        return self._load_batch_from_disk(batch_num)

    def __repr__(self):
        if self._use_preload:
            return (f"BatchedDataset(samples={self.num_samples}, "
                    f"batches={self.num_batches}, "
                    f"cached={self.batches_to_cache}, "
                    f"ram_budget={self.batches_to_cache * self.batch_size_gb:.1f}GB)")
        else:
            return (f"BatchedDataset(samples={self.num_samples}, "
                    f"batches={self.num_batches}, "
                    f"samples_per_batch={self.samples_per_batch}, "
                    f"cache_size={self._cache_size})")
