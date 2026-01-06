"""
Torch-backed dataset with shared memory support for multiprocessing.

This module provides efficient PyTorch tensor-based datasets optimized for
multiprocessing with DataLoader workers. All tensors are stored in shared
memory to enable zero-copy access from worker processes.

Classes
-------
TorchDataset
    Pure torch tensor dataset with shared memory for efficient training.
BatchWriter
    Accumulates samples and writes batch files to disk for streaming datasets.

Examples
--------
Creating and saving a TorchDataset:

>>> images = torch.randn(100, 256, 256, 3, dtype=torch.float32)
>>> labels = torch.randint(0, 2, (100, 256, 256), dtype=torch.uint8)
>>> dataset = TorchDataset(images, labels)
>>> dataset.save_to_disk('dataset.pt')

Loading a saved dataset:

>>> dataset = TorchDataset.load_from_disk('dataset.pt')
>>> dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

Using BatchWriter for streaming:

>>> writer = BatchWriter('output_dir', samples_per_batch=100)
>>> for batch_dataset in generate_batches():
...     writer.add_batch(batch_dataset)
>>> writer.finalize()

Notes
-----
All tensors are stored in shared memory using `.share_memory_()` to enable
zero-copy access from DataLoader worker processes. This provides significant
performance benefits compared to pickling tensors across process boundaries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class TorchDataset:
    """
    Pure torch tensor dataset for efficient training with DataLoader workers.

    Stores images and labels as PyTorch tensors in shared memory, enabling
    zero-copy access from DataLoader worker processes. Compatible with
    SAMDataset interface for drop-in replacement.

    Parameters
    ----------
    images : torch.Tensor
        Image tensor of shape (N, H, W, 3) and dtype=float32.
    labels : torch.Tensor
        Label mask tensor of shape (N, H, W) and dtype=uint8.
    metadata : Optional[Dict[str, Any]], default=None
        Optional dictionary of metadata (preprocessing params, statistics, etc.).

    Attributes
    ----------
    images : torch.Tensor
        Shared memory tensor containing images.
    labels : torch.Tensor
        Shared memory tensor containing labels.
    metadata : Dict[str, Any]
        Metadata dictionary.

    Raises
    ------
    AssertionError
        If images and labels have different lengths or incorrect dtypes.

    Examples
    --------
    >>> images = torch.randn(100, 256, 256, 3, dtype=torch.float32)
    >>> labels = torch.randint(0, 2, (100, 256, 256), dtype=torch.uint8)
    >>> metadata = {'patch_size': 256, 'stretch': 'SQRT'}
    >>> dataset = TorchDataset(images, labels, metadata)
    >>> len(dataset)
    100

    Notes
    -----
    Tensors are automatically moved to shared memory using `.share_memory_()`,
    which enables zero-copy access from forked DataLoader worker processes.
    This avoids expensive tensor serialization/deserialization overhead.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize torch dataset with shared memory tensors.

        Parameters
        ----------
        images : torch.Tensor
            Image tensor of shape (N, H, W, 3) and dtype=float32.
        labels : torch.Tensor
            Label mask tensor of shape (N, H, W) and dtype=uint8.
        metadata : Optional[Dict[str, Any]], default=None
            Optional metadata dictionary.

        Raises
        ------
        AssertionError
            If images and labels have different lengths or incorrect dtypes.
        """
        assert len(images) == len(labels), "Images and labels must have same length"
        assert images.dtype == torch.float32, f"Images must be float32, got {images.dtype}"
        assert labels.dtype == torch.uint8, f"Labels must be uint8, got {labels.dtype}"

        # Store tensors in shared memory for zero-copy worker access
        self.images = images.share_memory_()
        self.labels = labels.share_memory_()
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """
        Get number of samples in dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.

        Returns sample in format compatible with SAMDataset expectations.
        Uses `.contiguous()` to ensure memory layout is compatible with SAM2
        (no copy if already contiguous).

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'image' : torch.Tensor of shape (H, W, 3)
                Image tensor.
            - 'label' : torch.Tensor of shape (H, W)
                Label mask tensor.

        Examples
        --------
        >>> dataset = TorchDataset(images, labels)
        >>> sample = dataset[0]
        >>> sample['image'].shape
        torch.Size([256, 256, 3])
        """
        return {"image": self.images[idx].contiguous(), "label": self.labels[idx].contiguous()}

    def save_to_disk(self, path: str) -> None:
        """
        Save dataset to .pt file.

        Parameters
        ----------
        path : str
            Path where .pt file will be saved. Parent directories are created
            if they don't exist.

        Examples
        --------
        >>> dataset = TorchDataset(images, labels)
        >>> dataset.save_to_disk('data/dataset.pt')
        Saved TorchDataset to data/dataset.pt
          100 samples, 0.75 GB

        Notes
        -----
        Saves tensors and metadata to a single .pt file using torch.save.
        Prints size information to stdout.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({"images": self.images, "labels": self.labels, "metadata": self.metadata}, path)

        size_gb = (
            self.images.element_size() * self.images.numel()
            + self.labels.element_size() * self.labels.numel()
        ) / 1e9
        print(f"Saved TorchDataset to {path}")
        print(f"  {len(self)} samples, {size_gb:.2f} GB")

    @classmethod
    def load_from_disk(cls, path: str) -> "TorchDataset":
        """
        Load dataset from .pt file.

        Parameters
        ----------
        path : str
            Path to .pt file created by save_to_disk.

        Returns
        -------
        TorchDataset
            Loaded dataset instance.

        Raises
        ------
        FileNotFoundError
            If file doesn't exist.

        Examples
        --------
        >>> dataset = TorchDataset.load_from_disk('data/dataset.pt')
        >>> len(dataset)
        100

        Notes
        -----
        Automatically moves loaded tensors to shared memory for efficient
        use with DataLoader workers.
        """
        data = torch.load(path)
        return cls(data["images"], data["labels"], data.get("metadata"))

    def __repr__(self) -> str:
        """
        String representation of dataset.

        Returns
        -------
        str
            Formatted string with dataset statistics.

        Examples
        --------
        >>> dataset = TorchDataset(images, labels)
        >>> print(dataset)
        TorchDataset(samples=100, image_shape=(256, 256, 3), size=0.75GB)
        """
        size_gb = (
            self.images.element_size() * self.images.numel()
            + self.labels.element_size() * self.labels.numel()
        ) / 1e9
        return (
            f"TorchDataset(samples={len(self)}, "
            f"image_shape={tuple(self.images.shape[1:])}, "
            f"size={size_gb:.2f}GB)"
        )


class BatchWriter:
    """
    Accumulates samples and writes batch files to disk.

    Efficient batch file writer that accumulates samples in memory and writes
    them as uncompressed .pt files for fast loading during training. Manages
    memory automatically by flushing when batch size is reached.

    Parameters
    ----------
    output_dir : str
        Directory where batch files will be written.
    samples_per_batch : int, default=100
        Number of samples to include in each batch file.

    Attributes
    ----------
    output_dir : Path
        Directory for batch files.
    samples_per_batch : int
        Target samples per batch file.
    accumulated_images : List[torch.Tensor]
        Buffer of accumulated image tensors.
    accumulated_labels : List[torch.Tensor]
        Buffer of accumulated label tensors.
    batch_file_idx : int
        Current batch file index.
    total_samples : int
        Total samples written so far.

    Examples
    --------
    >>> writer = BatchWriter('output_dir', samples_per_batch=100)
    >>> for batch_dataset in generate_batches():
    ...     writer.add_batch(batch_dataset)
    >>> writer.finalize()
    Wrote batch_000.pt: 100 patches (1.36 GB)
    Wrote batch_001.pt: 100 patches (1.36 GB)
    Total samples: 200

    Notes
    -----
    Call finalize() when done to flush remaining samples and write metadata.json.
    Batch files are written as uncompressed .pt files for maximum loading speed
    during training.
    """

    def __init__(self, output_dir: str, samples_per_batch: int = 100) -> None:
        """
        Initialize batch writer.

        Parameters
        ----------
        output_dir : str
            Directory to write batch files. Created if it doesn't exist.
        samples_per_batch : int, default=100
            Number of samples per batch file.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.samples_per_batch = samples_per_batch
        self.accumulated_images: List[torch.Tensor] = []
        self.accumulated_labels: List[torch.Tensor] = []
        self.batch_file_idx = 0
        self.total_samples = 0

    def add_batch(self, dataset: "TorchDataset") -> None:
        """
        Add samples from a TorchDataset batch.

        Accumulates samples in memory buffers. When accumulated samples reach
        samples_per_batch, automatically flushes to disk and clears memory.

        Parameters
        ----------
        dataset : TorchDataset
            TorchDataset instance with .images and .labels tensors.

        Examples
        --------
        >>> writer = BatchWriter('output_dir', samples_per_batch=100)
        >>> for i in range(5):
        ...     batch = generate_batch(20)  # Returns TorchDataset
        ...     writer.add_batch(batch)
        Wrote batch_000.pt: 100 patches (1.36 GB)

        Notes
        -----
        Automatically triggers _flush() when accumulated samples exceed
        samples_per_batch to manage memory efficiently.
        """
        self.accumulated_images.append(dataset.images)
        self.accumulated_labels.append(dataset.labels)

        # Check if we have enough to write a file
        current_size = sum(len(img) for img in self.accumulated_images)
        if current_size >= self.samples_per_batch:
            self._flush()

    def _flush(self) -> None:
        """
        Write ALL accumulated data to disk, clearing memory.

        Concatenates accumulated tensors, splits into batch files of
        samples_per_batch size, writes to disk, and clears accumulators
        to free memory.

        Notes
        -----
        This is an internal method called automatically by add_batch() or
        manually by finalize(). Clears accumulators immediately after
        concatenation to minimize peak memory usage.
        """
        if not self.accumulated_images:
            return

        # Concatenate all accumulated data
        images = torch.cat(self.accumulated_images)
        labels = torch.cat(self.accumulated_labels)

        # Clear accumulators immediately to free memory
        self.accumulated_images = []
        self.accumulated_labels = []

        # Write in chunks of samples_per_batch
        total_samples = len(images)
        for start_idx in range(0, total_samples, self.samples_per_batch):
            end_idx = min(start_idx + self.samples_per_batch, total_samples)

            images_chunk = images[start_idx:end_idx]
            labels_chunk = labels[start_idx:end_idx]

            batch_file = self.output_dir / f"batch_{self.batch_file_idx:03d}.pt"
            torch.save({"images": images_chunk, "labels": labels_chunk}, batch_file)

            size_gb = (
                images_chunk.element_size() * images_chunk.numel()
                + labels_chunk.element_size() * labels_chunk.numel()
            ) / 1e9
            print(f"    Wrote {batch_file.name}: {len(images_chunk)} patches ({size_gb:.2f} GB)")

            self.total_samples += len(images_chunk)
            self.batch_file_idx += 1

    def finalize(self) -> None:
        """
        Flush remaining samples and write metadata.

        Writes any remaining buffered samples to disk and creates metadata.json
        file with dataset statistics for use with BatchedDataset.

        Examples
        --------
        >>> writer = BatchWriter('output_dir', samples_per_batch=100)
        >>> for batch in batches:
        ...     writer.add_batch(batch)
        >>> writer.finalize()
        Batch writing complete:
          Total samples: 500
          Batch files: 5
          Metadata: output_dir/metadata.json

        Notes
        -----
        Always call this method when done writing batches to ensure all data
        is flushed and metadata.json is created. The metadata.json file is
        required by BatchedDataset for loading.
        """
        import json

        # Flush any remaining samples
        if self.accumulated_images:
            self._flush()

        # Write metadata
        metadata = {
            "num_samples": self.total_samples,
            "samples_per_batch": self.samples_per_batch,
            "num_batches": self.batch_file_idx,
            "image_shape": [1024, 1024, 3],
            "mask_shape": [1024, 1024],
            "dtype": "float32",
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\nBatch writing complete:")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Batch files: {self.batch_file_idx}")
        print(f"  Metadata: {metadata_path}")
