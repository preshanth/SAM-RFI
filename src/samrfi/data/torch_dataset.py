"""
Torch-backed dataset with shared memory support for multiprocessing
"""

from pathlib import Path

import torch


class TorchDataset:
    """
    Pure torch tensor dataset for efficient training with DataLoader workers.

    All tensors are stored in shared memory to enable zero-copy access
    from DataLoader worker processes.

    Compatible with SAMDataset - provides same interface as HF Dataset.

    Args:
        images: torch.Tensor of shape (N, H, W, 3) dtype=float32
        labels: torch.Tensor of shape (N, H, W) dtype=uint8
        metadata: optional dict of metadata (params, stats, etc.)
    """

    def __init__(self, images, labels, metadata=None):
        assert len(images) == len(labels), "Images and labels must have same length"
        assert images.dtype == torch.float32, f"Images must be float32, got {images.dtype}"
        assert labels.dtype == torch.uint8, f"Labels must be uint8, got {labels.dtype}"

        # Store tensors in shared memory for zero-copy worker access
        self.images = images.share_memory_()
        self.labels = labels.share_memory_()
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns dict compatible with SAMDataset expectations.

        .contiguous() ensures memory layout is compatible with SAM2
        (no copy if already contiguous).
        """
        return {"image": self.images[idx].contiguous(), "label": self.labels[idx].contiguous()}

    def save_to_disk(self, path):
        """Save to .pt file"""
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
    def load_from_disk(cls, path):
        """Load from .pt file"""
        data = torch.load(path)
        return cls(data["images"], data["labels"], data.get("metadata"))

    def __repr__(self):
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

    Writes uncompressed .pt files for fast loading during training.

    Usage:
        writer = BatchWriter(output_dir, samples_per_batch=100)
        for batch_dataset in generate_batches():
            writer.add_batch(batch_dataset)
        writer.finalize()  # Flush remaining + write metadata
    """

    def __init__(self, output_dir, samples_per_batch=100):
        """
        Initialize batch writer.

        Args:
            output_dir: Directory to write batch files
            samples_per_batch: Number of samples per batch file
        """
        from pathlib import Path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.samples_per_batch = samples_per_batch
        self.accumulated_images = []
        self.accumulated_labels = []
        self.batch_file_idx = 0
        self.total_samples = 0

    def add_batch(self, dataset):
        """
        Add samples from a TorchDataset batch.

        Args:
            dataset: TorchDataset instance with .images and .labels
        """
        self.accumulated_images.append(dataset.images)
        self.accumulated_labels.append(dataset.labels)

        # Check if we have enough to write a file
        current_size = sum(len(img) for img in self.accumulated_images)
        if current_size >= self.samples_per_batch:
            self._flush()

    def _flush(self):
        """Write ALL accumulated data to disk, clearing memory."""
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

    def finalize(self):
        """Flush remaining samples and write metadata."""
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
