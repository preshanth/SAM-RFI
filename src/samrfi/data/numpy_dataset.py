"""
Lightweight numpy-backed dataset (drop-in replacement for HuggingFace Dataset)
"""
import numpy as np
from pathlib import Path


class NumpyDataset:
    """
    Simple numpy array dataset for fast local training.

    Compatible with SAMDataset - provides same interface as HF Dataset.

    Args:
        images: numpy array of shape (N, H, W, 3) dtype=float32
        labels: numpy array of shape (N, H, W) dtype=uint8
        metadata: optional dict of metadata (params, stats, etc.)
    """

    def __init__(self, images, labels, metadata=None):
        assert len(images) == len(labels), "Images and labels must have same length"
        assert images.dtype == np.float32, f"Images must be float32, got {images.dtype}"
        assert labels.dtype == np.uint8, f"Labels must be uint8, got {labels.dtype}"

        self.images = images
        self.labels = labels
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Returns dict compatible with SAMDataset expectations"""
        return {
            "image": self.images[idx],
            "label": self.labels[idx]
        }

    def save_to_disk(self, path):
        """Save to compressed .npz file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            images=self.images,
            labels=self.labels,
            metadata=np.array([self.metadata], dtype=object)[0]  # Hack to save dict
        )
        print(f"Saved NumpyDataset to {path}")
        print(f"  {len(self)} samples, {self.images.nbytes / 1e9:.2f} GB")

    @classmethod
    def load_from_disk(cls, path):
        """Load from .npz file"""
        data = np.load(path, allow_pickle=True)
        metadata = data.get('metadata', None)
        if metadata is not None:
            metadata = metadata.item()  # Unpack from array wrapper
        return cls(data['images'], data['labels'], metadata)

    def __repr__(self):
        return (f"NumpyDataset(samples={len(self)}, "
                f"image_shape={self.images.shape[1:]}, "
                f"size={self.images.nbytes / 1e9:.2f}GB)")


class BatchWriter:
    """
    Accumulates samples and writes batch files to disk.

    Writes uncompressed .npz files for fast loading during training.

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
        import json
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
        Add samples from a NumpyDataset batch.

        Args:
            dataset: NumpyDataset instance with .images and .labels
        """
        self.accumulated_images.append(dataset.images)
        self.accumulated_labels.append(dataset.labels)

        # Check if we have enough to write a file
        current_size = sum(len(img) for img in self.accumulated_images)
        if current_size >= self.samples_per_batch:
            self._flush()

    def _flush(self):
        """Write accumulated data to batch_NNN.npz."""
        if not self.accumulated_images:
            return

        import numpy as np

        images = np.concatenate(self.accumulated_images)
        labels = np.concatenate(self.accumulated_labels)

        # Take exactly samples_per_batch (might have a few extra)
        num_to_write = min(len(images), self.samples_per_batch)
        images_to_write = images[:num_to_write]
        labels_to_write = labels[:num_to_write]

        # Write file (uncompressed for fast loading)
        batch_file = self.output_dir / f"batch_{self.batch_file_idx:03d}.npz"
        np.savez(batch_file, images=images_to_write, labels=labels_to_write)

        print(f"  Wrote {batch_file.name}: {num_to_write} samples ({images_to_write.nbytes / 1e9:.2f} GB)")

        # Track remainder if any
        remainder_images = images[num_to_write:]
        remainder_labels = labels[num_to_write:]

        if len(remainder_images) > 0:
            self.accumulated_images = [remainder_images]
            self.accumulated_labels = [remainder_labels]
        else:
            self.accumulated_images = []
            self.accumulated_labels = []

        self.total_samples += num_to_write
        self.batch_file_idx += 1

    def finalize(self):
        """Flush remaining samples and write metadata."""
        import json

        # Flush any remaining samples
        if self.accumulated_images:
            self._flush()

        # Write metadata
        metadata = {
            'num_samples': self.total_samples,
            'samples_per_batch': self.samples_per_batch,
            'num_batches': self.batch_file_idx,
            'image_shape': [1024, 1024, 3],
            'mask_shape': [1024, 1024],
            'dtype': 'float32'
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nBatch writing complete:")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Batch files: {self.batch_file_idx}")
        print(f"  Metadata: {metadata_path}")
