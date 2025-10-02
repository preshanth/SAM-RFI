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
