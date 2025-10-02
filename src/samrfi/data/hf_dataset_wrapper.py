"""
Conversion utilities between NumpyDataset and HuggingFace Dataset
"""
import numpy as np
from PIL import Image
from datasets import Dataset
from .numpy_dataset import NumpyDataset


class HFDatasetWrapper:
    """Convert between NumpyDataset and HuggingFace Dataset formats"""

    @staticmethod
    def from_numpy(numpy_dataset, batch_size=50):
        """
        Convert NumpyDataset → HuggingFace Dataset for publishing.

        Args:
            numpy_dataset: NumpyDataset instance
            batch_size: Process in batches to avoid 2GB Arrow limit (default: 50)

        Returns:
            HuggingFace Dataset
        """
        print(f"Converting NumpyDataset to HF Dataset...")
        print(f"  Processing {len(numpy_dataset)} samples in batches of {batch_size}")

        # Convert in batches to avoid Arrow 2GB limit
        all_datasets = []

        for i in range(0, len(numpy_dataset), batch_size):
            end = min(i + batch_size, len(numpy_dataset))
            batch_images = numpy_dataset.images[i:end]
            batch_labels = numpy_dataset.labels[i:end]

            dataset_dict = {
                "image": list(batch_images),  # HF expects list of arrays
                "label": [Image.fromarray(mask) for mask in batch_labels]
            }

            batch_dataset = Dataset.from_dict(dataset_dict)
            all_datasets.append(batch_dataset)
            print(f"    Batch {i//batch_size + 1}/{(len(numpy_dataset) + batch_size - 1)//batch_size}")

        # Concatenate all batches
        from datasets import concatenate_datasets
        hf_dataset = concatenate_datasets(all_datasets)

        # Attach metadata if present
        if numpy_dataset.metadata:
            hf_dataset.info.description = str(numpy_dataset.metadata)

        print(f"  ✓ HF Dataset created: {len(hf_dataset)} samples")
        return hf_dataset

    @staticmethod
    def to_numpy(hf_dataset):
        """
        Convert HuggingFace Dataset → NumpyDataset.

        Useful for loading published datasets into fast numpy format.

        Args:
            hf_dataset: HuggingFace Dataset with 'image' and 'label' fields

        Returns:
            NumpyDataset instance
        """
        print(f"Converting HF Dataset to NumpyDataset...")

        images = []
        labels = []

        for item in hf_dataset:
            img = np.array(item['image'], dtype=np.float32)
            label = np.array(item['label'], dtype=np.uint8)
            images.append(img)
            labels.append(label)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        # Extract metadata from info if present
        metadata = {}
        if hf_dataset.info.description:
            try:
                metadata = eval(hf_dataset.info.description)
            except:
                pass

        numpy_dataset = NumpyDataset(images, labels, metadata)
        print(f"  ✓ {numpy_dataset}")
        return numpy_dataset
