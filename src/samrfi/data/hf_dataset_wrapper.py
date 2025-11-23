"""
Conversion utilities between TorchDataset and HuggingFace Dataset
"""
import numpy as np
import torch
from PIL import Image
from datasets import Dataset
from .torch_dataset import TorchDataset


class HFDatasetWrapper:
    """Convert between TorchDataset and HuggingFace Dataset formats"""

    @staticmethod
    def from_numpy(dataset, batch_size=50):
        """
        Convert TorchDataset → HuggingFace Dataset for publishing.

        Args:
            dataset: TorchDataset instance
            batch_size: Process in batches to avoid 2GB Arrow limit (default: 50)

        Returns:
            HuggingFace Dataset
        """
        print(f"Converting dataset to HF Dataset...")
        print(f"  Processing {len(dataset)} samples in batches of {batch_size}")

        # Convert in batches to avoid Arrow 2GB limit
        all_datasets = []

        for i in range(0, len(dataset), batch_size):
            end = min(i + batch_size, len(dataset))
            batch_images = dataset.images[i:end]
            batch_labels = dataset.labels[i:end]

            # Convert torch tensors to numpy if needed
            if torch.is_tensor(batch_images):
                batch_images = batch_images.cpu().numpy()
            if torch.is_tensor(batch_labels):
                batch_labels = batch_labels.cpu().numpy()

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
        if dataset.metadata:
            hf_dataset.info.description = str(dataset.metadata)

        print(f"  ✓ HF Dataset created: {len(hf_dataset)} samples")
        return hf_dataset

    @staticmethod
    def to_numpy(hf_dataset):
        """
        Convert HuggingFace Dataset → TorchDataset.

        Useful for loading published datasets into fast torch format.

        Args:
            hf_dataset: HuggingFace Dataset with 'image' and 'label' fields

        Returns:
            TorchDataset instance
        """
        print(f"Converting HF Dataset to TorchDataset...")

        images = []
        labels = []

        for item in hf_dataset:
            img = np.array(item['image'], dtype=np.float32)
            label = np.array(item['label'], dtype=np.uint8)
            images.append(img)
            labels.append(label)

        images_np = np.array(images, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.uint8)

        # Convert to torch tensors
        images_tensor = torch.from_numpy(images_np)
        labels_tensor = torch.from_numpy(labels_np)

        # Extract metadata from info if present
        metadata = {}
        if hf_dataset.info.description:
            try:
                metadata = eval(hf_dataset.info.description)
            except:
                pass

        torch_dataset = TorchDataset(images_tensor, labels_tensor, metadata)
        print(f"  ✓ {torch_dataset}")
        return torch_dataset
