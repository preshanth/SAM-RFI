"""
Conversion utilities between TorchDataset/BatchedDataset and HuggingFace Dataset
"""

import numpy as np
import torch
from datasets import Dataset
from PIL import Image

from .torch_dataset import TorchDataset


class HFDatasetWrapper:
    """Convert between TorchDataset/BatchedDataset and HuggingFace Dataset formats"""

    @staticmethod
    def from_dataset(dataset, batch_size=50):
        """
        Convert any dataset (TorchDataset or BatchedDataset) → HuggingFace Dataset.

        Args:
            dataset: TorchDataset or BatchedDataset instance
            batch_size: Process in batches to avoid 2GB Arrow limit (default: 50)

        Returns:
            HuggingFace Dataset
        """
        # Detect dataset type
        if hasattr(dataset, "batch_files"):
            # BatchedDataset - load from batch files
            return HFDatasetWrapper._from_batched_dataset(dataset, batch_size)
        elif hasattr(dataset, "images") and hasattr(dataset, "labels"):
            # TorchDataset - direct access to tensors
            return HFDatasetWrapper._from_torch_dataset(dataset, batch_size)
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    @staticmethod
    def _from_batched_dataset(dataset, batch_size=50):
        """Convert BatchedDataset → HuggingFace Dataset"""
        print("Converting BatchedDataset to HF Dataset...")
        print(f"  Loading {len(dataset)} samples from {len(dataset.batch_files)} batch files")

        all_datasets = []
        processed = 0

        # Load all batch files
        for batch_file in dataset.batch_files:
            batch_data = torch.load(batch_file)
            batch_images = batch_data["images"]
            batch_labels = batch_data["labels"]

            # Convert to numpy
            if torch.is_tensor(batch_images):
                batch_images = batch_images.cpu().numpy()
            if torch.is_tensor(batch_labels):
                batch_labels = batch_labels.cpu().numpy()

            # Convert in smaller chunks to avoid Arrow limit
            for i in range(0, len(batch_images), batch_size):
                end = min(i + batch_size, len(batch_images))
                chunk_images = batch_images[i:end]
                chunk_labels = batch_labels[i:end]

                dataset_dict = {
                    "image": list(chunk_images),
                    "label": [
                        Image.fromarray((mask * 255).astype(np.uint8)) for mask in chunk_labels
                    ],
                }

                chunk_dataset = Dataset.from_dict(dataset_dict)
                all_datasets.append(chunk_dataset)
                processed += len(chunk_images)
                print(f"    Progress: {processed}/{len(dataset)} samples")

        # Concatenate all chunks
        from datasets import concatenate_datasets

        hf_dataset = concatenate_datasets(all_datasets)

        # Attach metadata
        if hasattr(dataset, "metadata") and dataset.metadata:
            hf_dataset.info.description = str(dataset.metadata)

        print(f"  ✓ HF Dataset created: {len(hf_dataset)} samples")
        return hf_dataset

    @staticmethod
    def _from_torch_dataset(dataset, batch_size=50):
        """Convert TorchDataset → HuggingFace Dataset (legacy)"""
        return HFDatasetWrapper.from_numpy(dataset, batch_size)

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
        print("Converting dataset to HF Dataset...")
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
                "label": [Image.fromarray(mask) for mask in batch_labels],
            }

            batch_dataset = Dataset.from_dict(dataset_dict)
            all_datasets.append(batch_dataset)
            print(f"    Batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")

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
        print("Converting HF Dataset to TorchDataset...")

        images = []
        labels = []

        for item in hf_dataset:
            img = np.array(item["image"], dtype=np.float32)
            label = np.array(item["label"], dtype=np.uint8)
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
            except Exception:
                pass

        torch_dataset = TorchDataset(images_tensor, labels_tensor, metadata)
        print(f"  ✓ {torch_dataset}")
        return torch_dataset
