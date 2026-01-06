"""
Conversion utilities between TorchDataset/BatchedDataset and HuggingFace Dataset.

This module provides bidirectional conversion between SAM-RFI's internal dataset
formats (TorchDataset, BatchedDataset) and HuggingFace's Dataset format. This
enables publishing datasets to HuggingFace Hub and loading published datasets
for training.

Classes
-------
HFDatasetWrapper
    Converter between TorchDataset/BatchedDataset and HuggingFace Dataset formats.

Examples
--------
>>> from samrfi.data.hf_dataset_wrapper import HFDatasetWrapper
>>> from samrfi.data.torch_dataset import TorchDataset
>>> import torch
>>>
>>> # Create a TorchDataset
>>> images = torch.randn(100, 1024, 1024, 3)
>>> labels = torch.randint(0, 2, (100, 1024, 1024), dtype=torch.uint8)
>>> torch_dataset = TorchDataset(images, labels)
>>>
>>> # Convert to HuggingFace format
>>> hf_dataset = HFDatasetWrapper.from_dataset(torch_dataset)
>>>
>>> # Push to HuggingFace Hub
>>> hf_dataset.push_to_hub("username/dataset-name")
>>>
>>> # Load back to TorchDataset
>>> loaded_dataset = HFDatasetWrapper.to_numpy(hf_dataset)

Notes
-----
The conversion process handles the 2GB Arrow format limit by processing data
in batches. The default batch size of 50 works well for typical patch sizes
(128-1024 pixels).

HuggingFace Dataset format uses PIL Images for labels to ensure compatibility
with the HuggingFace ecosystem. Conversion to/from numpy arrays is handled
automatically.

See Also
--------
samrfi.data.torch_dataset.TorchDataset : In-memory PyTorch dataset
datasets.Dataset : HuggingFace Dataset format
"""

import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, Union, Optional

from datasets import Dataset

from .torch_dataset import TorchDataset


class HFDatasetWrapper:
    """
    Converter between TorchDataset/BatchedDataset and HuggingFace Dataset formats.

    This class provides static methods for bidirectional conversion between
    SAM-RFI's internal dataset formats and HuggingFace's Dataset format. It
    automatically detects the source dataset type and handles batched conversion
    to avoid Arrow format size limits.

    Methods
    -------
    from_dataset(dataset, batch_size=50)
        Convert TorchDataset or BatchedDataset to HuggingFace Dataset.
    from_numpy(dataset, batch_size=50)
        Convert TorchDataset to HuggingFace Dataset (legacy method).
    to_numpy(hf_dataset)
        Convert HuggingFace Dataset back to TorchDataset.

    Examples
    --------
    >>> from samrfi.data.hf_dataset_wrapper import HFDatasetWrapper
    >>> # Convert to HuggingFace format
    >>> hf_dataset = HFDatasetWrapper.from_dataset(my_torch_dataset)
    >>> # Convert back to TorchDataset
    >>> torch_dataset = HFDatasetWrapper.to_numpy(hf_dataset)

    Notes
    -----
    The wrapper processes data in batches to avoid the Arrow format's 2GB limit
    per chunk. For large datasets (>1000 samples with 1024x1024 patches), adjust
    batch_size accordingly.

    See Also
    --------
    samrfi.data.torch_dataset.TorchDataset : In-memory PyTorch dataset
    datasets.Dataset : HuggingFace Dataset class
    """

    @staticmethod
    def from_dataset(dataset: Any, batch_size: int = 50) -> Dataset:
        """
        Convert any dataset (TorchDataset or BatchedDataset) to HuggingFace Dataset.

        Automatically detects the dataset type and applies the appropriate
        conversion method. Processes data in batches to avoid Arrow format
        size limits.

        Parameters
        ----------
        dataset : TorchDataset or BatchedDataset
            Source dataset to convert. Must have either `images` and `labels`
            attributes (TorchDataset) or `batch_files` attribute (BatchedDataset).
        batch_size : int, default=50
            Number of samples to process per batch. Smaller values use less
            memory but may be slower.

        Returns
        -------
        datasets.Dataset
            HuggingFace Dataset with 'image' and 'label' fields.

        Raises
        ------
        TypeError
            If dataset type is not TorchDataset or BatchedDataset.

        Examples
        --------
        >>> from samrfi.data.hf_dataset_wrapper import HFDatasetWrapper
        >>> hf_dataset = HFDatasetWrapper.from_dataset(torch_dataset, batch_size=50)
        >>> len(hf_dataset)
        1000
        >>> hf_dataset[0].keys()
        dict_keys(['image', 'label'])

        Notes
        -----
        The conversion process:
        1. Detects dataset type (TorchDataset vs BatchedDataset)
        2. Loads data in batches to avoid memory limits
        3. Converts labels to PIL Images for HuggingFace compatibility
        4. Concatenates all batches into a single Dataset
        5. Preserves metadata in dataset.info.description

        See Also
        --------
        from_numpy : Legacy method for TorchDataset conversion
        to_numpy : Convert HuggingFace Dataset back to TorchDataset
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
    def _from_batched_dataset(dataset: Any, batch_size: int = 50) -> Dataset:
        """
        Convert BatchedDataset to HuggingFace Dataset.

        Internal method for converting BatchedDataset format (multiple .pt files)
        to HuggingFace Dataset format.

        Parameters
        ----------
        dataset : BatchedDataset
            Source dataset with batch_files attribute.
        batch_size : int, default=50
            Number of samples to process per chunk.

        Returns
        -------
        datasets.Dataset
            HuggingFace Dataset with all batches concatenated.

        Notes
        -----
        This method loads all batch files sequentially and concatenates them
        into a single HuggingFace Dataset. Progress is printed during loading.
        """
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
    def _from_torch_dataset(dataset: Any, batch_size: int = 50) -> Dataset:
        """
        Convert TorchDataset to HuggingFace Dataset (internal method).

        Parameters
        ----------
        dataset : TorchDataset
            Source dataset with images and labels tensors.
        batch_size : int, default=50
            Number of samples to process per chunk.

        Returns
        -------
        datasets.Dataset
            HuggingFace Dataset.

        Notes
        -----
        This is an internal method. Use from_dataset() instead.
        """
        return HFDatasetWrapper.from_numpy(dataset, batch_size)

    @staticmethod
    def from_numpy(dataset: Any, batch_size: int = 50) -> Dataset:
        """
        Convert TorchDataset to HuggingFace Dataset for publishing.

        This is a legacy method that directly converts TorchDataset to
        HuggingFace format. For new code, use from_dataset() which
        automatically detects the dataset type.

        Parameters
        ----------
        dataset : TorchDataset
            TorchDataset instance with images and labels tensors.
        batch_size : int, default=50
            Number of samples to process per batch to avoid 2GB Arrow limit.

        Returns
        -------
        datasets.Dataset
            HuggingFace Dataset with 'image' and 'label' fields.

        Examples
        --------
        >>> from samrfi.data.hf_dataset_wrapper import HFDatasetWrapper
        >>> hf_dataset = HFDatasetWrapper.from_numpy(torch_dataset, batch_size=50)
        >>> print(f"Converted {len(hf_dataset)} samples")

        Notes
        -----
        Labels are converted to PIL Images (grayscale) for HuggingFace
        compatibility. Metadata is preserved in dataset.info.description
        if present in the source dataset.

        See Also
        --------
        from_dataset : Recommended method that auto-detects dataset type
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
    def to_numpy(hf_dataset: Dataset) -> TorchDataset:
        """
        Convert HuggingFace Dataset to TorchDataset.

        Converts a HuggingFace Dataset (typically loaded from HuggingFace Hub)
        back to SAM-RFI's TorchDataset format for fast training.

        Parameters
        ----------
        hf_dataset : datasets.Dataset
            HuggingFace Dataset with 'image' and 'label' fields.

        Returns
        -------
        TorchDataset
            TorchDataset instance with images and labels as PyTorch tensors.

        Examples
        --------
        >>> from datasets import load_dataset
        >>> from samrfi.data.hf_dataset_wrapper import HFDatasetWrapper
        >>> # Load from HuggingFace Hub
        >>> hf_dataset = load_dataset("username/dataset-name", split="train")
        >>> # Convert to TorchDataset for training
        >>> torch_dataset = HFDatasetWrapper.to_numpy(hf_dataset)
        >>> print(torch_dataset)
        TorchDataset(num_samples=1000, ...)

        Notes
        -----
        The conversion process:
        1. Loads all samples from HuggingFace Dataset
        2. Converts PIL Images to numpy arrays
        3. Stacks into single tensors
        4. Creates TorchDataset with PyTorch tensors
        5. Preserves metadata from dataset.info.description

        This method is useful for loading published datasets from HuggingFace
        Hub for local training with SAM-RFI.

        See Also
        --------
        from_dataset : Convert TorchDataset to HuggingFace Dataset
        samrfi.data.torch_dataset.TorchDataset : Output format
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
