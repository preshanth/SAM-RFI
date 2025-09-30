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

    def __init__(self, dataset, processor):
        """
        Initialize SAM dataset.

        Args:
            dataset: HuggingFace Dataset with 'image' and 'label' fields
            processor: SAM2Processor from transformers
        """
        self.dataset = dataset
        self.processor = processor

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
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # Get bounding box from mask
        bbox = self._get_bounding_box(ground_truth_mask)

        # Process image and prompt
        inputs = self.processor(
            image,
            input_boxes=[[bbox]],
            return_tensors="pt"
        )

        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth mask
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

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
            # Empty mask - return center box
            H, W = mask.shape
            return [W//4, H//4, 3*W//4, 3*H//4]

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add random perturbation (±20 pixels)
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return [x_min, y_min, x_max, y_max]