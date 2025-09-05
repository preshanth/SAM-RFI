"""
SAM Adapter Base Interface

Abstract base class for SAM version adapters to provide clean abstraction
across different SAM versions (SAM1, SAM2, future SAM3).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from pathlib import Path


class SAMAdapter(ABC):
    """Abstract base class for SAM version adapters"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self, checkpoint_path: str, config: Dict[str, Any]) -> None:
        """
        Load pre-trained model from checkpoint

        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary with model-specific settings
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on single image

        Args:
            image: Input image as numpy array
            points: Point prompts for segmentation
            boxes: Bounding box prompts
            masks: Mask prompts for refinement
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing 'masks', 'scores', 'logits'
        """
        pass

    @abstractmethod
    def predict_patches(
        self,
        patches: List[np.ndarray],
        patch_positions: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on image patches for large images

        Args:
            patches: List of image patches
            patch_positions: Position of each patch in original image
            **kwargs: Additional parameters

        Returns:
            List of prediction results for each patch
        """
        pass

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Required input image size for this SAM version"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """SAM version identifier (e.g., 'sam1', 'sam2')"""
        pass

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "version": self.version,
            "device": self.device,
            "input_size": self.input_size,
            "is_loaded": self.is_loaded,
        }

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess image for SAM input

        Args:
            image: Input image array

        Returns:
            Processed image ready for SAM
        """
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Convert single channel to RGB
            image = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            raise ValueError(f"Image must have 1 or 3 channels, got {image.shape[2]}")

        # Ensure float32 type
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize to 0-255 range if needed
        if image.max() <= 1.0:
            image = image * 255.0

        return image

    def create_default_prompts(
        self, image: np.ndarray, num_points: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Create default prompts for RFI detection

        Args:
            image: Input image
            num_points: Number of points to generate

        Returns:
            Dictionary with default prompts
        """
        height, width = image.shape[:2]

        # Create points at high-intensity regions (likely RFI)
        if len(image.shape) == 3:
            intensity = np.mean(image, axis=2)
        else:
            intensity = image

        # Find high-intensity regions
        threshold = np.percentile(intensity, 95)  # Top 5% of pixels
        high_intensity_mask = intensity > threshold

        if np.any(high_intensity_mask):
            y_coords, x_coords = np.where(high_intensity_mask)
            if len(y_coords) > num_points:
                # Randomly sample points
                indices = np.random.choice(len(y_coords), num_points, replace=False)
                y_coords = y_coords[indices]
                x_coords = x_coords[indices]

            points = np.column_stack([x_coords, y_coords])  # SAM expects (x, y) format
        else:
            # Fallback: random points
            points = np.random.randint(0, min(height, width), (num_points, 2))

        # Create bounding box covering the entire image
        bbox = np.array([0, 0, width, height])

        return {
            "points": points,
            "bbox": bbox,
            "point_labels": np.ones(len(points), dtype=int),  # All positive points
        }
