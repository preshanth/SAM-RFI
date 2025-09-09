"""
Generic Tiling Processor

Handles splitting arbitrary-sized images into patches and reconstruction.
Pure numpy operations with no domain-specific knowledge.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TilingProcessor:
    """Generic tiling operations for arbitrary image sizes"""
    
    def __init__(self, patch_size: int = 256):
        """
        Initialize tiling processor
        
        Args:
            patch_size: Size of square patches to create (default 256 for SAM2)
        """
        self.patch_size = patch_size
        
    def create_patches(self, data: np.ndarray, overlap: int = 0) -> Dict[str, Any]:
        """
        Split image into patches
        
        Args:
            data: Input image array [..., H, W] - handles any number of leading dimensions
            overlap: Pixel overlap between patches (default 0)
            
        Returns:
            Dict containing:
            - patches: Array of patches [..., n_patches, patch_size, patch_size]  
            - metadata: Info needed for reconstruction
        """
        original_shape = data.shape
        *leading_dims, height, width = original_shape
        
        if height != width:
            raise ValueError(f"Expected square image, got {height}x{width}")
            
        if height % self.patch_size != 0:
            raise ValueError(f"Image size {height} not divisible by patch_size {self.patch_size}")
        
        patches_per_side = height // self.patch_size
        n_patches = patches_per_side * patches_per_side
        
        # Reshape to isolate patches
        # [..., H, W] -> [..., patches_per_side, patch_size, patches_per_side, patch_size]
        reshaped = data.reshape(*leading_dims, patches_per_side, self.patch_size, 
                               patches_per_side, self.patch_size)
        
        # Transpose to group patches together
        # [..., patches_per_side, patch_size, patches_per_side, patch_size] 
        # -> [..., patches_per_side, patches_per_side, patch_size, patch_size]
        transposed = reshaped.transpose(*range(len(leading_dims)), 
                                       len(leading_dims), len(leading_dims)+2,
                                       len(leading_dims)+1, len(leading_dims)+3)
        
        # Flatten patch dimensions: [..., n_patches, patch_size, patch_size]
        patches = transposed.reshape(*leading_dims, n_patches, self.patch_size, self.patch_size)
        
        metadata = {
            'original_shape': original_shape,
            'patches_per_side': patches_per_side,
            'n_patches': n_patches,
            'patch_size': self.patch_size,
            'overlap': overlap
        }
        
        logger.debug(f"Created {n_patches} patches of size {self.patch_size}x{self.patch_size} "
                    f"from {height}x{width} image")
        
        return {
            'patches': patches,
            'metadata': metadata
        }
    
    def reconstruct_from_patches(self, patches: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Reconstruct full image from patches
        
        Args:
            patches: Patch array [..., n_patches, patch_size, patch_size]
            metadata: Metadata from create_patches
            
        Returns:
            Reconstructed image with original shape
        """
        original_shape = metadata['original_shape']
        patches_per_side = metadata['patches_per_side']
        n_patches = metadata['n_patches']
        patch_size = metadata['patch_size']
        
        *leading_dims, _, _ = original_shape
        
        # Verify patch array shape
        expected_patch_shape = (*leading_dims, n_patches, patch_size, patch_size)
        if patches.shape != expected_patch_shape:
            raise ValueError(f"Expected patches shape {expected_patch_shape}, got {patches.shape}")
        
        # Reshape patches back to grid: [..., n_patches, patch_size, patch_size]
        # -> [..., patches_per_side, patches_per_side, patch_size, patch_size]
        patch_grid = patches.reshape(*leading_dims, patches_per_side, patches_per_side, 
                                    patch_size, patch_size)
        
        # Transpose back: [..., patches_per_side, patches_per_side, patch_size, patch_size]
        # -> [..., patches_per_side, patch_size, patches_per_side, patch_size]
        transposed = patch_grid.transpose(*range(len(leading_dims)),
                                         len(leading_dims), len(leading_dims)+2,
                                         len(leading_dims)+1, len(leading_dims)+3)
        
        # Reshape back to original: [..., H, W]
        reconstructed = transposed.reshape(original_shape)
        
        logger.debug(f"Reconstructed {original_shape} image from {n_patches} patches")
        
        return reconstructed
    
    def get_patch_coordinates(self, patch_idx: int, patches_per_side: int) -> Tuple[int, int]:
        """
        Get (row, col) coordinates for a patch index
        
        Args:
            patch_idx: Index of patch (0 to n_patches-1)
            patches_per_side: Number of patches per side
            
        Returns:
            (row, col) coordinates in patch grid
        """
        row = patch_idx // patches_per_side
        col = patch_idx % patches_per_side
        return row, col
    
    def get_pixel_coordinates(self, patch_idx: int, patches_per_side: int, 
                             local_y: int, local_x: int) -> Tuple[int, int]:
        """
        Convert patch-local coordinates to global image coordinates
        
        Args:
            patch_idx: Index of patch
            patches_per_side: Number of patches per side
            local_y, local_x: Coordinates within patch
            
        Returns:
            (global_y, global_x) coordinates in full image
        """
        patch_row, patch_col = self.get_patch_coordinates(patch_idx, patches_per_side)
        
        global_y = patch_row * self.patch_size + local_y
        global_x = patch_col * self.patch_size + local_x
        
        return global_y, global_x