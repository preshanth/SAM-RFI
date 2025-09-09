"""
SAM Inference Engine

Handles SAM2 model operations including patch processing and mask union computation.
Wraps SAM2Adapter for clean interface.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch

logger = logging.getLogger(__name__)


class SAMInferenceEngine:
    """SAM2 inference engine for RFI detection"""
    
    def __init__(self, device: str = "cuda", variant: str = "large", 
                 local_model_path: Optional[str] = None):
        """
        Initialize SAM inference engine
        
        Args:
            device: CUDA device to use  
            variant: SAM2 model variant (tiny, small, base_plus, large)
            local_model_path: Path to local model files (None = download from HF)
        """
        self.device = device
        self.variant = variant
        self.local_model_path = local_model_path
        
        # Will be initialized on first use
        self._sam_adapter = None
        
    def _ensure_model_loaded(self):
        """Lazy load SAM2 model on first use"""
        if self._sam_adapter is None:
            from ..adapters import SAM2Adapter
            
            logger.info(f"Loading SAM2 {self.variant} on {self.device}")
            self._sam_adapter = SAM2Adapter(device=self.device, variant=self.variant)
            
            if self.local_model_path:
                self._sam_adapter.load_model(local_model_path=self.local_model_path)
            else:
                self._sam_adapter.load_model()
                
            logger.info("SAM2 model loaded successfully")
    
    def predict_patches(self, patches: np.ndarray, 
                       generate_prompts: bool = True) -> List[np.ndarray]:
        """
        Run inference on image patches
        
        Args:
            patches: Array of patches [n_patches, channels, height, width] 
            generate_prompts: Whether to generate prompts automatically
            
        Returns:
            List of masks, one per patch [height, width]
        """
        self._ensure_model_loaded()
        
        n_patches = patches.shape[0]
        masks = []
        
        logger.info(f"Running SAM2 inference on {n_patches} patches")
        
        for i, patch in enumerate(patches):
            # Convert to HWC format expected by SAM2Adapter
            if patch.ndim == 3 and patch.shape[0] == 3:  # CHW -> HWC
                patch_hwc = patch.transpose(1, 2, 0)
            else:
                patch_hwc = patch
                
            # Ensure uint8 format
            if patch_hwc.dtype != np.uint8:
                if patch_hwc.max() <= 1.0:  # [0,1] range
                    patch_hwc = (patch_hwc * 255).astype(np.uint8)
                else:  # Already [0,255] range
                    patch_hwc = patch_hwc.astype(np.uint8)
            
            if generate_prompts:
                # Use untrained SAM - just predict everything as potential RFI
                # This will give us raw SAM2 predictions to evaluate pipeline
                result = self._sam_adapter.predict_single(patch_hwc)
                
                # Extract best mask from SAM2 multi-mask output
                if isinstance(result, dict) and 'masks' in result:
                    mask = result['masks'][0]  # Take first/best mask
                else:
                    mask = result  # Assume direct mask return
                    
            else:
                # For trained models, we would pass actual prompts here
                # For now, use automatic mode
                result = self._sam_adapter.predict_single(patch_hwc)
                mask = result['masks'][0] if isinstance(result, dict) else result
            
            # Ensure binary mask
            if mask.dtype != bool:
                mask = mask > 0.5
                
            masks.append(mask.astype(np.float32))
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i+1}/{n_patches} patches")
        
        logger.info(f"SAM2 inference complete - generated {len(masks)} masks")
        return masks
    
    def compute_union_mask(self, mask_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute union of multiple masks (same approach as training)
        
        Args:
            mask_list: List of masks to combine
            
        Returns:
            Single binary mask representing union of all inputs
        """
        if not mask_list:
            raise ValueError("Empty mask list provided")
        
        if len(mask_list) == 1:
            return mask_list[0]
        
        # Stack masks and take element-wise maximum (union)
        stacked = np.stack(mask_list, axis=0)  # [n_masks, H, W]
        union_mask = np.max(stacked, axis=0)   # [H, W]
        
        logger.debug(f"Combined {len(mask_list)} masks into union mask")
        return union_mask
    
    def predict_and_union(self, patches: np.ndarray) -> np.ndarray:
        """
        Convenience method: predict patches and compute union
        
        Args:
            patches: Array of patches to process
            
        Returns:
            Single union mask combining all patch predictions
        """
        patch_masks = self.predict_patches(patches)
        union_mask = self.compute_union_mask(patch_masks)
        return union_mask
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if self._sam_adapter is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "variant": self.variant, 
            "device": self.device,
            "local_path": self.local_model_path
        }
    
    def generate_simple_prompts(self, image_shape: Tuple[int, int], 
                               n_points: int = 10) -> Dict[str, Any]:
        """
        Generate simple point prompts for untrained SAM
        
        Args:
            image_shape: (height, width) of image
            n_points: Number of random points to generate
            
        Returns:
            Dict with points and labels for SAM2
        """
        h, w = image_shape
        
        # Generate random points
        points = np.random.randint(0, min(h, w), size=(n_points, 2))
        labels = np.ones(n_points)  # All positive prompts
        
        return {
            "input_points": points,
            "input_labels": labels
        }