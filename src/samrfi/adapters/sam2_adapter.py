"""
SAM2 Adapter Implementation

Primary SAM2 implementation optimized for RFI detection with support for
both V100 (memory-constrained) and H200 (time-constrained) training.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    build_sam2 = None
    SAM2ImagePredictor = None
    logging.warning(f"SAM2 dependencies not available: {e}")

from .base import SAMAdapter

logger = logging.getLogger(__name__)


class SAM2Adapter(SAMAdapter):
    """SAM2 implementation - primary adapter for RFI detection"""
    
    def __init__(self, device: str = 'cuda', variant: str = 'large'):
        """
        Initialize SAM2 adapter
        
        Args:
            device: Device to run on ('cuda', 'cpu')
            variant: SAM2 model variant ('tiny', 'small', 'base_plus', 'large')
        """
        super().__init__(device)
        self.variant = variant
        self.predictor = None
        self._input_size = 1024  # SAM2 standard
        
        if not TORCH_AVAILABLE:
            raise ImportError("SAM2 dependencies not available. Please install torch and sam2.")
        
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
    
    @property
    def version(self) -> str:
        return f"sam2_{self.variant}"
    
    @property
    def input_size(self) -> int:
        return self._input_size
    
    def load_model(self, checkpoint_path: str, config: Dict[str, Any]) -> None:
        """
        Load SAM2 model from checkpoint
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint
            config: Configuration with 'config_path' for SAM2 config file
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        config_path = config.get('config_path', self._get_default_config())
        
        try:
            # Build SAM2 model
            logger.info(f"Loading SAM2 model: {self.variant}")
            self.model = build_sam2(config_path, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            self.is_loaded = True
            
            logger.info(f"SAM2 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            self.is_loaded = False
            raise
    
    def predict(self, 
                image: np.ndarray, 
                points: Optional[np.ndarray] = None,
                boxes: Optional[np.ndarray] = None,
                masks: Optional[np.ndarray] = None,
                threshold: float = 0.5,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        Run SAM2 inference on single image
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            points: Point prompts as (N, 2) array
            boxes: Bounding box as (4,) array [x1, y1, x2, y2]
            masks: Previous mask for refinement
            threshold: Threshold for binary mask creation
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'masks', 'scores', 'logits'
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate and preprocess image
        image_rgb = self.validate_image(image)
        
        # Set image for prediction
        self.predictor.set_image(image_rgb)
        
        # Create default prompts if none provided
        if points is None and boxes is None:
            prompts = self.create_default_prompts(image_rgb)
            points = prompts['points']
            boxes = prompts['bbox']
            point_labels = prompts['point_labels']
        else:
            point_labels = np.ones(len(points)) if points is not None else None
        
        try:
            # Run prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=boxes,
                mask_input=masks,
                multimask_output=False,
                **kwargs
            )
            
            # Create binary masks
            binary_masks = (masks > threshold).astype(np.uint8)
            
            return {
                'masks': binary_masks,
                'scores': scores,
                'logits': logits,
                'raw_masks': masks
            }
            
        except Exception as e:
            logger.error(f"SAM2 prediction failed: {e}")
            raise
    
    def predict_patches(self, 
                       patches: List[np.ndarray], 
                       patch_positions: Optional[List[Tuple[int, int]]] = None,
                       threshold: float = 0.5,
                       **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Run SAM2 inference on multiple patches
        
        Args:
            patches: List of image patches
            patch_positions: Position of each patch in original image
            threshold: Threshold for binary mask creation
            **kwargs: Additional parameters
            
        Returns:
            List of prediction results for each patch
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for i, patch in enumerate(patches):
            try:
                # Get patch-specific prompts
                prompts = self.create_default_prompts(patch)
                
                # Run prediction on patch
                result = self.predict(
                    patch,
                    points=prompts['points'],
                    boxes=prompts['bbox'],
                    threshold=threshold,
                    **kwargs
                )
                
                # Add patch information
                result['patch_index'] = i
                if patch_positions:
                    result['patch_position'] = patch_positions[i]
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to process patch {i}: {e}")
                # Add empty result to maintain list alignment
                results.append({
                    'masks': np.zeros((1, *patch.shape[:2]), dtype=np.uint8),
                    'scores': np.array([0.0]),
                    'logits': np.zeros((1, *patch.shape[:2])),
                    'patch_index': i,
                    'error': str(e)
                })
        
        return results
    
    def _get_default_config(self) -> str:
        """Get default config file path for the variant"""
        config_mapping = {
            'tiny': 'sam2.1_hiera_t.yaml',
            'small': 'sam2.1_hiera_s.yaml', 
            'base_plus': 'sam2.1_hiera_b+.yaml',
            'large': 'sam2.1_hiera_l.yaml'
        }
        
        return config_mapping.get(self.variant, 'sam2.1_hiera_l.yaml')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if not torch.cuda.is_available() or self.device == 'cpu':
            return {'device': 'cpu', 'memory_used_gb': 0.0, 'memory_total_gb': 0.0}
        
        memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            'device': self.device,
            'memory_used_gb': memory_used,
            'memory_total_gb': memory_total,
            'memory_utilization': memory_used / memory_total if memory_total > 0 else 0.0
        }
    
    def optimize_for_hardware(self, memory_gb: int, time_limit_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Get optimization recommendations based on hardware constraints
        
        Args:
            memory_gb: Available GPU memory
            time_limit_hours: Time constraint (if any)
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'batch_size': 1,
            'gradient_accumulation': 8,
            'mixed_precision': True,
            'gradient_checkpointing': False
        }
        
        if memory_gb <= 16:  # V100 constraints
            recommendations.update({
                'batch_size': 1,
                'gradient_accumulation': 8,
                'gradient_checkpointing': True,
                'recommended_variant': 'small'
            })
        else:  # H200 or similar
            recommendations.update({
                'batch_size': 4,
                'gradient_accumulation': 4,
                'gradient_checkpointing': False,
                'recommended_variant': 'large'
            })
        
        if time_limit_hours and time_limit_hours <= 2:
            # Time-constrained (H200 scenario)
            recommendations.update({
                'batch_size': max(4, recommendations['batch_size']),
                'gradient_accumulation': 2,
                'max_epochs': 50
            })
        
        return recommendations


# Auto-register SAM2 variants when module is imported
from .registry import register_sam_adapter

if TORCH_AVAILABLE:
    # Register different SAM2 variants
    variants = ['tiny', 'small', 'base_plus', 'large']
    
    for variant in variants:
        register_sam_adapter(
            f'sam2_{variant}',
            lambda variant=variant, **kwargs: SAM2Adapter(variant=variant, **kwargs),
            metadata={
                'description': f'SAM2 {variant} variant',
                'input_size': 1024,
                'memory_requirements': {
                    'tiny': '8GB',
                    'small': '12GB', 
                    'base_plus': '16GB',
                    'large': '20GB+'
                }.get(variant, 'Unknown')
            }
        )
    
    # Register default SAM2 (large variant)
    register_sam_adapter('sam2', SAM2Adapter, metadata={
        'description': 'Default SAM2 (large variant)',
        'input_size': 1024,
        'memory_requirements': '20GB+'
    })
    
    logger.info("SAM2 adapters registered successfully")
else:
    logger.warning("SAM2 adapters not registered - dependencies unavailable")