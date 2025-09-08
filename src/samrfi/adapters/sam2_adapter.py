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
    from PIL import Image
    
    # Try transformers approach first
    try:
        from transformers import Sam2Model, Sam2Processor
        USE_TRANSFORMERS = True
    except ImportError:
        USE_TRANSFORMERS = False
        Sam2Model = None
        Sam2Processor = None
        logger.info("Sam2Model not available in transformers, will use fallback")
    
    # Try official SAM2 approach as fallback
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor  
        from huggingface_hub import hf_hub_download
        USE_OFFICIAL_SAM2 = True
    except ImportError:
        USE_OFFICIAL_SAM2 = False
        build_sam2 = None
        SAM2ImagePredictor = None
        hf_hub_download = None

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    USE_TRANSFORMERS = False
    USE_OFFICIAL_SAM2 = False
    torch = None
    Image = None
    logging.warning(f"SAM2 dependencies not available: {e}")

from .base import SAMAdapter

logger = logging.getLogger(__name__)


class SAM2Adapter(SAMAdapter):
    """SAM2 implementation - primary adapter for RFI detection"""

    def __init__(self, device: str = "cuda", variant: str = "large"):
        """
        Initialize SAM2 adapter

        Args:
            device: Device to run on ('cuda', 'cpu')
            variant: SAM2 model variant ('tiny', 'small', 'base_plus', 'large')
        """
        super().__init__(device)
        self.variant = variant
        self.model = None
        self.processor = None
        self._input_size = 1024  # SAM2 standard

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if not USE_TRANSFORMERS and not USE_OFFICIAL_SAM2:
            raise ImportError(
                "SAM2 not available. Please install either:\n"
                "1. transformers>=4.46 (pip install transformers>=4.46), or\n" 
                "2. Official SAM2 (pip install git+https://github.com/facebookresearch/segment-anything-2.git)"
            )

        # Validate device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

    @property
    def version(self) -> str:
        return f"sam2_{self.variant}"

    @property
    def input_size(self) -> int:
        return self._input_size

    def load_model(self, checkpoint_path: str = None, config: Dict[str, Any] = None, 
                   local_model_path: str = None) -> None:
        """
        Load SAM2 model (local path > transformers > official SAM2)

        Args:
            checkpoint_path: Optional checkpoint path for official SAM2
            config: Configuration dict (optional)
            local_model_path: Path to local model directory (overrides HuggingFace)
        """
        try:
            if local_model_path:
                self._load_from_local(local_model_path)
            elif USE_TRANSFORMERS:
                self._load_from_transformers()
            elif USE_OFFICIAL_SAM2:
                self._load_from_official_sam2(checkpoint_path, config)
            else:
                raise ImportError("No SAM2 implementation available")

        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            self.is_loaded = False
            raise
    
    def _load_from_transformers(self) -> None:
        """Load using transformers library"""
        model_id = self._get_hf_model_id()
        
        logger.info(f"Loading SAM2 {self.variant} from transformers: {model_id}")
        
        self.model = Sam2Model.from_pretrained(model_id).to(self.device)
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.is_loaded = True
        
        logger.info(f"SAM2 model loaded via transformers on {self.device}")
    
    def _load_from_local(self, local_model_path: str) -> None:
        """Load model from local directory with fallback logic"""
        from pathlib import Path
        
        model_path = Path(local_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {local_model_path}")
        
        logger.info(f"Loading SAM2 {self.variant} from local path: {local_model_path}")
        
        # Try to load using transformers from local directory first
        if USE_TRANSFORMERS:
            try:
                self.model = Sam2Model.from_pretrained(str(model_path)).to(self.device)
                self.processor = Sam2Processor.from_pretrained(str(model_path))
                self.is_loaded = True
                logger.info(f"SAM2 model loaded from local path via transformers on {self.device}")
                return
            except Exception as e:
                logger.error(f"Failed to load from local path with transformers: {e}")
        
        # Fallback to official SAM2 loading if transformers failed
        if USE_OFFICIAL_SAM2:
            try:
                # Look for the official SAM2 checkpoint file
                pt_files = list(model_path.glob("*.pt"))
                config_files = list(model_path.glob("*.yaml"))
                
                if not pt_files:
                    raise FileNotFoundError(f"No .pt checkpoint file found in {model_path}")
                
                checkpoint_path = str(pt_files[0])  # Use first .pt file found
                
                # Use local config if available, otherwise use default
                if config_files:
                    config_path = str(config_files[0])
                else:
                    config_path = self._get_config_file_path()
                
                logger.info(f"Falling back to official SAM2 loading")
                logger.info(f"Checkpoint: {checkpoint_path}")
                logger.info(f"Config: {config_path}")
                
                self.model = build_sam2(config_path, checkpoint_path, device=self.device)
                self.predictor = SAM2ImagePredictor(self.model)
                self.is_loaded = True
                logger.info(f"SAM2 model loaded from local path via official SAM2 on {self.device}")
                return
                
            except Exception as e:
                logger.error(f"Failed to load from local path with official SAM2: {e}")
        
        # If both methods failed, raise the error
        raise RuntimeError(f"Failed to load SAM2 model from {local_model_path} using both transformers and official SAM2 methods")
    
    def _load_from_official_sam2(self, checkpoint_path: str = None, config: Dict[str, Any] = None) -> None:
        """Load using official SAM2 repository"""
        if checkpoint_path is None:
            checkpoint_path = self._download_sam2_checkpoint()
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        config_path = self._get_config_file_path()
        
        logger.info(f"Loading SAM2 {self.variant} from official repo")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Config: {config_path}")
        
        self.model = build_sam2(config_path, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.is_loaded = True
        
        logger.info(f"SAM2 model loaded via official repo on {self.device}")
    
    def _download_sam2_checkpoint(self) -> str:
        """Download SAM2 checkpoint from HuggingFace"""
        if not hf_hub_download:
            raise ImportError("huggingface_hub required for download")
        
        # SAM2.1 model filenames
        model_files = {
            "tiny": "sam2.1_hiera_tiny.pt",
            "small": "sam2.1_hiera_small.pt", 
            "base_plus": "sam2.1_hiera_base_plus.pt",
            "large": "sam2.1_hiera_large.pt"
        }
        
        if self.variant not in model_files:
            logger.warning(f"Unknown variant {self.variant}, using large")
            self.variant = "large"
        
        filename = model_files[self.variant]
        repo_id = f"facebook/sam2.1-hiera-{self.variant}" if self.variant != "base_plus" else "facebook/sam2.1-hiera-base-plus"
        
        logger.info(f"Downloading SAM2.1 {self.variant} checkpoint...")
        
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None
        )
        
        logger.info(f"Downloaded checkpoint to: {checkpoint_path}")
        return checkpoint_path
    
    def _get_config_file_path(self) -> str:
        """Get config file for official SAM2"""
        # These configs are typically bundled with the SAM2 package
        config_files = {
            "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml", 
            "large": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
        
        return config_files.get(self.variant, config_files["large"])
    
    def _get_hf_model_id(self) -> str:
        """Get HuggingFace model ID for the variant"""
        hf_models = {
            "tiny": "facebook/sam2-hiera-tiny",
            "small": "facebook/sam2-hiera-small", 
            "base_plus": "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large"
        }
        
        if self.variant not in hf_models:
            logger.warning(f"Unknown variant {self.variant}, defaulting to large")
            return hf_models["large"]
        
        return hf_models[self.variant]
    

    def predict(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
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
            points = prompts["points"]
            boxes = prompts["bbox"]
            point_labels = prompts["point_labels"]
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
                **kwargs,
            )

            # Create binary masks
            binary_masks = (masks > threshold).astype(np.uint8)

            return {
                "masks": binary_masks,
                "scores": scores,
                "logits": logits,
                "raw_masks": masks,
            }

        except Exception as e:
            logger.error(f"SAM2 prediction failed: {e}")
            raise

    def predict_patches(
        self,
        patches: List[np.ndarray],
        patch_positions: Optional[List[Tuple[int, int]]] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
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
                    points=prompts["points"],
                    boxes=prompts["bbox"],
                    threshold=threshold,
                    **kwargs,
                )

                # Add patch information
                result["patch_index"] = i
                if patch_positions:
                    result["patch_position"] = patch_positions[i]

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to process patch {i}: {e}")
                # Add empty result to maintain list alignment
                results.append(
                    {
                        "masks": np.zeros((1, *patch.shape[:2]), dtype=np.uint8),
                        "scores": np.array([0.0]),
                        "logits": np.zeros((1, *patch.shape[:2])),
                        "patch_index": i,
                        "error": str(e),
                    }
                )

        return results

    def _get_default_config(self) -> str:
        """Get default config file path for the variant"""
        config_mapping = {
            "tiny": "sam2.1_hiera_t.yaml",
            "small": "sam2.1_hiera_s.yaml",
            "base_plus": "sam2.1_hiera_b+.yaml",
            "large": "sam2.1_hiera_l.yaml",
        }

        return config_mapping.get(self.variant, "sam2.1_hiera_l.yaml")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if not torch.cuda.is_available() or self.device == "cpu":
            return {"device": "cpu", "memory_used_gb": 0.0, "memory_total_gb": 0.0}

        memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            "device": self.device,
            "memory_used_gb": memory_used,
            "memory_total_gb": memory_total,
            "memory_utilization": (
                memory_used / memory_total if memory_total > 0 else 0.0
            ),
        }

    def optimize_for_hardware(
        self, memory_gb: int, time_limit_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations based on hardware constraints

        Args:
            memory_gb: Available GPU memory
            time_limit_hours: Time constraint (if any)

        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "batch_size": 1,
            "gradient_accumulation": 8,
            "mixed_precision": True,
            "gradient_checkpointing": False,
        }

        if memory_gb <= 16:  # V100 constraints
            recommendations.update(
                {
                    "batch_size": 1,
                    "gradient_accumulation": 8,
                    "gradient_checkpointing": True,
                    "recommended_variant": "small",
                }
            )
        else:  # H200 or similar
            recommendations.update(
                {
                    "batch_size": 4,
                    "gradient_accumulation": 4,
                    "gradient_checkpointing": False,
                    "recommended_variant": "large",
                }
            )

        if time_limit_hours and time_limit_hours <= 2:
            # Time-constrained (H200 scenario)
            recommendations.update(
                {
                    "batch_size": max(4, recommendations["batch_size"]),
                    "gradient_accumulation": 2,
                    "max_epochs": 50,
                }
            )

        return recommendations


# Auto-register SAM2 variants when module is imported
from .registry import register_sam_adapter

if TORCH_AVAILABLE:
    # Register different SAM2 variants  
    variants = ["tiny", "small", "base_plus", "large"]

    def create_variant_class(variant_name):
        class VariantSAM2Adapter(SAM2Adapter):
            def __init__(self, **kwargs):
                super().__init__(variant=variant_name, **kwargs)
        return VariantSAM2Adapter

    for variant in variants:
        adapter_class = create_variant_class(variant)
        register_sam_adapter(
            f"sam2_{variant}",
            adapter_class,
            metadata={
                "description": f"SAM2 {variant} variant",
                "input_size": 1024,
                "memory_requirements": {
                    "tiny": "8GB",
                    "small": "12GB",
                    "base_plus": "16GB",
                    "large": "20GB+",
                }.get(variant, "Unknown"),
            },
        )

    # Register default SAM2 (large variant)
    register_sam_adapter(
        "sam2",
        SAM2Adapter,
        metadata={
            "description": "Default SAM2 (large variant)",
            "input_size": 1024,
            "memory_requirements": "20GB+",
        },
    )

    logger.info("SAM2 adapters registered successfully")
else:
    logger.warning("SAM2 adapters not registered - dependencies unavailable")
