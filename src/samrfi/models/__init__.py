"""
SAM-RFI Models Module

Model management and training functionality.
"""

try:
    from .training import GPUOptimizedTrainer, load_training_config, get_recommended_config
    TRAINING_AVAILABLE = True
except ImportError:
    GPUOptimizedTrainer = None
    load_training_config = None
    get_recommended_config = None
    TRAINING_AVAILABLE = False

__all__ = [
    'GPUOptimizedTrainer',
    'load_training_config', 
    'get_recommended_config'
] if TRAINING_AVAILABLE else []