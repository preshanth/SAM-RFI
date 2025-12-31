"""
Data module - MS loading, preprocessing, and dataset creation

NOTE: Core data utilities (MSLoader, Preprocessor, TorchDataset, BatchWriter)
have been moved to rfi_toolbox for sharing across ML methods.
This module provides forward-compatibility imports.
"""

from rfi_toolbox.datasets.batched_dataset import BatchWriter, TorchDataset

# Forward imports from rfi_toolbox (shared utilities)
try:
    from rfi_toolbox.io.ms_loader import MSLoader
except ImportError:
    MSLoader = None  # CASA not available
from rfi_toolbox.preprocessing.preprocessor import GPUPreprocessor, Preprocessor

# SAM2-specific modules (stay in samrfi)
from .adaptive_patcher import AdaptivePatcher, check_ms_compatibility
from .gpu_dataset import GPUBatchTransformDataset, GPUTransformDataset
from .gpu_transforms import GPUTransforms, create_gpu_transforms
from .hf_dataset_wrapper import HFDatasetWrapper
from .ram_dataset import RAMCachedDataset
from .sam_dataset import BatchedDataset, SAMDataset

__all__ = [
    # Shared utilities (from rfi_toolbox)
    "MSLoader",
    "Preprocessor",
    "GPUPreprocessor",
    "TorchDataset",
    "BatchWriter",
    # SAM2-specific
    "SAMDataset",
    "BatchedDataset",
    "HFDatasetWrapper",
    "AdaptivePatcher",
    "check_ms_compatibility",
    "GPUTransforms",
    "create_gpu_transforms",
    "GPUTransformDataset",
    "GPUBatchTransformDataset",
    "RAMCachedDataset",
]
