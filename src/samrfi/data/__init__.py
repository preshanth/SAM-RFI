"""
Data module - SAM2-specific dataset wrappers and utilities

NOTE: Core data utilities (MSLoader, Preprocessor, TorchDataset, BatchWriter)
are in rfi_toolbox. Import directly:
    from rfi_toolbox.io import MSLoader
    from rfi_toolbox.preprocessing import Preprocessor
    from rfi_toolbox.datasets import BatchWriter, TorchDataset
"""

# SAM2-specific modules
from .adaptive_patcher import AdaptivePatcher, check_ms_compatibility
from .gpu_dataset import GPUBatchTransformDataset, GPUTransformDataset
from .gpu_transforms import GPUTransforms, create_gpu_transforms
from .hf_dataset_wrapper import HFDatasetWrapper
from .ram_dataset import RAMCachedDataset
from .sam_dataset import BatchedDataset, SAMDataset

__all__ = [
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
