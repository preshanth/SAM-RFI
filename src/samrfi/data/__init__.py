"""
Data module - MS loading, preprocessing, and dataset creation
"""

from .adaptive_patcher import AdaptivePatcher, check_ms_compatibility
from .gpu_dataset import GPUBatchTransformDataset, GPUTransformDataset
from .gpu_transforms import GPUTransforms, create_gpu_transforms
from .hf_dataset_wrapper import HFDatasetWrapper
from .preprocessor import GPUPreprocessor, Preprocessor
from .ram_dataset import RAMCachedDataset
from .sam_dataset import BatchedDataset, SAMDataset
from .torch_dataset import BatchWriter, TorchDataset

__all__ = [
    "Preprocessor",
    "GPUPreprocessor",
    "SAMDataset",
    "BatchedDataset",
    "TorchDataset",
    "BatchWriter",
    "HFDatasetWrapper",
    "AdaptivePatcher",
    "check_ms_compatibility",
    "GPUTransforms",
    "create_gpu_transforms",
    "GPUTransformDataset",
    "GPUBatchTransformDataset",
    "RAMCachedDataset",
]

# Note: MSLoader requires CASA and is not imported by default
# Use: from samrfi.data.ms_loader import MSLoader
