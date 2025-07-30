"""
SAM-RFI Datasets Module

HuggingFace-compatible dataset creation and management.
"""

try:
    from .rfi_dataset import RFIDatasetCreator
    HF_AVAILABLE = True
except ImportError:
    RFIDatasetCreator = None
    HF_AVAILABLE = False

__all__ = ['RFIDatasetCreator'] if HF_AVAILABLE else []