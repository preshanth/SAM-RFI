"""
SAM-RFI Datasets Module

Dataset handling, synthetic data generation, and HuggingFace integration.
"""

# Core synthetic data generation (always available)
from .synthetic_ms import ObservationConfig, RFIConfig, SyntheticVisibilityGenerator
from .ms_writer import MSWriter
from .generator import SyntheticDatasetGenerator

# HuggingFace integration (optional)
try:
    from .rfi_dataset import RFIDatasetCreator

    HF_AVAILABLE = True
except ImportError:
    RFIDatasetCreator = None
    HF_AVAILABLE = False

__all__ = [
    "ObservationConfig",
    "RFIConfig",
    "SyntheticVisibilityGenerator",
    "MSWriter",
    "SyntheticDatasetGenerator",
]

if HF_AVAILABLE:
    __all__.append("RFIDatasetCreator")
