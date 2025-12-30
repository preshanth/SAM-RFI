"""
Utility modules for SAM-RFI
"""

from .errors import (
    CheckpointMismatchError,
    ConfigValidationError,
    DataShapeError,
    ModelLoadError,
    SAMRFIError,
)
from .logger import logger, setup_logger

# Note: ModelCache requires transformers and is not imported by default
# Use: from samrfi.utils.model_cache import ModelCache

__all__ = [
    "logger",
    "setup_logger",
    "SAMRFIError",
    "DataShapeError",
    "CheckpointMismatchError",
    "ModelLoadError",
    "ConfigValidationError",
]
