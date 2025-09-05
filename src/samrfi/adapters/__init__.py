"""
SAM-RFI Adapters Module

SAM version adapters and registry system.
"""

from .base import SAMAdapter
from .registry import (
    SAMRegistry,
    register_sam_adapter,
    get_sam_adapter,
    list_sam_versions,
)

# Import SAM2 adapter (auto-registers variants)
try:
    from .sam2_adapter import SAM2Adapter

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

__all__ = [
    "SAMAdapter",
    "SAMRegistry",
    "register_sam_adapter",
    "get_sam_adapter",
    "list_sam_versions",
    "SAM2Adapter" if SAM2_AVAILABLE else None,
]

# Remove None from __all__
__all__ = [item for item in __all__ if item is not None]
