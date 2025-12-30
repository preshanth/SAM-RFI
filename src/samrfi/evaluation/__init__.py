"""
Evaluation metrics and validation tools for RFI segmentation
"""

from .metrics import (
    compute_dice,
    compute_f1,
    compute_iou,
    compute_precision,
    compute_recall,
    evaluate_segmentation,
)
from .statistics import (
    compute_calcquality,
    compute_ffi,
    compute_statistics,
    print_statistics_comparison,
)

__all__ = [
    "compute_iou",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_dice",
    "evaluate_segmentation",
    "compute_statistics",
    "compute_ffi",
    "compute_calcquality",
    "print_statistics_comparison",
]

# Optional CASA dependency for MS injection
try:
    from .ms_injection import inject_synthetic_data

    __all__.append("inject_synthetic_data")
except ImportError:
    pass  # CASA not available
