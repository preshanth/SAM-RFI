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
from .ms_injection import inject_synthetic_data
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
    "inject_synthetic_data",
    "compute_statistics",
    "compute_ffi",
    "compute_calcquality",
    "print_statistics_comparison",
]
