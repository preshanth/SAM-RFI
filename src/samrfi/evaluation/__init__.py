"""
Evaluation metrics and validation tools for RFI segmentation

NOTE: Core metrics (IoU, F1, Dice, FFI, statistics) have been moved to rfi_toolbox
for sharing across ML methods. This module provides forward-compatibility imports.
"""

# Forward imports from rfi_toolbox (shared metrics)
from rfi_toolbox.evaluation import (
    compute_dice,
    compute_f1,
    compute_ffi,
    compute_iou,
    compute_precision,
    compute_recall,
    compute_statistics,
    evaluate_segmentation,
    print_statistics_comparison,
)
from rfi_toolbox.io import inject_synthetic_data

# SAM2-specific evaluation (if any remain in local files)
# Currently all metrics are in rfi_toolbox

__all__ = [
    "compute_iou",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_dice",
    "evaluate_segmentation",
    "compute_statistics",
    "compute_ffi",
    "print_statistics_comparison",
    "inject_synthetic_data",
]
