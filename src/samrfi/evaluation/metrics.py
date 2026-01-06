"""
Segmentation metrics for RFI detection evaluation.

This module provides standard binary segmentation metrics for evaluating
RFI detection performance by comparing predicted masks against ground truth.
All functions accept both PyTorch tensors and NumPy arrays, automatically
converting to NumPy arrays for computation.

The module includes:
- IoU (Intersection over Union / Jaccard Index)
- Precision, Recall, F1 Score
- Dice Coefficient
- Combined evaluation function

All metrics return values in [0, 1] where 1 indicates perfect agreement.
"""

from typing import Dict, Union

import numpy as np
import torch

# Type alias for input arrays
ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """
    Convert torch tensor or numpy array to numpy array.

    Parameters
    ----------
    arr : torch.Tensor or np.ndarray
        Input array to convert.

    Returns
    -------
    np.ndarray
        NumPy array representation of input.

    Examples
    --------
    >>> import torch
    >>> tensor = torch.tensor([1, 2, 3])
    >>> _to_numpy(tensor)
    array([1, 2, 3])
    >>> arr = np.array([4, 5, 6])
    >>> _to_numpy(arr)
    array([4, 5, 6])
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_iou(pred: ArrayLike, true: ArrayLike) -> float:
    """
    Compute Intersection over Union (IoU) / Jaccard Index.

    IoU measures the overlap between predicted and ground truth masks.
    Formula: IoU = |Intersection| / |Union|

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Predicted binary mask. Will be converted to boolean.
    true : torch.Tensor or np.ndarray
        Ground truth binary mask. Will be converted to boolean.

    Returns
    -------
    float
        IoU score in [0, 1].
        Returns 1.0 if both masks are empty (perfect agreement).

    Notes
    -----
    Empty masks (both pred and true all False) return 1.0 to indicate
    perfect agreement (neither detected any RFI).

    Examples
    --------
    >>> pred = np.array([[1, 1, 0], [0, 1, 0]])
    >>> true = np.array([[1, 0, 0], [0, 1, 1]])
    >>> compute_iou(pred, true)
    0.4

    >>> # Both empty - perfect agreement
    >>> compute_iou(np.zeros((2, 2)), np.zeros((2, 2)))
    1.0
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0  # Both masks empty = perfect agreement

    return float(intersection / union)


def compute_precision(pred: ArrayLike, true: ArrayLike) -> float:
    """
    Compute Precision = TP / (TP + FP).

    Precision measures what fraction of predicted RFI is actually RFI.
    Answers: "Of all the RFI we detected, how much was real?"

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Predicted binary mask. Will be converted to boolean.
    true : torch.Tensor or np.ndarray
        Ground truth binary mask. Will be converted to boolean.

    Returns
    -------
    float
        Precision in [0, 1].
        Returns 1.0 if no predictions on clean data (correct abstention).
        Returns 0.0 if no predictions but RFI exists (failure to detect).

    Notes
    -----
    Edge case handling:
    - No predictions + no RFI: 1.0 (correct abstention)
    - No predictions + RFI present: 0.0 (missed detection)
    - All predictions correct: 1.0 (perfect precision)

    Examples
    --------
    >>> pred = np.array([1, 1, 0, 0])
    >>> true = np.array([1, 0, 0, 1])
    >>> compute_precision(pred, true)  # 1 TP, 1 FP
    0.5

    >>> # No predictions on clean data
    >>> compute_precision(np.zeros(4), np.zeros(4))
    1.0
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    if tp + fp == 0:
        # No predictions made
        if fn == 0:
            # No RFI in ground truth = correct abstention
            return 1.0
        else:
            # RFI exists but not detected = failure
            return 0.0

    return float(tp / (tp + fp))


def compute_recall(pred: ArrayLike, true: ArrayLike) -> float:
    """
    Compute Recall = TP / (TP + FN) = Sensitivity = True Positive Rate.

    Recall measures what fraction of actual RFI is detected.
    Answers: "Of all the RFI that exists, how much did we detect?"

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Predicted binary mask. Will be converted to boolean.
    true : torch.Tensor or np.ndarray
        Ground truth binary mask. Will be converted to boolean.

    Returns
    -------
    float
        Recall in [0, 1].
        Returns 1.0 if no RFI in ground truth (perfect recall trivially).

    Notes
    -----
    If ground truth contains no RFI (all False), recall is defined as 1.0
    since there is no RFI to miss.

    Examples
    --------
    >>> pred = np.array([1, 1, 0, 0])
    >>> true = np.array([1, 0, 0, 1])
    >>> compute_recall(pred, true)  # 1 TP, 1 FN
    0.5

    >>> # No RFI to detect
    >>> compute_recall(np.ones(4), np.zeros(4))
    1.0
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    tp = np.logical_and(pred, true).sum()
    fn = np.logical_and(~pred, true).sum()

    if tp + fn == 0:
        return 1.0  # No RFI to detect = perfect recall

    return float(tp / (tp + fn))


def compute_f1(pred: ArrayLike, true: ArrayLike) -> float:
    """
    Compute F1 Score = 2 * (Precision * Recall) / (Precision + Recall).

    F1 is the harmonic mean of precision and recall, providing a balanced
    measure of detection performance.

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Predicted binary mask. Will be converted to boolean.
    true : torch.Tensor or np.ndarray
        Ground truth binary mask. Will be converted to boolean.

    Returns
    -------
    float
        F1 score in [0, 1].
        Returns 0.0 if both precision and recall are 0.

    Notes
    -----
    F1 score is equivalent to Dice coefficient for binary segmentation.
    Harmonic mean penalizes extreme values, requiring both precision
    and recall to be high for a good F1 score.

    Examples
    --------
    >>> pred = np.array([1, 1, 0, 0])
    >>> true = np.array([1, 0, 0, 1])
    >>> compute_f1(pred, true)  # Precision=0.5, Recall=0.5
    0.5

    >>> # Perfect detection
    >>> compute_f1(np.array([1, 0, 1]), np.array([1, 0, 1]))
    1.0
    """
    precision = compute_precision(pred, true)
    recall = compute_recall(pred, true)

    if precision + recall == 0:
        return 0.0

    return float(2 * (precision * recall) / (precision + recall))


def compute_dice(pred: ArrayLike, true: ArrayLike) -> float:
    """
    Compute Dice Coefficient = 2 * TP / (2 * TP + FP + FN).

    Dice coefficient measures overlap between masks. Equivalent to F1 score
    for binary segmentation. Commonly used in medical image segmentation.

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Predicted binary mask. Will be converted to boolean.
    true : torch.Tensor or np.ndarray
        Ground truth binary mask. Will be converted to boolean.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
        Returns 1.0 if both masks are empty (perfect agreement).

    Notes
    -----
    Dice coefficient is mathematically equivalent to F1 score for binary
    segmentation. It emphasizes regions where both masks agree.

    Examples
    --------
    >>> pred = np.array([[1, 1, 0], [0, 1, 0]])
    >>> true = np.array([[1, 0, 0], [0, 1, 1]])
    >>> compute_dice(pred, true)  # 2 TP, 2 FP, 1 FN
    0.5

    >>> # Both empty
    >>> compute_dice(np.zeros((2, 2)), np.zeros((2, 2)))
    1.0
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    if 2 * tp + fp + fn == 0:
        return 1.0  # Both masks empty = perfect agreement

    return float((2 * tp) / (2 * tp + fp + fn))


def evaluate_segmentation(pred: ArrayLike, true: ArrayLike) -> Dict[str, float]:
    """
    Compute all segmentation metrics at once.

    Convenience function that computes IoU, precision, recall, F1, and Dice
    coefficient in a single call.

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Predicted binary mask. Will be converted to boolean.
    true : torch.Tensor or np.ndarray
        Ground truth binary mask. Will be converted to boolean.

    Returns
    -------
    dict
        Dictionary with keys: 'iou', 'precision', 'recall', 'f1', 'dice'.
        All values are floats in [0, 1].

    Examples
    --------
    >>> pred = np.array([1, 1, 0, 0])
    >>> true = np.array([1, 0, 0, 1])
    >>> metrics = evaluate_segmentation(pred, true)
    >>> metrics['precision']
    0.5
    >>> metrics['recall']
    0.5
    >>> metrics['f1']
    0.5

    >>> # Perfect detection
    >>> pred = np.array([[1, 1], [0, 0]])
    >>> true = np.array([[1, 1], [0, 0]])
    >>> metrics = evaluate_segmentation(pred, true)
    >>> all(v == 1.0 for v in metrics.values())
    True
    """
    return {
        "iou": compute_iou(pred, true),
        "precision": compute_precision(pred, true),
        "recall": compute_recall(pred, true),
        "f1": compute_f1(pred, true),
        "dice": compute_dice(pred, true),
    }
