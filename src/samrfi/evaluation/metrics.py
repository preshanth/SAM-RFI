"""
Segmentation metrics for RFI detection evaluation

Standard binary segmentation metrics comparing predicted masks vs ground truth.
Accepts both torch tensors and numpy arrays (converts to numpy internally).
"""

import numpy as np
import torch


def _to_numpy(arr):
    """Convert torch tensor or numpy array to numpy array"""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_iou(pred, true):
    """
    Intersection over Union (IoU) / Jaccard Index

    Args:
        pred: Predicted binary mask (torch.Tensor or numpy array)
        true: Ground truth binary mask (torch.Tensor or numpy array)

    Returns:
        float: IoU score in [0, 1], or 1.0 if both masks are empty
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0  # Both masks empty = perfect agreement

    return intersection / union


def compute_precision(pred, true):
    """
    Precision = TP / (TP + FP)

    What fraction of predicted RFI is actually RFI?

    Args:
        pred: Predicted binary mask (torch.Tensor or numpy array)
        true: Ground truth binary mask (torch.Tensor or numpy array)

    Returns:
        float: Precision in [0, 1]
               Returns 1.0 if no predictions on clean data (correct abstention)
               Returns 0.0 if no predictions on RFI data (failure to detect)
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

    return tp / (tp + fp)


def compute_recall(pred, true):
    """
    Recall = TP / (TP + FN) = Sensitivity = True Positive Rate

    What fraction of actual RFI is detected?

    Args:
        pred: Predicted binary mask (torch.Tensor or numpy array)
        true: Ground truth binary mask (torch.Tensor or numpy array)

    Returns:
        float: Recall in [0, 1], or 1.0 if no RFI in ground truth
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    tp = np.logical_and(pred, true).sum()
    fn = np.logical_and(~pred, true).sum()

    if tp + fn == 0:
        return 1.0  # No RFI to detect = perfect recall

    return tp / (tp + fn)


def compute_f1(pred, true):
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

    Harmonic mean of precision and recall.

    Args:
        pred: Predicted binary mask (torch.Tensor or numpy array)
        true: Ground truth binary mask (torch.Tensor or numpy array)

    Returns:
        float: F1 score in [0, 1]
    """
    precision = compute_precision(pred, true)
    recall = compute_recall(pred, true)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_dice(pred, true):
    """
    Dice Coefficient = 2 * TP / (2 * TP + FP + FN)

    Equivalent to F1 score for binary segmentation.

    Args:
        pred: Predicted binary mask (torch.Tensor or numpy array)
        true: Ground truth binary mask (torch.Tensor or numpy array)

    Returns:
        float: Dice coefficient in [0, 1]
    """
    pred = _to_numpy(pred).astype(bool)
    true = _to_numpy(true).astype(bool)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    if 2 * tp + fp + fn == 0:
        return 1.0  # Both masks empty = perfect agreement

    return (2 * tp) / (2 * tp + fp + fn)


def evaluate_segmentation(pred, true):
    """
    Compute all segmentation metrics at once.

    Args:
        pred: Predicted binary mask (torch.Tensor or numpy array)
        true: Ground truth binary mask (torch.Tensor or numpy array)

    Returns:
        dict: Dictionary with keys: 'iou', 'precision', 'recall', 'f1', 'dice'
    """
    return {
        "iou": compute_iou(pred, true),
        "precision": compute_precision(pred, true),
        "recall": compute_recall(pred, true),
        "f1": compute_f1(pred, true),
        "dice": compute_dice(pred, true),
    }
