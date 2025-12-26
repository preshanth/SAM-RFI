"""
Unit tests for evaluation metrics (IoU, Precision, Recall, F1, Dice).

Tests the metrics module used for validation and comparison.
"""

import pytest
import numpy as np
import torch
from samrfi.evaluation import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_dice,
    evaluate_segmentation,
)


class TestMetricsBasic:
    """Test basic metric computations with simple cases."""

    def test_perfect_match(self):
        """Test metrics with perfect prediction (all correct)."""
        pred = np.array([1, 1, 0, 0], dtype=bool)
        gt = np.array([1, 1, 0, 0], dtype=bool)

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["iou"] == 1.0, "Perfect match should have IoU=1.0"
        assert metrics["precision"] == 1.0, "Perfect match should have Precision=1.0"
        assert metrics["recall"] == 1.0, "Perfect match should have Recall=1.0"
        assert metrics["f1"] == 1.0, "Perfect match should have F1=1.0"
        assert metrics["dice"] == 1.0, "Perfect match should have Dice=1.0"

    def test_no_overlap(self):
        """Test metrics with no overlap (worst case)."""
        pred = np.array([1, 1, 0, 0], dtype=bool)
        gt = np.array([0, 0, 1, 1], dtype=bool)

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["iou"] == 0.0, "No overlap should have IoU=0.0"
        assert metrics["precision"] == 0.0, "No overlap should have Precision=0.0"
        assert metrics["recall"] == 0.0, "No overlap should have Recall=0.0"
        assert metrics["f1"] == 0.0, "No overlap should have F1=0.0"
        assert metrics["dice"] == 0.0, "No overlap should have Dice=0.0"

    def test_partial_overlap(self):
        """Test metrics with partial overlap."""
        # TP=1, FP=1, FN=1
        pred = np.array([1, 1, 0], dtype=bool)
        gt = np.array([1, 0, 1], dtype=bool)

        iou = compute_iou(pred, gt)
        precision = compute_precision(pred, gt)
        recall = compute_recall(pred, gt)

        # IoU = TP / (TP + FP + FN) = 1 / (1+1+1) = 1/3
        assert np.isclose(iou, 1/3), f"Expected IoU=1/3, got {iou}"

        # Precision = TP / (TP + FP) = 1 / (1+1) = 0.5
        assert np.isclose(precision, 0.5), f"Expected Precision=0.5, got {precision}"

        # Recall = TP / (TP + FN) = 1 / (1+1) = 0.5
        assert np.isclose(recall, 0.5), f"Expected Recall=0.5, got {recall}"


class TestMetricsEdgeCases:
    """Test edge cases and special conditions."""

    def test_all_zeros_prediction(self):
        """Test when prediction has no positive labels."""
        pred = np.zeros(10, dtype=bool)
        gt = np.ones(10, dtype=bool)

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["precision"] == 0.0, "No predictions should have Precision=0"
        assert metrics["recall"] == 0.0, "No TP should have Recall=0"
        assert metrics["iou"] == 0.0, "No TP should have IoU=0"

    def test_all_zeros_ground_truth(self):
        """Test when ground truth has no positive labels."""
        pred = np.ones(10, dtype=bool)
        gt = np.zeros(10, dtype=bool)

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["precision"] == 0.0, "All FP should have Precision=0"
        assert metrics["recall"] == 1.0 or np.isnan(metrics["recall"]), "No positives in GT"
        assert metrics["iou"] == 0.0, "No TP should have IoU=0"

    def test_both_empty(self):
        """Test when both prediction and ground truth are empty."""
        pred = np.zeros(10, dtype=bool)
        gt = np.zeros(10, dtype=bool)

        metrics = evaluate_segmentation(pred, gt)

        # Perfect agreement on "no RFI"
        assert metrics["iou"] == 1.0, "Both empty should be perfect match"
        assert metrics["precision"] == 1.0, "Both empty should have Precision=1"
        assert metrics["recall"] == 1.0, "Both empty should have Recall=1"


class TestMetricsWithTorch:
    """Test that metrics work with torch tensors."""

    def test_torch_tensor_input(self):
        """Test that metrics accept torch tensors."""
        pred = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
        gt = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["iou"] == 1.0
        assert metrics["f1"] == 1.0

    def test_torch_float_tensor(self):
        """Test that metrics handle float tensors (convert to bool)."""
        pred = torch.tensor([0.9, 0.8, 0.1, 0.2], dtype=torch.float32)
        gt = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

        # Should threshold at 0.5
        metrics = evaluate_segmentation(pred > 0.5, gt > 0.5)

        assert metrics["iou"] == 1.0


class TestMetricsMultidimensional:
    """Test metrics with 2D and 3D arrays (waterfalls)."""

    def test_2d_waterfall(self):
        """Test metrics on 2D waterfall (channels × times)."""
        # Create 2D RFI pattern
        pred = np.zeros((256, 256), dtype=bool)
        pred[100:150, :] = True  # Narrowband RFI

        gt = np.zeros((256, 256), dtype=bool)
        gt[100:150, :] = True  # Same pattern

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["iou"] == 1.0, "Identical 2D patterns should have IoU=1"

    def test_3d_multi_pol(self):
        """Test metrics on 3D data (pols × channels × times)."""
        # 4 polarizations
        pred = np.zeros((4, 256, 256), dtype=bool)
        pred[:, 100:150, :] = True

        gt = np.zeros((4, 256, 256), dtype=bool)
        gt[:, 100:150, :] = True

        metrics = evaluate_segmentation(pred, gt)

        assert metrics["iou"] == 1.0, "Identical 3D patterns should have IoU=1"


class TestF1DiceEquivalence:
    """Test that F1 and Dice scores are equivalent."""

    def test_f1_equals_dice(self):
        """F1 and Dice should be mathematically equivalent."""
        # Random prediction and ground truth
        np.random.seed(42)
        pred = np.random.rand(100) > 0.5
        gt = np.random.rand(100) > 0.5

        f1 = compute_f1(pred, gt)
        dice = compute_dice(pred, gt)

        np.testing.assert_allclose(f1, dice, atol=1e-10,
                                   err_msg="F1 and Dice should be equivalent")


class TestMetricsRealWorldScenarios:
    """Test metrics on realistic RFI detection scenarios."""

    def test_aoflagger_overflagging(self):
        """Simulate AOFLAGGER overflagging (high recall, low precision)."""
        # Ground truth: narrow RFI
        gt = np.zeros(1000, dtype=bool)
        gt[400:450] = True  # 50 RFI samples

        # Prediction: overflagging
        pred = np.zeros(1000, dtype=bool)
        pred[350:500] = True  # 150 flagged (includes GT + extra)

        metrics = evaluate_segmentation(pred, gt)

        # Should have:
        # - High recall (caught all real RFI)
        # - Low precision (many false positives)
        assert metrics["recall"] >= 0.9, "Should catch most RFI"
        assert metrics["precision"] < 0.5, "Should have low precision (overflagging)"

    def test_conservative_flagging(self):
        """Simulate conservative flagging (high precision, low recall)."""
        # Ground truth: broad RFI
        gt = np.zeros(1000, dtype=bool)
        gt[300:600] = True  # 300 RFI samples

        # Prediction: conservative
        pred = np.zeros(1000, dtype=bool)
        pred[400:500] = True  # 100 flagged (only obvious RFI)

        metrics = evaluate_segmentation(pred, gt)

        # Should have:
        # - Low recall (missed some RFI)
        # - High precision (what we flagged was correct)
        assert metrics["recall"] < 0.5, "Should miss some RFI"
        assert metrics["precision"] >= 0.9, "What's flagged should be correct"


class TestMetricsBenchmark:
    """Benchmark expected performance against literature."""

    def test_samrfi_target_performance(self):
        """Verify metrics are in SAM-RFI expected range (F1 > 0.75)."""
        # Simulate good SAM-RFI performance
        gt = np.zeros(10000, dtype=bool)
        gt[3000:4000] = True  # 1000 RFI samples

        # Good prediction: 900 correct, 50 FP, 100 missed
        pred = np.zeros(10000, dtype=bool)
        pred[3000:3900] = True  # 900 TP
        pred[5000:5050] = True  # 50 FP

        metrics = evaluate_segmentation(pred, gt)

        # Expected: Precision = 900/(900+50) = 0.947
        #           Recall = 900/1000 = 0.9
        #           F1 = 2 * 0.947 * 0.9 / (0.947 + 0.9) ≈ 0.923

        assert metrics["f1"] > 0.75, f"SAM-RFI should achieve F1 > 0.75, got {metrics['f1']}"
        assert metrics["precision"] > 0.8, "SAM-RFI should have high precision"
        assert metrics["recall"] > 0.8, "SAM-RFI should have high recall"
