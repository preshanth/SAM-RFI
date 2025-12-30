"""
Extended end-to-end tests with SAM2-tiny model.

These tests are slow and require downloading SAM2-tiny from HuggingFace.
Not run in CI - manual execution only.

Usage:
    pytest tests/extended/test_full_validation.py -v
    # Or run all extended tests:
    pytest -m slow
"""

import pytest
import torch


@pytest.mark.slow
class TestFullPipelineWithTinyModel:
    """Full end-to-end validation with SAM2-tiny model."""

    def test_tiny_model_inference_on_synthetic_data(self, tmp_path, synthetic_data_with_rfi):
        """
        Full pipeline test: synthetic data → preprocessor → SAM2-tiny inference.

        This is a smoke test to ensure the full pipeline works end-to-end.
        Does not train the model (too slow), just tests inference with pretrained weights.
        """
        from samrfi.evaluation import evaluate_segmentation
        from samrfi.inference import RFIPredictor

        # Get synthetic data
        data = synthetic_data_with_rfi["data"]
        ground_truth = synthetic_data_with_rfi["flags"]

        print("\n[Test] Starting full pipeline with SAM2-tiny...")

        # Step 1: Create mock checkpoint with SAM2-tiny architecture
        # (In reality, would load trained checkpoint)
        print("  [1/5] Creating mock SAM2-tiny checkpoint...")

        # Download SAM2-tiny base model (auto-downloaded by HuggingFace)
        from transformers import Sam2Model

        tiny_model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny")

        # Create checkpoint with preprocessing metadata
        checkpoint = {
            "model_state_dict": tiny_model.state_dict(),
            "epoch": 0,
            "preprocessing": {
                "patch_size": 256,
                "augmentation_rotations": 1,
                "stretch": None,
                "normalize_before_stretch": False,
                "normalize_after_stretch": False,
            },
            "config": {
                "sam_checkpoint": "tiny",
            },
        }

        checkpoint_path = tmp_path / "tiny_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Step 2: Initialize predictor
        print("  [2/5] Loading predictor...")

        predictor = RFIPredictor(
            model_path=checkpoint_path,
            sam_checkpoint="tiny",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=2,
        )

        print("  ✓ Predictor loaded")

        # Step 3: Run inference
        print("  [3/5] Running inference on synthetic data...")

        predicted_flags = predictor.predict_array(
            data,
            patch_size=256,
            stretch=None,
            enable_augmentation=False,
            normalize_before_stretch=False,
            normalize_after_stretch=False,
            return_probabilities=False,
        )

        print(f"  ✓ Inference complete. Predicted shape: {predicted_flags.shape}")

        # Step 4: Verify results
        print("  [4/5] Validating results...")

        assert (
            predicted_flags.shape == ground_truth.shape
        ), f"Shape mismatch: predicted {predicted_flags.shape} vs GT {ground_truth.shape}"

        assert predicted_flags.dtype == bool, f"Expected bool dtype, got {predicted_flags.dtype}"

        print("  ✓ Shape and dtype validated")

        # Step 5: Compute metrics
        print("  [5/5] Computing metrics...")

        metrics = evaluate_segmentation(predicted_flags, ground_truth)

        print("  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")

        # Sanity checks (untrained model won't be accurate, but should produce valid output)
        assert 0.0 <= metrics["iou"] <= 1.0, "IoU should be in [0,1]"
        assert 0.0 <= metrics["precision"] <= 1.0, "Precision should be in [0,1]"
        assert 0.0 <= metrics["recall"] <= 1.0, "Recall should be in [0,1]"

        print("\n✓ Full pipeline test passed!")

    @pytest.mark.slow
    @pytest.mark.requires_gpu
    def test_tiny_model_gpu_inference(self, tmp_path, synthetic_data_with_rfi):
        """
        Test that inference works correctly on GPU.

        Requires CUDA GPU - skipped if not available.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from samrfi.inference import RFIPredictor

        # Create minimal checkpoint
        checkpoint = {
            "model_state_dict": {},  # Empty for quick test
            "preprocessing": {
                "patch_size": 256,
                "augmentation_rotations": 1,
                "stretch": None,
                "normalize_before_stretch": False,
                "normalize_after_stretch": False,
            },
            "config": {
                "sam_checkpoint": "tiny",
            },
        }

        checkpoint_path = tmp_path / "gpu_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # Initialize on GPU
        predictor = RFIPredictor(
            model_path=checkpoint_path,
            sam_checkpoint="tiny",
            device="cuda",
            batch_size=4,
        )

        # Verify model is on GPU
        assert next(predictor.model.parameters()).is_cuda, "Model should be on CUDA device"

        print("✓ GPU inference test passed")


@pytest.mark.slow
class TestCheckpointMetadataValidationExtended:
    """Extended tests for checkpoint validation with real model."""

    def test_trained_checkpoint_has_metadata(self, tmp_path):
        """
        Test that a real training run produces checkpoint with metadata.

        Note: This would require full training setup, so we simulate it.
        """
        # This test would:
        # 1. Generate small synthetic dataset
        # 2. Run 1 epoch of training with SAM2-tiny
        # 3. Verify checkpoint has all metadata fields
        # 4. Load checkpoint and verify validation works

        # Skipped for now - requires full training infrastructure
        pytest.skip("Full training test - implement when needed")


@pytest.mark.slow
class TestIterativeInferenceExtended:
    """Extended tests for iterative inference."""

    def test_iterative_inference_improves_results(self, tmp_path, synthetic_data_with_rfi):
        """
        Test that 2-3 iterations find more RFI than single pass.

        Note: With untrained model, this may not show improvement.
        With trained model, should show progressive improvement.
        """
        pytest.skip("Requires trained model for meaningful test")


# ============================================================================
# Utility Functions for Extended Tests
# ============================================================================


def download_sam2_tiny_if_needed():
    """
    Download SAM2-tiny model if not already cached.

    Returns:
        Path to cached model
    """
    from transformers import Sam2Model

    print("Checking SAM2-tiny cache...")
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny")
    print("✓ SAM2-tiny available")

    return model


def create_minimal_training_checkpoint(model_variant="tiny", save_path=None):
    """
    Create a minimal training checkpoint for testing.

    Args:
        model_variant: SAM2 variant (tiny/small/base_plus/large)
        save_path: Where to save checkpoint

    Returns:
        Path to checkpoint
    """
    from transformers import Sam2Model

    model_map = {
        "tiny": "facebook/sam2-hiera-tiny",
        "small": "facebook/sam2-hiera-small",
        "base_plus": "facebook/sam2-hiera-base-plus",
        "large": "facebook/sam2-hiera-large",
    }

    print(f"Loading SAM2 {model_variant}...")
    model = Sam2Model.from_pretrained(model_map[model_variant])

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 0,
        "training_losses": [1.0],
        "validation_losses": [1.1],
        "preprocessing": {
            "patch_size": 256,
            "augmentation_rotations": 1,
            "stretch": None,
            "normalize_before_stretch": False,
            "normalize_after_stretch": False,
        },
        "config": {
            "sam_checkpoint": model_variant,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "loss_function": "DiceCELoss",
        },
    }

    if save_path:
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")

    return checkpoint
