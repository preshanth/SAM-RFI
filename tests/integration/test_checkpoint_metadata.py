"""
Integration tests for checkpoint metadata validation.

Tests that preprocessing config is correctly saved during training
and validated during inference.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestCheckpointMetadataSave:
    """Test that training saves preprocessing metadata to checkpoints."""

    def test_checkpoint_contains_preprocessing_metadata(self, mock_checkpoint):
        """Test that checkpoint file contains preprocessing metadata."""
        checkpoint = torch.load(mock_checkpoint)

        # Should have preprocessing field
        assert "preprocessing" in checkpoint, "Checkpoint missing preprocessing metadata"

        # Should contain all required fields
        required_fields = [
            "patch_size",
            "augmentation_rotations",
            "stretch",
            "normalize_before_stretch",
            "normalize_after_stretch",
        ]

        for field in required_fields:
            assert field in checkpoint["preprocessing"], \
                f"Preprocessing metadata missing field: {field}"

    def test_checkpoint_backward_compatible(self, mock_checkpoint):
        """Test that checkpoint maintains backward compatibility (patch_size at top level)."""
        checkpoint = torch.load(mock_checkpoint)

        # Should have both old and new format
        assert "patch_size" in checkpoint, "Backward compatibility: patch_size at top level"
        assert "preprocessing" in checkpoint, "New format: preprocessing dict"

        # Values should match
        assert checkpoint["patch_size"] == checkpoint["preprocessing"]["patch_size"]


class TestCheckpointMetadataValidation:
    """Test that inference validates preprocessing parameters against checkpoint."""

    def test_validation_raises_on_patch_size_mismatch(self, mock_checkpoint):
        """Test that mismatched patch_size raises ValueError."""
        from samrfi.inference import RFIPredictor

        # This should raise - checkpoint has patch_size=1024
        with pytest.raises(ValueError, match="Patch size mismatch"):
            predictor = RFIPredictor(
                model_path=mock_checkpoint,
                sam_checkpoint="tiny",
                device="cpu",
            )
            # Call validation with wrong patch_size
            predictor._validate_preprocessing_params(
                patch_size=128,  # Mismatch!
                stretch=None,
            )

    def test_validation_warns_on_stretch_mismatch(self, mock_checkpoint, capsys):
        """Test that mismatched stretch function prints warning."""
        from samrfi.inference import RFIPredictor

        predictor = RFIPredictor(
            model_path=mock_checkpoint,
            sam_checkpoint="tiny",
            device="cpu",
        )

        # Should print warning (not raise)
        predictor._validate_preprocessing_params(
            patch_size=1024,  # Matches
            stretch="SQRT",   # Mismatch (checkpoint has None)
        )

        # Check warning was printed
        captured = capsys.readouterr()
        assert "WARNING" in captured.out, "Should print warning for stretch mismatch"
        assert "stretch" in captured.out.lower()

    def test_validation_succeeds_on_match(self, mock_checkpoint):
        """Test that matching parameters pass validation."""
        from samrfi.inference import RFIPredictor

        predictor = RFIPredictor(
            model_path=mock_checkpoint,
            sam_checkpoint="tiny",
            device="cpu",
        )

        # Should not raise - parameters match checkpoint
        predictor._validate_preprocessing_params(
            patch_size=1024,
            stretch=None,
            normalize_before_stretch=False,
            normalize_after_stretch=False,
        )

    def test_validation_skips_old_checkpoints(self, tmp_path):
        """Test that validation gracefully handles old checkpoints without preprocessing metadata."""
        from samrfi.inference import RFIPredictor

        # Create old-style checkpoint (no preprocessing field)
        old_checkpoint = {
            "model_state_dict": {},
            "patch_size": 1024,
            "config": {
                "sam_checkpoint": "tiny",
            },
        }

        old_checkpoint_path = tmp_path / "old_checkpoint.pth"
        torch.save(old_checkpoint, old_checkpoint_path)

        predictor = RFIPredictor(
            model_path=old_checkpoint_path,
            sam_checkpoint="tiny",
            device="cpu",
        )

        # Should not raise - validation skipped for old checkpoints
        predictor._validate_preprocessing_params(
            patch_size=128,  # Different from checkpoint, but validation skipped
            stretch="SQRT",
        )


class TestCheckpointMetadataInference:
    """Test that validation is called during actual inference."""

    def test_predict_ms_validates_parameters(self, mock_checkpoint, tmp_path, monkeypatch):
        """Test that predict_ms calls validation before inference."""
        from samrfi.inference import RFIPredictor

        predictor = RFIPredictor(
            model_path=mock_checkpoint,
            sam_checkpoint="tiny",
            device="cpu",
        )

        # Track if validation was called
        validation_called = {"called": False}

        original_validate = predictor._validate_preprocessing_params

        def mock_validate(*args, **kwargs):
            validation_called["called"] = True
            return original_validate(*args, **kwargs)

        monkeypatch.setattr(predictor, "_validate_preprocessing_params", mock_validate)

        # Mock MSLoader to avoid needing real MS file
        class MockMSLoader:
            def __init__(self, *args):
                self.data = np.random.randn(1, 4, 1024, 1024) + 1j * np.random.randn(1, 4, 1024, 1024)
                self.magnitude = np.abs(self.data)

            def load(self, *args, **kwargs):
                pass

            def save_flags(self, *args):
                pass

        monkeypatch.setattr("samrfi.inference.predictor.MSLoader", MockMSLoader)

        # Attempt inference (will fail at SAM2 model, but validation should run)
        try:
            predictor.predict_ms(
                ms_path="dummy.ms",
                patch_size=1024,
                stretch=None,
                save_flags=False,
            )
        except Exception:
            pass  # Expected to fail (no real model), but validation should have run

        assert validation_called["called"], "Validation should be called during predict_ms"


class TestCheckpointMetadataDisplay:
    """Test that checkpoint metadata is displayed to user."""

    def test_checkpoint_info_displayed(self, mock_checkpoint, capsys):
        """Test that loading checkpoint prints preprocessing info."""
        from samrfi.inference import RFIPredictor

        predictor = RFIPredictor(
            model_path=mock_checkpoint,
            sam_checkpoint="tiny",
            device="cpu",
        )

        # Check that preprocessing config was printed
        captured = capsys.readouterr()
        assert "preprocessing config" in captured.out.lower() or "Checkpoint" in captured.out
        # Should display at least patch_size
        assert "patch_size" in captured.out or "1024" in captured.out
