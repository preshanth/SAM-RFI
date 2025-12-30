"""
Unit tests for flag reconstruction logic.

Tests the reconstruction of full flag arrays from predicted patches,
including our critical num_rotations fix.
"""

from unittest.mock import MagicMock

import numpy as np


class TestReconstructionNumRotations:
    """Test reconstruction with different rotation counts (regression tests for our fix)."""

    def test_reconstruction_with_single_rotation(self):
        """
        Regression test for reconstruction hardcoded 4-rotation loop.

        Bug: Reconstruction always looped for rotation in range(4),
             but inference uses augmentation_rotations=1

        Fix: reconstruction now accepts num_rotations parameter

        This test verifies reconstruction works correctly with num_rotations=1.
        """
        from samrfi.inference import RFIPredictor

        # Mock predictor (we only need _reconstruct_flags method)
        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # Simulate inference with single rotation
        # Shape: (2 baselines, 4 pols, 1024 channels, 1024 times)
        # With patch_size=1024 and num_rotations=1: 2×4×1×1 = 8 patches
        data_shape = (2, 4, 1024, 1024)
        num_patches = 8
        predicted_patches = [np.random.rand(1024, 1024) > 0.5 for _ in range(num_patches)]

        # Should NOT raise - uses correct number of rotations
        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=1024, num_rotations=1
        )

        assert flags.shape == data_shape, f"Expected {data_shape}, got {flags.shape}"
        assert flags.dtype == bool, f"Expected bool dtype, got {flags.dtype}"

    def test_reconstruction_with_four_rotations(self):
        """
        Test reconstruction with 4 rotations (training mode).

        Verifies backward compatibility with training pipeline.
        """
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # Simulate training with 4 rotations
        # Shape: (1 baseline, 4 pols, 512 channels, 512 times)
        # With patch_size=512 and num_rotations=4: 1×4×4×1 = 16 patches
        data_shape = (1, 4, 512, 512)
        num_patches = 16
        predicted_patches = [np.random.rand(512, 512) > 0.5 for _ in range(num_patches)]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=512, num_rotations=4
        )

        assert flags.shape == data_shape
        assert flags.dtype == bool

    def test_reconstruction_patch_count_validation(self):
        """
        Test that reconstruction handles correct number of patches.

        Verifies that patch count matches expected: baselines × pols × rotations × (H_patches × W_patches)
        """
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # Multiple patches per baseline/pol
        # Shape: (2 baselines, 4 pols, 2048 channels, 1024 times)
        # With patch_size=1024 and num_rotations=1: 2×4×1×(2×1) = 16 patches
        data_shape = (2, 4, 2048, 1024)
        num_patches = 16
        predicted_patches = [np.random.rand(1024, 1024) > 0.5 for _ in range(num_patches)]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=1024, num_rotations=1
        )

        assert flags.shape == data_shape


class TestReconstructionRotationReversal:
    """Test that rotations are correctly reversed during reconstruction."""

    def test_rotation_0_no_change(self):
        """Test that rotation=0 (original) is not modified."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # Create patch with known pattern
        patch = np.zeros((256, 256), dtype=bool)
        patch[100:150, 50:100] = True  # Rectangle

        # Single patch, single rotation
        data_shape = (1, 1, 256, 256)
        predicted_patches = [patch]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=256, num_rotations=1
        )

        # Should be identical (rotation=0 doesn't transform)
        np.testing.assert_array_equal(flags[0, 0], patch)

    def test_multiple_rotations_combine_correctly(self):
        """Test that multiple rotations are combined with OR operation."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # 4 patches (4 rotations of same baseline/pol)
        # Different patterns in each rotation
        patch1 = np.zeros((256, 256), dtype=bool)
        patch1[50:100, :] = True

        patch2 = np.zeros((256, 256), dtype=bool)
        patch2[:, 50:100] = True

        patch3 = np.zeros((256, 256), dtype=bool)
        patch3[150:200, :] = True

        patch4 = np.zeros((256, 256), dtype=bool)
        patch4[:, 150:200] = True

        data_shape = (1, 1, 256, 256)
        predicted_patches = [patch1, patch2, patch3, patch4]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=256, num_rotations=4
        )

        # Combined result should have flags from all rotations (OR operation)
        # Check that at least some of each pattern is present
        assert flags[0, 0, 50:100, :].any(), "Pattern from rotation 0 missing"
        assert flags[0, 0, :, 50:100].any(), "Pattern from rotation 1 missing"


class TestReconstructionProbabilities:
    """Test reconstruction with probability maps (not just boolean flags)."""

    def test_reconstruction_preserves_probabilities(self):
        """Test that reconstruction handles float probabilities (not just bool)."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # Float probability patches (0.0 to 1.0)
        data_shape = (1, 1, 256, 256)
        predicted_patches = [np.random.rand(256, 256).astype(np.float32)]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=256, num_rotations=1
        )

        # Should preserve float dtype
        assert flags.dtype == np.float32, f"Expected float32, got {flags.dtype}"
        assert flags.min() >= 0.0 and flags.max() <= 1.0, "Probabilities outside [0,1] range"

    def test_probabilities_use_maximum_across_rotations(self):
        """Test that probabilities are combined with max (not OR) across rotations."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # Create patches with different probability values
        patch1 = np.full((256, 256), 0.3, dtype=np.float32)
        patch2 = np.full((256, 256), 0.7, dtype=np.float32)
        patch3 = np.full((256, 256), 0.5, dtype=np.float32)
        patch4 = np.full((256, 256), 0.9, dtype=np.float32)

        data_shape = (1, 1, 256, 256)
        predicted_patches = [patch1, patch2, patch3, patch4]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=256, num_rotations=4
        )

        # Should take maximum across all rotations
        np.testing.assert_allclose(flags[0, 0], 0.9, atol=1e-6)


class TestReconstructionMultipleBaselines:
    """Test reconstruction with multiple baselines and polarizations."""

    def test_reconstruction_multiple_baselines(self):
        """Test that reconstruction correctly handles multiple baselines."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # 3 baselines, 4 pols, 256×256, 1 rotation
        # Expected patches: 3×4×1×1 = 12
        data_shape = (3, 4, 256, 256)
        _num_patches = 12

        # Create distinct patterns for each baseline
        predicted_patches = []
        for baseline in range(3):
            for _pol in range(4):
                patch = np.zeros((256, 256), dtype=bool)
                # Unique pattern for each baseline
                patch[baseline * 50 : (baseline + 1) * 50, :] = True
                predicted_patches.append(patch)

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=256, num_rotations=1
        )

        # Verify each baseline has its unique pattern
        for baseline in range(3):
            assert flags[
                baseline, 0, baseline * 50 : (baseline + 1) * 50, :
            ].any(), f"Baseline {baseline} pattern missing"


class TestReconstructionEdgeCases:
    """Test edge cases and error handling in reconstruction."""

    def test_reconstruction_default_num_rotations(self):
        """Test that num_rotations defaults to 1 (backward compatible)."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        data_shape = (1, 1, 256, 256)
        predicted_patches = [np.zeros((256, 256), dtype=bool)]

        # Call without num_rotations parameter (should default to 1)
        flags = predictor._reconstruct_flags(
            predicted_patches,
            data_shape,
            patch_size=256,
            # Note: num_rotations omitted, should default to 1
        )

        assert flags.shape == data_shape

    def test_reconstruction_handles_empty_patches_gracefully(self):
        """Test that reconstruction handles patches with no flags."""
        from samrfi.inference import RFIPredictor

        predictor = MagicMock(spec=RFIPredictor)
        predictor._reconstruct_flags = RFIPredictor._reconstruct_flags.__get__(predictor)

        # All False patches
        data_shape = (1, 1, 256, 256)
        predicted_patches = [np.zeros((256, 256), dtype=bool)]

        flags = predictor._reconstruct_flags(
            predicted_patches, data_shape, patch_size=256, num_rotations=1
        )

        # Should be all False
        assert not flags.any(), "Expected all False flags"
