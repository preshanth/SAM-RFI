"""
Unit tests for Preprocessor (patchification, feature extraction, normalization).

Tests the preprocessing pipeline including our critical patchification fix.
"""

import pytest
import numpy as np
import torch
from samrfi.data import Preprocessor


class TestPatchification:
    """Test patchification logic, including regression tests for our fixes."""

    def test_patchification_2048x1024_with_patch_1024(self, synthetic_waterfall_large):
        """
        Regression test for patchification bug.

        Bug: Shape (2048, 1024) with patch_size=1024 incorrectly skipped patchification
             due to condition: if patch_size >= min(waterfall_shape)

        Fix: Changed to: if waterfall_shape[0] <= patch_size and waterfall_shape[1] <= patch_size

        This test verifies that (2048, 1024) with patch_size=1024 creates 2 patches.
        """
        # Create data: (1 baseline, 4 pols, 2048 channels, 1024 times)
        data = synthetic_waterfall_large[np.newaxis, ...]  # Add baseline dimension

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=1024,
            enable_augmentation=False,  # 1 rotation (inference mode)
            inference_mode=True,  # Skip blank removal, shuffling
        )

        # Expected: 1 baseline × 4 pols × 1 rotation × (2×1 patches) = 8 patches
        assert len(dataset) == 8, f"Expected 8 patches, got {len(dataset)}"

        # Verify patch shape
        sample = dataset[0]
        assert sample["images"].shape == (1024, 1024, 3), "Patch should be 1024×1024×3"

    def test_patchification_1024x1024_no_split(self, synthetic_waterfall_medium):
        """
        Test that 1024×1024 with patch_size=1024 creates 1 patch (no splitting).

        This is the boundary case - waterfall exactly matches patch size.
        """
        data = synthetic_waterfall_medium[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=1024,
            enable_augmentation=False,
            inference_mode=True,
        )

        # Expected: 1 baseline × 4 pols × 1 rotation × (1×1 patches) = 4 patches
        assert len(dataset) == 4, f"Expected 4 patches, got {len(dataset)}"

    def test_patchification_256x256_creates_patches(self, synthetic_waterfall_medium):
        """
        Test that 1024×1024 with patch_size=256 creates 16 patches.
        """
        data = synthetic_waterfall_medium[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        # Expected: 1 baseline × 4 pols × 1 rotation × (4×4 patches) = 64 patches
        assert len(dataset) == 64, f"Expected 64 patches, got {len(dataset)}"

    def test_patchification_with_augmentation(self, synthetic_waterfall_medium):
        """
        Test that augmentation creates 4× more patches.
        """
        data = synthetic_waterfall_medium[np.newaxis, ...]

        # Without augmentation
        preprocessor_no_aug = Preprocessor(data, flags=None)
        dataset_no_aug = preprocessor_no_aug.create_dataset(
            patch_size=512,
            enable_augmentation=False,
            augmentation_rotations=1,
            inference_mode=True,
        )

        # With augmentation
        preprocessor_aug = Preprocessor(data, flags=None)
        dataset_aug = preprocessor_aug.create_dataset(
            patch_size=512,
            enable_augmentation=True,
            augmentation_rotations=4,
            inference_mode=False,  # Training mode (allows shuffling)
        )

        # Expected: 4× more patches with 4-way augmentation
        assert len(dataset_aug) == 4 * len(dataset_no_aug), \
            f"Augmented dataset should be 4× larger. Got {len(dataset_aug)} vs {len(dataset_no_aug)}"


class TestFeatureExtraction:
    """Test 3-channel feature extraction from complex data."""

    def test_complex_to_3channel(self, synthetic_waterfall_small):
        """Test that complex data is converted to 3-channel (gradient, log_amp, phase)."""
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        sample = dataset[0]
        images = sample["images"]

        # Check shape: (H, W, 3) channels
        assert images.shape == (256, 256, 3), f"Expected (256, 256, 3), got {images.shape}"

        # Check dtype
        assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"

        # Check channels are in valid range (after ImageNet normalization)
        # ImageNet norm: (x - mean) / std, so values can be negative
        assert not np.isnan(images.numpy()).any(), "NaN values in image channels"
        assert not np.isinf(images.numpy()).any(), "Inf values in image channels"

    def test_channel_extraction_preserves_shape(self, synthetic_waterfall_small):
        """Test that channel extraction preserves spatial dimensions."""
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=128,
            enable_augmentation=False,
            inference_mode=True,
        )

        # All patches should have same shape
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert sample["images"].shape == (128, 128, 3), \
                f"Patch {i} has incorrect shape: {sample['images'].shape}"


class TestNormalization:
    """Test normalization options."""

    def test_sam2_normalization_applied(self, synthetic_waterfall_small):
        """Test that SAM2 ImageNet normalization is applied."""
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        sample = dataset[0]
        images = sample["images"]

        # After ImageNet normalization: (pixel - mean) / std
        # Mean ~= [0.485, 0.456, 0.406], Std ~= [0.229, 0.224, 0.225]
        # So normalized values typically in range [-3, 3] (most data within 3 stds)

        # Check reasonable range (allow outliers)
        assert images.min() > -10, f"Values too negative: {images.min()}"
        assert images.max() < 10, f"Values too positive: {images.max()}"


class TestMetadata:
    """Test that preprocessing metadata is correctly stored."""

    def test_metadata_stored_in_dataset(self, synthetic_waterfall_small):
        """Test that preprocessing parameters are stored in dataset metadata."""
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            stretch=None,
            flag_sigma=5,
            enable_augmentation=False,
            augmentation_rotations=1,
            normalize_before_stretch=False,
            normalize_after_stretch=False,
        )

        # Check metadata
        assert hasattr(dataset, "metadata"), "Dataset should have metadata attribute"

        metadata = dataset.metadata
        assert metadata["patch_size"] == 256
        assert metadata["stretch"] is None
        assert metadata["flag_sigma"] == 5
        assert metadata["augmentation_rotations"] == 1
        assert metadata["normalize_before_stretch"] is False
        assert metadata["normalize_after_stretch"] is False

    def test_metadata_includes_augmentation_rotations(self, synthetic_waterfall_small):
        """Regression test: metadata MUST include augmentation_rotations for reconstruction."""
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=True,
            augmentation_rotations=4,
        )

        # Critical: augmentation_rotations must be in metadata
        assert "augmentation_rotations" in dataset.metadata, \
            "augmentation_rotations missing from metadata (needed for reconstruction)"

        assert dataset.metadata["augmentation_rotations"] == 4


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_preprocessor_handles_3d_input(self, synthetic_waterfall_small):
        """Test that preprocessor accepts 3D input (pols, channels, times)."""
        # No baseline dimension
        data = synthetic_waterfall_small

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        # Should work - preprocessor adds baseline dimension internally
        assert len(dataset) == 4  # 4 pols, no augmentation

    def test_preprocessor_handles_4d_input(self, synthetic_waterfall_small):
        """Test that preprocessor accepts 4D input (baselines, pols, channels, times)."""
        # Add baseline dimension
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        assert len(dataset) == 4  # 1 baseline × 4 pols
