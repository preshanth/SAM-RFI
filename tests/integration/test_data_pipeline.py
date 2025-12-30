"""
Integration tests for data pipeline (end-to-end preprocessing).

Tests the full pipeline: data → preprocessor → dataset → predictions.
"""

import numpy as np
import torch


class TestPreprocessingPipeline:
    """Test full preprocessing pipeline."""

    def test_complex_data_to_dataset(self, synthetic_data_with_rfi):
        """Test complete pipeline from complex data to TorchDataset."""
        from samrfi.data import Preprocessor

        data = synthetic_data_with_rfi["data"]
        flags = synthetic_data_with_rfi["flags"]

        preprocessor = Preprocessor(data, flags=flags)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            stretch=None,
            enable_augmentation=False,
            use_custom_flags=True,
            inference_mode=True,
        )

        # Verify dataset created
        assert len(dataset) > 0, "Dataset should have samples"

        # Verify sample format
        sample = dataset[0]
        assert "image" in sample, "Sample missing image"
        assert "label" in sample, "Sample missing label"

        # Verify shapes
        assert sample["image"].shape == (256, 256, 3), "Image shape incorrect"
        assert sample["label"].shape == (256, 256), "Label shape incorrect"

        # Verify metadata preserved
        assert hasattr(dataset, "metadata"), "Dataset should have metadata"
        assert dataset.metadata["patch_size"] == 256
        assert dataset.metadata["augmentation_rotations"] == 4

    def test_preprocessing_metadata_consistency(self, synthetic_waterfall_small):
        """Test that metadata remains consistent through preprocessing."""
        from samrfi.data import Preprocessor

        data = synthetic_waterfall_small[np.newaxis, ...]

        config = {
            "patch_size": 256,
            "stretch": None,
            "flag_sigma": 5,
            "enable_augmentation": False,
            "augmentation_rotations": 1,
            "normalize_before_stretch": False,
            "normalize_after_stretch": False,
        }

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(**config)

        # Check that critical config values are in metadata
        # Note: enable_augmentation is not saved (implicit from augmentation_rotations)
        expected_metadata = {
            "patch_size": 256,
            "stretch": None,
            "flag_sigma": 5,
            "augmentation_rotations": 1,
            "normalize_before_stretch": False,
            "normalize_after_stretch": False,
        }
        for key, value in expected_metadata.items():
            assert (
                dataset.metadata[key] == value
            ), f"Metadata mismatch: {key} = {dataset.metadata[key]}, expected {value}"


class TestInferencePipelineIntegration:
    """Test integration of preprocessing with inference."""

    def test_metadata_flows_to_reconstruction(self, synthetic_data_with_rfi):
        """Test that metadata flows from preprocessing to reconstruction."""
        from samrfi.data import Preprocessor

        data = synthetic_data_with_rfi["data"]

        # Preprocess with specific augmentation_rotations
        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            augmentation_rotations=2,  # Non-standard value
            enable_augmentation=True,
            inference_mode=False,
        )

        # Verify metadata has correct value
        assert (
            dataset.metadata["augmentation_rotations"] == 2
        ), "Augmentation rotations should be stored in metadata"

        # This metadata should be used during reconstruction
        # (tested separately in unit tests)

    def test_end_to_end_inference_metadata(self, mock_torch_dataset, tmp_path):
        """Test that metadata is preserved end-to-end in inference."""

        # Dataset has metadata
        assert hasattr(mock_torch_dataset, "metadata")
        assert mock_torch_dataset.metadata["augmentation_rotations"] == 1

        # Save and reload
        save_path = tmp_path / "test_dataset.pt"
        mock_torch_dataset.save_to_disk(save_path)

        # Load and verify metadata preserved
        loaded = torch.load(save_path)
        assert "metadata" in loaded
        assert loaded["metadata"]["augmentation_rotations"] == 1


class TestPipelineRobustness:
    """Test pipeline handles edge cases robustly."""

    def test_pipeline_handles_single_baseline(self, synthetic_waterfall_small):
        """Test pipeline works with single baseline."""
        from samrfi.data import Preprocessor

        # Single baseline
        data = synthetic_waterfall_small[np.newaxis, ...]

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        assert len(dataset) == 4, "Should have 4 patches (4 pols, 1 baseline, no augmentation)"

    def test_pipeline_handles_multiple_baselines(self, synthetic_waterfall_small):
        """Test pipeline works with multiple baselines."""
        from samrfi.data import Preprocessor

        # 3 baselines
        data = np.stack([synthetic_waterfall_small] * 3)

        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        assert len(dataset) == 12, "Should have 12 patches (4 pols × 3 baselines)"

    def test_pipeline_preserves_data_integrity(self, synthetic_data_with_rfi):
        """Test that preprocessing doesn't corrupt data."""
        from samrfi.data import Preprocessor

        data = synthetic_data_with_rfi["data"]
        original_data = data.copy()

        preprocessor = Preprocessor(data, flags=None)
        _dataset = preprocessor.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,
        )

        # Original data should be unchanged
        np.testing.assert_array_equal(
            data, original_data, err_msg="Preprocessing should not modify input data"
        )


class TestAugmentationConsistency:
    """Test that augmentation is applied consistently."""

    def test_augmentation_matches_num_rotations(self, synthetic_waterfall_small):
        """Test that enabling augmentation creates expected number of patches."""
        from samrfi.data import Preprocessor

        data = synthetic_waterfall_small[np.newaxis, ...]

        # Test different rotation counts
        for num_rotations in [1, 2, 4]:
            preprocessor = Preprocessor(data, flags=None)
            dataset = preprocessor.create_dataset(
                patch_size=256,
                enable_augmentation=(num_rotations > 1),
                augmentation_rotations=num_rotations,
                inference_mode=False,
            )

            # Expected: 1 baseline × 4 pols × num_rotations × 1 patch = 4 * num_rotations
            expected_patches = 4 * num_rotations

            # Note: blank removal might reduce count, but with synthetic data should be minimal
            assert (
                len(dataset) >= expected_patches * 0.9
            ), f"With {num_rotations} rotations, expected ~{expected_patches} patches, got {len(dataset)}"

    def test_inference_mode_disables_blank_removal(self, synthetic_waterfall_small):
        """Test that inference mode preserves all patches (no blank removal)."""
        from samrfi.data import Preprocessor

        data = synthetic_waterfall_small[np.newaxis, ...]

        # Training mode (blank removal enabled)
        preprocessor_train = Preprocessor(data, flags=None)
        dataset_train = preprocessor_train.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=False,  # Training mode
        )

        # Inference mode (blank removal disabled)
        preprocessor_infer = Preprocessor(data, flags=None)
        dataset_infer = preprocessor_infer.create_dataset(
            patch_size=256,
            enable_augmentation=False,
            inference_mode=True,  # Inference mode
        )

        # Inference mode should have same or more patches (no removal)
        assert len(dataset_infer) >= len(
            dataset_train
        ), "Inference mode should preserve all patches"
