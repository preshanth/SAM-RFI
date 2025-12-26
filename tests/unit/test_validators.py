"""
Unit tests for configuration validators.

Tests that validators catch invalid configs and provide helpful messages.
"""

import pytest
from pathlib import Path
from samrfi.config.validators import (
    validate_preprocessing_config,
    validate_training_config,
    validate_paths_exist,
    validate_all,
)
from samrfi.utils.errors import ConfigValidationError


class TestPreprocessingConfigValidator:
    """Test preprocessing configuration validation."""

    def test_valid_preprocessing_config(self):
        """Test that valid config passes validation."""
        config = {
            "patch_size": 1024,
            "stretch": "SQRT",
            "augmentation_rotations": 4,
        }
        assert validate_preprocessing_config(config) is True

    def test_valid_preprocessing_config_defaults(self):
        """Test that validation works with missing optional fields."""
        config = {}  # All defaults
        assert validate_preprocessing_config(config) is True

    def test_invalid_patch_size_too_small(self):
        """Test that invalid patch_size raises error."""
        config = {"patch_size": 64}  # Too small
        with pytest.raises(ConfigValidationError, match="patch_size"):
            validate_preprocessing_config(config)

    def test_invalid_patch_size_not_power_of_2(self):
        """Test that non-power-of-2 patch_size raises error."""
        config = {"patch_size": 333}  # Not a power of 2
        with pytest.raises(ConfigValidationError, match="patch_size"):
            validate_preprocessing_config(config)

    def test_valid_patch_sizes(self):
        """Test all valid patch sizes."""
        valid_sizes = [128, 256, 512, 1024]
        for size in valid_sizes:
            config = {"patch_size": size}
            assert validate_preprocessing_config(config) is True

    def test_invalid_stretch_function(self):
        """Test that invalid stretch function raises error."""
        config = {"stretch": "INVALID"}
        with pytest.raises(ConfigValidationError, match="stretch"):
            validate_preprocessing_config(config)

    def test_valid_stretch_functions(self):
        """Test all valid stretch functions."""
        valid_stretches = [None, "SQRT", "LOG10"]
        for stretch in valid_stretches:
            config = {"stretch": stretch}
            assert validate_preprocessing_config(config) is True

    def test_invalid_augmentation_rotations(self):
        """Test that invalid augmentation_rotations raises error."""
        config = {"augmentation_rotations": 3}  # Not 1, 2, or 4
        with pytest.raises(ConfigValidationError, match="augmentation_rotations"):
            validate_preprocessing_config(config)

    def test_valid_augmentation_rotations(self):
        """Test all valid augmentation_rotations values."""
        valid_rotations = [1, 2, 4]
        for rotations in valid_rotations:
            config = {"augmentation_rotations": rotations}
            assert validate_preprocessing_config(config) is True


class TestTrainingConfigValidator:
    """Test training configuration validation."""

    def test_valid_training_config(self):
        """Test that valid config passes validation."""
        config = {
            "sam_checkpoint": "large",
            "batch_size": 8,
            "learning_rate": 1e-4,
        }
        assert validate_training_config(config) is True

    def test_valid_training_config_defaults(self):
        """Test that validation works with missing optional fields."""
        config = {}  # All defaults
        assert validate_training_config(config) is True

    def test_invalid_sam_checkpoint(self):
        """Test that invalid SAM checkpoint raises error."""
        config = {"sam_checkpoint": "xlarge"}  # Doesn't exist
        with pytest.raises(ConfigValidationError, match="sam_checkpoint"):
            validate_training_config(config)

    def test_valid_sam_checkpoints(self):
        """Test all valid SAM checkpoint sizes."""
        valid_checkpoints = ["tiny", "small", "base_plus", "large"]
        for checkpoint in valid_checkpoints:
            config = {"sam_checkpoint": checkpoint}
            assert validate_training_config(config) is True

    def test_invalid_batch_size_too_small(self):
        """Test that batch_size < 1 raises error."""
        config = {"batch_size": 0}
        with pytest.raises(ConfigValidationError, match="batch_size"):
            validate_training_config(config)

    def test_invalid_batch_size_too_large(self):
        """Test that batch_size > 128 raises error."""
        config = {"batch_size": 256}
        with pytest.raises(ConfigValidationError, match="batch_size"):
            validate_training_config(config)

    def test_valid_batch_sizes(self):
        """Test reasonable batch sizes."""
        valid_sizes = [1, 4, 8, 16, 32, 64, 128]
        for size in valid_sizes:
            config = {"batch_size": size}
            assert validate_training_config(config) is True

    def test_invalid_learning_rate_zero(self):
        """Test that learning_rate = 0 raises error."""
        config = {"learning_rate": 0.0}
        with pytest.raises(ConfigValidationError, match="learning_rate"):
            validate_training_config(config)

    def test_invalid_learning_rate_negative(self):
        """Test that negative learning_rate raises error."""
        config = {"learning_rate": -0.001}
        with pytest.raises(ConfigValidationError, match="learning_rate"):
            validate_training_config(config)

    def test_invalid_learning_rate_too_large(self):
        """Test that learning_rate > 1 raises error."""
        config = {"learning_rate": 1.5}
        with pytest.raises(ConfigValidationError, match="learning_rate"):
            validate_training_config(config)

    def test_valid_learning_rates(self):
        """Test reasonable learning rates."""
        valid_lrs = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]
        for lr in valid_lrs:
            config = {"learning_rate": lr}
            assert validate_training_config(config) is True


class TestPathValidator:
    """Test path existence validation."""

    def test_validate_existing_dataset_path(self, tmp_path):
        """Test that existing dataset path passes validation."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        config = {"dataset": str(dataset_dir)}
        assert validate_paths_exist(config) is True

    def test_validate_nonexistent_dataset_path(self):
        """Test that nonexistent dataset path raises error."""
        config = {"dataset": "/nonexistent/path/dataset"}
        with pytest.raises(ConfigValidationError, match="Dataset path does not exist"):
            validate_paths_exist(config)

    def test_validate_existing_ms_path(self, tmp_path):
        """Test that existing MS path passes validation."""
        ms_path = tmp_path / "observation.ms"
        ms_path.mkdir()  # MS is a directory

        config = {"ms_path": str(ms_path)}
        assert validate_paths_exist(config) is True

    def test_validate_nonexistent_ms_path(self):
        """Test that nonexistent MS path raises error."""
        config = {"ms_path": "/nonexistent/observation.ms"}
        with pytest.raises(ConfigValidationError, match="Measurement set does not exist"):
            validate_paths_exist(config)

    def test_validate_existing_model_path(self, tmp_path):
        """Test that existing model path passes validation."""
        model_path = tmp_path / "model.pth"
        model_path.touch()

        config = {"model_path": str(model_path)}
        assert validate_paths_exist(config) is True

    def test_validate_nonexistent_model_path(self):
        """Test that nonexistent model path raises error."""
        config = {"model_path": "/nonexistent/model.pth"}
        with pytest.raises(ConfigValidationError, match="Model checkpoint does not exist"):
            validate_paths_exist(config)

    def test_validate_empty_config(self):
        """Test that config with no paths passes validation."""
        config = {}
        assert validate_paths_exist(config) is True

    def test_validate_multiple_paths(self, tmp_path):
        """Test validation with multiple paths."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        config = {
            "dataset": str(dataset_dir),
            "model_path": str(model_path),
        }
        assert validate_paths_exist(config) is True


class TestValidateAll:
    """Test combined validator."""

    def test_validate_all_with_config_object(self, tmp_path):
        """Test validate_all with config object (has attributes)."""
        # Create mock config object
        class MockConfig:
            def __init__(self):
                self.processing = {
                    "patch_size": 1024,
                    "stretch": None,
                    "augmentation_rotations": 4,
                }
                self.training = {
                    "sam_checkpoint": "large",
                    "batch_size": 8,
                    "learning_rate": 1e-4,
                }

        config = MockConfig()
        assert validate_all(config) is True

    def test_validate_all_catches_preprocessing_error(self):
        """Test that validate_all catches preprocessing errors."""
        class MockConfig:
            def __init__(self):
                self.processing = {"patch_size": 333}  # Invalid

        config = MockConfig()
        with pytest.raises(ConfigValidationError, match="patch_size"):
            validate_all(config)

    def test_validate_all_catches_training_error(self):
        """Test that validate_all catches training errors."""
        class MockConfig:
            def __init__(self):
                self.processing = {"patch_size": 1024}
                self.training = {"batch_size": 0}  # Invalid

        config = MockConfig()
        with pytest.raises(ConfigValidationError, match="batch_size"):
            validate_all(config)

    def test_validate_all_catches_path_error(self):
        """Test that validate_all catches path errors."""
        config = {"dataset": "/nonexistent/path"}
        with pytest.raises(ConfigValidationError, match="Dataset path"):
            validate_all(config)

    def test_validate_all_with_dict(self, tmp_path):
        """Test validate_all with plain dict."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        config = {"dataset": str(dataset_dir)}
        assert validate_all(config) is True


class TestValidatorErrorMessages:
    """Test that error messages are helpful."""

    def test_patch_size_error_shows_valid_options(self):
        """Test that patch_size error shows valid options."""
        config = {"patch_size": 333}
        try:
            validate_preprocessing_config(config)
        except ConfigValidationError as e:
            message = str(e)
            assert "128" in message
            assert "256" in message
            assert "512" in message
            assert "1024" in message
            assert "333" in message  # Shows what was provided

    def test_sam_checkpoint_error_shows_valid_options(self):
        """Test that sam_checkpoint error shows valid options."""
        config = {"sam_checkpoint": "invalid"}
        try:
            validate_training_config(config)
        except ConfigValidationError as e:
            message = str(e)
            assert "tiny" in message
            assert "small" in message
            assert "base_plus" in message
            assert "large" in message
            assert "invalid" in message  # Shows what was provided

    def test_path_error_shows_missing_path(self):
        """Test that path error shows which path is missing."""
        config = {"dataset": "/some/nonexistent/path/dataset"}
        try:
            validate_paths_exist(config)
        except ConfigValidationError as e:
            message = str(e)
            assert "/some/nonexistent/path/dataset" in message
