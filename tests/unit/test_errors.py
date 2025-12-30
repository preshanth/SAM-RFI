"""
Unit tests for custom error classes.

Tests that error messages are informative and include helpful context.
"""

import pytest

from samrfi.utils.errors import (
    CheckpointMismatchError,
    ConfigValidationError,
    DataShapeError,
    ModelLoadError,
    SAMRFIError,
)


class TestBaseError:
    """Test base SAMRFIError class."""

    def test_base_error_is_exception(self):
        """Test that SAMRFIError inherits from Exception."""
        assert issubclass(SAMRFIError, Exception)

    def test_base_error_can_be_raised(self):
        """Test that SAMRFIError can be raised and caught."""
        with pytest.raises(SAMRFIError, match="test message"):
            raise SAMRFIError("test message")


class TestDataShapeError:
    """Test DataShapeError class."""

    def test_data_shape_error_basic(self):
        """Test DataShapeError with basic shape mismatch."""
        error = DataShapeError(expected="(256, 256)", got="(128, 128)")

        assert "expected (256, 256)" in str(error)
        assert "got (128, 128)" in str(error)

    def test_data_shape_error_with_context(self):
        """Test DataShapeError includes context."""
        error = DataShapeError(
            expected="(4, 1024, 1024)", got="(2, 1024, 1024)", context="Input to Preprocessor"
        )

        message = str(error)
        assert "expected (4, 1024, 1024)" in message
        assert "got (2, 1024, 1024)" in message
        assert "Input to Preprocessor" in message

    def test_data_shape_error_is_samrfi_error(self):
        """Test that DataShapeError inherits from SAMRFIError."""
        assert issubclass(DataShapeError, SAMRFIError)


class TestCheckpointMismatchError:
    """Test CheckpointMismatchError class."""

    def test_checkpoint_mismatch_patch_size(self):
        """Test CheckpointMismatchError for patch_size mismatch."""
        error = CheckpointMismatchError(
            param_name="patch_size", checkpoint_value=1024, inference_value=128
        )

        message = str(error)
        assert "CHECKPOINT MISMATCH" in message
        assert "patch_size" in message
        assert "1024" in message
        assert "128" in message
        assert "Solution" in message
        assert "--patch-size 1024" in message  # Suggests fix

    def test_checkpoint_mismatch_stretch(self):
        """Test CheckpointMismatchError for stretch mismatch."""
        error = CheckpointMismatchError(
            param_name="stretch", checkpoint_value="SQRT", inference_value=None
        )

        message = str(error)
        assert "stretch" in message
        assert "SQRT" in message
        assert "None" in message
        assert "--stretch SQRT" in message

    def test_checkpoint_mismatch_converts_underscores(self):
        """Test that parameter names convert underscores to hyphens in CLI suggestion."""
        error = CheckpointMismatchError(
            param_name="normalize_before_stretch", checkpoint_value=True, inference_value=False
        )

        message = str(error)
        # Should suggest --normalize-before-stretch (not --normalize_before_stretch)
        assert "--normalize-before-stretch" in message

    def test_checkpoint_mismatch_is_samrfi_error(self):
        """Test that CheckpointMismatchError inherits from SAMRFIError."""
        assert issubclass(CheckpointMismatchError, SAMRFIError)


class TestModelLoadError:
    """Test ModelLoadError class."""

    def test_model_load_error_basic(self):
        """Test ModelLoadError with path and reason."""
        error = ModelLoadError(model_path="/path/to/model.pth", reason="File not found")

        message = str(error)
        assert "/path/to/model.pth" in message
        assert "File not found" in message
        assert "Troubleshooting" in message

    def test_model_load_error_includes_suggestions(self):
        """Test that ModelLoadError includes helpful troubleshooting steps."""
        error = ModelLoadError(model_path="model.pth", reason="Invalid checkpoint format")

        message = str(error)
        assert "1." in message  # Numbered troubleshooting steps
        assert "2." in message
        assert "file exists" in message.lower()
        assert "pytorch" in message.lower()

    def test_model_load_error_is_samrfi_error(self):
        """Test that ModelLoadError inherits from SAMRFIError."""
        assert issubclass(ModelLoadError, SAMRFIError)


class TestConfigValidationError:
    """Test ConfigValidationError class."""

    def test_config_validation_error_basic(self):
        """Test ConfigValidationError with basic message."""
        with pytest.raises(ConfigValidationError, match="Invalid patch_size"):
            raise ConfigValidationError("Invalid patch_size: 333")

    def test_config_validation_error_is_samrfi_error(self):
        """Test that ConfigValidationError inherits from SAMRFIError."""
        assert issubclass(ConfigValidationError, SAMRFIError)


class TestErrorInheritance:
    """Test error inheritance hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Test that all custom errors inherit from SAMRFIError."""
        custom_errors = [
            DataShapeError,
            CheckpointMismatchError,
            ModelLoadError,
            ConfigValidationError,
        ]

        for error_class in custom_errors:
            assert issubclass(
                error_class, SAMRFIError
            ), f"{error_class.__name__} should inherit from SAMRFIError"

    def test_all_errors_inherit_from_exception(self):
        """Test that all custom errors inherit from Exception."""
        custom_errors = [
            SAMRFIError,
            DataShapeError,
            CheckpointMismatchError,
            ModelLoadError,
            ConfigValidationError,
        ]

        for error_class in custom_errors:
            assert issubclass(
                error_class, Exception
            ), f"{error_class.__name__} should inherit from Exception"

    def test_catch_specific_error(self):
        """Test that specific errors can be caught independently."""
        try:
            raise DataShapeError("(256, 256)", "(128, 128)")
        except DataShapeError as e:
            assert "256" in str(e)
        except SAMRFIError:
            pytest.fail("Should catch DataShapeError, not base SAMRFIError")

    def test_catch_base_error(self):
        """Test that base error catches all custom errors."""
        errors_to_test = [
            DataShapeError("(1, 2)", "(3, 4)"),
            CheckpointMismatchError("param", "val1", "val2"),
            ModelLoadError("path.pth", "reason"),
            ConfigValidationError("invalid"),
        ]

        for error in errors_to_test:
            try:
                raise error
            except SAMRFIError:
                pass  # Expected
            else:
                pytest.fail(f"{type(error).__name__} should be catchable as SAMRFIError")
