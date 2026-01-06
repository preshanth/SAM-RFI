"""
Custom exception classes for SAM-RFI with helpful error messages.

This module defines a hierarchy of exception classes for SAM-RFI operations,
providing context-rich error messages with actionable suggestions for fixes.
All exceptions inherit from SAMRFIError, allowing for targeted exception handling.

Classes
-------
SAMRFIError
    Base exception for all SAM-RFI errors.
DataShapeError
    Raised when data has unexpected shape during processing or inference.
CheckpointMismatchError
    Raised when checkpoint configuration doesn't match inference parameters.
ModelLoadError
    Raised when model checkpoint fails to load from disk.
ConfigValidationError
    Raised when configuration validation fails.

Examples
--------
>>> from samrfi.utils.errors import DataShapeError
>>> import numpy as np
>>> data = np.zeros((256, 256))
>>> expected_shape = (128, 128, 3)
>>> if data.shape != expected_shape:
...     raise DataShapeError(expected_shape, data.shape, "Input preprocessing")
"""

from typing import Any


class SAMRFIError(Exception):
    """
    Base exception for all SAM-RFI errors.

    All custom exceptions in the SAM-RFI package inherit from this class,
    allowing for targeted exception handling at different levels of specificity.

    Examples
    --------
    >>> try:
    ...     # SAM-RFI operation
    ...     pass
    ... except SAMRFIError as e:
    ...     print(f"SAM-RFI error occurred: {e}")
    """

    pass


class DataShapeError(SAMRFIError):
    """
    Raised when data has unexpected shape.

    This exception is raised during preprocessing or inference when array
    dimensions don't match expected values, with context about where the
    error occurred.

    Parameters
    ----------
    expected : tuple or str
        Expected shape or shape description.
    got : tuple or str
        Actual shape received.
    context : str, optional
        Additional context about where the error occurred (e.g., 'Input preprocessing',
        'Model inference'). Default is empty string.

    Attributes
    ----------
    expected : tuple or str
        Expected shape or shape description.
    got : tuple or str
        Actual shape received.
    context : str
        Context information.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.zeros((256, 256))
    >>> expected = (128, 128, 3)
    >>> raise DataShapeError(expected, data.shape, "RGB conversion")
    Traceback (most recent call last):
        ...
    samrfi.utils.errors.DataShapeError: Data shape error: expected (128, 128, 3), got (256, 256)
    Context: RGB conversion
    """

    def __init__(self, expected: Any, got: Any, context: str = "") -> None:
        self.expected = expected
        self.got = got
        self.context = context

        msg = f"Data shape error: expected {expected}, got {got}"
        if context:
            msg += f"\nContext: {context}"
        super().__init__(msg)


class CheckpointMismatchError(SAMRFIError):
    """
    Raised when checkpoint configuration doesn't match inference parameters.

    This exception is raised when attempting to run inference with parameters
    that differ from those used during model training. Provides clear guidance
    on which parameter to adjust.

    Parameters
    ----------
    param_name : str
        Name of the mismatched parameter (e.g., 'patch_size', 'stretch').
    checkpoint_value : Any
        Value used during model training.
    inference_value : Any
        Value being used for inference.

    Attributes
    ----------
    param_name : str
        Name of the mismatched parameter.
    checkpoint_value : Any
        Value from checkpoint.
    inference_value : Any
        Value from inference config.

    Examples
    --------
    >>> raise CheckpointMismatchError('patch_size', 256, 128)
    Traceback (most recent call last):
        ...
    samrfi.utils.errors.CheckpointMismatchError:
    ============================================================
    CHECKPOINT MISMATCH ERROR
    ============================================================
    Parameter: patch_size
      Model trained with: 256
      Inference trying to use: 128
    <BLANKLINE>
    Solution: Use --patch-size 256
    ============================================================
    """

    def __init__(self, param_name: str, checkpoint_value: Any, inference_value: Any) -> None:
        self.param_name = param_name
        self.checkpoint_value = checkpoint_value
        self.inference_value = inference_value

        msg = (
            f"\n{'='*60}\n"
            f"CHECKPOINT MISMATCH ERROR\n"
            f"{'='*60}\n"
            f"Parameter: {param_name}\n"
            f"  Model trained with: {checkpoint_value}\n"
            f"  Inference trying to use: {inference_value}\n"
            f"\n"
            f"Solution: Use --{param_name.replace('_', '-')} {checkpoint_value}\n"
            f"{'='*60}"
        )
        super().__init__(msg)


class ModelLoadError(SAMRFIError):
    """
    Raised when model checkpoint fails to load.

    This exception is raised when a PyTorch model checkpoint cannot be loaded,
    with detailed troubleshooting steps for common issues.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint file.
    reason : str
        Detailed reason for the failure (exception message).

    Attributes
    ----------
    model_path : str
        Path to the failed checkpoint.
    reason : str
        Failure reason.

    Examples
    --------
    >>> raise ModelLoadError('/path/to/model.pth', 'File not found')
    Traceback (most recent call last):
        ...
    samrfi.utils.errors.ModelLoadError: Failed to load model from: /path/to/model.pth
    Reason: File not found
    <BLANKLINE>
    Troubleshooting:
      1. Check that file exists and is readable
      2. Verify it's a valid PyTorch checkpoint (.pth)
      3. Ensure checkpoint was saved with compatible PyTorch version
      4. Try loading with allow_partial_load=True (not recommended)
    """

    def __init__(self, model_path: str, reason: str) -> None:
        self.model_path = model_path
        self.reason = reason

        msg = (
            f"Failed to load model from: {model_path}\n"
            f"Reason: {reason}\n"
            f"\n"
            f"Troubleshooting:\n"
            f"  1. Check that file exists and is readable\n"
            f"  2. Verify it's a valid PyTorch checkpoint (.pth)\n"
            f"  3. Ensure checkpoint was saved with compatible PyTorch version\n"
            f"  4. Try loading with allow_partial_load=True (not recommended)"
        )
        super().__init__(msg)


class ConfigValidationError(SAMRFIError):
    """
    Raised when configuration validation fails.

    This exception is raised when configuration parameters fail validation
    checks before training or data generation begins, allowing early detection
    of invalid settings.

    Examples
    --------
    >>> from samrfi.utils.errors import ConfigValidationError
    >>> patch_size = 200  # Invalid, must be power of 2
    >>> if patch_size not in [128, 256, 512, 1024]:
    ...     raise ConfigValidationError(f"Invalid patch_size: {patch_size}")
    Traceback (most recent call last):
        ...
    samrfi.utils.errors.ConfigValidationError: Invalid patch_size: 200
    """

    pass
