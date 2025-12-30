"""
Custom exception classes for SAM-RFI with helpful error messages.

Provides context-rich errors with suggestions for fixes.
"""


class SAMRFIError(Exception):
    """Base exception for all SAM-RFI errors."""

    pass


class DataShapeError(SAMRFIError):
    """Raised when data has unexpected shape."""

    def __init__(self, expected, got, context=""):
        msg = f"Data shape error: expected {expected}, got {got}"
        if context:
            msg += f"\nContext: {context}"
        super().__init__(msg)


class CheckpointMismatchError(SAMRFIError):
    """Raised when checkpoint doesn't match inference config."""

    def __init__(self, param_name, checkpoint_value, inference_value):
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
    """Raised when model fails to load."""

    def __init__(self, model_path, reason):
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
    """Raised when configuration validation fails."""

    pass
