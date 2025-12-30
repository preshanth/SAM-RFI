"""
Configuration validation for SAM-RFI.

Validates config parameters early to provide clear error messages
before expensive operations like training or data generation.
"""

from pathlib import Path

from samrfi.utils.errors import ConfigValidationError


def validate_preprocessing_config(config):
    """
    Validate preprocessing configuration.

    Args:
        config: Preprocessing config dict with keys like patch_size, stretch, etc.

    Raises:
        ConfigValidationError: If config is invalid

    Returns:
        True if valid
    """
    # Patch size must be power of 2
    patch_size = config.get("patch_size", 128)
    if patch_size not in [128, 256, 512, 1024]:
        raise ConfigValidationError(f"patch_size must be 128, 256, 512, or 1024. Got: {patch_size}")

    # Stretch must be valid
    stretch = config.get("stretch")
    if stretch not in [None, "SQRT", "LOG10"]:
        raise ConfigValidationError(f"stretch must be None, 'SQRT', or 'LOG10'. Got: {stretch}")

    # Augmentation rotations
    aug_rot = config.get("augmentation_rotations", 4)
    if aug_rot not in [1, 2, 4]:
        raise ConfigValidationError(f"augmentation_rotations must be 1, 2, or 4. Got: {aug_rot}")

    return True


def validate_training_config(config):
    """
    Validate training configuration.

    Args:
        config: Training config dict

    Raises:
        ConfigValidationError: If config is invalid

    Returns:
        True if valid
    """
    # SAM checkpoint
    sam_checkpoint = config.get("sam_checkpoint", "large")
    if sam_checkpoint not in ["tiny", "small", "base_plus", "large"]:
        raise ConfigValidationError(
            f"sam_checkpoint must be tiny/small/base_plus/large. Got: {sam_checkpoint}"
        )

    # Batch size reasonable
    batch_size = config.get("batch_size", 8)
    if batch_size < 1 or batch_size > 128:
        raise ConfigValidationError(f"batch_size must be 1-128. Got: {batch_size}")

    # Learning rate reasonable
    lr = config.get("learning_rate", 1e-4)
    if lr <= 0 or lr > 1:
        raise ConfigValidationError(f"learning_rate must be in (0, 1]. Got: {lr}")

    return True


def validate_paths_exist(config):
    """
    Validate that paths in config exist.

    Args:
        config: Config dict potentially containing file/directory paths

    Raises:
        ConfigValidationError: If paths don't exist

    Returns:
        True if valid
    """
    # Check dataset path
    if "dataset" in config:
        dataset_path = Path(config["dataset"])
        if not dataset_path.exists():
            raise ConfigValidationError(f"Dataset path does not exist: {dataset_path}")

    # Check MS path
    if "ms_path" in config:
        ms_path = Path(config["ms_path"])
        if not ms_path.exists():
            raise ConfigValidationError(f"Measurement set does not exist: {ms_path}")

    # Check model path
    if "model_path" in config:
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise ConfigValidationError(f"Model checkpoint does not exist: {model_path}")

    return True


def validate_all(config):
    """
    Run all applicable validators on config.

    Args:
        config: Complete config object with processing, training, etc. sections

    Raises:
        ConfigValidationError: If any validation fails

    Returns:
        True if valid
    """
    # Validate preprocessing section if present
    if hasattr(config, "processing"):
        validate_preprocessing_config(config.processing)

    # Validate training section if present
    if hasattr(config, "training"):
        validate_training_config(config.training)

    # Validate paths (works with flat dict or object)
    config_dict = config.__dict__ if hasattr(config, "__dict__") else config
    validate_paths_exist(config_dict)

    return True
