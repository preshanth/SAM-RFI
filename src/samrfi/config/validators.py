"""
Configuration validation for SAM-RFI.

Validates configuration parameters early to provide clear error messages
before expensive operations like training or data generation.
"""

from pathlib import Path
from typing import Any, Dict, Union

from samrfi.utils.errors import ConfigValidationError


def validate_preprocessing_config(config: Union[Dict[str, Any], Any]) -> bool:
    """
    Validate preprocessing configuration parameters.

    Parameters
    ----------
    config : dict or object
        Preprocessing configuration with parameters like patch_size, stretch, etc.
        Can be a dictionary or object with attribute access.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ConfigValidationError
        If any configuration parameter is invalid.

    Examples
    --------
    >>> config = {'patch_size': 256, 'stretch': 'SQRT'}
    >>> validate_preprocessing_config(config)
    True
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


def validate_training_config(config: Union[Dict[str, Any], Any]) -> bool:
    """
    Validate training configuration parameters.

    Parameters
    ----------
    config : dict or object
        Training configuration with parameters like sam_checkpoint, batch_size, etc.
        Can be a dictionary or object with attribute access.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ConfigValidationError
        If any configuration parameter is invalid.

    Examples
    --------
    >>> config = {'sam_checkpoint': 'large', 'batch_size': 8, 'learning_rate': 1e-4}
    >>> validate_training_config(config)
    True
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


def validate_paths_exist(config: Union[Dict[str, Any], Any]) -> bool:
    """
    Validate that file and directory paths in configuration exist.

    Parameters
    ----------
    config : dict or object
        Configuration potentially containing file/directory paths.
        Can be a dictionary or object with attribute access.

    Returns
    -------
    bool
        True if all paths exist.

    Raises
    ------
    ConfigValidationError
        If any specified path doesn't exist.

    Examples
    --------
    >>> config = {'dataset': '/path/to/dataset', 'ms_path': '/path/to/ms'}
    >>> validate_paths_exist(config)  # doctest: +SKIP
    True
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


def validate_all(config: Union[Dict[str, Any], Any]) -> bool:
    """
    Run all applicable validators on configuration.

    Validates preprocessing, training, and path existence based on
    which sections are present in the configuration.

    Parameters
    ----------
    config : dict or object
        Complete configuration object with processing, training, etc. sections.
        Can be a dictionary or object with attribute access.

    Returns
    -------
    bool
        True if all validations pass.

    Raises
    ------
    ConfigValidationError
        If any validation check fails.

    Examples
    --------
    >>> config = {
    ...     'processing': {'patch_size': 256, 'stretch': 'SQRT'},
    ...     'training': {'sam_checkpoint': 'large', 'batch_size': 8}
    ... }
    >>> validate_all(config)
    True
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
