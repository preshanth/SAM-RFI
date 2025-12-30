"""
Shared pytest fixtures for SAM-RFI tests.

Fixtures are used across unit, integration, and extended tests.
"""

import numpy as np
import pytest
import torch

# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def synthetic_waterfall_small():
    """
    Small synthetic complex waterfall for fast testing.

    Returns:
        Complex array of shape (4 pols, 256 channels, 256 times)
    """
    np.random.seed(42)
    # Complex visibility data
    real = np.random.randn(4, 256, 256)
    imag = np.random.randn(4, 256, 256)
    return real + 1j * imag


@pytest.fixture
def synthetic_waterfall_medium():
    """
    Medium synthetic complex waterfall for patchification testing.

    Returns:
        Complex array of shape (4 pols, 1024 channels, 1024 times)
    """
    np.random.seed(42)
    real = np.random.randn(4, 1024, 1024)
    imag = np.random.randn(4, 1024, 1024)
    return real + 1j * imag


@pytest.fixture
def synthetic_waterfall_large():
    """
    Large synthetic waterfall for testing our patchification fix.

    Returns:
        Complex array of shape (4 pols, 2048 channels, 1024 times)
    """
    np.random.seed(42)
    real = np.random.randn(4, 2048, 1024)
    imag = np.random.randn(4, 2048, 1024)
    return real + 1j * imag


@pytest.fixture
def synthetic_flags_small():
    """
    Matching RFI flags for small waterfall.

    Returns:
        Boolean array of shape (4 pols, 256 channels, 256 times)
    """
    flags = np.zeros((4, 256, 256), dtype=bool)

    # Inject synthetic RFI patterns
    # Narrowband RFI (persistent across time)
    flags[:, 100:120, :] = True

    # Broadband burst (short duration)
    flags[:, :, 150:170] = True

    # Scattered RFI
    flags[:, 200:210, 80:90] = True

    return flags


@pytest.fixture
def synthetic_flags_medium():
    """
    Matching RFI flags for medium waterfall.

    Returns:
        Boolean array of shape (4, 1024, 1024)
    """
    flags = np.zeros((4, 1024, 1024), dtype=bool)

    # Narrowband RFI
    flags[:, 400:450, :] = True

    # Broadband burst
    flags[:, :, 600:650] = True

    return flags


@pytest.fixture
def synthetic_data_with_rfi():
    """
    Full synthetic dataset: waterfall + flags + ground truth.

    Returns:
        Dict with 'data', 'flags', 'shape' keys
    """
    np.random.seed(42)

    # Shape: (2 baselines, 4 pols, 256 channels, 256 times)
    shape = (2, 4, 256, 256)

    # Create complex data
    real = np.random.randn(*shape) * 1.0  # 1 mJy noise
    imag = np.random.randn(*shape) * 1.0
    data = real + 1j * imag

    # Inject bright RFI
    flags = np.zeros(shape, dtype=bool)
    flags[0, :, 100:120, :] = True  # Baseline 0: narrowband
    flags[1, :, :, 150:170] = True  # Baseline 1: broadband

    # Add RFI signal (10000 Jy, much brighter than noise)
    rfi_signal = 10000.0 * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    data[flags] = rfi_signal[flags]

    return {
        "data": data,
        "flags": flags,
        "shape": shape,
    }


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def mock_checkpoint(tmp_path):
    """
    Create a mock model checkpoint with preprocessing metadata.

    Args:
        tmp_path: pytest temp directory fixture

    Returns:
        Path to checkpoint file
    """
    checkpoint = {
        "model_state_dict": {},  # Empty for mock
        "optimizer_state_dict": {},
        "epoch": 10,
        "training_losses": [1.0, 0.8, 0.6, 0.5],
        "validation_losses": [1.1, 0.9, 0.7, 0.6],
        "patch_size": 1024,
        "preprocessing": {
            "patch_size": 1024,
            "augmentation_rotations": 4,
            "stretch": None,
            "normalize_before_stretch": False,
            "normalize_after_stretch": False,
        },
        "config": {
            "sam_checkpoint": "tiny",
            "learning_rate": 1e-4,
            "batch_size": 8,
            "loss_function": "DiceCELoss",
        },
    }

    checkpoint_path = tmp_path / "mock_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def mock_checkpoint_mismatch(tmp_path):
    """
    Create a mock checkpoint with mismatched preprocessing config.

    Used to test validation errors.
    """
    checkpoint = {
        "model_state_dict": {},
        "patch_size": 128,  # Mismatch!
        "preprocessing": {
            "patch_size": 128,
            "augmentation_rotations": 4,
            "stretch": "SQRT",  # Mismatch!
            "normalize_before_stretch": True,
            "normalize_after_stretch": False,
        },
        "config": {
            "sam_checkpoint": "tiny",
        },
    }

    checkpoint_path = tmp_path / "mismatch_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


# ============================================================================
# Dataset Fixtures
# ============================================================================


@pytest.fixture
def mock_torch_dataset():
    """
    Create a small TorchDataset for testing.

    Returns:
        TorchDataset with 10 samples
    """
    from samrfi.data import TorchDataset

    # Create mock images (10 samples, 256×256×3)
    images = torch.randn(10, 256, 256, 3, dtype=torch.float32)

    # Create mock labels (10 samples, 256×256)
    labels = torch.randint(0, 2, (10, 256, 256), dtype=torch.uint8)

    # Metadata
    metadata = {
        "patch_size": 256,
        "stretch": None,
        "flag_sigma": 5,
        "normalize_before_stretch": False,
        "normalize_after_stretch": False,
        "augmentation_rotations": 1,
    }

    return TorchDataset(images, labels, metadata)


# ============================================================================
# pytest Configuration Helpers
# ============================================================================


def pytest_configure(config):
    """
    Configure pytest markers for test categories.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "requires_gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "requires_casa: requires CASA installation")


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary output directory for test artifacts.

    Automatically cleaned up after test.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir
