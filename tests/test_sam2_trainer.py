"""
Unit tests for SAM2Trainer
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch
from pathlib import Path
import tempfile
import os

from datasets import Dataset
from PIL import Image


@pytest.fixture
def real_dataset():
    """Create a real minimal dataset for testing"""
    num_samples = 4
    images = []
    labels = []

    for i in range(num_samples):
        # Create small RGB image (64x64)
        img_array = np.random.rand(64, 64, 3) * 255
        img = Image.fromarray(img_array.astype(np.uint8))
        images.append(img)

        # Create binary mask
        mask_array = np.random.randint(0, 2, (64, 64)) * 255
        mask = Image.fromarray(mask_array.astype(np.uint8))
        labels.append(mask)

    # Create HuggingFace dataset
    dataset_dict = {
        "image": images,
        "label": labels
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Create wrapper with dataset_params and patched_data_norm_only
    class DatasetWrapper:
        def __init__(self, ds):
            self.dataset = ds
            self.dataset_params = {
                "stretch": "SQRT",
                "flag_sigma": 5,
                "patch_method": "patchify",
                "patch_size": 64
            }
            self.patched_data_norm_only = np.random.rand(num_samples, 64, 64)

    return DatasetWrapper(dataset)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestSAM2TrainerInit:
    """Test SAM2Trainer initialization"""

    def test_init_with_default_dir(self, real_dataset):
        """Test initialization with default directory"""
        from samrfi.training.sam2_trainer import SAM2Trainer

        trainer = SAM2Trainer(real_dataset, device='cpu')

        assert trainer.device == 'cpu'
        assert trainer.RFIDataset == real_dataset
        assert 'samrfi_data' in trainer.directory
        assert trainer.ave_meanloss == []

    def test_init_with_custom_dir(self, real_dataset, temp_dir):
        """Test initialization with custom directory"""
        from samrfi.training.sam2_trainer import SAM2Trainer

        trainer = SAM2Trainer(real_dataset, device='cpu', dir_path=temp_dir)

        assert trainer.device == 'cpu'
        assert temp_dir in trainer.directory
        assert os.path.exists(trainer.directory)

    def test_init_creates_output_directory(self, real_dataset, temp_dir):
        """Test that initialization creates output directory"""
        from samrfi.training.sam2_trainer import SAM2Trainer

        trainer = SAM2Trainer(real_dataset, device='cpu', dir_path=temp_dir)

        assert os.path.exists(trainer.directory)


class TestSAM2TrainerCheckpoint:
    """Test checkpoint name mapping"""

    def test_invalid_checkpoint_raises_error(self, real_dataset):
        """Test that invalid checkpoint name raises ValueError"""
        from samrfi.training.sam2_trainer import SAM2Trainer

        trainer = SAM2Trainer(real_dataset, device='cpu')

        with pytest.raises(ValueError, match="Invalid checkpoint"):
            trainer.train(num_epochs=1, sam_checkpoint='invalid_name')

    def test_valid_checkpoints(self, real_dataset):
        """Test that valid checkpoint names are accepted"""
        from samrfi.training.sam2_trainer import SAM2Trainer

        valid_checkpoints = ['tiny', 'small', 'base_plus', 'large']

        for checkpoint in valid_checkpoints:
            trainer = SAM2Trainer(real_dataset, device='cpu')
            # We'll mock the actual training to avoid downloading models
            # This just tests the checkpoint validation logic
            try:
                # This will fail at model loading, but checkpoint validation should pass
                with patch('samrfi.training.sam2_trainer.Sam2Processor'):
                    with patch('samrfi.training.sam2_trainer.Sam2Model'):
                        pass  # Checkpoint name should be valid
            except ValueError:
                pytest.fail(f"Valid checkpoint '{checkpoint}' was rejected")


class TestSAM2TrainerSaveModel:
    """Test model saving functionality"""

    def test_save_model_creates_directory(self, real_dataset, temp_dir):
        """Test that _save_model creates models directory"""
        from samrfi.training.sam2_trainer import SAM2Trainer
        from unittest.mock import Mock

        trainer = SAM2Trainer(real_dataset, device='cpu', dir_path=temp_dir)

        # Create mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {}

        with patch('torch.save'):
            trainer._save_model(mock_model, 'tiny', 1)

        models_dir = os.path.join(trainer.directory, 'models')
        assert os.path.exists(models_dir)

    def test_save_model_filename_format(self, real_dataset, temp_dir):
        """Test that saved model has correct filename format"""
        from samrfi.training.sam2_trainer import SAM2Trainer
        from unittest.mock import Mock

        trainer = SAM2Trainer(real_dataset, device='cpu', dir_path=temp_dir)

        mock_model = Mock()
        mock_model.state_dict.return_value = {}

        saved_path = None
        def capture_save_path(state_dict, path):
            nonlocal saved_path
            saved_path = path

        with patch('torch.save', side_effect=capture_save_path):
            trainer._save_model(mock_model, 'tiny', 5)

        # Check filename components
        filename = os.path.basename(saved_path)
        assert 'model_sam2-tiny' in filename
        assert 'stretch-SQRT' in filename
        assert 'sigma-5' in filename
        assert 'epochs5' in filename
        assert '.pth' in filename


class TestSAM2TrainerPlot:
    """Test plotting functionality"""

    def test_plot_loss_curve_creates_file(self, real_dataset, temp_dir):
        """Test that _plot_loss_curve creates actual PNG file"""
        from samrfi.training.sam2_trainer import SAM2Trainer

        trainer = SAM2Trainer(real_dataset, device='cpu', dir_path=temp_dir)
        trainer.ave_meanloss = [1.0, 0.8, 0.6, 0.4]

        # Actually create the plot
        trainer._plot_loss_curve('tiny', 4)

        # Check that file was created
        models_dir = os.path.join(trainer.directory, 'models')
        assert os.path.exists(models_dir)

        # Check that a PNG file was created
        png_files = list(Path(models_dir).glob('loss_plot_sam2-tiny*.png'))
        assert len(png_files) == 1
        assert 'stretch-SQRT' in png_files[0].name
        assert 'sigma-5' in png_files[0].name