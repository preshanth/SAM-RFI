"""
Unit tests for ConfigLoader
"""

import pytest
import yaml
import tempfile
from pathlib import Path

from samrfi.config.config_loader import ConfigLoader, TrainingConfig


class TestTrainingConfig:
    """Test TrainingConfig dataclass validation"""

    def test_default_config_valid(self):
        """Test that default configuration is valid"""
        config = TrainingConfig()

        assert config.model_checkpoint == 'large'
        assert config.num_epochs == 5
        assert config.batch_size == 4
        assert config.learning_rate == 1e-5
        assert config.device == 'cuda'

    def test_invalid_model_checkpoint(self):
        """Test that invalid model checkpoint raises ValueError"""
        with pytest.raises(ValueError, match="Invalid model_checkpoint"):
            TrainingConfig(model_checkpoint='invalid')

    def test_invalid_stretch(self):
        """Test that invalid stretch raises ValueError"""
        with pytest.raises(ValueError, match="Invalid stretch"):
            TrainingConfig(stretch='INVALID')

    def test_invalid_device(self):
        """Test that invalid device raises ValueError"""
        with pytest.raises(ValueError, match="Invalid device"):
            TrainingConfig(device='invalid')

    def test_negative_num_epochs(self):
        """Test that negative num_epochs raises ValueError"""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TrainingConfig(num_epochs=-1)

    def test_negative_batch_size(self):
        """Test that negative batch_size raises ValueError"""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)

    def test_negative_learning_rate(self):
        """Test that negative learning_rate raises ValueError"""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.001)

    def test_negative_flag_sigma(self):
        """Test that negative flag_sigma raises ValueError"""
        with pytest.raises(ValueError, match="flag_sigma must be positive"):
            TrainingConfig(flag_sigma=0)

    def test_negative_patch_size(self):
        """Test that negative patch_size raises ValueError"""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            TrainingConfig(patch_size=-128)

    def test_valid_checkpoints(self):
        """Test that all valid checkpoints are accepted"""
        valid_checkpoints = ['tiny', 'small', 'base_plus', 'large']

        for checkpoint in valid_checkpoints:
            config = TrainingConfig(model_checkpoint=checkpoint)
            assert config.model_checkpoint == checkpoint

    def test_valid_stretches(self):
        """Test that all valid stretches are accepted"""
        valid_stretches = ['SQRT', 'LOG10']

        for stretch in valid_stretches:
            config = TrainingConfig(stretch=stretch)
            assert config.stretch == stretch


class TestConfigLoader:
    """Test ConfigLoader class"""

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigLoader.load('/nonexistent/path/config.yaml')

    def test_load_empty_file(self):
        """Test that loading empty file raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(None, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Empty configuration file"):
                ConfigLoader.load(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_yaml(self):
        """Test that loading invalid YAML raises YAMLError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                ConfigLoader.load(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_valid_config(self):
        """Test loading valid configuration file"""
        config_dict = {
            'model': {
                'checkpoint': 'tiny',
                'freeze_encoders': True
            },
            'training': {
                'num_epochs': 10,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'device': 'cpu'
            },
            'dataset': {
                'stretch': 'LOG10',
                'flag_sigma': 8,
                'patch_size': 256
            },
            'output': {
                'dir_path': '/tmp/test',
                'save_plots': False
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = ConfigLoader.load(temp_path)

            assert config.model_checkpoint == 'tiny'
            assert config.num_epochs == 10
            assert config.batch_size == 8
            assert config.learning_rate == 1e-4
            assert config.device == 'cpu'
            assert config.stretch == 'LOG10'
            assert config.flag_sigma == 8
            assert config.patch_size == 256
            assert config.dir_path == '/tmp/test'
            assert config.save_plots == False
        finally:
            Path(temp_path).unlink()

    def test_load_partial_config_uses_defaults(self):
        """Test that partial config uses default values for missing fields"""
        config_dict = {
            'training': {
                'num_epochs': 20
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = ConfigLoader.load(temp_path)

            # Check provided value
            assert config.num_epochs == 20

            # Check defaults are used
            assert config.model_checkpoint == 'large'
            assert config.batch_size == 4
            assert config.learning_rate == 1e-5
        finally:
            Path(temp_path).unlink()

    def test_save_config(self):
        """Test saving configuration to YAML file"""
        config = TrainingConfig(
            model_checkpoint='small',
            num_epochs=15,
            batch_size=2,
            device='cpu'
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            ConfigLoader.save(config, temp_path)

            # Load back and verify
            loaded_config = ConfigLoader.load(temp_path)

            assert loaded_config.model_checkpoint == 'small'
            assert loaded_config.num_epochs == 15
            assert loaded_config.batch_size == 2
            assert loaded_config.device == 'cpu'
        finally:
            Path(temp_path).unlink()

    def test_create_default_config(self):
        """Test creating default configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            ConfigLoader.create_default_config(temp_path)

            # Verify file exists
            assert Path(temp_path).exists()

            # Load and verify it's valid
            config = ConfigLoader.load(temp_path)
            assert config.model_checkpoint == 'large'
            assert config.num_epochs == 5
        finally:
            Path(temp_path).unlink()

    def test_flatten_config(self):
        """Test _flatten_config method"""
        nested_dict = {
            'model': {'checkpoint': 'tiny'},
            'training': {'num_epochs': 5, 'batch_size': 4}
        }

        flat = ConfigLoader._flatten_config(nested_dict)

        assert flat['model_checkpoint'] == 'tiny'
        assert flat['num_epochs'] == 5
        assert flat['batch_size'] == 4

    def test_roundtrip_save_load(self):
        """Test that save->load preserves configuration"""
        original_config = TrainingConfig(
            model_checkpoint='base_plus',
            num_epochs=25,
            batch_size=6,
            learning_rate=5e-5,
            stretch='LOG10',
            flag_sigma=10,
            patch_size=512
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            ConfigLoader.save(original_config, temp_path)

            # Load
            loaded_config = ConfigLoader.load(temp_path)

            # Verify all fields match
            assert loaded_config.model_checkpoint == original_config.model_checkpoint
            assert loaded_config.num_epochs == original_config.num_epochs
            assert loaded_config.batch_size == original_config.batch_size
            assert loaded_config.learning_rate == original_config.learning_rate
            assert loaded_config.stretch == original_config.stretch
            assert loaded_config.flag_sigma == original_config.flag_sigma
            assert loaded_config.patch_size == original_config.patch_size
        finally:
            Path(temp_path).unlink()