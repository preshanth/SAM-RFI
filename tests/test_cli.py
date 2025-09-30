"""
Unit tests for CLI interface
"""

import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

from samrfi.config.config_loader import ConfigLoader


@pytest.fixture
def temp_config():
    """Create temporary config file"""
    config_dict = {
        'model': {'checkpoint': 'tiny'},
        'training': {'num_epochs': 2, 'batch_size': 1, 'device': 'cpu'},
        'dataset': {'stretch': 'SQRT', 'flag_sigma': 5, 'patch_size': 64}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink()


@pytest.fixture
def temp_ms_path(tmp_path):
    """Create temporary MS path"""
    ms_path = tmp_path / "test.ms"
    ms_path.mkdir()
    return str(ms_path)


class TestCLICreateConfig:
    """Test create-config command"""

    def test_create_config_default_name(self):
        """Test creating config with default name"""
        from samrfi.cli import create_config_command

        args = Mock()
        args.output = None

        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                create_config_command(args)

                # Check default file created
                assert Path('sam2_config.yaml').exists()
            finally:
                os.chdir(original_cwd)

    def test_create_config_custom_name(self):
        """Test creating config with custom name"""
        from samrfi.cli import create_config_command

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'custom_config.yaml'

            args = Mock()
            args.output = str(output_path)

            create_config_command(args)

            assert output_path.exists()

            # Verify it's a valid config
            config = ConfigLoader.load(str(output_path))
            assert config.model_checkpoint == 'large'


class TestCLIValidateConfig:
    """Test validate-config command"""

    def test_validate_valid_config(self, temp_config):
        """Test validating a valid config file"""
        from samrfi.cli import validate_config_command

        args = Mock()
        args.config = temp_config

        result = validate_config_command(args)

        assert result == 0  # Success

    def test_validate_nonexistent_config(self):
        """Test validating nonexistent config file"""
        from samrfi.cli import validate_config_command

        args = Mock()
        args.config = '/nonexistent/config.yaml'

        result = validate_config_command(args)

        assert result == 1  # Failure

    def test_validate_invalid_config(self):
        """Test validating invalid config file"""
        from samrfi.cli import validate_config_command

        # Create invalid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'model': {'checkpoint': 'invalid_checkpoint'}}, f)
            temp_path = f.name

        try:
            args = Mock()
            args.config = temp_path

            result = validate_config_command(args)

            assert result == 1  # Failure
        finally:
            Path(temp_path).unlink()


class TestCLITrainCommand:
    """Test train command validation"""

    def test_train_command_requires_dataset(self, temp_config):
        """Test that train command requires dataset path"""
        from samrfi.cli import train_command

        args = Mock()
        args.config = temp_config
        args.dataset = None  # Missing
        args.device = None
        args.output_dir = None

        with pytest.raises(ValueError, match="--dataset is required"):
            train_command(args)


class TestCLIMain:
    """Test main CLI entry point"""

    def test_main_no_command_shows_help(self):
        """Test that calling CLI with no command shows help"""
        from samrfi.cli import main

        with patch('sys.argv', ['samrfi']):
            with patch('sys.exit') as mock_exit:
                try:
                    main()
                except SystemExit:
                    pass

    def test_main_create_config_command(self):
        """Test main with create-config command"""
        from samrfi.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_config.yaml'

            with patch('sys.argv', ['samrfi', 'create-config', '--output', str(output_path)]):
                result = main()

                assert result == 0
                assert output_path.exists()

    def test_main_validate_config_command(self, temp_config):
        """Test main with validate-config command"""
        from samrfi.cli import main

        with patch('sys.argv', ['samrfi', 'validate-config', '--config', temp_config]):
            result = main()

            assert result == 0  # Valid config

