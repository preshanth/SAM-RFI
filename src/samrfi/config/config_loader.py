"""
Configuration loader for SAM-RFI training
Handles YAML config files with validation
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Training configuration dataclass with validation"""

    # Model configuration
    model_checkpoint: str = 'large'
    freeze_encoders: bool = True

    # Training hyperparameters
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    device: str = 'cuda'

    # Dataset configuration
    stretch: str = 'SQRT'
    flag_sigma: int = 5
    patch_method: str = 'patchify'
    patch_size: int = 128
    num_patches: Optional[int] = None
    apply_stretching: bool = True
    custom_flag: bool = True

    # Output configuration
    dir_path: str = './samrfi_data'
    save_plots: bool = True
    plot_dpi: int = 300

    # MS loading configuration
    num_antennas: Optional[int] = None
    data_mode: str = 'DATA'

    def __post_init__(self):
        """Validate configuration values"""
        # Validate model checkpoint
        valid_checkpoints = ['tiny', 'small', 'base_plus', 'large']
        if self.model_checkpoint not in valid_checkpoints:
            raise ValueError(
                f"Invalid model_checkpoint '{self.model_checkpoint}'. "
                f"Must be one of: {valid_checkpoints}"
            )

        # Validate stretch
        valid_stretches = ['SQRT', 'LOG10']
        if self.stretch not in valid_stretches:
            raise ValueError(
                f"Invalid stretch '{self.stretch}'. "
                f"Must be one of: {valid_stretches}"
            )

        # Validate device
        valid_devices = ['cuda', 'cpu']
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Must be one of: {valid_devices}"
            )

        # Validate numeric ranges
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.flag_sigma <= 0:
            raise ValueError(f"flag_sigma must be positive, got {self.flag_sigma}")

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")


class ConfigLoader:
    """
    Load and validate YAML configuration files for SAM-RFI training
    """

    @staticmethod
    def load(config_path: str) -> TrainingConfig:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            TrainingConfig object with validated parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        with open(config_file, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Failed to parse YAML config: {e}")

        if config_dict is None:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Flatten nested structure
        flat_config = ConfigLoader._flatten_config(config_dict)

        # Create and validate TrainingConfig
        try:
            config = TrainingConfig(**flat_config)
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameters: {e}")

        return config

    @staticmethod
    def _flatten_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested YAML structure to match TrainingConfig fields

        Example:
            Input: {'model': {'checkpoint': 'large'}, 'training': {'num_epochs': 5}}
            Output: {'model_checkpoint': 'large', 'num_epochs': 5}
        """
        flat = {}

        # Model section
        if 'model' in config_dict:
            model_config = config_dict['model']
            flat['model_checkpoint'] = model_config.get('checkpoint', 'large')
            flat['freeze_encoders'] = model_config.get('freeze_encoders', True)

        # Training section
        if 'training' in config_dict:
            training_config = config_dict['training']
            flat['num_epochs'] = training_config.get('num_epochs', 5)
            flat['batch_size'] = training_config.get('batch_size', 4)
            flat['learning_rate'] = training_config.get('learning_rate', 1e-5)
            flat['weight_decay'] = training_config.get('weight_decay', 0.0)
            flat['device'] = training_config.get('device', 'cuda')

        # Dataset section
        if 'dataset' in config_dict:
            dataset_config = config_dict['dataset']
            flat['stretch'] = dataset_config.get('stretch', 'SQRT')
            flat['flag_sigma'] = dataset_config.get('flag_sigma', 5)
            flat['patch_method'] = dataset_config.get('patch_method', 'patchify')
            flat['patch_size'] = dataset_config.get('patch_size', 128)
            flat['num_patches'] = dataset_config.get('num_patches', None)
            flat['apply_stretching'] = dataset_config.get('apply_stretching', True)
            flat['custom_flag'] = dataset_config.get('custom_flag', True)

        # Output section
        if 'output' in config_dict:
            output_config = config_dict['output']
            flat['dir_path'] = output_config.get('dir_path', './samrfi_data')
            flat['save_plots'] = output_config.get('save_plots', True)
            flat['plot_dpi'] = output_config.get('plot_dpi', 300)

        # MS loading section
        if 'ms_loading' in config_dict:
            ms_config = config_dict['ms_loading']
            flat['num_antennas'] = ms_config.get('num_antennas', None)
            flat['data_mode'] = ms_config.get('data_mode', 'DATA')

        return flat

    @staticmethod
    def save(config: TrainingConfig, output_path: str):
        """
        Save TrainingConfig to YAML file

        Args:
            config: TrainingConfig object
            output_path: Path to save YAML file
        """
        # Convert to nested structure
        config_dict = {
            'model': {
                'checkpoint': config.model_checkpoint,
                'freeze_encoders': config.freeze_encoders
            },
            'training': {
                'num_epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'device': config.device
            },
            'dataset': {
                'stretch': config.stretch,
                'flag_sigma': config.flag_sigma,
                'patch_method': config.patch_method,
                'patch_size': config.patch_size,
                'num_patches': config.num_patches,
                'apply_stretching': config.apply_stretching,
                'custom_flag': config.custom_flag
            },
            'output': {
                'dir_path': config.dir_path,
                'save_plots': config.save_plots,
                'plot_dpi': config.plot_dpi
            }
        }

        # Add ms_loading only if num_antennas is set
        if config.num_antennas is not None:
            config_dict['ms_loading'] = {
                'num_antennas': config.num_antennas,
                'data_mode': config.data_mode
            }

        # Write YAML
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


    @staticmethod
    def create_default_config(output_path: str):
        """
        Create a default configuration file

        Args:
            output_path: Path to save default config YAML
        """
        default_config = TrainingConfig()
        ConfigLoader.save(default_config, output_path)