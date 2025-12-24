"""
Configuration loader for SAM-RFI training and data generation
Handles YAML config files with validation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DataConfig:
    """
    Flexible config wrapper for data generation
    Preserves nested YAML structure and supports both dict and attribute access
    """

    def __init__(self, data: dict):
        self._data = data
        # Recursively wrap nested dicts
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DataConfig(value))
            else:
                setattr(self, key, value)

    # Dict-like operations for compatibility
    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def items(self):
        return self._data.items()


@dataclass
class TrainingConfig:
    """Training configuration dataclass with validation"""

    # Model configuration
    model_checkpoint: str = "large"
    freeze_encoders: bool = True

    # Training hyperparameters
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    device: str = "cuda"

    # Optimizer settings
    optimizer: str = "adam"
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    momentum: float = 0.9

    # Loss function settings
    loss_function: str = "dicece"
    loss_sigmoid: bool = True
    loss_squared_pred: bool = True
    loss_reduction: str = "mean"

    # Model architecture
    multimask_output: bool = False
    freeze_vision_encoder: bool = True
    freeze_prompt_encoder: bool = True

    # Data augmentation
    bbox_perturbation: int = 20

    # DataLoader settings
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

    # Training optimization
    log_interval: int = 100
    cuda_cache_clear_interval: int = 100

    # Dataset configuration
    stretch: str | None = "SQRT"
    flag_sigma: int = 5
    patch_method: str = "patchify"
    patch_size: int = 128
    num_patches: int | None = None
    apply_stretching: bool = True
    custom_flag: bool = True

    # Output configuration
    dir_path: str = "./samrfi_data"
    save_plots: bool = True
    plot_dpi: int = 300
    plot: bool = True
    save_model: bool = True

    # MS loading configuration
    num_antennas: int | None = None
    data_mode: str = "DATA"

    def __post_init__(self):
        """Validate configuration values (skip validation for None values)"""
        # Validate model checkpoint (required)
        if self.model_checkpoint is not None:
            valid_checkpoints = ["tiny", "small", "base_plus", "large"]
            if self.model_checkpoint not in valid_checkpoints:
                raise ValueError(
                    f"Invalid model_checkpoint '{self.model_checkpoint}'. "
                    f"Must be one of: {valid_checkpoints}"
                )

        # Validate stretch (None/null is valid for synthetic data)
        if self.stretch is not None:
            valid_stretches = ["SQRT", "LOG10"]
            if self.stretch not in valid_stretches:
                raise ValueError(
                    f"Invalid stretch '{self.stretch}'. "
                    f"Must be one of: {valid_stretches} or null"
                )

        # Validate device (required)
        if self.device is not None:
            valid_devices = ["cuda", "cpu"]
            if self.device not in valid_devices:
                raise ValueError(
                    f"Invalid device '{self.device}'. " f"Must be one of: {valid_devices}"
                )

        # Validate numeric ranges (only if not None)
        if self.num_epochs is not None and self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.flag_sigma is not None and self.flag_sigma <= 0:
            raise ValueError(f"flag_sigma must be positive, got {self.flag_sigma}")

        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")


class ConfigLoader:
    """
    Load and validate YAML configuration files for SAM-RFI
    """

    @staticmethod
    def load_training(config_path: str) -> TrainingConfig:
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
        with open(config_file) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Failed to parse YAML config: {e}") from e

        if config_dict is None:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Flatten nested structure
        flat_config = ConfigLoader._flatten_config(config_dict)

        # Create and validate TrainingConfig
        try:
            config = TrainingConfig(**flat_config)
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameters: {e}") from e

        return config

    @staticmethod
    def _flatten_config(config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten nested YAML structure to match TrainingConfig fields

        Example:
            Input: {'model': {'checkpoint': 'large'}, 'training': {'num_epochs': 5}}
            Output: {'model_checkpoint': 'large', 'num_epochs': 5}
        """
        flat = {}

        # Model section
        if "model" in config_dict:
            model_config = config_dict["model"]
            flat["model_checkpoint"] = model_config.get("checkpoint", "large")
            flat["freeze_encoders"] = model_config.get("freeze_encoders", True)

        # Training section
        if "training" in config_dict:
            training_config = config_dict["training"]
            # Basic training params
            flat["num_epochs"] = training_config.get("num_epochs", 5)
            flat["batch_size"] = training_config.get("batch_size", 4)
            flat["learning_rate"] = training_config.get("learning_rate", 1e-5)
            flat["weight_decay"] = training_config.get("weight_decay", 0.0)
            flat["device"] = training_config.get("device", "cuda")

            # Model checkpoint (can be in training or model section)
            if "model_checkpoint" in training_config:
                flat["model_checkpoint"] = training_config["model_checkpoint"]

            # Optimizer settings
            flat["optimizer"] = training_config.get("optimizer", "adam")
            flat["adam_betas"] = training_config.get("adam_betas", (0.9, 0.999))
            flat["adam_eps"] = training_config.get("adam_eps", 1e-8)
            flat["momentum"] = training_config.get("momentum", 0.9)

            # Loss function settings
            flat["loss_function"] = training_config.get("loss_function", "dicece")
            flat["loss_sigmoid"] = training_config.get("loss_sigmoid", True)
            flat["loss_squared_pred"] = training_config.get("loss_squared_pred", True)
            flat["loss_reduction"] = training_config.get("loss_reduction", "mean")

            # Model architecture
            flat["multimask_output"] = training_config.get("multimask_output", False)
            flat["freeze_vision_encoder"] = training_config.get("freeze_vision_encoder", True)
            flat["freeze_prompt_encoder"] = training_config.get("freeze_prompt_encoder", True)

            # Data augmentation
            flat["bbox_perturbation"] = training_config.get("bbox_perturbation", 20)

            # DataLoader settings
            flat["num_workers"] = training_config.get("num_workers", 0)
            flat["prefetch_factor"] = training_config.get("prefetch_factor", 2)
            flat["persistent_workers"] = training_config.get("persistent_workers", True)
            flat["pin_memory"] = training_config.get("pin_memory", True)

            # Training optimization
            flat["log_interval"] = training_config.get("log_interval", 100)
            flat["cuda_cache_clear_interval"] = training_config.get(
                "cuda_cache_clear_interval", 100
            )

            # Output settings (can be in training or output section)
            if "plot" in training_config:
                flat["plot"] = training_config["plot"]
            if "save_model" in training_config:
                flat["save_model"] = training_config["save_model"]
            if "output_dir" in training_config:
                flat["dir_path"] = training_config["output_dir"]

        # Dataset section
        if "dataset" in config_dict:
            dataset_config = config_dict["dataset"]
            # Handle stretch (can be "SQRT", "LOG10", null, or None)
            stretch = dataset_config.get("stretch", "SQRT")
            flat["stretch"] = None if stretch in (None, "null", "None") else stretch
            flat["flag_sigma"] = dataset_config.get("flag_sigma", 5)
            flat["patch_method"] = dataset_config.get("patch_method", "patchify")
            flat["patch_size"] = dataset_config.get("patch_size", 128)
            flat["num_patches"] = dataset_config.get("num_patches", None)
            flat["apply_stretching"] = dataset_config.get("apply_stretching", True)
            flat["custom_flag"] = dataset_config.get("custom_flag", True)

        # Processing section (for data generation configs)
        if "processing" in config_dict:
            processing_config = config_dict["processing"]
            # Handle stretch (can be in processing section instead of dataset)
            if "stretch" in processing_config:
                stretch = processing_config["stretch"]
                flat["stretch"] = None if stretch in (None, "null", "None") else stretch
            if "flag_sigma" in processing_config:
                flat["flag_sigma"] = processing_config["flag_sigma"]
            if "patch_size" in processing_config:
                flat["patch_size"] = processing_config["patch_size"]
            if "apply_stretching" in processing_config:
                flat["apply_stretching"] = processing_config["apply_stretching"]

        # Output section
        if "output" in config_dict:
            output_config = config_dict["output"]
            flat["dir_path"] = output_config.get("dir_path", "./samrfi_data")
            flat["save_plots"] = output_config.get("save_plots", True)
            flat["plot_dpi"] = output_config.get("plot_dpi", 300)

        # MS loading section
        if "ms_loading" in config_dict:
            ms_config = config_dict["ms_loading"]
            flat["num_antennas"] = ms_config.get("num_antennas", None)
            flat["data_mode"] = ms_config.get("data_mode", "DATA")

        return flat

    @staticmethod
    def load_data(config_path: str) -> DataConfig:
        """
        Load data generation configuration from YAML file
        Preserves nested structure for flexible data generation

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DataConfig object with nested structure

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        with open(config_file) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Failed to parse YAML config: {e}") from e

        if config_dict is None:
            raise ValueError(f"Empty configuration file: {config_path}")

        return DataConfig(config_dict)

    @staticmethod
    def load(config_path: str) -> TrainingConfig:
        """
        Load training configuration (alias for load_training)
        Maintained for backwards compatibility

        Args:
            config_path: Path to YAML configuration file

        Returns:
            TrainingConfig object with validated parameters
        """
        return ConfigLoader.load_training(config_path)

    @staticmethod
    def save(config: TrainingConfig, output_path: str):
        """
        Save TrainingConfig to YAML file

        Args:
            config: TrainingConfig object
            output_path: Path to save YAML file
        """
        # Convert to nested structure matching actual config files
        config_dict = {
            "training": {
                # Basic params
                "device": config.device,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "model_checkpoint": config.model_checkpoint,
                # Optimizer settings
                "optimizer": config.optimizer,
                "weight_decay": config.weight_decay,
                "adam_betas": list(config.adam_betas),
                "adam_eps": config.adam_eps,
                # Loss function
                "loss_function": config.loss_function,
                "loss_sigmoid": config.loss_sigmoid,
                "loss_squared_pred": config.loss_squared_pred,
                "loss_reduction": config.loss_reduction,
                # Model architecture
                "multimask_output": config.multimask_output,
                "freeze_vision_encoder": config.freeze_vision_encoder,
                "freeze_prompt_encoder": config.freeze_prompt_encoder,
                # Data augmentation
                "bbox_perturbation": config.bbox_perturbation,
                # DataLoader settings
                "num_workers": config.num_workers,
                "prefetch_factor": config.prefetch_factor,
                "persistent_workers": config.persistent_workers,
                "pin_memory": config.pin_memory,
                # Training optimization
                "log_interval": config.log_interval,
                "cuda_cache_clear_interval": config.cuda_cache_clear_interval,
                # Output
                "plot": config.plot,
                "save_model": config.save_model,
            },
            "dataset": {
                "stretch": config.stretch,
                "flag_sigma": config.flag_sigma,
                "patch_method": config.patch_method,
                "patch_size": config.patch_size,
                "num_patches": config.num_patches,
                "apply_stretching": config.apply_stretching,
                "custom_flag": config.custom_flag,
            },
            "output": {
                "dir_path": config.dir_path,
                "save_plots": config.save_plots,
                "plot_dpi": config.plot_dpi,
            },
        }

        # Add ms_loading only if num_antennas is set
        if config.num_antennas is not None:
            config_dict["ms_loading"] = {
                "num_antennas": config.num_antennas,
                "data_mode": config.data_mode,
            }

        # Write YAML
        with open(output_path, "w") as f:
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
