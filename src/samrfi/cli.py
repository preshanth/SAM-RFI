"""
Command-line interface for SAM-RFI.

This module provides the main CLI entry point for SAM-RFI operations including
data generation, model training, prediction, evaluation, and publishing to
HuggingFace Hub.

Functions
---------
generate_data_command
    Generate synthetic or MS-based training datasets.
train_command
    Train SAM2 models on pre-generated datasets.
predict_command
    Apply trained models for RFI prediction on measurement sets.
evaluate_command
    Evaluate prediction accuracy against ground truth.
publish_command
    Publish datasets or models to HuggingFace Hub.
create_config_command
    Create default YAML configuration files.
validate_config_command
    Validate YAML configuration files.
load_dataset
    Load datasets from disk (BatchedDataset or RAMCachedDataset).
main
    Main CLI entry point and argument parser.

Examples
--------
Generate a synthetic training dataset:

>>> # Command line
>>> samrfi generate-data --source synthetic --config configs/synthetic.yaml --output ./data

Train a model:

>>> # Command line
>>> samrfi train --config configs/training.yaml --dataset ./data/exact_masks

Predict RFI flags:

>>> # Command line
>>> samrfi predict --model ./models/sam2_rfi.pth --input observation.ms

Notes
-----
The CLI is organized around subcommands that correspond to major workflows:
- generate-data: Dataset creation from synthetic or measurement set sources
- train: Model training with validation support
- predict: RFI flagging with single-pass or iterative modes
- evaluate: Metrics computation against ground truth
- publish: Dataset/model publishing to HuggingFace Hub
- create-config: Configuration file generation
- validate-config: Configuration validation

See Also
--------
samrfi.config.config_loader : Configuration loading and validation
samrfi.training.sam2_trainer : SAM2 model training
samrfi.inference : RFI prediction and flagging
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import validate_all
from .config.config_loader import ConfigLoader
from .data import MSLoader
from .data_generation.ms_generator import MSDataGenerator
from .data_generation.synthetic_generator import SyntheticDataGenerator
from .evaluation.metrics import evaluate_segmentation
from .inference import RFIPredictor
from .training.sam2_trainer import SAM2Trainer
from .utils import logger, setup_logger
from .utils.errors import ConfigValidationError


def generate_data_command(args: argparse.Namespace) -> None:
    """
    Execute data generation command.

    Generates training/validation datasets from either synthetic RFI
    simulations or real measurement set observations. Creates both
    exact ground truth masks and MAD-based masks.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - config : str
            Path to YAML configuration file
        - source : str
            Data source type ('synthetic' or 'ms')
        - output : str
            Output directory path for generated datasets

    Raises
    ------
    ValueError
        If source is not 'synthetic' or 'ms'.
    FileNotFoundError
        If configuration file doesn't exist.

    Examples
    --------
    Generate synthetic dataset:

    >>> # Command line
    >>> samrfi generate-data --source synthetic \\
    ...     --config configs/synthetic_train_4k.yaml \\
    ...     --output ./datasets/train_4k

    Generate dataset from measurement set:

    >>> # Command line
    >>> samrfi generate-data --source ms \\
    ...     --config configs/ms_data.yaml \\
    ...     --output ./datasets/my_ms_data

    Notes
    -----
    Output directory structure:
    - exact_masks/ : Perfect ground truth masks
    - mad_masks/ : Median Absolute Deviation based masks

    See Also
    --------
    SyntheticDataGenerator : Synthetic RFI data generation
    MSDataGenerator : Measurement set data generation
    """
    print("=" * 60)
    print("SAM-RFI Data Generation")
    print("=" * 60)

    # Load data generation config
    print(f"\nLoading configuration from: {args.config}")
    config = ConfigLoader.load_data(args.config)

    if args.source == "synthetic":
        print("\nGenerating synthetic dataset...")
        generator = SyntheticDataGenerator(config)
        generator.generate(output_path=args.output)
    elif args.source == "ms":
        print("\nGenerating dataset from Measurement Set...")
        generator = MSDataGenerator(config)
        generator.generate(output_path=args.output)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print("  exact_masks/ - Perfect ground truth")
    print("  mad_masks/ - MAD-based masks")


def load_dataset(path: str) -> Any:
    """
    Load dataset from batched .pt directory.

    Automatically detects and loads either BatchedDataset (preprocessed)
    or RAMCachedDataset (raw) formats based on metadata.json format field.

    Parameters
    ----------
    path : str
        Path to dataset directory containing batch_*.pt files and metadata.json.

    Returns
    -------
    BatchedDataset or RAMCachedDataset
        Loaded dataset ready for training or validation.

    Raises
    ------
    ValueError
        If path is not a directory, missing metadata.json, or invalid format.

    Examples
    --------
    Load preprocessed dataset:

    >>> dataset = load_dataset('./datasets/train_4k/exact_masks')
    Loading BatchedDataset (preprocessed format) from ./datasets/train_4k/exact_masks

    Load raw dataset with GPU transforms:

    >>> dataset = load_dataset('./datasets/raw_data')
    Loading RAMCachedDataset (raw format) from ./datasets/raw_data

    Notes
    -----
    Supported formats:
    - BatchedDataset (preprocessed): batch_*.pt + metadata.json with format='preprocessed'
    - RAMCachedDataset (raw): batch_*.pt + metadata.json with format='raw'

    Legacy single .pt files are no longer supported. Use generate-data command
    to create modern batched datasets.

    Expected directory structure:
    - dataset_dir/
      - batch_000.pt
      - batch_001.pt
      - ...
      - metadata.json

    See Also
    --------
    BatchedDataset : Streaming preprocessed dataset loader
    RAMCachedDataset : RAM-cached raw dataset with GPU transforms
    """
    from samrfi.data import BatchedDataset

    path = Path(path)

    # Must be a directory
    if not path.is_dir():
        raise ValueError(
            f"Dataset must be a directory, got: {path}\n\n"
            f"Legacy single .pt files are no longer supported.\n"
            f"Regenerate your dataset with:\n"
            f"  samrfi generate-data --source [synthetic|ms] --config <config> --output <path>"
        )

    # Must have metadata.json
    metadata_file = path / "metadata.json"
    if not metadata_file.exists():
        raise ValueError(
            f"Invalid dataset directory: {path}\n"
            f"Missing metadata.json file.\n\n"
            f"Expected BatchedDataset format:\n"
            f"  {path}/\n"
            f"  ├── batch_000.pt\n"
            f"  ├── batch_001.pt\n"
            f"  ├── ...\n"
            f"  └── metadata.json\n\n"
            f"If this is an old HuggingFace dataset, regenerate with:\n"
            f"  samrfi generate-data --source [synthetic|ms] --config <config> --output <path>"
        )

    # Load metadata and determine format
    import json

    with open(metadata_file) as f:
        metadata = json.load(f)

    data_format = metadata.get("format", "preprocessed")

    if data_format == "raw":
        # Raw batches: load into RAM + GPU transforms on-the-fly
        from samrfi.data import RAMCachedDataset

        print(f"  Loading RAMCachedDataset (raw format) from {path}")
        return RAMCachedDataset(path, device="cuda")
    else:
        # Preprocessed batches: streaming from disk
        print(f"  Loading BatchedDataset (preprocessed format) from {path}")
        return BatchedDataset(path)


def train_command(args: argparse.Namespace) -> None:
    """
    Execute training command on pre-generated dataset.

    Trains SAM2 models for RFI detection using pre-generated training
    datasets. Supports optional validation dataset and checkpoint resumption.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - config : str
            Path to YAML training configuration file
        - dataset : str
            Path to training dataset directory
        - validation_dataset : str, optional
            Path to validation dataset directory
        - resume : str, optional
            Path to checkpoint file to resume training from
        - device : str, optional
            Device override ('cuda' or 'cpu')
        - output_dir : str, optional
            Output directory override for models and plots

    Raises
    ------
    ValueError
        If dataset path is missing or invalid.
    ConfigValidationError
        If configuration validation fails.

    Examples
    --------
    Train with basic configuration:

    >>> # Command line
    >>> samrfi train --config configs/sam2_training.yaml \\
    ...     --dataset ./datasets/train_4k/exact_masks

    Train with validation dataset:

    >>> # Command line
    >>> samrfi train --config configs/sam2_training.yaml \\
    ...     --dataset ./datasets/train_4k/exact_masks \\
    ...     --validation-dataset ./datasets/val_1k/exact_masks

    Resume training from checkpoint:

    >>> # Command line
    >>> samrfi train --config configs/sam2_training.yaml \\
    ...     --dataset ./datasets/train_4k/exact_masks \\
    ...     --resume ./models/checkpoint_epoch_10.pth

    Notes
    -----
    The training process:
    1. Loads and validates configuration
    2. Loads training dataset (and optional validation dataset)
    3. Initializes SAM2 model and trainer
    4. Trains for specified epochs with optional validation
    5. Saves model checkpoints and training plots

    Models are saved to: <output_dir>/models/
    Plots are saved to: <output_dir>/plots/

    See Also
    --------
    SAM2Trainer : SAM2 model training implementation
    ConfigLoader : Configuration loading and validation
    load_dataset : Dataset loading utility
    """

    print("=" * 60)
    print("SAM-RFI SAM2 Training")
    print("=" * 60)

    # Load configuration
    logger.info(f"\nLoading configuration from: {args.config}")
    config = ConfigLoader.load(args.config)

    # Validate configuration
    try:
        validate_all(config)
        logger.info("Configuration validation passed")
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Override config with command-line arguments
    if not args.dataset:
        raise ValueError("--dataset is required for training (path to HuggingFace dataset)")

    if args.device:
        config.device = args.device

    if args.output_dir:
        config.dir_path = args.output_dir

    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Dataset: {args.dataset}")
    if args.validation_dataset:
        print(f"  Validation dataset: {args.validation_dataset}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print(f"  Model: sam2-{config.model_checkpoint}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"  Loaded {len(dataset)} training patches")

    # Load validation dataset if provided
    val_dataset = None
    if args.validation_dataset:
        print(f"\nLoading validation dataset from: {args.validation_dataset}")
        val_dataset = load_dataset(args.validation_dataset)
        print(f"  Loaded {len(val_dataset)} validation patches")

    # Create minimal wrapper for SAM2Trainer compatibility
    class DatasetWrapper:
        def __init__(self, ds):
            self.dataset = ds

    dataset_wrapper = DatasetWrapper(dataset)

    # Train model
    print("\nInitializing SAM2 trainer...")
    trainer = SAM2Trainer(dataset_wrapper, device=config.device, dir_path=config.dir_path)

    losses = trainer.train(
        # Basic training params
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        sam_checkpoint=config.model_checkpoint,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        # Optimizer settings
        optimizer=config.optimizer,
        adam_betas=config.adam_betas,
        adam_eps=config.adam_eps,
        momentum=config.momentum,
        # Loss function settings
        loss_function=config.loss_function,
        loss_sigmoid=config.loss_sigmoid,
        loss_squared_pred=config.loss_squared_pred,
        loss_reduction=config.loss_reduction,
        # Model architecture
        multimask_output=config.multimask_output,
        freeze_vision_encoder=config.freeze_vision_encoder,
        freeze_prompt_encoder=config.freeze_prompt_encoder,
        # Data augmentation
        bbox_perturbation=config.bbox_perturbation,
        # DataLoader settings
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
        # Training optimization
        log_interval=config.log_interval,
        cuda_cache_clear_interval=config.cuda_cache_clear_interval,
        # Output settings
        plot=config.plot,
        save_model=config.save_model,
        validation_dataset=val_dataset,
        model_path=args.resume,  # Resume from checkpoint if provided
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Handle different return formats
    if isinstance(losses, dict):
        print(f"Final train loss: {losses['train'][-1]:.6f}")
        print(f"Best train loss: {min(losses['train']):.6f}")
        print(f"Final val loss: {losses['val'][-1]:.6f}")
        print(f"Best val loss: {min(losses['val']):.6f}")
    else:
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Best loss: {min(losses):.6f}")

    print(f"Models saved to: {config.dir_path}/models/")


def create_config_command(args: argparse.Namespace) -> None:
    """
    Create default configuration file.

    Generates a YAML configuration file with default training parameters
    that can be customized for specific training workflows.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - output : str, optional
            Output path for configuration file (default: 'sam2_config.yaml')

    Examples
    --------
    Create default configuration:

    >>> # Command line
    >>> samrfi create-config

    Create configuration with custom path:

    >>> # Command line
    >>> samrfi create-config --output my_config.yaml

    Notes
    -----
    The generated configuration includes all TrainingConfig fields with
    default values. Edit the file to customize:
    - Model settings (checkpoint size, frozen encoders)
    - Training hyperparameters (epochs, batch size, learning rate)
    - Optimizer configuration (Adam/SGD, weight decay)
    - Loss function settings
    - Dataset preprocessing (stretch, patch size, sigma)
    - Output settings (save paths, plotting)

    See Also
    --------
    ConfigLoader.create_default_config : Configuration file generator
    TrainingConfig : Complete configuration schema
    """
    output_path = args.output or "sam2_config.yaml"

    print(f"Creating default configuration: {output_path}")
    ConfigLoader.create_default_config(output_path)
    print(f"✓ Configuration file created: {output_path}")
    print("\nEdit this file to customize training parameters, then run:")
    print(f"  samrfi train --config {output_path} --ms-path <path-to-ms>")


def validate_config_command(args: argparse.Namespace) -> int:
    """
    Validate configuration file.

    Checks YAML configuration file for syntax errors, missing fields,
    and invalid parameter values. Prints configuration summary if valid.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - config : str
            Path to YAML configuration file to validate

    Returns
    -------
    int
        Exit code: 0 if valid, 1 if invalid.

    Examples
    --------
    Validate training configuration:

    >>> # Command line
    >>> samrfi validate-config --config configs/sam2_training.yaml
    ✓ Configuration is valid

    Configuration summary:
      Model: sam2-large
      Epochs: 10
      Batch size: 8
      Learning rate: 0.0001
      Device: cuda

    Notes
    -----
    Validation checks:
    - YAML syntax parsing
    - Required fields present
    - Value types correct (int, float, str, bool)
    - Enum values valid (model checkpoint, device, optimizer, etc.)
    - Numeric ranges reasonable (positive epochs, learning rate < 1)

    See Also
    --------
    ConfigLoader.load : Configuration loading with validation
    validate_all : Full configuration validation suite
    """
    print(f"Validating configuration: {args.config}")

    try:
        config = ConfigLoader.load(args.config)
        print("✓ Configuration is valid")
        print("\nConfiguration summary:")
        print(f"  Model: sam2-{config.model_checkpoint}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Device: {config.device}")
        return 0
    except Exception as e:
        print(f"✗ Configuration is invalid: {e}")
        return 1


def publish_command(args: argparse.Namespace) -> None:
    """
    Dispatcher for publishing datasets or models to HuggingFace Hub.

    Routes to appropriate publishing function based on --type argument
    (dataset or model).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type : str
            Publication type ('dataset' or 'model')
        - Additional arguments passed to specific publish functions

    Raises
    ------
    ValueError
        If publish type is not 'dataset' or 'model'.

    See Also
    --------
    publish_dataset_command : Dataset publishing to HuggingFace Hub
    publish_model_command : Model publishing to HuggingFace Hub
    """
    publish_type = getattr(args, "type", "dataset")

    if publish_type == "dataset":
        publish_dataset_command(args)
    elif publish_type == "model":
        publish_model_command(args)
    else:
        raise ValueError(f"Unknown publish type: {publish_type}")


def publish_dataset_command(args: argparse.Namespace) -> None:
    """
    Publish dataset to HuggingFace Hub.

    Converts BatchedDataset or RAMCachedDataset to HuggingFace Dataset
    format and uploads to the Hub for sharing and reproducibility.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - input : str
            Path to local dataset directory
        - repo_id : str
            HuggingFace repository ID (username/repo-name)
        - private : bool
            Whether to make repository private
        - token : str, optional
            HuggingFace API token (or use HF_TOKEN env var)
        - batch_size : int
            Batch size for conversion (default: 50)

    Examples
    --------
    Publish public dataset:

    >>> # Command line
    >>> samrfi publish --type dataset \\
    ...     --input ./datasets/train_4k/exact_masks \\
    ...     --repo-id username/sam-rfi-dataset

    Publish private dataset with token:

    >>> # Command line
    >>> samrfi publish --type dataset \\
    ...     --input ./datasets/train_4k/exact_masks \\
    ...     --repo-id username/sam-rfi-dataset \\
    ...     --private --token hf_xxxxx

    Notes
    -----
    Publishing process:
    1. Load local dataset (auto-detect format)
    2. Convert to HuggingFace Dataset format
    3. Upload to HuggingFace Hub
    4. Generate dataset card with metadata

    The published dataset can be loaded with:
    >>> from datasets import load_dataset
    >>> dataset = load_dataset('username/sam-rfi-dataset')

    See Also
    --------
    HFDatasetWrapper : HuggingFace dataset conversion wrapper
    load_dataset : Local dataset loading
    """
    from .data.hf_dataset_wrapper import HFDatasetWrapper

    print("=" * 60)
    print("SAM-RFI Dataset Publishing")
    print("=" * 60)

    # Load dataset (auto-detect format)
    print(f"\nLoading dataset from {args.input}")
    dataset = load_dataset(args.input)
    print(f"  Loaded: {type(dataset).__name__}")

    # Convert to HF format
    print("\nConverting to HuggingFace Dataset format...")
    hf_dataset = HFDatasetWrapper.from_dataset(dataset, batch_size=args.batch_size)

    # Push to hub
    print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
    hf_dataset.push_to_hub(args.repo_id, private=args.private, token=args.token)

    print("\n" + "=" * 60)
    print("✓ Dataset Published!")
    print("=" * 60)
    print(f"URL: https://huggingface.co/datasets/{args.repo_id}")


def publish_model_command(args: argparse.Namespace) -> None:
    """
    Publish trained model to HuggingFace Hub.

    Uploads trained SAM2 model checkpoint to HuggingFace Hub with
    auto-generated model card containing training metadata and usage examples.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - input : str
            Path to local model checkpoint (.pth file)
        - repo_id : str
            HuggingFace repository ID (username/repo-name)
        - model_size : str, optional
            Model size ('tiny', 'small', 'base_plus', 'large')
            Auto-detected from checkpoint if not specified
        - private : bool
            Whether to make repository private
        - token : str, optional
            HuggingFace API token (or use HF_TOKEN env var)

    Raises
    ------
    ValueError
        If model size cannot be detected and not specified.

    Examples
    --------
    Publish model with auto-detection:

    >>> # Command line
    >>> samrfi publish --type model \\
    ...     --input ./models/sam2_rfi_best.pth \\
    ...     --repo-id username/sam-rfi-models

    Publish with explicit model size:

    >>> # Command line
    >>> samrfi publish --type model \\
    ...     --input ./models/sam2_rfi_best.pth \\
    ...     --repo-id username/sam-rfi-models \\
    ...     --model-size large

    Notes
    -----
    Publishing process:
    1. Load checkpoint and extract metadata
    2. Auto-detect model size from checkpoint config
    3. Generate model card with training info and usage examples
    4. Create HuggingFace repository (if doesn't exist)
    5. Upload model to {model_size}/model.pth
    6. Upload README.md with model card

    Model organization on Hub:
    - repo-name/
      - tiny/model.pth
      - small/model.pth
      - base_plus/model.pth
      - large/model.pth
      - README.md

    The published model can be used with:
    >>> samrfi predict --model username/sam-rfi-models/large --input obs.ms

    See Also
    --------
    generate_model_card : Model card generation
    RFIPredictor : Model loading and inference
    """
    import torch
    from huggingface_hub import HfApi, create_repo

    from .utils.model_card import generate_model_card

    print("=" * 60)
    print("SAM-RFI Model Publishing")
    print("=" * 60)

    # Load checkpoint to extract metadata
    print(f"\nLoading checkpoint from: {args.input}")
    checkpoint = torch.load(args.input, map_location="cpu")
    print("  ✓ Checkpoint loaded")

    # Auto-detect model size from config (or use --model-size)
    model_size = args.model_size or checkpoint.get("config", {}).get("sam_checkpoint", "unknown")

    if model_size == "unknown":
        logger.warning("Could not detect model size from checkpoint. Use --model-size flag.")
        raise ValueError(
            "Model size required for upload. Use --model-size {tiny,small,base_plus,large}"
        )

    print(f"  Detected model size: {model_size}")

    # Generate model card
    print("\nGenerating model card...")
    model_card = generate_model_card(checkpoint, model_size)
    print("  ✓ Model card generated")

    # Create repo if doesn't exist
    print(f"\nPreparing HuggingFace repository: {args.repo_id}")
    api = HfApi(token=args.token)

    try:
        create_repo(
            args.repo_id, repo_type="model", exist_ok=True, private=args.private, token=args.token
        )
        print("  ✓ Repository ready")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Upload model file to size-specific subdirectory
    print(f"\nUploading model to {model_size}/model.pth...")
    try:
        api.upload_file(
            path_or_fileobj=args.input,
            path_in_repo=f"{model_size}/model.pth",
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
        )
        print("  ✓ Model uploaded")
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise

    # Upload model card (README.md)
    print("\nUploading model card (README.md)...")
    try:
        from io import BytesIO

        model_card_bytes = BytesIO(model_card.encode("utf-8"))
        api.upload_file(
            path_or_fileobj=model_card_bytes,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token,
        )
        print("  ✓ Model card uploaded")
    except Exception as e:
        logger.error(f"Failed to upload model card: {e}")
        raise

    print("\n" + "=" * 60)
    print("✓ Model Published!")
    print("=" * 60)
    print(f"Model size: {model_size}")
    print(f"URL: https://huggingface.co/{args.repo_id}")
    print(f"Path in repo: {model_size}/model.pth")
    print("\nUsage:")
    print(f"  samrfi predict --model {args.repo_id}/{model_size} --input observation.ms")


def predict_command(args: argparse.Namespace) -> None:
    """
    Execute RFI prediction command.

    Applies trained SAM2 model to flag RFI in measurement sets.
    Supports single-pass and iterative flagging modes with adaptive
    or fixed probability thresholds.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - model : str
            Path to trained model (.pth) or HuggingFace repo ID
        - input : str
            Path to input measurement set
        - checkpoint : str
            SAM2 checkpoint size ('tiny', 'small', 'base_plus', 'large')
        - iterations : int, optional
            Number of iterative flagging passes (default: 1 = single-pass)
        - num_antennas : int, optional
            Number of antennas to load (default: all)
        - patch_size : int
            Patch size in pixels (default: 128)
        - stretch : str
            Stretch function ('SQRT', 'LOG10', 'None')
        - threshold : float, optional
            RFI probability threshold (default: None = adaptive mean)
        - device : str
            Compute device ('cuda' or 'cpu')
        - batch_size : int
            Batch size for inference (default: 4)
        - apply_existing : bool
            Apply existing MS flags before prediction
        - no_save : bool
            Don't save flags to MS (prediction only)

    Examples
    --------
    Single-pass prediction with local model:

    >>> # Command line
    >>> samrfi predict --model ./models/sam2_rfi.pth --input observation.ms

    Single-pass with HuggingFace model:

    >>> # Command line
    >>> samrfi predict --model polarimetic/sam-rfi/large --input observation.ms

    Iterative flagging (3 passes):

    >>> # Command line
    >>> samrfi predict --model ./models/sam2_rfi.pth \\
    ...     --input observation.ms --iterations 3

    Fixed threshold prediction:

    >>> # Command line
    >>> samrfi predict --model ./models/sam2_rfi.pth \\
    ...     --input observation.ms --threshold 0.5

    Prediction without saving flags:

    >>> # Command line
    >>> samrfi predict --model ./models/sam2_rfi.pth \\
    ...     --input observation.ms --no-save

    Notes
    -----
    Flagging modes:
    - Single-pass (iterations=1): One forward pass through all data
    - Iterative (iterations>1): Multiple passes, refining flags each iteration

    Threshold modes:
    - Adaptive (threshold=None): Uses mean of predicted probabilities
    - Fixed (threshold=0.0-1.0): Uses specified threshold value

    The prediction process:
    1. Load measurement set and trained model
    2. Extract patches from visibility data
    3. Run SAM2 inference to predict RFI probabilities
    4. Apply threshold to generate binary flags
    5. Save flags to measurement set (unless --no-save)

    See Also
    --------
    RFIPredictor : Prediction and inference implementation
    RFIPredictor.predict_ms : Single-pass prediction
    RFIPredictor.predict_iterative : Iterative prediction
    """
    print("=" * 60)
    print("SAM-RFI RFI Prediction")
    print("=" * 60)

    # Load predictor
    print(f"\nLoading model from: {args.model}")
    predictor = RFIPredictor(
        model_path=args.model,
        sam_checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Convert "None" string to None
    stretch = None if args.stretch == "None" else args.stretch

    # Convert threshold to None if not specified or "None"
    threshold = (
        None
        if not hasattr(args, "threshold") or args.threshold is None or args.threshold == "None"
        else args.threshold
    )

    # Log threshold setting
    if threshold is None:
        print("\nThreshold: Adaptive (will use mean of probabilities)")
    else:
        print(f"\nThreshold: {threshold:.3f} (fixed)")

    # Determine if iterative
    num_iterations = args.iterations if args.iterations else 1
    is_iterative = num_iterations > 1

    if is_iterative:
        print(f"\nMode: Iterative flagging ({num_iterations} passes)")
        flags = predictor.predict_iterative(
            ms_path=args.input,
            num_iterations=num_iterations,
            num_antennas=args.num_antennas,
            patch_size=args.patch_size,
            stretch=stretch,
            save_flags=not args.no_save,
            apply_existing_flags=args.apply_existing,
            threshold=threshold,
        )
    else:
        print("\nMode: Single-pass flagging")
        flags = predictor.predict_ms(
            ms_path=args.input,
            num_antennas=args.num_antennas,
            patch_size=args.patch_size,
            stretch=stretch,
            apply_existing_flags=args.apply_existing,
            save_flags=not args.no_save,
            threshold=threshold,
        )

    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print("=" * 60)
    print(f"Total flagged: {flags.sum()/flags.size*100:.2f}%")
    if not args.no_save:
        print(f"Flags saved to: {args.input}")


def evaluate_command(args: argparse.Namespace) -> int:
    """
    Execute evaluation command.

    Computes segmentation metrics by comparing predicted RFI flags
    against ground truth masks. Saves results to CSV.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - input : str
            Path to measurement set with predicted flags
        - ground_truth : str
            Path to ground truth .npy file
        - output : str
            Output CSV file path (default: 'metrics.csv')

    Returns
    -------
    int
        Exit code: 0 if successful, 1 if error.

    Examples
    --------
    Evaluate predictions against ground truth:

    >>> # Command line
    >>> samrfi evaluate \\
    ...     --input observation.ms \\
    ...     --ground-truth ground_truth.npy \\
    ...     --output metrics.csv

    Notes
    -----
    Computed metrics:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    - IoU (Jaccard): TP / (TP + FP + FN)
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Specificity: TN / (TN + FP)

    Where:
    - TP: True Positives (correctly flagged RFI)
    - TN: True Negatives (correctly unflagged clean data)
    - FP: False Positives (incorrectly flagged clean data)
    - FN: False Negatives (missed RFI)

    Output CSV format:
    - ms_path: Path to measurement set
    - ground_truth_path: Path to ground truth file
    - precision, recall, f1, iou, accuracy, specificity: Metric values

    See Also
    --------
    evaluate_segmentation : Metrics computation implementation
    MSLoader.load_flags : Flag loading from measurement sets
    """
    print("=" * 60)
    print("SAM-RFI Evaluation")
    print("=" * 60)

    # Load ground truth
    print(f"\n[1/3] Loading ground truth from: {args.ground_truth}")
    ground_truth = np.load(args.ground_truth)
    print(f"  Ground truth shape: {ground_truth.shape}")
    gt_percent = np.sum(ground_truth) / ground_truth.size * 100
    print(f"  Ground truth RFI: {gt_percent:.2f}%")

    # Load predicted flags from MS
    print(f"\n[2/3] Loading predicted flags from MS: {args.input}")
    loader = MSLoader(args.input)
    loader.load()
    predicted_flags = loader.load_flags()
    print(f"  Predicted flags shape: {predicted_flags.shape}")
    pred_percent = np.sum(predicted_flags) / predicted_flags.size * 100
    print(f"  Predicted RFI: {pred_percent:.2f}%")

    # Check shape compatibility
    if ground_truth.shape != predicted_flags.shape:
        print("\n✗ Error: Shape mismatch!")
        print(f"  Ground truth: {ground_truth.shape}")
        print(f"  Predicted: {predicted_flags.shape}")
        return 1

    # Compute metrics
    print("\n[3/3] Computing metrics...")
    metrics = evaluate_segmentation(predicted_flags, ground_truth)

    # Display metrics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper():12s}: {value:.4f}")

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])
    df.insert(0, "ms_path", args.input)
    df.insert(1, "ground_truth_path", args.ground_truth)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Metrics saved to: {output_path}")
    print("=" * 60)


def main() -> int:
    """
    Main CLI entry point.

    Parses command-line arguments and dispatches to appropriate command
    handlers for SAM-RFI operations.

    Returns
    -------
    int
        Exit code: 0 if successful, 1 if error.

    Examples
    --------
    Display help:

    >>> # Command line
    >>> samrfi --help

    Run a command:

    >>> # Command line
    >>> samrfi train --config config.yaml --dataset ./data

    Notes
    -----
    Available commands:
    - generate-data: Generate training datasets
    - train: Train SAM2 models
    - predict: Apply models for RFI flagging
    - evaluate: Compute metrics against ground truth
    - publish: Upload datasets/models to HuggingFace Hub
    - create-config: Generate default configuration files
    - validate-config: Validate configuration files

    Global options (available for all commands):
    - --log-level: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)
    - --log-file: Write logs to file in addition to console

    See Also
    --------
    generate_data_command : Data generation
    train_command : Model training
    predict_command : RFI prediction
    evaluate_command : Metrics evaluation
    publish_command : HuggingFace Hub publishing
    """
    parser = argparse.ArgumentParser(
        description="SAM-RFI: SAM2 training and prediction for Radio Frequency Interference detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic dataset
  samrfi generate-data --source synthetic --config configs/synthetic_train_4k.yaml --output ./datasets/train_4k

  # Generate dataset from MS
  samrfi generate-data --source ms --config configs/ms_data.yaml --output ./datasets/my_ms_data

  # Train with pre-generated dataset (.pt format)
  samrfi train --config configs/sam2_training.yaml --dataset ./datasets/train_4k/exact_masks.pt

  # Train with validation
  samrfi train --config configs/sam2_training.yaml --dataset ./datasets/train_4k/exact_masks.pt --validation-dataset ./datasets/val_1k/exact_masks.pt

  # Publish dataset to HuggingFace Hub
  samrfi publish --type dataset --input ./datasets/train_4k/exact_masks.pt --repo-id username/sam-rfi-dataset

  # Publish trained model to HuggingFace Hub
  samrfi publish --type model --input ./models/sam2_rfi_best.pth --repo-id username/sam-rfi-models

  # Predict (single pass) - local model
  samrfi predict --model ./models/sam2_rfi.pth --input observation.ms

  # Predict (single pass) - HuggingFace model
  samrfi predict --model polarimetic/sam-rfi/large --input observation.ms

  # Predict (iterative - 3 passes)
  samrfi predict --model ./models/sam2_rfi.pth --input observation.ms --iterations 3
        """,
    )

    # Global logging arguments (available for all commands)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to file (in addition to console)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate data command
    generate_parser = subparsers.add_parser(
        "generate-data", help="Generate training dataset from MS or synthetic"
    )
    generate_parser.add_argument(
        "--source", required=True, choices=["synthetic", "ms"], help="Data source: synthetic or ms"
    )
    generate_parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    generate_parser.add_argument(
        "--output", required=True, help="Output directory for generated dataset"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train SAM2 model on RFI data")
    train_parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    train_parser.add_argument(
        "--dataset", required=True, help="Path to pre-generated dataset (.pt or HF format)"
    )
    train_parser.add_argument(
        "--validation-dataset", help="Path to validation dataset (.pt or HF format, optional)"
    )
    train_parser.add_argument(
        "--resume", help="Path to checkpoint to resume training from (.pth file)"
    )
    train_parser.add_argument(
        "--device", choices=["cuda", "cpu"], help="Device to use (overrides config)"
    )
    train_parser.add_argument("--output-dir", help="Output directory (overrides config)")

    # Create config command
    create_parser = subparsers.add_parser("create-config", help="Create default configuration file")
    create_parser.add_argument(
        "--output", "-o", help="Output path for config file (default: sam2_config.yaml)"
    )

    # Validate config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate configuration file")
    validate_parser.add_argument("--config", required=True, help="Path to YAML configuration file")

    # Publish command
    publish_parser = subparsers.add_parser(
        "publish", help="Publish dataset or model to HuggingFace Hub"
    )
    publish_parser.add_argument(
        "--type",
        choices=["dataset", "model"],
        default="dataset",
        help="Publish dataset or trained model (default: dataset)",
    )
    publish_parser.add_argument(
        "--input", required=True, help="Path to .pt dataset or .pth model checkpoint"
    )
    publish_parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (username/repo-name)",
    )
    publish_parser.add_argument("--private", action="store_true", help="Make repository private")
    publish_parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")
    publish_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="[Dataset only] Batch size for conversion (default: 50)",
    )
    publish_parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base_plus", "large"],
        help="[Model only] Model size (auto-detected from checkpoint if not specified)",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Apply trained model to flag RFI")
    predict_parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.pth file) OR HuggingFace repo ID (e.g., user/repo/large)",
    )
    predict_parser.add_argument("--input", required=True, help="Path to input measurement set")
    predict_parser.add_argument(
        "--checkpoint",
        default="large",
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM2 checkpoint size (default: large)",
    )
    predict_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterative flagging passes (default: 1 = single pass)",
    )
    predict_parser.add_argument(
        "--num-antennas", type=int, help="Number of antennas to load (default: all)"
    )
    predict_parser.add_argument(
        "--patch-size", type=int, default=128, help="Patch size (default: 128)"
    )
    predict_parser.add_argument(
        "--stretch",
        default="SQRT",
        choices=["SQRT", "LOG10", "None"],
        help="Stretch function (default: SQRT, use None for synthetic data)",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="RFI probability threshold (default: None = adaptive/mean, range: 0.0-1.0)",
    )
    predict_parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device (default: cuda)"
    )
    predict_parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    predict_parser.add_argument(
        "--apply-existing",
        action="store_true",
        help="Apply existing MS flags before prediction",
    )
    predict_parser.add_argument(
        "--no-save", action="store_true", help="Do not save flags to MS (prediction only)"
    )

    # Evaluate parser
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate predictions against ground truth"
    )
    evaluate_parser.add_argument(
        "--input", required=True, help="Path to measurement set with predicted flags"
    )
    evaluate_parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth .npy file"
    )
    evaluate_parser.add_argument(
        "--output", default="metrics.csv", help="Output CSV file path (default: metrics.csv)"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging (after parsing args, before any commands)
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level, log_file=args.log_file)

    # Execute command
    try:
        if args.command == "generate-data":
            generate_data_command(args)
            return 0
        elif args.command == "train":
            train_command(args)
            return 0
        elif args.command == "create-config":
            create_config_command(args)
            return 0
        elif args.command == "validate-config":
            return validate_config_command(args)
        elif args.command == "publish":
            publish_command(args)
            return 0
        elif args.command == "predict":
            predict_command(args)
            return 0
        elif args.command == "evaluate":
            evaluate_command(args)
            return 0
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
