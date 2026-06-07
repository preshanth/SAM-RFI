"""
Command-line interface for SAM-RFI training
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rfi_toolbox.io import MSLoader

from .config import validate_all
from .config.config_loader import ConfigLoader
from .data_generation.ms_generator import MSDataGenerator
from .data_generation.synthetic_generator import SyntheticDataGenerator
from .evaluation.metrics import evaluate_segmentation
from .inference import RFIPredictor
from .training.sam2_trainer import SAM2Trainer
from .utils import logger, setup_logger
from .utils.errors import ConfigValidationError


def generate_data_command(args):
    """Execute data generation command"""
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


def load_dataset(path):
    """
    Load dataset from batched .pt directory (BatchedDataset or RAMCachedDataset).

    Supported formats:
    - BatchedDataset (preprocessed): Contains batch_*.pt + metadata.json
    - RAMCachedDataset (raw): Contains batch_*.pt + metadata.json with format='raw'
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


def train_command(args):
    """Execute training command on pre-generated dataset"""

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
        patience=config.patience,
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


def create_config_command(args):
    """Create default configuration file"""
    output_path = args.output or "sam2_config.yaml"

    print(f"Creating default configuration: {output_path}")
    ConfigLoader.create_default_config(output_path)
    print(f"✓ Configuration file created: {output_path}")
    print("\nEdit this file to customize training parameters, then run:")
    print(f"  samrfi train --config {output_path} --ms-path <path-to-ms>")


def validate_config_command(args):
    """Validate configuration file"""
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


def publish_command(args):
    """Dispatcher for publishing datasets or models to HuggingFace Hub"""
    publish_type = getattr(args, "type", "dataset")

    if publish_type == "dataset":
        publish_dataset_command(args)
    elif publish_type == "model":
        publish_model_command(args)
    else:
        raise ValueError(f"Unknown publish type: {publish_type}")


def publish_dataset_command(args):
    """Publish dataset (BatchedDataset or TorchDataset) to HuggingFace Hub"""
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


def publish_model_command(args):
    """Publish trained model to HuggingFace Hub"""
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


def predict_command(args):
    """Execute prediction command"""
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
    per_baseline = args.per_baseline

    if per_baseline:
        # Per-baseline mode (low memory)
        if is_iterative:
            print(f"\nMode: Iterative per-baseline ({num_iterations} passes, low memory)")
            predictor.predict_iterative_per_baseline(
                ms_path=args.input,
                num_iterations=num_iterations,
                num_antennas=args.num_antennas,
                patch_size=args.patch_size,
                stretch=stretch,
                save_flags=not args.no_save,
                threshold=threshold,
            )
        else:
            print("\nMode: Per-baseline flagging (low memory)")
            predictor.predict_ms_per_baseline(
                ms_path=args.input,
                num_antennas=args.num_antennas,
                patch_size=args.patch_size,
                stretch=stretch,
                save_flags=not args.no_save,
                threshold=threshold,
            )
        print("\n" + "=" * 60)
        print("Prediction Complete!")
        print("=" * 60)
        if not args.no_save:
            print(f"Flags saved to: {args.input}")
    else:
        # Original mode (greedy)
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


def evaluate_command(args):
    """Execute evaluation command - compute metrics given ground truth and predicted flags"""
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


def list_models_command(args):
    """List models in a HuggingFace repository"""
    from .utils.model_cache import ModelCache

    print("=" * 60)
    print("SAM-RFI Model Listing")
    print("=" * 60)

    cache = ModelCache()

    print(f"\nRepository: {args.repo}")
    print(f"Pattern: {args.pattern}")
    print("\nFetching model list from HuggingFace...")

    try:
        models = cache.list_repo_models(args.repo, pattern=args.pattern)

        if not models:
            print(f"\n✗ No models found matching pattern '{args.pattern}'")
            print(f"  Repository: {args.repo}")
            return 1

        print(f"\n✓ Found {len(models)} model(s):")
        print("-" * 60)

        for model in models:
            filename = model["filename"]
            size_info = f"{model['size_mb']:.1f} MB" if model["size_mb"] else "size unknown"
            print(f"  {filename:50} {size_info}")

        print("-" * 60)
        print("\nTo download a model:")
        print(f"  samrfi download-model --repo {args.repo} --model <filename> --output <directory>")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def download_model_command(args):
    """Download model from HuggingFace repository to custom directory"""
    from .utils.model_cache import ModelCache

    print("=" * 60)
    print("SAM-RFI Model Download")
    print("=" * 60)

    # Validate that output looks like a directory, not a filename
    if args.output.endswith(".pth") or args.output.endswith(".pt"):
        print("\n✗ Error: --output should be a directory path, not a filename")
        print(f"\n  You provided: --output {args.output}")
        print("\n  Correct usage:")
        print(f"    samrfi download-model --repo {args.repo} --model {args.model} --output ./")
        if args.output.endswith((".pth", ".pt")):
            print(
                f"    samrfi download-model --repo {args.repo} --model {args.model} --output ./ --name {Path(args.output).name}"
            )
        print("\n  --output = directory path (e.g., ./ or /nfs/models/)")
        print("  --name   = custom filename (optional)")
        return 1

    cache = ModelCache()

    try:
        downloaded_path = cache.download_from_repo(
            repo_id=args.repo,
            filename=args.model,
            output_dir=args.output,
            local_name=args.name,
            show_progress=True,
        )

        print("\n" + "=" * 60)
        print("✓ Download Complete!")
        print("=" * 60)
        print(f"Model saved to: {downloaded_path}")
        print("\nUsage:")
        print(f"  samrfi predict --model {downloaded_path} --input observation.ms")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def main():
    """Main CLI entry point"""
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

  # List models in HuggingFace repository
  samrfi list-models --repo polarimetric/sam-rfi

  # Download model to current directory (--output is directory path)
  samrfi download-model --repo polarimetric/sam-rfi --model large/model.pth --output ./

  # Download to specific directory
  samrfi download-model --repo polarimetric/sam-rfi --model large/model.pth --output /nfs/models/

  # Download with custom filename (--name is the filename)
  samrfi download-model --repo polarimetric/sam-rfi --model large/model.pth --output ./ --name production.pth
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
    predict_parser.add_argument(
        "--per-baseline",
        action="store_true",
        help="Process one baseline at a time (low memory usage)",
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

    # List models parser
    list_models_parser = subparsers.add_parser(
        "list-models", help="List models in a HuggingFace repository"
    )
    list_models_parser.add_argument(
        "--repo", required=True, help="HuggingFace repo ID (e.g., polarimetric/sam-rfi)"
    )
    list_models_parser.add_argument(
        "--pattern",
        default="*.pth",
        help="File pattern to filter models (default: *.pth)",
    )

    # Download model parser
    download_model_parser = subparsers.add_parser(
        "download-model", help="Download model from HuggingFace to custom directory"
    )
    download_model_parser.add_argument(
        "--repo", required=True, help="HuggingFace repo ID (e.g., polarimetric/sam-rfi)"
    )
    download_model_parser.add_argument(
        "--model",
        required=True,
        help="Model filename in repository (e.g., large/model.pth or sigma5_sqrt.pth)",
    )
    download_model_parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory path (NOT filename). Examples: ./ or /nfs/shared/models/",
    )
    download_model_parser.add_argument(
        "--name",
        metavar="FILENAME",
        help="Custom filename for downloaded model (e.g., my_model.pth). If not specified, uses original filename from repo.",
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
        elif args.command == "list-models":
            return list_models_command(args)
        elif args.command == "download-model":
            return download_model_command(args)
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
