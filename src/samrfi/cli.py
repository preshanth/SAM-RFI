"""
Command-line interface for SAM-RFI training
"""

import argparse
import sys
from pathlib import Path

from samrfi.data import MSLoader, Preprocessor
from .training.sam2_trainer import SAM2Trainer
from .config.config_loader import ConfigLoader, TrainingConfig, DataConfig
from .inference import RFIPredictor
from .data_generation.synthetic_generator import SyntheticDataGenerator
from .data_generation.ms_generator import MSDataGenerator
from .data.numpy_dataset import NumpyDataset


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
    print(f"  exact_masks/ - Perfect ground truth")
    print(f"  mad_masks/ - MAD-based masks")


def load_dataset(path):
    """Load dataset from either .npz (numpy) or HF format"""
    path = Path(path)

    if path.suffix == '.npz':
        print(f"  Loading NumpyDataset from {path}")
        return NumpyDataset.load_from_disk(path)
    else:
        # Assume HF dataset directory (backward compatibility)
        from datasets import load_from_disk
        print(f"  Loading HuggingFace Dataset from {path}")
        return load_from_disk(path)


def train_command(args):
    """Execute training command on pre-generated dataset"""

    print("=" * 60)
    print("SAM-RFI SAM2 Training")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = ConfigLoader.load(args.config)

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
    print(f"\nInitializing SAM2 trainer...")
    trainer = SAM2Trainer(dataset_wrapper, device=config.device, dir_path=config.dir_path)

    losses = trainer.train(
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        sam_checkpoint=config.model_checkpoint,
        learning_rate=config.learning_rate,
        plot=config.save_plots,
        validation_dataset=val_dataset,
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


def publish_dataset_command(args):
    """Publish numpy dataset to HuggingFace Hub"""
    from .data.hf_dataset_wrapper import HFDatasetWrapper

    print("=" * 60)
    print("SAM-RFI Dataset Publishing")
    print("=" * 60)

    # Load numpy dataset
    print(f"\nLoading numpy dataset from {args.input}")
    numpy_dataset = NumpyDataset.load_from_disk(args.input)
    print(f"  {numpy_dataset}")

    # Convert to HF format
    print(f"\nConverting to HuggingFace Dataset format...")
    hf_dataset = HFDatasetWrapper.from_numpy(numpy_dataset, batch_size=args.batch_size)

    # Push to hub
    print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
    hf_dataset.push_to_hub(
        args.repo_id,
        private=args.private,
        token=args.token
    )

    print("\n" + "=" * 60)
    print("✓ Dataset Published!")
    print("=" * 60)
    print(f"URL: https://huggingface.co/datasets/{args.repo_id}")


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
            stretch=args.stretch,
            save_flags=not args.no_save,
        )
    else:
        print(f"\nMode: Single-pass flagging")
        flags = predictor.predict_ms(
            ms_path=args.input,
            num_antennas=args.num_antennas,
            patch_size=args.patch_size,
            stretch=args.stretch,
            apply_existing_flags=args.apply_existing,
            save_flags=not args.no_save,
        )

    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print("=" * 60)
    print(f"Total flagged: {flags.sum()/flags.size*100:.2f}%")
    if not args.no_save:
        print(f"Flags saved to: {args.input}")


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

  # Train with pre-generated dataset (.npz format)
  samrfi train --config configs/sam2_training.yaml --dataset ./datasets/train_4k/exact_masks.npz

  # Train with validation
  samrfi train --config configs/sam2_training.yaml --dataset ./datasets/train_4k/exact_masks.npz --validation-dataset ./datasets/val_1k/exact_masks.npz

  # Publish dataset to HuggingFace Hub
  samrfi publish --input ./datasets/train_4k/exact_masks.npz --repo-id username/sam-rfi-dataset

  # Predict (single pass)
  samrfi predict --model ./models/sam2_rfi.pth --input observation.ms

  # Predict (iterative - 3 passes)
  samrfi predict --model ./models/sam2_rfi.pth --input observation.ms --iterations 3
        """,
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
        "--dataset", required=True, help="Path to pre-generated dataset (.npz or HF format)"
    )
    train_parser.add_argument("--validation-dataset", help="Path to validation dataset (.npz or HF format, optional)")
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
    publish_parser = subparsers.add_parser("publish", help="Publish dataset to HuggingFace Hub")
    publish_parser.add_argument("--input", required=True, help="Path to .npz dataset")
    publish_parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (username/dataset-name)")
    publish_parser.add_argument("--private", action="store_true", help="Make dataset private")
    publish_parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")
    publish_parser.add_argument("--batch-size", type=int, default=50, help="Batch size for conversion (default: 50)")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Apply trained model to flag RFI")
    predict_parser.add_argument("--model", required=True, help="Path to trained model (.pth file)")
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
        choices=["SQRT", "LOG10"],
        help="Stretch function (default: SQRT)",
    )
    predict_parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device (default: cuda)"
    )
    predict_parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    predict_parser.add_argument(
        "--apply-existing",
        action="store_true",
        help="Apply existing MS flags before prediction (single-pass only)",
    )
    predict_parser.add_argument(
        "--no-save", action="store_true", help="Do not save flags to MS (prediction only)"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

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
            publish_dataset_command(args)
            return 0
        elif args.command == "predict":
            predict_command(args)
            return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
