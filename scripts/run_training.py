#!/usr/bin/env python3
"""
Simple training pipeline: generate datasets → train SAM2

Usage:
    python scripts/run_training.py --config configs/training_config.yaml
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.samrfi.training.sam2_trainer import SAM2Trainer
from src.samrfi.data import BatchedDataset
from src.samrfi.data_generation import SyntheticDataGenerator
from src.samrfi.config.config_loader import ConfigLoader


def setup_logging(output_dir):
    """Setup logging to both console and file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"training_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file


class DatasetWrapper:
    """Simple wrapper for SAM2Trainer compatibility"""
    def __init__(self, dataset):
        self.dataset = dataset


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--skip-generation", action="store_true", help="Skip dataset generation")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup logging to file and console
    log_file = setup_logging(config['training']['output_dir'])

    logger.info("="*60)
    logger.info("SAM-RFI Training Pipeline")
    logger.info("="*60)
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Step 1: Generate training dataset
    if not args.skip_generation:
        logger.info("\n[1/3] Generating training dataset...")
        train_gen_config_path = config['data']['train_generation_config']
        train_output = config['data']['train_dataset']

        logger.info(f"  Config: {train_gen_config_path}")
        logger.info(f"  Output: {train_output}")

        # Load generation config and generate
        train_gen_config = ConfigLoader.load_data(train_gen_config_path)
        generator = SyntheticDataGenerator(train_gen_config)
        generator.generate(output_path=train_output)

        # Step 2: Generate validation dataset
        if 'val_generation_config' in config['data']:
            logger.info("\n[2/3] Generating validation dataset...")
            val_gen_config_path = config['data']['val_generation_config']
            val_output = config['data']['val_dataset']

            logger.info(f"  Config: {val_gen_config_path}")
            logger.info(f"  Output: {val_output}")

            val_gen_config = ConfigLoader.load_data(val_gen_config_path)
            generator = SyntheticDataGenerator(val_gen_config)
            generator.generate(output_path=val_output)
    else:
        logger.info("\n[1/3] Skipping dataset generation...")

    # Step 3: Train
    logger.info("\n[3/3] Training SAM2...")

    # Load datasets with streaming workers
    train_path = Path(config['data']['train_dataset']) / config['data']['mask_type']
    logger.info(f"Loading training dataset: {train_path}")
    train_dataset = BatchedDataset(train_path)
    logger.info(f"  {train_dataset}")

    val_dataset = None
    if 'val_dataset' in config['data']:
        val_path = Path(config['data']['val_dataset']) / config['data']['mask_type']
        logger.info(f"Loading validation dataset: {val_path}")
        val_dataset = BatchedDataset(val_path)
        logger.info(f"  {val_dataset}")

    # Wrap datasets
    train_wrapper = DatasetWrapper(train_dataset)

    # Create trainer
    trainer = SAM2Trainer(
        rfidataset_instance=train_wrapper,
        device=config['training']['device'],
        dir_path=config['training']['output_dir']
    )

    # Train
    logger.info(f"\nStarting training:")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Model: {config['training']['model_checkpoint']}")

    # Extract training config with defaults
    train_cfg = config['training']

    trainer.train(
        num_epochs=train_cfg['num_epochs'],
        batch_size=train_cfg['batch_size'],
        sam_checkpoint=train_cfg['model_checkpoint'],
        learning_rate=train_cfg['learning_rate'],
        validation_dataset=val_dataset,
        # Optimizer
        optimizer=train_cfg.get('optimizer', 'adam'),
        weight_decay=train_cfg.get('weight_decay', 0.0),
        adam_betas=tuple(train_cfg.get('adam_betas', [0.9, 0.999])),
        adam_eps=train_cfg.get('adam_eps', 1e-8),
        momentum=train_cfg.get('momentum', 0.9),
        # Loss function
        loss_function=train_cfg.get('loss_function', 'dicece'),
        loss_sigmoid=train_cfg.get('loss_sigmoid', True),
        loss_squared_pred=train_cfg.get('loss_squared_pred', True),
        loss_reduction=train_cfg.get('loss_reduction', 'mean'),
        # Model architecture
        multimask_output=train_cfg.get('multimask_output', False),
        freeze_vision_encoder=train_cfg.get('freeze_vision_encoder', True),
        freeze_prompt_encoder=train_cfg.get('freeze_prompt_encoder', True),
        # Data augmentation
        bbox_perturbation=train_cfg.get('bbox_perturbation', 20),
        # DataLoader
        num_workers=train_cfg.get('num_workers', 0),
        prefetch_factor=train_cfg.get('prefetch_factor', 2),
        persistent_workers=train_cfg.get('persistent_workers', True),
        pin_memory=train_cfg.get('pin_memory', True),
        # Training optimization
        log_interval=train_cfg.get('log_interval', 100),
        cuda_cache_clear_interval=train_cfg.get('cuda_cache_clear_interval', 100),
        # Output
        plot=train_cfg.get('plot', True),
        save_model=train_cfg.get('save_model', True)
    )

    logger.info("\n✓ Training complete!")
    logger.info(f"  Output: {config['training']['output_dir']}")
    logger.info(f"  Log file: {log_file}")


if __name__ == "__main__":
    main()
