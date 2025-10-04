#!/usr/bin/env python3
"""
Simple training pipeline: generate datasets → train SAM2

Usage:
    python scripts/run_training.py --config configs/training_config.yaml
"""

import argparse
import sys
import yaml
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.samrfi.training.sam2_trainer import SAM2Trainer
from src.samrfi.data import BatchedDataset


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

    print("="*60)
    print("SAM-RFI Training Pipeline")
    print("="*60)

    # Step 1: Generate training dataset
    if not args.skip_generation:
        print("\n[1/3] Generating training dataset...")
        train_gen_config = config['data']['train_generation_config']
        train_output = config['data']['train_dataset']

        cmd = [
            'samrfi', 'generate-data',
            '--source', 'synthetic',
            '--config', train_gen_config,
            '--output', train_output
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Step 2: Generate validation dataset
        if 'val_generation_config' in config['data']:
            print("\n[2/3] Generating validation dataset...")
            val_gen_config = config['data']['val_generation_config']
            val_output = config['data']['val_dataset']

            cmd = [
                'samrfi', 'generate-data',
                '--source', 'synthetic',
                '--config', val_gen_config,
                '--output', val_output
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    else:
        print("\n[1/3] Skipping dataset generation...")

    # Step 3: Train
    print("\n[3/3] Training SAM2...")

    # Load datasets with large cache to keep all batches in RAM
    train_path = Path(config['data']['train_dataset']) / config['data']['mask_type']
    print(f"Loading training dataset: {train_path}")
    train_dataset = BatchedDataset(train_path, cache_size=9999)  # Cache all batches
    print(f"  {len(train_dataset)} samples")

    val_dataset = None
    if 'val_dataset' in config['data']:
        val_path = Path(config['data']['val_dataset']) / config['data']['mask_type']
        print(f"Loading validation dataset: {val_path}")
        val_dataset = BatchedDataset(val_path, cache_size=9999)  # Cache all batches
        print(f"  {len(val_dataset)} samples")

    # Wrap datasets
    train_wrapper = DatasetWrapper(train_dataset)

    # Create trainer
    trainer = SAM2Trainer(
        rfidataset_instance=train_wrapper,
        device=config['training']['device'],
        dir_path=config['training']['output_dir']
    )

    # Train
    print(f"\nStarting training:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Model: {config['training']['model_checkpoint']}")

    trainer.train(
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        sam_checkpoint=config['training']['model_checkpoint'],
        learning_rate=config['training']['learning_rate'],
        validation_dataset=val_dataset,
        plot=True,
        save_model=True
    )

    print("\n✓ Training complete!")
    print(f"  Output: {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
