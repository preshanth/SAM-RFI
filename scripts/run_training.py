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

    # Load datasets
    cache_size = config['training'].get('cache_size', 3)
    train_path = Path(config['data']['train_dataset']) / config['data']['mask_type']
    print(f"Loading training dataset: {train_path}")
    train_dataset = BatchedDataset(train_path, cache_size=cache_size)
    print(f"  {len(train_dataset)} samples")

    val_dataset = None
    if 'val_dataset' in config['data']:
        val_path = Path(config['data']['val_dataset']) / config['data']['mask_type']
        print(f"Loading validation dataset: {val_path}")
        val_dataset = BatchedDataset(val_path, cache_size=cache_size)
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
        # LoRA settings
        use_lora=train_cfg.get('use_lora', False),
        lora_rank=train_cfg.get('lora_rank', 16),
        lora_alpha=train_cfg.get('lora_alpha', 32),
        lora_dropout=train_cfg.get('lora_dropout', 0.1),
        lora_target_modules=train_cfg.get('lora_target_modules', ["q_proj", "v_proj"]),
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

    print("\n✓ Training complete!")
    print(f"  Output: {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
