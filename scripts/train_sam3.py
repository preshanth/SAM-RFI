#!/usr/bin/env python3
"""
DEPRECATED: This script is deprecated. Use scripts/run_training.py instead.

Old usage:
    python scripts/train_sam3.py --config configs/sam3_training.yaml

New usage:
    python scripts/run_training.py --config configs/h100_sam3_config.yaml

The new unified backend (run_training.py) handles both SAM2 and SAM3 with:
- Hardware-centric configs (H100, A100, V100, etc.)
- Consistent logging and output structure
- Same DataLoader optimizations
- Auto-detection of model type from config

This script remains for backward compatibility but will be removed in a future release.
--------------------------------------------------------------------------------

SAM3 Training Script with Full Experiment Tracking (DEPRECATED)

Trains SAM3 (Segment Anything Model 3) for RFI detection using visual prompts.
SAM3 is a unified 840M parameter model (vs SAM2's 4 variants).

Features:
- Train/validation loss tracking and export to .npz
- Best model checkpointing (lowest val loss)
- Experiment configuration saving
- Resume training from checkpoint
- Structured logging
- Encoder freezing (840M → 33M trainable params)

Usage (DEPRECATED):
    python scripts/train_sam3.py --config configs/sam3_training.yaml
    python scripts/train_sam3.py --config configs/sam3_training.yaml --resume output/sam3/checkpoint_epoch5.pth
"""

import warnings
warnings.warn(
    "\n\n"
    "=" * 80 + "\n"
    "DEPRECATION WARNING: scripts/train_sam3.py is deprecated!\n"
    "\n"
    "Please use the unified training script instead:\n"
    "  Old: python scripts/train_sam3.py --config configs/sam3_training.yaml\n"
    "  New: python scripts/run_training.py --config configs/h100_sam3_config.yaml\n"
    "\n"
    "Benefits of the new script:\n"
    "  - Unified backend for both SAM2 and SAM3\n"
    "  - Hardware-centric configs (H100, A100, V100, etc.)\n"
    "  - Consistent logging and output structure\n"
    "  - Same optimized DataLoader for both models\n"
    "\n"
    "This script will continue to work but will be removed in a future release.\n"
    "=" * 80 + "\n",
    DeprecationWarning,
    stacklevel=2
)

import argparse
import sys
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import monai
from transformers import Sam3Processor, Sam3Model
from torch.nn.functional import interpolate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.samrfi.data import SAMDataset, NumpyDataset, BatchedDataset
from src.samrfi.data_generation import SyntheticDataGenerator
from src.samrfi.config.config_loader import ConfigLoader


class ExperimentTracker:
    """Track and save experiment results"""

    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1

        # Save config
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save git commit hash if available
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            (self.output_dir / "git_commit.txt").write_text(git_hash)
        except:
            pass

        # Initialize log file
        self.log_file = self.output_dir / "training_log.txt"
        self.log(f"Experiment started: {datetime.now()}")
        self.log(f"Config: {config}")

    def log(self, message):
        """Log message to file and stdout"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def record_epoch(self, epoch, train_loss, val_loss=None):
        """Record losses for an epoch"""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        # Save losses after each epoch
        self.save_losses()

        # Log
        msg = f"Epoch {epoch}: train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.6f}"
            if val_loss < self.best_val_loss:
                msg += " (BEST!)"
                self.best_val_loss = val_loss
                self.best_epoch = epoch
        self.log(msg)

    def save_losses(self):
        """Save losses to .npz file"""
        losses_path = self.output_dir / "losses.npz"
        data = {
            'epochs': np.arange(1, len(self.train_losses) + 1),
            'train_loss': np.array(self.train_losses),
        }
        if self.val_losses:
            data['val_loss'] = np.array(self.val_losses)
            data['best_val_loss'] = self.best_val_loss
            data['best_epoch'] = self.best_epoch

        np.savez(losses_path, **data)

    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.log(f"Checkpoint saved: {checkpoint_path}")

        # Save final model
        if epoch == self.config['training']['num_epochs']:
            final_path = self.output_dir / "model_final.pth"
            torch.save(model.state_dict(), final_path)
            self.log(f"Final model saved: {final_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / "model_best.pth"
            torch.save(model.state_dict(), best_path)
            self.log(f"Best model saved: {best_path} (val_loss={self.best_val_loss:.6f})")


def load_dataset(path):
    """Load dataset from .npz, batched, or HF format"""
    path = Path(path)

    # Check if it's a directory with batch files
    if path.is_dir() and any(path.glob('batch_*.npz')):
        return BatchedDataset(path)
    # Check if it's a single .npz file
    elif path.suffix == '.npz':
        return NumpyDataset.load_from_disk(path)
    # Otherwise assume HF format
    else:
        from datasets import load_from_disk
        return load_from_disk(path)


def train_epoch(model, dataloader, optimizer, seg_loss, device, tracker, epoch):
    """Train for one epoch"""
    model.train()
    epoch_losses = []

    for batch_idx, batch in enumerate(dataloader, 1):
        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            input_boxes=batch["input_boxes"].to(device),
            multimask_output=False,
        )

        # Get predictions and ground truth
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        if len(ground_truth_masks.shape) == 3:
            ground_truth_masks = ground_truth_masks.unsqueeze(1)

        # Resize ground truth to match predicted mask size
        predicted_mask_size = predicted_masks.shape[-2:]
        ground_truth_masks_resized = interpolate(
            ground_truth_masks,
            size=predicted_mask_size,
            mode="bilinear",
            align_corners=False,
        )

        # Compute loss
        loss = seg_loss(predicted_masks, ground_truth_masks_resized)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        # Cleanup
        del outputs, predicted_masks, ground_truth_masks, ground_truth_masks_resized, loss, batch

        # Log progress
        if batch_idx % 50 == 0:
            avg_loss = np.mean(epoch_losses[-50:])
            tracker.log(f"  Epoch {epoch} [Train] [{batch_idx}/{len(dataloader)}] loss={avg_loss:.6f}")

        # Periodic CUDA cleanup
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    return np.mean(epoch_losses)


def validate_epoch(model, dataloader, seg_loss, device, tracker, epoch):
    """Validate for one epoch"""
    model.eval()
    epoch_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, 1):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False,
            )

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            if len(ground_truth_masks.shape) == 3:
                ground_truth_masks = ground_truth_masks.unsqueeze(1)

            predicted_mask_size = predicted_masks.shape[-2:]
            ground_truth_masks_resized = interpolate(
                ground_truth_masks,
                size=predicted_mask_size,
                mode="bilinear",
                align_corners=False,
            )

            loss = seg_loss(predicted_masks, ground_truth_masks_resized)
            epoch_losses.append(loss.item())

            # Cleanup
            del outputs, predicted_masks, ground_truth_masks, ground_truth_masks_resized, loss, batch

            # Log progress
            if batch_idx % 50 == 0:
                avg_loss = np.mean(epoch_losses[-50:])
                tracker.log(f"  Epoch {epoch} [Val] [{batch_idx}/{len(dataloader)}] loss={avg_loss:.6f}")

            # Periodic CUDA cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

    return np.mean(epoch_losses)


def main():
    parser = argparse.ArgumentParser(description="Train SAM3 for RFI detection with full experiment tracking")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Override device from config")
    parser.add_argument("--skip-generation", action="store_true", help="Skip dataset generation (use existing)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override device if specified
    if args.device:
        config['training']['device'] = args.device

    device = config['training']['device']
    print(f"\n{'='*60}")
    print(f"SAM3 Training - Experiment: {config['experiment']['name']}")
    print(f"{'='*60}\n")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        output_dir=config['experiment']['output_dir'],
        config=config
    )

    # Generate datasets if needed
    if not args.skip_generation:
        # Generate training dataset
        if 'train_generation_config' in config['data']:
            tracker.log(f"\n[Step 1/3] Generating training dataset...")
            train_gen_config_path = config['data']['train_generation_config']
            train_output = config['data']['train_dataset']

            tracker.log(f"  Config: {train_gen_config_path}")
            tracker.log(f"  Output: {train_output}")

            train_gen_config = ConfigLoader.load_data(train_gen_config_path)
            generator = SyntheticDataGenerator(train_gen_config)
            generator.generate(output_path=train_output)
            tracker.log(f"  ✓ Training dataset generated")

        # Generate validation dataset
        if 'val_generation_config' in config['data']:
            tracker.log(f"\n[Step 2/3] Generating validation dataset...")
            val_gen_config_path = config['data']['val_generation_config']
            val_output = config['data']['val_dataset']

            tracker.log(f"  Config: {val_gen_config_path}")
            tracker.log(f"  Output: {val_output}")

            val_gen_config = ConfigLoader.load_data(val_gen_config_path)
            generator = SyntheticDataGenerator(val_gen_config)
            generator.generate(output_path=val_output)
            tracker.log(f"  ✓ Validation dataset generated")
    else:
        tracker.log("\n[Skipped] Dataset generation (using existing datasets)")

    # Load datasets
    tracker.log(f"\n[Step 3/3] Loading datasets for training...")
    tracker.log(f"Loading training dataset: {config['data']['train_dataset']}")
    train_dataset = load_dataset(config['data']['train_dataset'])
    tracker.log(f"  Loaded {len(train_dataset)} training samples")

    val_dataset = None
    if 'val_dataset' in config['data'] and config['data']['val_dataset']:
        tracker.log(f"Loading validation dataset: {config['data']['val_dataset']}")
        val_dataset = load_dataset(config['data']['val_dataset'])
        tracker.log(f"  Loaded {len(val_dataset)} validation samples")

    # Load SAM3 model (unified checkpoint)
    tracker.log(f"Loading SAM3 model: facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    tracker.log(f"  Total parameters: {total_params/1e6:.1f}M")

    # Freeze encoders (only train mask decoder)
    if config['model'].get('freeze_encoders', True):
        for name, param in model.named_parameters():
            if "vision_encoder" in name or "prompt_encoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tracker.log(f"  Frozen vision and prompt encoders")
        tracker.log(f"  Trainable parameters: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")

    # Create dataloaders
    train_dataloader = DataLoader(
        SAMDataset(train_dataset, processor, bbox_perturbation=20),
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            SAMDataset(val_dataset, processor, bbox_perturbation=0),
            batch_size=config['training']['batch_size'],
            shuffle=False
        )

    # Setup optimizer and loss
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    # Move model to device
    model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        tracker.log(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        tracker.log(f"  Resumed from epoch {checkpoint['epoch']}")

    # Training loop
    tracker.log(f"Starting training: {config['training']['num_epochs']} epochs")
    tracker.log(f"  Batch size: {config['training']['batch_size']}")
    tracker.log(f"  Learning rate: {config['training']['learning_rate']}")
    tracker.log(f"  Device: {device}")

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        tracker.log(f"\nEpoch {epoch}/{config['training']['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, seg_loss, device, tracker, epoch)

        # Validate
        val_loss = None
        if val_dataloader:
            val_loss = validate_epoch(model, val_dataloader, seg_loss, device, tracker, epoch)

        # Record epoch
        tracker.record_epoch(epoch, train_loss, val_loss)

        # Save checkpoint
        is_best = val_loss is not None and val_loss < tracker.best_val_loss
        if epoch % config['experiment'].get('save_every_n_epochs', 5) == 0 or epoch == config['training']['num_epochs']:
            tracker.save_checkpoint(model, optimizer, epoch, is_best=is_best)

        # CUDA cleanup
        torch.cuda.empty_cache()

    tracker.log(f"\nTraining complete!")
    if tracker.val_losses:
        tracker.log(f"Best validation loss: {tracker.best_val_loss:.6f} (epoch {tracker.best_epoch})")


if __name__ == "__main__":
    main()
