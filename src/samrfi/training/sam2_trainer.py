"""
SAM2 Trainer - Clean implementation using transformers library
Mirrors the working SAM1 training approach
"""

import os
import gc
import time
from pathlib import Path
from datetime import datetime
from statistics import mean

import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import Adam
import monai

from transformers import Sam2Processor, Sam2Model

import numpy as np
import matplotlib.pyplot as plt

from samrfi.data import SAMDataset


def _log_progress(batch_idx, total_batches, start_time, prefix="", current_loss=None):
    """
    Log training progress without TQDM overhead.

    Logs every 100 batches or at completion to avoid excessive output.
    """
    if batch_idx % 100 == 0 or batch_idx == total_batches:
        elapsed = time.time() - start_time
        rate = batch_idx / elapsed if elapsed > 0 else 0
        eta_sec = (total_batches - batch_idx) / rate if rate > 0 else 0

        loss_str = f", Loss: {current_loss:.6f}" if current_loss is not None else ""
        print(f"{prefix}[{batch_idx}/{total_batches}] "
              f"Rate: {rate:.1f} batch/s, ETA: {eta_sec/60:.1f}m{loss_str}")


class SAM2Trainer:
    """
    SAM2 training using HuggingFace transformers library.
    Simple, clean implementation that mirrors working SAM1 code.
    """

    def __init__(self, rfidataset_instance, device="cuda", dir_path=None):
        """
        Initialize SAM2 trainer

        Args:
            rfidataset_instance: RFIDataset instance with .dataset attribute
            device: 'cuda' or 'cpu'
            dir_path: Directory to save models (default: ./samrfi_data)
        """
        self.device = device
        self.RFIDataset = rfidataset_instance

        # Setup output directory
        if dir_path:
            if dir_path.endswith("/"):
                dir_path = dir_path[:-1]
            current_directory = str(dir_path)
        else:
            current_directory = os.getcwd()

        new_directory = os.path.join(current_directory, "samrfi_data")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        self.directory = new_directory
        self.ave_meanloss = []
        self.val_losses = None

    def train(
        self,
        num_epochs=3,
        batch_size=4,
        sam_checkpoint="large",
        learning_rate=1e-5,
        plot=True,
        model_path=None,
        trained_model_path=None,
        validation_dataset=None,
        save_model=True,
    ):
        """
        Train SAM2 model on RFI dataset

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            sam_checkpoint: 'tiny', 'small', 'base_plus', or 'large'
            learning_rate: Learning rate (default: 1e-5)
            plot: Whether to plot loss curve
            model_path: Path to pretrained model to resume from
            trained_model_path: Path to save trained model
            validation_dataset: Optional HuggingFace dataset for validation
            save_model: Whether to save model checkpoint (default: True, set False for validation)
        """

        # Map checkpoint names to HuggingFace model IDs
        checkpoint_map = {
            "tiny": "facebook/sam2-hiera-tiny",
            "small": "facebook/sam2-hiera-small",
            "base_plus": "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large",
        }

        if sam_checkpoint not in checkpoint_map:
            raise ValueError(
                f"Invalid checkpoint '{sam_checkpoint}'. Use: {list(checkpoint_map.keys())}"
            )

        model_name = checkpoint_map[sam_checkpoint]

        print(f"\nLoading SAM2 model: {model_name}")

        # Load processor and model from HuggingFace
        processor = Sam2Processor.from_pretrained(model_name)
        model = Sam2Model.from_pretrained(model_name)

        # Create dataset using SAMDataset wrapper
        train_dataset = SAMDataset(dataset=self.RFIDataset.dataset, processor=processor)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            prefetch_factor=2,
            persistent_workers=True
        )

        # Create validation dataloader if provided
        val_dataloader = None
        if validation_dataset is not None:
            val_dataset = SAMDataset(dataset=validation_dataset, processor=processor)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=12,
                prefetch_factor=2,
                persistent_workers=True
            )
            print(f"  Validation samples: {len(validation_dataset)}")

        # Freeze vision encoder and prompt encoder (only train mask decoder)
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        # Load pretrained weights if provided
        if model_path:
            print(f"Loading pretrained weights from: {model_path}")
            model.load_state_dict(torch.load(model_path))

        # Setup optimizer and loss
        optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

        # Move model to device
        model.to(self.device)
        model.train()

        print(f"\nTraining SAM2 model...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_losses = []

            total_batches = len(train_dataloader)
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs} [Train]: Starting {total_batches} batches")

            for batch_idx, batch in enumerate(train_dataloader, 1):
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    input_boxes=batch["input_boxes"].to(self.device),
                    multimask_output=False,
                )

                # Get predictions and ground truth
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)

                # Ensure ground truth masks have correct dimensions
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

                # Extract loss value
                loss_value = loss.item()
                epoch_train_losses.append(loss_value)

                # CRITICAL: Explicit cleanup to prevent memory accumulation
                # Safe to delete after optimizer.step() - gradients stored in parameter.grad
                del outputs, predicted_masks, ground_truth_masks, ground_truth_masks_resized, loss, batch

                # Clear CUDA cache periodically to prevent fragmentation
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

                # Log progress
                _log_progress(batch_idx, total_batches, epoch_start_time,
                            f"Epoch {epoch+1}/{num_epochs} [Train] ", loss_value)

            # Calculate mean training loss
            epoch_mean_train_loss = mean(epoch_train_losses)
            train_losses.append(epoch_mean_train_loss)

            # Validation phase
            epoch_val_loss = None
            if val_dataloader is not None:
                model.eval()
                epoch_val_losses = []

                total_val_batches = len(val_dataloader)
                val_start_time = time.time()
                print(f"\nEpoch {epoch+1}/{num_epochs} [Val]: Starting {total_val_batches} batches")

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader, 1):
                        outputs = model(
                            pixel_values=batch["pixel_values"].to(self.device),
                            input_boxes=batch["input_boxes"].to(self.device),
                            multimask_output=False,
                        )

                        predicted_masks = outputs.pred_masks.squeeze(1)
                        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)

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
                        loss_value = loss.item()
                        epoch_val_losses.append(loss_value)

                        # CRITICAL: Explicit cleanup (same as training)
                        del outputs, predicted_masks, ground_truth_masks, ground_truth_masks_resized, loss, batch

                        # Clear CUDA cache periodically
                        if batch_idx % 100 == 0:
                            torch.cuda.empty_cache()

                        # Log progress
                        _log_progress(batch_idx, total_val_batches, val_start_time,
                                    f"Epoch {epoch+1}/{num_epochs} [Val] ", loss_value)

                epoch_val_loss = mean(epoch_val_losses)
                val_losses.append(epoch_val_loss)

            # Log epoch statistics
            log_msg = f"EPOCH: {epoch+1}/{num_epochs} | Train loss: {epoch_mean_train_loss:.6f}"
            if epoch_val_loss is not None:
                log_msg += f" | Val loss: {epoch_val_loss:.6f}"
            print(log_msg)

            # Force garbage collection at end of epoch
            gc.collect()
            torch.cuda.empty_cache()

        self.ave_meanloss = train_losses
        self.val_losses = val_losses if val_losses else None

        # Save model (skip during validation to save memory)
        if save_model:
            self._save_model(model, sam_checkpoint, num_epochs, trained_model_path)

        # Plot loss curve
        if plot:
            self._plot_loss_curve(sam_checkpoint, num_epochs)

        print(f"\nTraining complete!")

        # Return losses
        if self.val_losses:
            return {"train": train_losses, "val": val_losses}
        else:
            return train_losses

    def _save_model(self, model, sam_checkpoint, num_epochs, trained_model_path=None):
        """Save trained model with descriptive filename"""
        # Extract params from dataset if available (for backward compatibility)
        params = getattr(self.RFIDataset, 'dataset_params', None)

        if params:
            # Old format (legacy RFIDataset)
            stretch = params.get("stretch", "unknown")
            flag_sigma = params.get("flag_sigma", "unknown")
            patch_method = params.get("patch_method", "unknown")
            patch_size = params.get("patch_size", "unknown")
        else:
            # New format (NumpyDataset) - extract from metadata if available
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, 'metadata', {})
            stretch = metadata.get("stretch", "unknown")
            flag_sigma = metadata.get("flag_sigma", "unknown")
            patch_method = "numpy"
            patch_size = metadata.get("patch_size", "unknown")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"model_sam2-{sam_checkpoint}_"
            f"stretch-{stretch}_sigma-{flag_sigma}_"
            f"patch-{patch_method}_size-{patch_size}_"
            f"epochs{num_epochs}_{timestamp}.pth"
        )

        method_dir = os.path.join(self.directory, "models")
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)

        if trained_model_path:
            try:
                torch.save(model.state_dict(), trained_model_path)
                print(f"Model saved to: {trained_model_path}")
            except Exception as e:
                print(f"Could not save to {trained_model_path}: {e}")
                print(f"Saving to default location: {os.path.join(method_dir, filename)}")
                torch.save(model.state_dict(), os.path.join(method_dir, filename))
        else:
            save_path = os.path.join(method_dir, filename)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

    def _plot_loss_curve(self, sam_checkpoint, num_epochs):
        """Plot and save training and validation loss curves"""
        # Extract params from dataset if available (for backward compatibility)
        params = getattr(self.RFIDataset, 'dataset_params', None)

        if params:
            stretch = params.get("stretch", "unknown")
            flag_sigma = params.get("flag_sigma", "unknown")
            patch_method = params.get("patch_method", "unknown")
            patch_size = params.get("patch_size", "unknown")
        else:
            # New format (NumpyDataset)
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, 'metadata', {})
            stretch = metadata.get("stretch", "unknown")
            flag_sigma = metadata.get("flag_sigma", "unknown")
            patch_method = "numpy"
            patch_size = metadata.get("patch_size", "unknown")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        # Plot training loss
        epochs = range(1, len(self.ave_meanloss) + 1)
        ax.plot(
            epochs, self.ave_meanloss, label=f"Training Loss", color="blue", linewidth=2, marker="o"
        )

        # Plot validation loss if available
        if self.val_losses:
            ax.plot(
                epochs,
                self.val_losses,
                label=f"Validation Loss",
                color="red",
                linewidth=2,
                marker="s",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Loss", fontsize=12)

        # Title with dataset info
        title = f"SAM2-{sam_checkpoint} Training"
        if self.val_losses:
            title += " and Validation"
        # Get number of patches from dataset
        dataset = self.RFIDataset.dataset
        num_patches = len(dataset) if hasattr(dataset, '__len__') else "unknown"
        title += f" | {num_patches} patches"
        ax.set_title(title, fontsize=14)

        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        method_dir = os.path.join(self.directory, "models")
        os.makedirs(method_dir, exist_ok=True)  # Ensure directory exists

        filename = (
            f"loss_plot_sam2-{sam_checkpoint}_"
            f"stretch-{stretch}_sigma-{flag_sigma}_"
            f"patch-{patch_method}_size-{patch_size}_"
            f"epochs{num_epochs}_{timestamp}.png"
        )

        fig.savefig(os.path.join(method_dir, filename))
        print(f"Loss plot saved to: {os.path.join(method_dir, filename)}")
        plt.close()
