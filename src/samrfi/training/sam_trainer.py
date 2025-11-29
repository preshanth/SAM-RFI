"""
Unified SAM Trainer - Handles both SAM2 and SAM3 models
Clean implementation using transformers library with automatic model detection
"""

import os
import gc
import time
import logging
from pathlib import Path
from datetime import datetime
from statistics import mean

import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai

import numpy as np
import matplotlib.pyplot as plt

from samrfi.data import SAMDataset

# Get logger (configured by parent script or defaults to console if standalone)
logger = logging.getLogger(__name__)

# Ensure logging is configured (fallback for standalone use)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _log_progress(batch_idx, total_batches, start_time, prefix="", current_loss=None):
    """Log training progress without TQDM overhead"""
    elapsed = time.time() - start_time
    rate = batch_idx / elapsed if elapsed > 0 else 0

    # Format elapsed time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    elapsed_str = f"{elapsed_min}m{elapsed_sec:02d}s"

    loss_str = f", Loss: {current_loss:.6f}" if current_loss is not None else ""
    logger.info(f"{prefix}[{batch_idx}/{total_batches}] "
                f"Elapsed: {elapsed_str}, Rate: {rate:.2f} batch/s{loss_str}")


class SAMTrainer:
    """
    Unified SAM trainer for both SAM2 and SAM3 models.
    Automatically detects model type and loads appropriate architecture.

    SAM2: Supports tiny, small, base_plus, large variants
    SAM3: Single unified 840M parameter model
    """

    def __init__(self, rfidataset_instance, device="cuda", dir_path=None, model_type="sam2"):
        """
        Initialize SAM trainer

        Args:
            rfidataset_instance: RFIDataset instance with .dataset attribute
            device: 'cuda' or 'cpu'
            dir_path: Directory to save models (default: ./samrfi_data)
            model_type: 'sam2' or 'sam3' (default: 'sam2')
        """
        self.device = device
        self.RFIDataset = rfidataset_instance
        self.model_type = model_type.lower()

        if self.model_type not in ['sam2', 'sam3']:
            raise ValueError(f"Invalid model_type '{model_type}'. Use 'sam2' or 'sam3'")

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
        # Optimizer settings
        optimizer='adam',
        weight_decay=0.0,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        momentum=0.9,
        # Loss function settings
        loss_function='dicece',
        loss_sigmoid=True,
        loss_squared_pred=True,
        loss_reduction='mean',
        # Model architecture
        multimask_output=False,
        freeze_vision_encoder=True,
        freeze_prompt_encoder=True,
        # Data augmentation
        bbox_perturbation=20,
        # DataLoader settings
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        # Training optimization
        log_interval=100,
        cuda_cache_clear_interval=100,
        # Output settings
        plot=True,
        model_path=None,
        trained_model_path=None,
        validation_dataset=None,
        save_model=True,
    ):
        """
        Train SAM model (SAM2 or SAM3) on RFI dataset

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            sam_checkpoint: For SAM2: 'tiny', 'small', 'base_plus', 'large'. For SAM3: ignored (single model)
            learning_rate: Learning rate (default: 1e-5)
            optimizer: 'adam', 'adamw', or 'sgd'
            weight_decay: L2 regularization (default: 0.0)
            loss_function: 'dicece', 'dice', 'ce', or 'focal'
            freeze_vision_encoder: Freeze vision encoder weights
            freeze_prompt_encoder: Freeze prompt encoder weights
            bbox_perturbation: Random bbox shift for data augmentation (pixels)
            num_workers: DataLoader worker processes
            validation_dataset: Optional dataset for validation
            save_model: Whether to save model checkpoint
        """

        # Load model and processor based on model type
        if self.model_type == 'sam2':
            from transformers import Sam2Processor, Sam2Model

            checkpoint_map = {
                "tiny": "facebook/sam2-hiera-tiny",
                "small": "facebook/sam2-hiera-small",
                "base_plus": "facebook/sam2-hiera-base-plus",
                "large": "facebook/sam2-hiera-large",
            }

            if sam_checkpoint not in checkpoint_map:
                raise ValueError(
                    f"Invalid SAM2 checkpoint '{sam_checkpoint}'. Use: {list(checkpoint_map.keys())}"
                )

            model_name = checkpoint_map[sam_checkpoint]
            logger.info(f"\nLoading SAM2 model: {model_name}")
            processor = Sam2Processor.from_pretrained(model_name)
            model = Sam2Model.from_pretrained(model_name)

        else:  # sam3
            from transformers import Sam3Processor, Sam3Model

            model_name = "facebook/sam3"
            logger.info(f"\nLoading SAM3 model: {model_name}")
            logger.info(f"  Note: SAM3 has single 840M param model (no size variants)")
            if sam_checkpoint != "large":
                logger.info(f"  Requested checkpoint '{sam_checkpoint}' mapped to unified SAM3 model")

            processor = Sam3Processor.from_pretrained(model_name)
            model = Sam3Model.from_pretrained(model_name)

        # Create dataset using SAMDataset wrapper
        train_dataset = SAMDataset(
            dataset=self.RFIDataset.dataset,
            processor=processor,
            bbox_perturbation=bbox_perturbation
        )

        # Build DataLoader config
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        # Only add worker-specific settings if using workers
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            dataloader_kwargs['persistent_workers'] = persistent_workers

        train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)

        # Create validation dataloader if provided
        val_dataloader = None
        if validation_dataset is not None:
            val_dataset = SAMDataset(
                dataset=validation_dataset,
                processor=processor,
                bbox_perturbation=0  # No augmentation for validation
            )

            # Use same DataLoader config but no shuffle for validation
            val_kwargs = dataloader_kwargs.copy()
            val_kwargs['shuffle'] = False

            val_dataloader = DataLoader(val_dataset, **val_kwargs)
            logger.info(f"  Validation samples: {len(validation_dataset)}")

        # Freeze layers based on config
        for name, param in model.named_parameters():
            if freeze_vision_encoder and name.startswith("vision_encoder"):
                param.requires_grad_(False)
            if freeze_prompt_encoder and name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params/1e6:.1f}M")
        logger.info(f"  Trainable parameters: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")

        # Load pretrained weights if provided
        if model_path:
            logger.info(f"Loading pretrained weights from: {model_path}")
            model.load_state_dict(torch.load(model_path))

        # Setup optimizer
        trainable_param_list = [p for p in model.parameters() if p.requires_grad]

        if optimizer.lower() == 'adam':
            opt = Adam(trainable_param_list, lr=learning_rate, weight_decay=weight_decay,
                      betas=adam_betas, eps=adam_eps)
        elif optimizer.lower() == 'adamw':
            from torch.optim import AdamW
            opt = AdamW(trainable_param_list, lr=learning_rate, weight_decay=weight_decay,
                       betas=adam_betas, eps=adam_eps)
        elif optimizer.lower() == 'sgd':
            from torch.optim import SGD
            opt = SGD(trainable_param_list, lr=learning_rate, weight_decay=weight_decay,
                     momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'adam', 'adamw', or 'sgd'")

        # Setup loss function
        if loss_function.lower() == 'dicece':
            seg_loss = monai.losses.DiceCELoss(
                sigmoid=loss_sigmoid,
                squared_pred=loss_squared_pred,
                reduction=loss_reduction
            )
        elif loss_function.lower() == 'dice':
            seg_loss = monai.losses.DiceLoss(
                sigmoid=loss_sigmoid,
                squared_pred=loss_squared_pred,
                reduction=loss_reduction
            )
        elif loss_function.lower() == 'ce':
            from torch.nn import BCEWithLogitsLoss
            seg_loss = BCEWithLogitsLoss(reduction=loss_reduction)
        elif loss_function.lower() == 'focal':
            seg_loss = monai.losses.FocalLoss(reduction=loss_reduction)
        else:
            raise ValueError(f"Unknown loss: {loss_function}. Use 'dicece', 'dice', 'ce', or 'focal'")

        # Move model to device
        model.to(self.device)
        model.train()

        logger.info(f"\nTraining {self.model_type.upper()} model...")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Optimizer: {optimizer}")
        logger.info(f"  Loss function: {loss_function}")
        logger.info(f"  Device: {self.device}")

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_losses = []

            total_batches = len(train_dataloader)
            epoch_start_time = time.time()
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} [Train]: Starting {total_batches} batches")

            for batch_idx, batch in enumerate(train_dataloader, 1):
                # Forward pass - SAM3 requires text prompts or empty text_embeds
                if self.model_type == 'sam3':
                    # SAM3: visual-only prompting with empty text
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(self.device),
                        input_boxes=batch["input_boxes"].to(self.device),
                        text_embeds=torch.zeros(batch["pixel_values"].size(0), 1, 768, device=self.device),  # Empty text embeddings
                        multimask_output=multimask_output,
                    )
                else:
                    # SAM2: visual-only prompting
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(self.device),
                        input_boxes=batch["input_boxes"].to(self.device),
                        multimask_output=multimask_output,
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
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Extract loss value
                loss_value = loss.item()
                epoch_train_losses.append(loss_value)

                # CRITICAL: Explicit cleanup to prevent memory accumulation
                del outputs, predicted_masks, ground_truth_masks, ground_truth_masks_resized, loss, batch

                # Clear CUDA cache periodically to prevent fragmentation
                if cuda_cache_clear_interval > 0 and batch_idx % cuda_cache_clear_interval == 0:
                    torch.cuda.empty_cache()

                # Log progress
                if batch_idx % log_interval == 0 or batch_idx == total_batches:
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
                logger.info(f"\nEpoch {epoch+1}/{num_epochs} [Val]: Starting {total_val_batches} batches")

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader, 1):
                        # Forward pass - SAM3 requires text prompts or empty text_embeds
                        if self.model_type == 'sam3':
                            outputs = model(
                                pixel_values=batch["pixel_values"].to(self.device),
                                input_boxes=batch["input_boxes"].to(self.device),
                                text_embeds=torch.zeros(batch["pixel_values"].size(0), 1, 768, device=self.device),
                                multimask_output=multimask_output,
                            )
                        else:
                            outputs = model(
                                pixel_values=batch["pixel_values"].to(self.device),
                                input_boxes=batch["input_boxes"].to(self.device),
                                multimask_output=multimask_output,
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

                        # CRITICAL: Explicit cleanup
                        del outputs, predicted_masks, ground_truth_masks, ground_truth_masks_resized, loss, batch

                        # Clear CUDA cache periodically
                        if cuda_cache_clear_interval > 0 and batch_idx % cuda_cache_clear_interval == 0:
                            torch.cuda.empty_cache()

                        # Log progress
                        if batch_idx % log_interval == 0 or batch_idx == total_val_batches:
                            _log_progress(batch_idx, total_val_batches, val_start_time,
                                        f"Epoch {epoch+1}/{num_epochs} [Val] ", loss_value)

                epoch_val_loss = mean(epoch_val_losses)
                val_losses.append(epoch_val_loss)

            # Log epoch statistics
            log_msg = f"EPOCH: {epoch+1}/{num_epochs} | Train loss: {epoch_mean_train_loss:.6f}"
            if epoch_val_loss is not None:
                log_msg += f" | Val loss: {epoch_val_loss:.6f}"
            logger.info(log_msg)

            # Force garbage collection at end of epoch
            gc.collect()
            torch.cuda.empty_cache()

        self.ave_meanloss = train_losses
        self.val_losses = val_losses if val_losses else None

        # Save model
        if save_model:
            self._save_model(model, sam_checkpoint, num_epochs, trained_model_path)

        # Plot loss curve
        if plot:
            self._plot_loss_curve(sam_checkpoint, num_epochs)

        logger.info(f"\nTraining complete!")

        # Return losses
        if self.val_losses:
            return {"train": train_losses, "val": val_losses}
        else:
            return train_losses

    def _save_model(self, model, sam_checkpoint, num_epochs, trained_model_path=None):
        """Save trained model with descriptive filename"""
        # Extract params from dataset if available
        params = getattr(self.RFIDataset, 'dataset_params', None)

        if params:
            # Old format (legacy RFIDataset)
            stretch = params.get("stretch", "unknown")
            flag_sigma = params.get("flag_sigma", "unknown")
            patch_method = params.get("patch_method", "unknown")
            patch_size = params.get("patch_size", "unknown")
        else:
            # New format (TorchDataset)
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, 'metadata', {})
            stretch = metadata.get("stretch", "unknown")
            flag_sigma = metadata.get("flag_sigma", "unknown")
            patch_method = "torch"
            patch_size = metadata.get("patch_size", "unknown")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"model_{self.model_type}-{sam_checkpoint}_"
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
                logger.info(f"Model saved to: {trained_model_path}")
            except Exception as e:
                logger.info(f"Could not save to {trained_model_path}: {e}")
                logger.info(f"Saving to default location: {os.path.join(method_dir, filename)}")
                torch.save(model.state_dict(), os.path.join(method_dir, filename))
        else:
            save_path = os.path.join(method_dir, filename)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to: {save_path}")

    def _plot_loss_curve(self, sam_checkpoint, num_epochs):
        """Plot and save training and validation loss curves"""
        # Extract params from dataset if available
        params = getattr(self.RFIDataset, 'dataset_params', None)

        if params:
            stretch = params.get("stretch", "unknown")
            flag_sigma = params.get("flag_sigma", "unknown")
            patch_method = params.get("patch_method", "unknown")
            patch_size = params.get("patch_size", "unknown")
        else:
            # New format (TorchDataset)
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, 'metadata', {})
            stretch = metadata.get("stretch", "unknown")
            flag_sigma = metadata.get("flag_sigma", "unknown")
            patch_method = "torch"
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
        title = f"{self.model_type.upper()}-{sam_checkpoint} Training"
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
        os.makedirs(method_dir, exist_ok=True)

        filename = (
            f"loss_plot_{self.model_type}-{sam_checkpoint}_"
            f"stretch-{stretch}_sigma-{flag_sigma}_"
            f"patch-{patch_method}_size-{patch_size}_"
            f"epochs{num_epochs}_{timestamp}.png"
        )

        fig.savefig(os.path.join(method_dir, filename))
        logger.info(f"Loss plot saved to: {os.path.join(method_dir, filename)}")
        plt.close()


# Backward compatibility aliases
SAM2Trainer = SAM3Trainer = SAMTrainer
