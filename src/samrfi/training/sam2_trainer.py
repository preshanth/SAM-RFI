"""
SAM2 Trainer - Clean implementation using transformers library
Mirrors the working SAM1 training approach
"""

import gc
import logging
import multiprocessing
import os
import time
from datetime import datetime
from statistics import mean

import matplotlib.pyplot as plt
import monai
import torch
from torch.nn.functional import interpolate
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import Sam2Model, Sam2Processor

from samrfi.data import RAMCachedDataset, SAMDataset

# Get logger (configured by parent script or defaults to console if standalone)
logger = logging.getLogger(__name__)

# Ensure logging is configured (fallback for standalone use)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def _log_progress(batch_idx, total_batches, start_time, prefix="", current_loss=None):
    """
    Log training progress without TQDM overhead.
    """
    elapsed = time.time() - start_time
    rate = batch_idx / elapsed if elapsed > 0 else 0

    # Format elapsed time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    elapsed_str = f"{elapsed_min}m{elapsed_sec:02d}s"

    loss_str = f", Loss: {current_loss:.6f}" if current_loss is not None else ""
    logger.info(
        f"{prefix}[{batch_idx}/{total_batches}] "
        f"Elapsed: {elapsed_str}, Rate: {rate:.2f} batch/s{loss_str}"
    )


class SAM2Trainer:
    """
    SAM2 training using HuggingFace transformers library.
    Simple, clean implementation that mirrors working SAM1 code.
    """

    def __init__(self, rfidataset_instance, device="cuda", dir_path=None, use_gpu_transforms=False):
        """
        Initialize SAM2 trainer

        Args:
            rfidataset_instance: RFIDataset instance with .dataset attribute
                                OR GPUPreprocessor instance with .raw_patches attribute
            device: 'cuda' or 'cpu'
            dir_path: Directory to save models (default: ./samrfi_data)
            use_gpu_transforms: Use GPU-accelerated transforms (10-100x faster) (default: False)
        """
        self.device = device
        self.RFIDataset = rfidataset_instance
        self.use_gpu_transforms = use_gpu_transforms

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
        optimizer="adam",
        weight_decay=0.05,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        momentum=0.9,
        # Loss function settings
        loss_function="dicece",
        loss_sigmoid=True,
        loss_squared_pred=True,
        loss_reduction="mean",
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
        use_amp=False,
        accumulation_steps=1,
        # Output settings
        plot=True,
        model_path=None,
        trained_model_path=None,
        validation_dataset=None,
        save_model=True,
        patience=None,
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
            patience: Early-stopping patience in epochs. If set, training stops when the
                monitored loss (validation loss if a validation set is given, otherwise
                training loss) has not improved for `patience` consecutive epochs.
                Default None disables early stopping (no change to existing behavior).
            use_amp: Enable automatic mixed precision (fp16 autocast + GradScaler) on
                CUDA. Default False preserves full-fp32 behavior; ignored on CPU.
            accumulation_steps: Accumulate gradients over this many batches before each
                optimizer step, for an effective batch size of
                batch_size * accumulation_steps without the extra memory. Default 1.
        """

        # Fix multiprocessing for CUDA in workers (required for GPU transforms)
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

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

        logger.info(f"\nLoading SAM2 model: {model_name}")

        # Load processor and model from HuggingFace
        processor = Sam2Processor.from_pretrained(model_name)
        model = Sam2Model.from_pretrained(model_name)

        # Create dataset - GPU transforms or standard CPU pipeline
        if self.use_gpu_transforms:
            # GPU-accelerated transform pipeline (10-100x faster)
            from samrfi.data import GPUTransformDataset

            logger.info("  Using GPU-accelerated transform pipeline")

            # Check if we have raw patches from GPUPreprocessor
            if hasattr(self.RFIDataset, "raw_patches") and hasattr(self.RFIDataset, "raw_masks"):
                logger.info("  Using raw patches from GPUPreprocessor")
                logger.info(f"  Patches: {len(self.RFIDataset.raw_patches)}")

                train_dataset = GPUTransformDataset(
                    complex_patches=self.RFIDataset.raw_patches,
                    masks=self.RFIDataset.raw_masks,
                    device=self.device,
                    enable_augmentation=True,
                    stretch_type=None,  # Can be configured if needed
                    normalize_before_stretch=False,
                    normalize_after_stretch=False,
                    bbox_perturbation=bbox_perturbation,
                    pin_memory=pin_memory,
                )
            else:
                raise ValueError(
                    "use_gpu_transforms=True requires GPUPreprocessor instance "
                    "with raw_patches and raw_masks attributes. "
                    "Use: preprocessor = GPUPreprocessor(data, flags); "
                    "preprocessor.create_raw_patches()"
                )
        else:
            # Standard CPU pipeline (backward compatible)
            train_dataset = SAMDataset(
                dataset=self.RFIDataset.dataset,
                processor=processor,
                bbox_perturbation=bbox_perturbation,
            )

        # Build DataLoader config
        use_pin_memory = pin_memory and not isinstance(self.RFIDataset.dataset, RAMCachedDataset)
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": use_pin_memory,
        }
        # Only add worker-specific settings if using workers
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
            dataloader_kwargs["persistent_workers"] = persistent_workers

        train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)

        # Create validation dataloader if provided
        val_dataloader = None
        if validation_dataset is not None:
            val_dataset = SAMDataset(
                dataset=validation_dataset, processor=processor, bbox_perturbation=bbox_perturbation
            )

            # Use same DataLoader config but no shuffle for validation
            val_kwargs = dataloader_kwargs.copy()
            val_kwargs["shuffle"] = False

            val_dataloader = DataLoader(val_dataset, **val_kwargs)
            logger.info(f"  Validation samples: {len(validation_dataset)}")

        # Freeze layers based on config
        for name, param in model.named_parameters():
            if freeze_vision_encoder and name.startswith("vision_encoder"):
                param.requires_grad_(False)
            if freeze_prompt_encoder and name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        # Load pretrained weights or resume from checkpoint
        start_epoch = 0
        resume_train_losses = []
        resume_val_losses = []
        checkpoint_data = None

        if model_path:
            logger.info(f"Loading from: {model_path}")
            checkpoint_data = torch.load(model_path)

            # Check if full checkpoint or just state_dict
            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                # Full checkpoint format
                model.load_state_dict(checkpoint_data["model_state_dict"])
                start_epoch = checkpoint_data.get("epoch", -1) + 1
                resume_train_losses = checkpoint_data.get("training_losses", [])
                resume_val_losses = checkpoint_data.get("validation_losses", [])

                if start_epoch > 0 and start_epoch < num_epochs:
                    logger.info(f"  Resuming from epoch {start_epoch} (continuing to {num_epochs})")
                    logger.info(f"  Previous train loss: {resume_train_losses[-1]:.6f}")
                    if resume_val_losses:
                        logger.info(f"  Previous val loss: {resume_val_losses[-1]:.6f}")
                else:
                    logger.info("  Loading pretrained weights (starting from epoch 0)")
                    start_epoch = 0
            else:
                # Old format - just state_dict
                model.load_state_dict(checkpoint_data)
                logger.info("  Loaded model weights (old format)")
                start_epoch = 0

        # Setup optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if optimizer.lower() == "adam":
            opt = Adam(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=adam_betas,
                eps=adam_eps,
            )
        elif optimizer.lower() == "adamw":
            from torch.optim import AdamW

            opt = AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=adam_betas,
                eps=adam_eps,
            )
        elif optimizer.lower() == "sgd":
            from torch.optim import SGD

            opt = SGD(
                trainable_params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'adam', 'adamw', or 'sgd'")

        # Restore optimizer state if resuming
        if checkpoint_data and "optimizer_state_dict" in checkpoint_data and start_epoch > 0:
            opt.load_state_dict(checkpoint_data["optimizer_state_dict"])
            logger.info("  Restored optimizer state")

        # Automatic mixed precision: autocast region + loss scaler.
        # No-op when use_amp=False or when running on CPU.
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        use_amp = use_amp and device_type == "cuda"
        try:
            # Generic device API (torch >= 2.3)
            scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
        except (AttributeError, TypeError):
            # Fallback for torch 2.0-2.2 (CUDA-only scaler namespace)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if use_amp:
            logger.info("  Mixed precision (AMP) enabled")
        if accumulation_steps > 1:
            logger.info(
                f"  Gradient accumulation: {accumulation_steps} steps "
                f"(effective batch size {batch_size * accumulation_steps})"
            )

        # Setup loss function
        if loss_function.lower() == "dicece":
            seg_loss = monai.losses.DiceCELoss(
                sigmoid=loss_sigmoid, squared_pred=loss_squared_pred, reduction=loss_reduction
            )
        elif loss_function.lower() == "dice":
            seg_loss = monai.losses.DiceLoss(
                sigmoid=loss_sigmoid, squared_pred=loss_squared_pred, reduction=loss_reduction
            )
        elif loss_function.lower() == "ce":
            from torch.nn import BCEWithLogitsLoss

            seg_loss = BCEWithLogitsLoss(reduction=loss_reduction)
        elif loss_function.lower() == "focal":
            seg_loss = monai.losses.FocalLoss(reduction=loss_reduction)
        else:
            raise ValueError(
                f"Unknown loss: {loss_function}. Use 'dicece', 'dice', 'ce', or 'focal'"
            )

        # Move model to device
        model.to(self.device)
        model.train()

        # Extract preprocessing metadata for checkpoint saving
        params = getattr(self.RFIDataset, "dataset_params", None)
        if params:
            preprocessing_metadata = {
                "patch_size": params.get("patch_size", "unknown"),
                "augmentation_rotations": params.get("augmentation_rotations", 4),
                "stretch": params.get("stretch", None),
                "normalize_before_stretch": params.get("normalize_before_stretch", True),
                "normalize_after_stretch": params.get("normalize_after_stretch", False),
            }
        else:
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, "metadata", {})
            preprocessing_metadata = {
                "patch_size": metadata.get("patch_size", "unknown"),
                "augmentation_rotations": metadata.get("augmentation_rotations", 4),
                "stretch": metadata.get("stretch", None),
                "normalize_before_stretch": metadata.get("normalize_before_stretch", True),
                "normalize_after_stretch": metadata.get("normalize_after_stretch", False),
            }

        # Keep patch_size for backward compatibility and logging
        patch_size = preprocessing_metadata["patch_size"]

        logger.info("\nTraining SAM2 model...")
        logger.info(f"  Epochs: {num_epochs} (starting from {start_epoch})")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Patch size: {patch_size}")

        # Training loop - start from resume epoch if continuing
        train_losses = resume_train_losses.copy()
        val_losses = resume_val_losses.copy()

        for epoch in range(start_epoch, num_epochs):
            # Training phase
            model.train()
            epoch_train_losses = []

            total_batches = len(train_dataloader)
            epoch_start_time = time.time()
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} [Train]: Starting {total_batches} batches")

            opt.zero_grad()
            for batch_idx, batch in enumerate(train_dataloader, 1):
                # Forward pass (under autocast when AMP is enabled)
                with torch.amp.autocast(device_type, enabled=use_amp):
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

                    # Compute loss, normalized for gradient accumulation
                    loss = seg_loss(predicted_masks, ground_truth_masks_resized)
                    loss = loss / accumulation_steps

                # Backward with gradient scaling; step every accumulation_steps batches
                # (and on the final batch to flush any remainder).
                scaler.scale(loss).backward()
                if batch_idx % accumulation_steps == 0 or batch_idx == total_batches:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

                # Report the per-batch loss magnitude (undo accumulation scaling)
                loss_value = loss.item() * accumulation_steps
                epoch_train_losses.append(loss_value)

                # CRITICAL: Explicit cleanup to prevent memory accumulation
                # Safe to delete after optimizer.step() - gradients stored in parameter.grad
                del (
                    outputs,
                    predicted_masks,
                    ground_truth_masks,
                    ground_truth_masks_resized,
                    loss,
                    batch,
                )

                # Clear CUDA cache periodically to prevent fragmentation
                if cuda_cache_clear_interval > 0 and batch_idx % cuda_cache_clear_interval == 0:
                    torch.cuda.empty_cache()

                # Log progress
                if batch_idx % log_interval == 0 or batch_idx == total_batches:
                    _log_progress(
                        batch_idx,
                        total_batches,
                        epoch_start_time,
                        f"Epoch {epoch+1}/{num_epochs} [Train] ",
                        loss_value,
                    )

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
                logger.info(
                    f"\nEpoch {epoch+1}/{num_epochs} [Val]: Starting {total_val_batches} batches"
                )

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader, 1):
                        with torch.amp.autocast(device_type, enabled=use_amp):
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

                        # CRITICAL: Explicit cleanup (same as training)
                        del (
                            outputs,
                            predicted_masks,
                            ground_truth_masks,
                            ground_truth_masks_resized,
                            loss,
                            batch,
                        )

                        # Clear CUDA cache periodically
                        if (
                            cuda_cache_clear_interval > 0
                            and batch_idx % cuda_cache_clear_interval == 0
                        ):
                            torch.cuda.empty_cache()

                        # Log progress
                        if batch_idx % log_interval == 0 or batch_idx == total_val_batches:
                            _log_progress(
                                batch_idx,
                                total_val_batches,
                                val_start_time,
                                f"Epoch {epoch+1}/{num_epochs} [Val] ",
                                loss_value,
                            )

                epoch_val_loss = mean(epoch_val_losses)
                val_losses.append(epoch_val_loss)

            # Log epoch statistics
            log_msg = f"EPOCH: {epoch+1}/{num_epochs} | Train loss: {epoch_mean_train_loss:.6f}"
            if epoch_val_loss is not None:
                log_msg += f" | Val loss: {epoch_val_loss:.6f}"
            logger.info(log_msg)

            # Save best model based on the monitored loss: validation loss when a
            # validation set is provided, otherwise fall back to training loss so a
            # best checkpoint is still produced (and early stopping has a signal).
            if epoch_val_loss is not None:
                monitor_loss = epoch_val_loss
                monitor_name = "val_loss"
            else:
                monitor_loss = epoch_mean_train_loss
                monitor_name = "train_loss"

            if not hasattr(self, "best_monitor_loss"):
                self.best_monitor_loss = float("inf")
                self.best_epoch = epoch
                self.epochs_since_improvement = 0

            if monitor_loss < self.best_monitor_loss:
                self.best_monitor_loss = monitor_loss
                self.best_epoch = epoch
                self.epochs_since_improvement = 0
                # Preserve the legacy attribute name when validating
                if epoch_val_loss is not None:
                    self.best_val_loss = epoch_val_loss
                best_model_path = os.path.join(self.directory, "sam2_rfi_best.pth")
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                    "training_losses": train_losses[: epoch + 1 - start_epoch],
                    "validation_losses": val_losses[: epoch + 1 - start_epoch],
                    "patch_size": patch_size,  # Kept for backward compatibility
                    "preprocessing": preprocessing_metadata,
                    "config": {
                        "sam_checkpoint": sam_checkpoint,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "loss_function": loss_function,
                        "freeze_vision_encoder": freeze_vision_encoder,
                        "freeze_prompt_encoder": freeze_prompt_encoder,
                    },
                }
                torch.save(best_checkpoint, best_model_path)
                logger.info(
                    f"  💾 New best model saved ({monitor_name}: {monitor_loss:.6f}) -> {best_model_path}"
                )
            else:
                self.epochs_since_improvement += 1

            # Force garbage collection at end of epoch
            gc.collect()
            torch.cuda.empty_cache()

            # Early stopping (opt-in via `patience`; default None preserves behavior)
            if patience is not None and self.epochs_since_improvement >= patience:
                logger.info(
                    f"\nEarly stopping at epoch {epoch+1}: {monitor_name} has not improved "
                    f"for {patience} epoch(s) (best {monitor_name}: {self.best_monitor_loss:.6f} "
                    f"at epoch {self.best_epoch+1})."
                )
                break

        self.ave_meanloss = train_losses
        self.val_losses = val_losses if val_losses else None

        # Save model (skip during validation to save memory)
        if save_model:
            self._save_model(
                model,
                opt,
                num_epochs - 1,
                sam_checkpoint,
                learning_rate,
                batch_size,
                loss_function,
                patch_size,
                num_epochs,
                freeze_vision_encoder,
                freeze_prompt_encoder,
                trained_model_path,
            )

        # Plot loss curve
        if plot:
            self._plot_loss_curve(sam_checkpoint, num_epochs)

        logger.info("\nTraining complete!")

        # Return losses
        if self.val_losses:
            return {"train": train_losses, "val": val_losses}
        else:
            return train_losses

    def _save_model(
        self,
        model,
        optimizer,
        epoch,
        sam_checkpoint,
        learning_rate,
        batch_size,
        loss_function,
        patch_size,
        num_epochs,
        freeze_vision_encoder=True,
        freeze_prompt_encoder=True,
        trained_model_path=None,
    ):
        """Save trained model checkpoint with full training state"""
        # Extract params from dataset if available (for backward compatibility in filename)
        params = getattr(self.RFIDataset, "dataset_params", None)

        if params:
            # Old format (legacy RFIDataset)
            stretch = params.get("stretch", "unknown")
            flag_sigma = params.get("flag_sigma", "unknown")
            patch_method = params.get("patch_method", "unknown")
        else:
            # New format (TorchDataset) - extract from metadata if available
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, "metadata", {})
            stretch = metadata.get("stretch", "unknown")
            flag_sigma = metadata.get("flag_sigma", "unknown")
            patch_method = "torch"

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

        # Extract preprocessing metadata for checkpoint
        if params:
            preprocessing_metadata = {
                "patch_size": params.get("patch_size", patch_size),
                "augmentation_rotations": params.get("augmentation_rotations", 4),
                "stretch": params.get("stretch", stretch),
                "normalize_before_stretch": params.get("normalize_before_stretch", True),
                "normalize_after_stretch": params.get("normalize_after_stretch", False),
            }
        else:
            preprocessing_metadata = {
                "patch_size": metadata.get("patch_size", patch_size),
                "augmentation_rotations": metadata.get("augmentation_rotations", 4),
                "stretch": metadata.get("stretch", stretch),
                "normalize_before_stretch": metadata.get("normalize_before_stretch", True),
                "normalize_after_stretch": metadata.get("normalize_after_stretch", False),
            }

        # Create full checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "training_losses": self.ave_meanloss,
            "validation_losses": self.val_losses,
            "patch_size": patch_size,  # Kept for backward compatibility
            "preprocessing": preprocessing_metadata,
            "config": {
                "sam_checkpoint": sam_checkpoint,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "loss_function": loss_function,
                "freeze_vision_encoder": freeze_vision_encoder,
                "freeze_prompt_encoder": freeze_prompt_encoder,
            },
        }

        if trained_model_path:
            try:
                torch.save(checkpoint, trained_model_path)
                logger.info(f"Model checkpoint saved to: {trained_model_path}")
            except Exception as e:
                logger.info(f"Could not save to {trained_model_path}: {e}")
                logger.info(f"Saving to default location: {os.path.join(method_dir, filename)}")
                torch.save(checkpoint, os.path.join(method_dir, filename))
        else:
            save_path = os.path.join(method_dir, filename)
            torch.save(checkpoint, save_path)
            logger.info(f"Model checkpoint saved to: {save_path}")

    def _plot_loss_curve(self, sam_checkpoint, num_epochs):
        """Plot and save training and validation loss curves"""
        # Extract params from dataset if available (for backward compatibility)
        params = getattr(self.RFIDataset, "dataset_params", None)

        if params:
            stretch = params.get("stretch", "unknown")
            flag_sigma = params.get("flag_sigma", "unknown")
            patch_method = params.get("patch_method", "unknown")
            patch_size = params.get("patch_size", "unknown")
        else:
            # New format (TorchDataset)
            dataset = self.RFIDataset.dataset
            metadata = getattr(dataset, "metadata", {})
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
            epochs, self.ave_meanloss, label="Training Loss", color="blue", linewidth=2, marker="o"
        )

        # Plot validation loss if available
        if self.val_losses:
            ax.plot(
                epochs,
                self.val_losses,
                label="Validation Loss",
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
        num_patches = len(dataset) if hasattr(dataset, "__len__") else "unknown"
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
        logger.info(f"Loss plot saved to: {os.path.join(method_dir, filename)}")
        plt.close()
