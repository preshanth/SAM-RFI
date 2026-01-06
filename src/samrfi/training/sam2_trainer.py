"""
SAM2 model training for RFI detection.

This module provides a PyTorch-based trainer for fine-tuning Meta's SAM2 (Segment
Anything Model 2) on radio frequency interference (RFI) detection tasks. It uses
HuggingFace transformers library and supports flexible training configurations,
GPU-accelerated data transforms, validation, and checkpoint management.

Classes
-------
SAM2Trainer
    Main training class for SAM2 model fine-tuning.

Functions
---------
_log_progress
    Internal progress logging without TQDM overhead.

Examples
--------
Basic training workflow:

>>> from samrfi.data import RFIDataset
>>> from samrfi.training import SAM2Trainer
>>>
>>> # Create dataset
>>> dataset = RFIDataset()
>>> dataset.load_ms('observation.ms')
>>> dataset.create_dataset(patch_size=256)
>>>
>>> # Train model
>>> trainer = SAM2Trainer(dataset, device='cuda')
>>> losses = trainer.train(
...     num_epochs=10,
...     batch_size=8,
...     sam_checkpoint='large',
...     learning_rate=1e-5
... )

GPU-accelerated training with on-the-fly transforms:

>>> from samrfi.data import GPUPreprocessor
>>>
>>> # Use GPU-accelerated pipeline (10-100x faster)
>>> preprocessor = GPUPreprocessor(complex_data, masks)
>>> preprocessor.create_raw_patches(patch_size=256)
>>>
>>> trainer = SAM2Trainer(preprocessor, device='cuda', use_gpu_transforms=True)
>>> losses = trainer.train(batch_size=32)  # 4x larger batches possible

Notes
-----
- SAM2 training requires GPU with sufficient VRAM (8GB+ recommended)
- Training freezes vision and prompt encoders by default (only mask decoder trained)
- Supports multiple loss functions: DiceCE, Dice, Cross-Entropy, Focal
- Checkpoints include full training state for resuming
- GPU transforms provide 10-100x speedup over CPU pipeline
"""

import gc
import logging
import multiprocessing
import os
import time
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional, Union

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


def _log_progress(
    batch_idx: int,
    total_batches: int,
    start_time: float,
    prefix: str = "",
    current_loss: Optional[float] = None,
) -> None:
    """
    Log training progress without TQDM overhead.

    Provides lightweight progress logging that displays batch progress, elapsed time,
    processing rate, and optional loss values. Designed as a TQDM alternative to
    avoid additional dependencies and overhead.

    Parameters
    ----------
    batch_idx : int
        Current batch index (1-indexed).
    total_batches : int
        Total number of batches in epoch.
    start_time : float
        Epoch start time from time.time().
    prefix : str, optional
        Message prefix for log output (e.g., "Epoch 1/10 [Train]"), by default "".
    current_loss : float, optional
        Current batch loss value to display, by default None.

    Examples
    --------
    >>> import time
    >>> start = time.time()
    >>> _log_progress(100, 500, start, prefix="Epoch 1/10 [Train]", current_loss=0.234)
    [2025-01-15 10:30:45] Epoch 1/10 [Train][100/500] Elapsed: 2m15s, Rate: 0.74 batch/s, Loss: 0.234000

    Notes
    -----
    - Time elapsed displayed in minutes:seconds format
    - Processing rate calculated as batches per second
    - Loss display is optional and formatted to 6 decimal places
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
    PyTorch trainer for fine-tuning SAM2 model on RFI detection.

    Provides a clean, simple training interface using HuggingFace transformers
    library. Supports both CPU and GPU training, validation splits, checkpoint
    resuming, and GPU-accelerated data transforms. Designed to mirror SAM1
    training approach with modern best practices.

    Parameters
    ----------
    rfidataset_instance : RFIDataset or GPUPreprocessor
        Dataset instance containing training data. Can be either:
        - RFIDataset instance with `.dataset` attribute (CPU pipeline)
        - GPUPreprocessor instance with `.raw_patches` attribute (GPU pipeline)
    device : str, optional
        Training device: 'cuda' or 'cpu', by default 'cuda'.
    dir_path : str, optional
        Directory to save models and plots. If None, uses current working
        directory. Creates 'samrfi_data' subdirectory, by default None.
    use_gpu_transforms : bool, optional
        Enable GPU-accelerated on-the-fly transforms (10-100x faster than CPU).
        Requires GPUPreprocessor instance, by default False.

    Attributes
    ----------
    device : str
        Training device ('cuda' or 'cpu').
    RFIDataset : RFIDataset or GPUPreprocessor
        Dataset instance for training.
    use_gpu_transforms : bool
        Whether GPU-accelerated transforms are enabled.
    directory : str
        Output directory for saving models and plots.
    ave_meanloss : list of float
        Training loss history (mean loss per epoch).
    val_losses : list of float or None
        Validation loss history if validation dataset provided.
    best_val_loss : float
        Best validation loss seen (set during training if validation enabled).

    Examples
    --------
    Basic training with CPU transforms:

    >>> from samrfi.data import RFIDataset
    >>> dataset = RFIDataset()
    >>> dataset.load_ms('observation.ms')
    >>> dataset.create_dataset(patch_size=256)
    >>>
    >>> trainer = SAM2Trainer(dataset, device='cuda')
    >>> losses = trainer.train(num_epochs=10, batch_size=8)

    GPU-accelerated training (10-100x faster data pipeline):

    >>> from samrfi.data import GPUPreprocessor
    >>> preprocessor = GPUPreprocessor(complex_data, masks)
    >>> preprocessor.create_raw_patches(patch_size=256)
    >>>
    >>> trainer = SAM2Trainer(preprocessor, device='cuda', use_gpu_transforms=True)
    >>> losses = trainer.train(batch_size=32)  # 4x larger batches possible

    Training with validation and checkpoint resuming:

    >>> trainer = SAM2Trainer(dataset, device='cuda')
    >>> losses = trainer.train(
    ...     num_epochs=20,
    ...     batch_size=8,
    ...     validation_dataset=val_dataset,
    ...     model_path='checkpoint.pth'  # Resume from checkpoint
    ... )

    Notes
    -----
    - GPU transforms reduce storage by 75% (no pre-generated augmentations)
    - Training checkpoints include full state for resuming
    - Validation enabled automatically if validation_dataset provided
    - Best model saved separately during validation
    - Memory optimized with periodic cache clearing
    """

    def __init__(
        self,
        rfidataset_instance: Any,
        device: str = "cuda",
        dir_path: Optional[str] = None,
        use_gpu_transforms: bool = False,
    ) -> None:
        """
        Initialize SAM2 trainer with dataset and configuration.

        Sets up trainer instance with dataset, device configuration, output directory,
        and GPU transform settings. Initializes loss tracking attributes and prepares
        output directory structure.

        Parameters
        ----------
        rfidataset_instance : RFIDataset or GPUPreprocessor
            Dataset instance containing training data.
        device : str, optional
            Training device: 'cuda' or 'cpu', by default 'cuda'.
        dir_path : str, optional
            Directory to save models and plots, by default None (uses cwd).
        use_gpu_transforms : bool, optional
            Enable GPU-accelerated transforms, by default False.

        Notes
        -----
        Creates 'samrfi_data/models' subdirectory for checkpoints and plots.
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
        self.ave_meanloss: List[float] = []
        self.val_losses: Optional[List[float]] = None

    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 4,
        sam_checkpoint: str = "large",
        learning_rate: float = 1e-6,
        # Optimizer settings
        optimizer: str = "adam",
        weight_decay: float = 0.05,
        adam_betas: tuple = (0.9, 0.999),
        adam_eps: float = 1e-8,
        momentum: float = 0.9,
        # Loss function settings
        loss_function: str = "dicece",
        loss_sigmoid: bool = True,
        loss_squared_pred: bool = True,
        loss_reduction: str = "mean",
        # Model architecture
        multimask_output: bool = False,
        freeze_vision_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        # Data augmentation
        bbox_perturbation: int = 20,
        # DataLoader settings
        num_workers: int = 0,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        # Training optimization
        log_interval: int = 100,
        cuda_cache_clear_interval: int = 100,
        # Output settings
        plot: bool = True,
        model_path: Optional[str] = None,
        trained_model_path: Optional[str] = None,
        validation_dataset: Optional[Any] = None,
        save_model: bool = True,
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        Train SAM2 model on RFI detection dataset.

        Performs complete training workflow including model loading, dataset preparation,
        optimizer setup, training loop with optional validation, checkpoint saving, and
        loss visualization. Supports checkpoint resuming, validation splits, and multiple
        loss functions.

        Parameters
        ----------
        num_epochs : int, optional
            Number of training epochs, by default 3.
        batch_size : int, optional
            Training batch size (GPU memory permitting), by default 4.
        sam_checkpoint : str, optional
            SAM2 model size: 'tiny', 'small', 'base_plus', or 'large', by default 'large'.
        learning_rate : float, optional
            Learning rate for optimizer, by default 1e-6.
        optimizer : str, optional
            Optimizer type: 'adam', 'adamw', or 'sgd', by default 'adam'.
        weight_decay : float, optional
            L2 regularization weight decay, by default 0.05.
        adam_betas : tuple of float, optional
            Beta coefficients for Adam optimizer (beta1, beta2), by default (0.9, 0.999).
        adam_eps : float, optional
            Epsilon for numerical stability in Adam, by default 1e-8.
        momentum : float, optional
            Momentum factor for SGD optimizer, by default 0.9.
        loss_function : str, optional
            Loss function: 'dicece' (Dice+CrossEntropy), 'dice', 'ce', or 'focal',
            by default 'dicece'.
        loss_sigmoid : bool, optional
            Apply sigmoid to predictions before loss calculation, by default True.
        loss_squared_pred : bool, optional
            Use squared predictions in Dice loss, by default True.
        loss_reduction : str, optional
            Loss reduction method: 'mean' or 'sum', by default 'mean'.
        multimask_output : bool, optional
            Enable SAM2 multi-mask output mode, by default False.
        freeze_vision_encoder : bool, optional
            Freeze vision encoder weights (only train mask decoder), by default True.
        freeze_prompt_encoder : bool, optional
            Freeze prompt encoder weights, by default True.
        bbox_perturbation : int, optional
            Bounding box perturbation in pixels for data augmentation, by default 20.
        num_workers : int, optional
            Number of DataLoader workers (0=main process), by default 0.
        prefetch_factor : int, optional
            Number of batches to prefetch per worker (only if num_workers>0), by default 2.
        persistent_workers : bool, optional
            Keep workers alive between epochs (only if num_workers>0), by default True.
        pin_memory : bool, optional
            Pin memory for faster GPU transfer, by default True.
        log_interval : int, optional
            Log progress every N batches, by default 100.
        cuda_cache_clear_interval : int, optional
            Clear CUDA cache every N batches (0=disable), by default 100.
        plot : bool, optional
            Plot and save loss curves after training, by default True.
        model_path : str, optional
            Path to pretrained checkpoint to resume from, by default None.
        trained_model_path : str, optional
            Custom path to save final trained model, by default None (auto-generated).
        validation_dataset : Any, optional
            Validation dataset (same format as training dataset), by default None.
        save_model : bool, optional
            Save final model checkpoint (set False for validation-only runs), by default True.

        Returns
        -------
        list of float or dict
            If no validation: Returns list of training losses (one per epoch).
            If validation enabled: Returns dict with keys 'train' and 'val', each
            containing list of losses per epoch.

        Raises
        ------
        ValueError
            If sam_checkpoint not in ['tiny', 'small', 'base_plus', 'large'].
            If optimizer not in ['adam', 'adamw', 'sgd'].
            If loss_function not in ['dicece', 'dice', 'ce', 'focal'].
            If use_gpu_transforms=True but dataset is not GPUPreprocessor.

        Examples
        --------
        Basic training:

        >>> trainer = SAM2Trainer(dataset, device='cuda')
        >>> losses = trainer.train(num_epochs=10, batch_size=8)
        >>> print(f"Final loss: {losses[-1]:.4f}")

        Training with validation:

        >>> losses = trainer.train(
        ...     num_epochs=20,
        ...     batch_size=8,
        ...     validation_dataset=val_dataset
        ... )
        >>> print(f"Train: {losses['train'][-1]:.4f}, Val: {losses['val'][-1]:.4f}")

        Resume from checkpoint:

        >>> losses = trainer.train(
        ...     num_epochs=30,
        ...     model_path='checkpoint_epoch_10.pth'
        ... )

        Custom loss and optimizer:

        >>> losses = trainer.train(
        ...     loss_function='focal',
        ...     optimizer='adamw',
        ...     weight_decay=0.01,
        ...     learning_rate=1e-4
        ... )

        Notes
        -----
        - Training automatically freezes encoders (only mask decoder trained)
        - Checkpoints include full state: model, optimizer, losses, config
        - Best validation model saved separately if validation enabled
        - GPU memory optimized with periodic cache clearing
        - Supports checkpoint resuming with full state restoration
        - Loss curves automatically plotted and saved
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

            for batch_idx, batch in enumerate(train_dataloader, 1):
                # Forward pass
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

            # Save best model based on validation loss
            if epoch_val_loss is not None:
                if not hasattr(self, "best_val_loss"):
                    self.best_val_loss = float("inf")

                if epoch_val_loss < self.best_val_loss:
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
                        f"  💾 New best model saved (val_loss: {epoch_val_loss:.6f}) -> {best_model_path}"
                    )

            # Force garbage collection at end of epoch
            gc.collect()
            torch.cuda.empty_cache()

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
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        sam_checkpoint: str,
        learning_rate: float,
        batch_size: int,
        loss_function: str,
        patch_size: Union[int, str],
        num_epochs: int,
        freeze_vision_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        trained_model_path: Optional[str] = None,
    ) -> None:
        """
        Save trained model checkpoint with full training state.

        Creates comprehensive checkpoint file containing model weights, optimizer state,
        training history, preprocessing metadata, and training configuration. Supports
        both custom save paths and auto-generated filenames with timestamp and parameters.

        Parameters
        ----------
        model : torch.nn.Module
            Trained SAM2 model instance.
        optimizer : torch.optim.Optimizer
            Optimizer instance with current state.
        epoch : int
            Final epoch number (0-indexed).
        sam_checkpoint : str
            SAM2 model size ('tiny', 'small', 'base_plus', 'large').
        learning_rate : float
            Learning rate used for training.
        batch_size : int
            Batch size used for training.
        loss_function : str
            Loss function used ('dicece', 'dice', 'ce', 'focal').
        patch_size : int or str
            Patch size used for training (e.g., 256) or 'unknown'.
        num_epochs : int
            Total number of training epochs.
        freeze_vision_encoder : bool, optional
            Whether vision encoder was frozen, by default True.
        freeze_prompt_encoder : bool, optional
            Whether prompt encoder was frozen, by default True.
        trained_model_path : str, optional
            Custom path to save checkpoint. If None, auto-generates filename
            with timestamp and parameters, by default None.

        Notes
        -----
        Checkpoint structure:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state for resuming
        - epoch: Final epoch number
        - training_losses: List of training losses per epoch
        - validation_losses: List of validation losses (or None)
        - patch_size: Patch size (kept for backward compatibility)
        - preprocessing: Dict of preprocessing metadata
        - config: Dict of training configuration

        Auto-generated filename format:
        model_sam2-{checkpoint}_stretch-{stretch}_sigma-{sigma}_patch-{method}_size-{size}_epochs{n}_{timestamp}.pth

        Examples
        --------
        >>> # Called internally by train() method
        >>> trainer._save_model(
        ...     model, optimizer, epoch=9, sam_checkpoint='large',
        ...     learning_rate=1e-5, batch_size=8, loss_function='dicece',
        ...     patch_size=256, num_epochs=10
        ... )
        Model checkpoint saved to: ./samrfi_data/models/model_sam2-large_...pth
        """
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

    def _plot_loss_curve(self, sam_checkpoint: str, num_epochs: int) -> None:
        """
        Plot and save training and validation loss curves.

        Creates matplotlib figure showing training loss (and validation loss if available)
        over epochs. Saves high-resolution plot to models directory with auto-generated
        filename containing training parameters and timestamp.

        Parameters
        ----------
        sam_checkpoint : str
            SAM2 model size ('tiny', 'small', 'base_plus', 'large') for plot title.
        num_epochs : int
            Total number of training epochs for plot title.

        Notes
        -----
        - Plot dimensions: 12x6 inches at 300 DPI
        - Training loss: Blue line with circle markers
        - Validation loss: Red line with square markers (if available)
        - Includes dataset size in title
        - Auto-generated filename matches model checkpoint naming

        Filename format:
        loss_plot_sam2-{checkpoint}_stretch-{stretch}_sigma-{sigma}_patch-{method}_size-{size}_epochs{n}_{timestamp}.png

        Examples
        --------
        >>> # Called internally by train() method
        >>> trainer._plot_loss_curve(sam_checkpoint='large', num_epochs=10)
        Loss plot saved to: ./samrfi_data/models/loss_plot_sam2-large_...png
        """
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
