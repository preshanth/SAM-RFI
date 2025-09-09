"""
GPU-Optimized Training Pipeline

Training pipeline optimized for V100 (memory-constrained, time-flexible)
and H200 (time-constrained, memory-rich) constraints.
"""

import numpy as np
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time
import json
from datetime import datetime, timedelta

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.cuda.amp import GradScaler
    try:
        from torch.amp import autocast  # New syntax
    except ImportError:
        from torch.cuda.amp import autocast  # Fallback for older PyTorch

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    logging.warning(f"PyTorch not available: {e}")

logger = logging.getLogger(__name__)


class LossTracker:
    """Silent loss tracking and post-training visualization"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        logging_config = config.get("logging", {})
        self.collect_metrics = logging_config.get("collect_metrics", True)
        self.verbose_breakdown = logging_config.get("verbose_breakdown", False)
        self.export_on_completion = logging_config.get("export_on_completion", True)
        
        # Data storage
        self.history = {
            "steps": [],
            "epochs": [],
            "timestamps": [],
            "total_loss": [],
            "segmentation_loss": [],
            "iou_loss": [],
            "gaussianity_loss": [],
            "learning_rate": [],
            "gradient_norm": []
        }
        
        # Detailed gaussianity components (optional)
        if self.verbose_breakdown:
            self.history.update({
                "skewness_real": [],
                "skewness_imag": [],
                "kurtosis_real": [],
                "kurtosis_imag": [],
                "anderson_real": [],
                "anderson_imag": []
            })
        
        self.start_time = time.time()
        logger.info(f"LossTracker initialized - collect_metrics: {self.collect_metrics}")
    
    def record(self, step: int, epoch: int, losses: Dict[str, float], 
               learning_rate: float = None, gradient_norm: float = None,
               gaussianity_components: Dict[str, Dict[str, float]] = None):
        """Record training metrics"""
        if not self.collect_metrics:
            return
            
        self.history["steps"].append(step)
        self.history["epochs"].append(epoch)
        self.history["timestamps"].append(time.time() - self.start_time)
        self.history["total_loss"].append(losses.get("total", 0.0))
        self.history["segmentation_loss"].append(losses.get("segmentation", 0.0))
        self.history["iou_loss"].append(losses.get("iou", 0.0))
        self.history["gaussianity_loss"].append(losses.get("gaussianity", 0.0))
        self.history["learning_rate"].append(learning_rate or 0.0)
        self.history["gradient_norm"].append(gradient_norm or 0.0)
        
        # Optional detailed components
        if self.verbose_breakdown and gaussianity_components:
            skew = gaussianity_components.get("skewness", {})
            kurt = gaussianity_components.get("kurtosis", {})
            anderson = gaussianity_components.get("anderson_darling", {})
            
            self.history["skewness_real"].append(skew.get("real", 0.0))
            self.history["skewness_imag"].append(skew.get("imag", 0.0))
            self.history["kurtosis_real"].append(kurt.get("real", 0.0))
            self.history["kurtosis_imag"].append(kurt.get("imag", 0.0))
            self.history["anderson_real"].append(anderson.get("real", 0.0))
            self.history["anderson_imag"].append(anderson.get("imag", 0.0))
    
    def save_history(self):
        """Save training history to JSON"""
        if not self.collect_metrics:
            return
            
        history_file = self.output_dir / "training_history.json"
        
        # Add metadata
        metadata = {
            "config": self.config,
            "total_steps": len(self.history["steps"]),
            "total_epochs": max(self.history["epochs"]) + 1 if self.history["epochs"] else 0,
            "training_duration_hours": (time.time() - self.start_time) / 3600,
            "final_losses": {
                "total": self.history["total_loss"][-1] if self.history["total_loss"] else 0,
                "segmentation": self.history["segmentation_loss"][-1] if self.history["segmentation_loss"] else 0,
                "iou": self.history["iou_loss"][-1] if self.history["iou_loss"] else 0,
                "gaussianity": self.history["gaussianity_loss"][-1] if self.history["gaussianity_loss"] else 0
            }
        }
        
        export_data = {
            "metadata": metadata,
            "history": self.history
        }
        
        with open(history_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Training history saved: {history_file}")
        return history_file
    
    def generate_plots(self):
        """Generate comprehensive training plots on completion"""
        if not self.export_on_completion or not self.collect_metrics:
            return
            
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            logger.warning("matplotlib/pandas not available - skipping plot generation")
            return
        
        if not self.history["steps"]:
            logger.warning("No training history to plot")
            return
        
        plots_dir = self.output_dir / "training_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Convert to pandas for easier plotting
        df = pd.DataFrame(self.history)
        
        # 1. Loss Components Over Time
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(df["steps"], df["total_loss"], label="Total Loss", linewidth=2)
        plt.plot(df["steps"], df["segmentation_loss"], label="Segmentation Loss", alpha=0.8)
        plt.plot(df["steps"], df["iou_loss"], label="IoU Loss", alpha=0.8)
        plt.plot(df["steps"], df["gaussianity_loss"], label="Gaussianity Loss", alpha=0.8)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Loss Components Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Learning Rate Schedule
        plt.subplot(2, 2, 2)
        plt.plot(df["steps"], df["learning_rate"])
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        
        # 3. Gradient Norms
        plt.subplot(2, 2, 3)
        if any(df["gradient_norm"]):
            plt.plot(df["steps"], df["gradient_norm"])
            plt.xlabel("Training Steps")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norms")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Gradient norms not collected", 
                    ha="center", va="center", transform=plt.gca().transAxes)
        
        # 4. Loss Per Epoch (smoothed)
        plt.subplot(2, 2, 4)
        epoch_losses = df.groupby("epochs")["total_loss"].mean()
        plt.plot(epoch_losses.index, epoch_losses.values, 'o-')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Loss Per Epoch")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "training_overview.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Detailed Loss Components (separate plot)
        plt.figure(figsize=(10, 6))
        plt.plot(df["steps"], df["segmentation_loss"], label="Segmentation Loss", linewidth=2)
        plt.plot(df["steps"], df["iou_loss"] * 20, label="IoU Loss (×20)", linewidth=2)  # Scale up for visibility
        plt.plot(df["steps"], df["gaussianity_loss"] * 100, label="Gaussianity Loss (×100)", linewidth=2)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Individual Loss Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "loss_components.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to: {plots_dir}")
        
    def finalize(self):
        """Called on training completion - save data and generate plots"""
        if not self.collect_metrics:
            return
            
        self.save_history()
        if self.export_on_completion:
            self.generate_plots()
        
        # Summary statistics
        if self.history["total_loss"]:
            final_loss = self.history["total_loss"][-1]
            best_loss = min(self.history["total_loss"])
            improvement = ((self.history["total_loss"][0] - final_loss) / 
                          self.history["total_loss"][0] * 100)
            
            logger.info(f"Training Summary:")
            logger.info(f"  Final Loss: {final_loss:.6f}")
            logger.info(f"  Best Loss: {best_loss:.6f}")
            logger.info(f"  Improvement: {improvement:.2f}%")
            logger.info(f"  Total Steps: {len(self.history['steps'])}")
            logger.info(f"  Duration: {(time.time() - self.start_time)/3600:.2f} hours")


class GPUOptimizedTrainer:
    """Training pipeline optimized for V100/H200 constraints"""

    def __init__(self, config: Dict[str, Any], output_dir: Path = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for training")

        self.config = config
        self.gpu_type = config["hardware"]["target_gpu"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Profiling configuration
        self.enable_profiling = config.get("logging", {}).get("enable_profiling", False)
        self.profiling_frequency = config.get("logging", {}).get("profiling_frequency", 25)

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize loss tracker
        if output_dir is None:
            output_dir = Path("training_output")
        self.loss_tracker = LossTracker(config, output_dir)

        # Setup optimizations based on GPU type
        self.setup_memory_optimization()

        logger.info(f"Initialized trainer for {self.gpu_type}")

    def setup_memory_optimization(self):
        """Configure memory-efficient training based on GPU type"""
        if self.gpu_type == "V100":
            # V100: Memory-constrained but time-flexible
            self.enable_gradient_checkpointing = True
            self.use_mixed_precision = self.config.get("training", {}).get(
                "mixed_precision", False  # Default to False for safety
            )
            self.compile_model = False  # May use extra memory
            logger.info(f"V100 optimizations: gradient checkpointing, mixed_precision={self.use_mixed_precision}")

        elif self.gpu_type == "H200":
            # H200: Time-constrained but memory-rich
            self.enable_gradient_checkpointing = False
            self.use_mixed_precision = self.config.get("training", {}).get(
                "mixed_precision", True
            )
            self.compile_model = self.config.get("training", {}).get(
                "compile_model", True
            )
            logger.info("H200 optimizations: no checkpointing, model compilation")
            
        elif self.gpu_type == "GTX1080Ti":
            # GTX1080Ti: Memory-constrained consumer GPU
            self.enable_gradient_checkpointing = self.config.get("training", {}).get(
                "gradient_checkpointing", True
            )
            self.use_mixed_precision = self.config.get("training", {}).get(
                "mixed_precision", True
            )
            self.compile_model = self.config.get("training", {}).get(
                "compile_model", False
            )
            logger.info("GTX1080Ti optimizations: gradient checkpointing, mixed precision")
        
        else:
            # Default settings for unknown GPUs
            self.enable_gradient_checkpointing = True
            self.use_mixed_precision = self.config.get("training", {}).get(
                "mixed_precision", True
            )
            self.compile_model = False
            logger.info(f"Default optimizations for {self.gpu_type}: gradient checkpointing, mixed precision")

        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()

    def setup_model(self, sam_adapter, dataset_size: int):
        """Setup model, optimizer, and scheduler"""
        self.model = sam_adapter.model
        self.processor = getattr(sam_adapter, 'processor', None)

        # Enable gradient checkpointing if needed (skip for SAM2)
        if (self.enable_gradient_checkpointing and 
            hasattr(self.model, "gradient_checkpointing_enable") and 
            "Sam2Model" not in str(type(self.model))):
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
        elif "Sam2Model" in str(type(self.model)):
            logger.info("Skipping gradient checkpointing (not supported by SAM2)")

        # Compile model for H200 speed optimization
        if self.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        # Setup optimizer
        optimizer_config = self.config.get("optimizer", {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            eps=float(optimizer_config.get("eps", 1e-8)),
        )

        # Setup scheduler
        scheduler_config = self.config.get("scheduler", {})
        if scheduler_config.get("name") == "cosine":
            total_steps = self.estimate_total_steps(dataset_size)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        logger.info("Model, optimizer, and scheduler initialized")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.current_epoch = epoch  # Track current epoch for logging

        total_loss = 0.0
        num_batches = len(dataloader)
        batch_size = self.config["training"]["batch_size"]
        gradient_accumulation = self.config["training"]["gradient_accumulation"]

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # PROFILING: Data Loading Time (implicit in DataLoader iterator)
            data_start = time.time()
            
            # Move data to device
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            data_time = time.time() - data_start

            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast(device_type='cuda'):
                    loss = self.compute_loss(images, masks)
                    loss = loss / gradient_accumulation  # Scale loss for accumulation

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                loss = self.compute_loss(images, masks)
                loss = loss / gradient_accumulation
                loss.backward()
            
            # Log data loading time (epoch-aware)
            if self.enable_profiling:
                data_log_freq = self.profiling_frequency
                if epoch > 0:
                    data_log_freq = data_log_freq * 20
                if batch_idx % data_log_freq == 0:
                    logger.info(f"Data Loading: {data_time*1000:.1f}ms")

            # Update weights after accumulation
            if (batch_idx + 1) % gradient_accumulation == 0:
                if self.use_mixed_precision:
                    # Unscale gradients before clipping (if needed)
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * gradient_accumulation

            # Record loss in tracker (silently) and reduced console logging
            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
            
            # Extract loss components for tracking
            if hasattr(self, '_last_loss_components'):
                loss_dict = self._last_loss_components
            else:
                loss_dict = {"total": loss.item() * gradient_accumulation}
            
            # Record in loss tracker
            self.loss_tracker.record(
                step=self.global_step,
                epoch=epoch,
                losses=loss_dict,
                learning_rate=current_lr
            )

            # Clean console logging (much less frequent)
            log_frequency = self.config["logging"]["log_every_n_steps"]
            if epoch > 0:
                log_frequency = log_frequency * 10  # 10x less frequent after epoch 0
            
            if batch_idx % log_frequency == 0:
                elapsed = time.time() - start_time
                self.log_training_progress(
                    epoch,
                    batch_idx,
                    num_batches,
                    loss.item() * gradient_accumulation,
                    elapsed,
                )

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time

        return {
            "loss": avg_loss,
            "epoch_time": epoch_time,
            "samples_per_second": len(dataloader.dataset) / epoch_time,
        }

    def compute_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute SAM2 training loss (segmentation + score loss)
        Based on working implementation from samrfi/rfitraining.py
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        # Use SAM2 transformers approach (primary) or official SAM2 (fallback)
        if hasattr(self, 'processor') and self.processor is not None:
            # Use transformers SAM2 approach
            return self._compute_sam2_loss(images, masks)
        
        # Fallback to official SAM2 if available
        try:
            from sam2 import SAM2ImagePredictor
            import numpy as np
        except ImportError:
            # Final fallback - this should not happen in normal operation
            raise RuntimeError("Neither transformers processor nor official SAM2 available")
        
        total_loss = 0.0
        batch_size = images.shape[0]
        eps = 1e-6
        
        # Process each sample in the batch (SAM2 typically processes images individually)
        for i in range(batch_size):
            try:
                # Get single image and mask
                image = images[i]  # [3, H, W]
                gt_mask = masks[i]  # [H, W]
                
                # Convert to numpy for SAM2 predictor (needs [H, W, 3] format)
                np_image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                gt_mask_np = gt_mask.cpu().numpy()
                
                # Generate prompts from ground truth mask
                input_points, input_labels, bounding_box = self._generate_prompts_from_mask(gt_mask_np)
                
                if input_points is None or len(input_points) == 0:
                    # Skip if no valid points found
                    continue
                
                # Create temporary predictor for this sample
                # Note: This is not optimal for training - should be refactored for batch processing
                predictor = SAM2ImagePredictor(self.model)
                predictor.set_image(np_image)
                
                # Generate prompts and get predictions
                bounding_box_tensor = torch.tensor([bounding_box], device=self.device).float().unsqueeze(0)
                
                # Prepare prompts (adapted from working implementation)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points, input_labels, box=bounding_box_tensor, 
                    mask_logits=None, normalize_coords=False
                )
                
                if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0:
                    continue
                
                # SAM2 forward pass
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels),
                    boxes=unnorm_box,
                    masks=None
                )
                
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                
                # Post-process predictions
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                prd_mask = torch.sigmoid(prd_masks[:, 0]).squeeze(0)
                prd_mask = 1 - prd_mask  # Invert as in working implementation
                
                # Convert ground truth to tensor
                gt_mask_tensor = torch.tensor(gt_mask_np, device=self.device, dtype=torch.float32)
                
                # Segmentation Loss: Binary Cross-Entropy
                seg_loss = (-gt_mask_tensor * torch.log(prd_mask + eps) - 
                           (1 - gt_mask_tensor) * torch.log(1 - prd_mask + eps)).mean()
                
                # Score Loss: IoU prediction loss
                intersection = (gt_mask_tensor * (prd_mask > 0.5)).sum()
                union = gt_mask_tensor.sum() + (prd_mask > 0.5).sum() - intersection
                iou = intersection / (union + eps)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                
                # Combined loss for this sample
                sample_loss = seg_loss + 0.05 * score_loss
                total_loss += sample_loss
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Average loss across batch
        if batch_size > 0:
            total_loss = total_loss / batch_size
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss
    
    def _generate_prompts_from_mask(self, mask_np):
        """Generate SAM2 prompts from ground truth mask"""
        import numpy as np
        
        # Find RFI regions
        rows, cols = np.where(mask_np > 0)
        
        if len(rows) == 0:
            return None, None, None
            
        # Sample random points from RFI regions (limit to avoid too many points)
        num_points = min(128, len(rows))  # From working implementation
        indices = np.random.choice(len(rows), size=num_points, replace=False)
        
        input_points = np.column_stack((rows[indices], cols[indices]))
        input_labels = np.ones(num_points, dtype=int)
        
        # Generate bounding box
        bounding_box = [cols.min(), rows.min(), cols.max(), rows.max()]
        bounding_box = [float(coord) for coord in bounding_box]
        
        return input_points, input_labels, bounding_box
    
    def _compute_sam2_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute loss using optimized batch SAM2 processing with profiling"""
        batch_size, channels, height, width = images.shape
        eps = 1e-6
        
        # Access model and processor from training setup
        sam2_model = self.model
        sam2_processor = getattr(self, 'processor', None)
        
        if sam2_processor is None:
            raise RuntimeError("SAM2 processor not available. Check training setup.")
        
        # Prepare batch data structures
        if self.enable_profiling:
            prompt_start = time.time()
            
        batch_points = []
        batch_labels = []
        batch_boxes = []
        valid_indices = []
        
        # Generate prompts for all samples
        for i in range(batch_size):
            mask_np = masks[i].cpu().numpy()
            input_points, input_labels, input_boxes = self._generate_prompts_from_mask(mask_np)
            
            if input_points is not None:
                batch_points.append([input_points.tolist()])
                batch_labels.append([input_labels.tolist()])
                batch_boxes.append([input_boxes] if input_boxes else None)
                valid_indices.append(i)
        
        if self.enable_profiling:
            prompt_time = time.time() - prompt_start
        
        if not valid_indices:
            # No valid RFI samples in batch
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get valid images and scale to [0,255] range for SAM2 processor
        valid_images = images[valid_indices]  # [N, 3, H, W]
        
        # Check if images are in [0,1] range and scale to [0,255]
        if valid_images.max() <= 1.0:
            valid_images = valid_images * 255.0
        
        # SAM2 Processor
        if self.enable_profiling:
            processor_start = time.time()
            
        inputs = sam2_processor(
            images=valid_images,  # Direct tensor input
            input_points=batch_points,
            input_labels=batch_labels,
            input_boxes=batch_boxes,
            return_tensors="pt"
        )
        
        if self.enable_profiling:
            processor_time = time.time() - processor_start
        
        # Move inputs to device
        if self.enable_profiling:
            device_start = time.time()
            
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(self.device)
                
        if self.enable_profiling:
            device_time = time.time() - device_start
        
        # SAM2 Forward Pass
        if self.enable_profiling:
            forward_start = time.time()
            
        with torch.set_grad_enabled(True):
            outputs = sam2_model(**inputs)
            
        if self.enable_profiling:
            forward_time = time.time() - forward_start
        
        # 4x256 Tiling: Process as 256x256 tiles for true high-resolution
        if self.enable_profiling:
            tiling_start = time.time()
            
        # Tile the images and process each tile through SAM2
        tiled_outputs = self._process_with_tiling(
            valid_images, batch_points, batch_labels, batch_boxes, 
            sam2_processor, sam2_model
        )
        
        # Replace outputs with tiled results
        outputs.pred_masks = tiled_outputs.pred_masks
        outputs.iou_scores = tiled_outputs.iou_scores
        
        if self.enable_profiling:
            tiling_time = time.time() - tiling_start
        
        # Loss Computation
        if self.enable_profiling:
            loss_start = time.time()
            
        loss = self._compute_batch_loss(outputs, masks[valid_indices], valid_indices, images[valid_indices], eps)
        
        if self.enable_profiling:
            loss_time = time.time() - loss_start
        
        # Log timing breakdown (epoch-aware frequency)
        if (self.enable_profiling and hasattr(self, 'global_step')):
            # Determine current epoch (approximate)
            current_epoch = getattr(self, 'current_epoch', 0)
            profiling_freq = self.profiling_frequency
            
            # Reduce profiling frequency after epoch 0
            if current_epoch > 0:
                profiling_freq = profiling_freq * 20  # Much less frequent profiling
            
            if self.global_step % profiling_freq == 0:
                total_time = prompt_time + processor_time + device_time + forward_time + tiling_time + loss_time
                logger.info(f"PROFILING - Step {self.global_step}:")
                logger.info(f"  Prompt Gen:     {prompt_time*1000:.1f}ms ({prompt_time/total_time*100:.1f}%)")
                logger.info(f"  SAM2 Processor: {processor_time*1000:.1f}ms ({processor_time/total_time*100:.1f}%)")
                logger.info(f"  Device Move:    {device_time*1000:.1f}ms ({device_time/total_time*100:.1f}%)")
                logger.info(f"  Forward Pass:   {forward_time*1000:.1f}ms ({forward_time/total_time*100:.1f}%)")
                logger.info(f"  Tiling:         {tiling_time*1000:.1f}ms ({tiling_time/total_time*100:.1f}%)")
                logger.info(f"  Loss Compute:   {loss_time*1000:.1f}ms ({loss_time/total_time*100:.1f}%)")
                logger.info(f"  TOTAL:          {total_time*1000:.1f}ms")
        
        return loss
    
    def _compute_batch_loss(self, outputs, gt_masks: torch.Tensor, valid_indices: list, images: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Union-based training approach: compute union of all masks for single loss computation"""
        import torch.nn.functional as F
        
        pred_masks = outputs.pred_masks  # SAM2: [batch, 1, num_masks, H, W]
        iou_scores = outputs.iou_scores  # SAM2: [batch, num_masks]
        
        batch_size = pred_masks.shape[0]
        
        # SAM2 outputs have shape:
        # pred_masks: [batch_size, point_batch_size, num_masks, H, W] 
        # iou_scores: [batch_size, point_batch_size, num_masks]
        
        # Ensure gt_masks is float
        gt_masks = gt_masks.float()  # [batch, H, W]
        
        # UNION APPROACH: Compute union of all masks for training and inference
        # This provides a single coherent learning signal
        
        # Compute union of all masks: [batch, point_batch, num_masks, H, W] -> [batch, H, W]
        # Take max across all masks (union for comprehensive RFI detection)
        union_logits = pred_masks[:, 0, :, :, :].max(dim=1)[0]  # [batch, H, W]
        
        # Handle dimension mismatches if needed
        if union_logits.shape[-2:] != gt_masks.shape[-2:]:
            union_logits = F.interpolate(
                union_logits.unsqueeze(1),  # [batch, 1, H, W]
                size=gt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [batch, H, W]
        
        # Compute segmentation loss on union
        segmentation_loss = F.binary_cross_entropy_with_logits(union_logits, gt_masks, reduction='mean')
        
        # Compute IoU score loss on union
        union_probs = torch.sigmoid(union_logits)  # [batch, H, W] 
        union_binary = (union_probs > 0.5).float()  # [batch, H, W]
        
        # Compute intersection and union for each sample
        intersection = (gt_masks * union_binary).sum(dim=(1, 2))  # [batch]
        union_area = gt_masks.sum(dim=(1, 2)) + union_binary.sum(dim=(1, 2)) - intersection  # [batch]
        actual_iou = intersection / (union_area + eps)  # [batch]
        
        # For IoU score prediction, use the max IoU score across all masks
        max_iou_scores = iou_scores[:, 0, :].max(dim=1)[0]  # [batch] - max across masks
        predicted_iou_scores = torch.sigmoid(max_iou_scores)  # [batch]
        
        iou_score_loss = torch.abs(predicted_iou_scores - actual_iou).mean()
        
        # GAUSSIANITY LOSS: Compute on union of all masks (final residual)
        # The combined RFI removal should leave Gaussian residuals
        gaussianity_loss = self._compute_gaussianity_loss(pred_masks, gt_masks, images)
        
        # Combined loss with configurable weights - union approach
        total_loss = segmentation_loss + 0.05 * iou_score_loss + gaussianity_loss
        
        # Store loss components for tracker (replaces verbose logging)
        self._last_loss_components = {
            "total": total_loss.item(),
            "segmentation": segmentation_loss.item(),
            "iou": iou_score_loss.item(),
            "gaussianity": gaussianity_loss.item()
        }
        
        # Optional verbose breakdown (configurable)
        if self.loss_tracker.verbose_breakdown and hasattr(self, 'global_step'):
            current_epoch = getattr(self, 'current_epoch', 0)
            log_freq = 200 if current_epoch == 0 else 1000  # Much less frequent
            
            if self.global_step % log_freq == 0:
                logger.info(f"LOSS BREAKDOWN - Step {self.global_step}:")
                logger.info(f"  Segmentation: {avg_seg_loss:.4f} | IoU: {avg_score_loss:.4f} | Gaussianity: {gaussianity_loss:.4f} | Total: {total_loss:.4f}")
        
        return total_loss
    
    def _process_with_tiling(self, images, batch_points, batch_labels, batch_boxes, processor, model):
        """Process images using tiling strategy for high-resolution processing"""
        batch_size, channels, height, width = images.shape
        tile_size = 256  # SAM2 native resolution - split 1024x1024 into 4x256x256 tiles
        
        # Create tiles: [batch, channels, tile_size, tile_size] for each of 4 tiles
        tiles = []
        tile_coords = [(0, 0), (0, tile_size), (tile_size, 0), (tile_size, tile_size)]
        
        for y, x in tile_coords:
            tile = images[:, :, y:y+tile_size, x:x+tile_size]  # [batch, channels, tile_size, tile_size]
            tiles.append(tile)
        
        # Process tiles sequentially to avoid memory explosion
        tile_results = []
        
        for i, tile in enumerate(tiles):
            # Process one tile at a time: [batch, channels, tile_size, tile_size]
            single_tile_inputs = processor(
                images=tile,
                input_points=batch_points,  # Same prompts per tile
                input_labels=batch_labels,
                input_boxes=batch_boxes,
                return_tensors="pt"
            )
            
            # Move to device
            for key in single_tile_inputs:
                if torch.is_tensor(single_tile_inputs[key]):
                    single_tile_inputs[key] = single_tile_inputs[key].to(self.device)
            
            # Forward pass on single tile
            with torch.set_grad_enabled(True):
                single_tile_output = model(**single_tile_inputs)
            
            tile_results.append(single_tile_output)
        
        # Combine results after processing
        combined_masks = torch.cat([r.pred_masks for r in tile_results], dim=0)
        combined_scores = torch.cat([r.iou_scores for r in tile_results], dim=0)
        
        # Create combined outputs structure
        class CombinedOutputs:
            def __init__(self, pred_masks, iou_scores):
                self.pred_masks = pred_masks
                self.iou_scores = iou_scores
        
        tiled_outputs = CombinedOutputs(combined_masks, combined_scores)
        
        # Reconstruct full-size masks from tiles
        pred_masks = self._reconstruct_from_tiles(tiled_outputs.pred_masks, batch_size, height, width)
        iou_scores = self._reconstruct_scores_from_tiles(tiled_outputs.iou_scores, batch_size)
        
        # Create output structure
        class TiledOutputs:
            def __init__(self, pred_masks, iou_scores):
                self.pred_masks = pred_masks
                self.iou_scores = iou_scores
        
        return TiledOutputs(pred_masks, iou_scores)
    
    def _reconstruct_from_tiles(self, tiled_masks, batch_size, height, width):
        """Reconstruct full-size masks from 4 tiles"""
        tile_size = 256  # SAM2 native output resolution
        num_tiles = 4
        
        # tiled_masks: [batch*4, point_batch, num_masks, tile_size, tile_size]
        _, point_batch, num_masks, _, _ = tiled_masks.shape
        
        # Reshape to separate batch and tiles: [batch, 4, point_batch, num_masks, tile_size, tile_size]  
        reshaped = tiled_masks.view(batch_size, num_tiles, point_batch, num_masks, tile_size, tile_size)
        
        # Initialize full mask
        full_masks = torch.zeros(batch_size, point_batch, num_masks, height, width, 
                                device=tiled_masks.device, dtype=tiled_masks.dtype)
        
        # Place tiles back into full mask
        tile_coords = [(0, 0), (0, tile_size), (tile_size, 0), (tile_size, tile_size)]
        for i, (y, x) in enumerate(tile_coords):
            full_masks[:, :, :, y:y+tile_size, x:x+tile_size] = reshaped[:, i, :, :, :, :]
        
        return full_masks
    
    def _reconstruct_scores_from_tiles(self, tiled_scores, batch_size):
        """Reconstruct IoU scores from tiles by averaging"""
        num_tiles = 4
        
        # tiled_scores: [batch*4, point_batch, num_masks]
        _, point_batch, num_masks = tiled_scores.shape
        
        # Reshape: [batch, 4, point_batch, num_masks]
        reshaped = tiled_scores.view(batch_size, num_tiles, point_batch, num_masks)
        
        # Average scores across tiles
        averaged_scores = reshaped.mean(dim=1)  # [batch, point_batch, num_masks]
        
        return averaged_scores
    
    def _compute_gaussianity_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Compute gaussianity loss on union of masks using real residuals - vectorized for speed"""
        
        # Check if gaussianity loss is enabled in config
        if not hasattr(self, 'config') or not self.config.get('loss', {}).get('gaussianity', {}).get('enabled', False):
            return torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
        
        gaussianity_config = self.config['loss']['gaussianity']
        overall_weight = gaussianity_config.get('weight', 0.1)
        measures_config = gaussianity_config.get('measures', {})
        
        # Compute union of all masks: [batch, point_batch, num_masks, H, W] -> [batch, H, W]
        # Take max across all masks (union for comprehensive RFI detection)
        # First take point_batch_size=0, then max across num_masks dimension
        union_masks = pred_masks[:, 0, :, :, :].max(dim=1)[0]  # [batch, H, W]
        union_probs = torch.sigmoid(union_masks)  # Convert logits to probabilities
        union_binary = (union_probs > 0.5).float()  # Threshold to binary mask
        
        # Use real residuals: corrupted_data with our RFI removal applied
        # The goal is to test if corrupted_data[union_mask == 0] is Gaussian
        clean_regions = 1.0 - union_binary  # [batch, H, W] - regions we predict as clean
        
        # Extract real and imaginary parts from input images
        # Assuming images are complex: [batch, 2, H, W] where dim 1 = [real, imag]
        # OR [batch, 3, H, W] where we treat first 2 channels as real/imag
        if images.shape[1] >= 2:
            real_part = images[:, 0, :, :]  # [batch, H, W]
            imag_part = images[:, 1, :, :]  # [batch, H, W] 
        else:
            # If only 1 channel, treat as real data
            real_part = images[:, 0, :, :]  # [batch, H, W]
            imag_part = torch.zeros_like(real_part)  # Zero imaginary part
        
        # Apply our RFI removal: keep only regions we predict as clean
        masked_real = real_part * clean_regions  # [batch, H, W]
        masked_imag = imag_part * clean_regions  # [batch, H, W]
        
        total_gaussianity_loss = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
        loss_count = 0
        
        # SKEWNESS LOSS - vectorized across batch (BOUNDED)
        skew_loss_real = skew_loss_imag = skew_loss = torch.tensor(0.0, device=pred_masks.device)
        if measures_config.get('skewness', {}).get('enabled', False):
            skew_weight = measures_config['skewness'].get('weight', 1.0)
            skew_target = measures_config['skewness'].get('target', 0.0)
            
            skew_loss_real = self._bounded_skewness_loss(masked_real, skew_target)
            skew_loss_imag = self._bounded_skewness_loss(masked_imag, skew_target)
            skew_loss = (skew_loss_real + skew_loss_imag) / 2
            
            total_gaussianity_loss = total_gaussianity_loss + skew_weight * skew_loss
            loss_count += 1
        
        # KURTOSIS LOSS - vectorized across batch (BOUNDED)
        kurt_loss_real = kurt_loss_imag = kurt_loss = torch.tensor(0.0, device=pred_masks.device)
        if measures_config.get('kurtosis', {}).get('enabled', False):
            kurt_weight = measures_config['kurtosis'].get('weight', 1.0)
            kurt_target = measures_config['kurtosis'].get('target', 3.0)
            
            kurt_loss_real = self._bounded_kurtosis_loss(masked_real, kurt_target)
            kurt_loss_imag = self._bounded_kurtosis_loss(masked_imag, kurt_target)
            kurt_loss = (kurt_loss_real + kurt_loss_imag) / 2
            
            total_gaussianity_loss = total_gaussianity_loss + kurt_weight * kurt_loss
            loss_count += 1
            
        # ANDERSON-DARLING LOSS - vectorized across batch (BOUNDED)
        ad_loss_real = ad_loss_imag = ad_loss = torch.tensor(0.0, device=pred_masks.device)
        if measures_config.get('anderson_darling', {}).get('enabled', False):
            ad_weight = measures_config['anderson_darling'].get('weight', 2.0)
            
            ad_loss_real = self._bounded_anderson_darling_loss(masked_real)
            ad_loss_imag = self._bounded_anderson_darling_loss(masked_imag)
            ad_loss = (ad_loss_real + ad_loss_imag) / 2
            
            total_gaussianity_loss = total_gaussianity_loss + ad_weight * ad_loss
            loss_count += 1
        
        # Apply overall weight and normalize by number of measures
        if loss_count > 0:
            final_loss = overall_weight * (total_gaussianity_loss / loss_count)
        else:
            final_loss = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
        
        # Store individual components for logging (store as attributes)
        self._last_gaussianity_components = {
            'skewness_real': float(skew_loss_real.detach()) if torch.is_tensor(skew_loss_real) else 0.0,
            'skewness_imag': float(skew_loss_imag.detach()) if torch.is_tensor(skew_loss_imag) else 0.0,
            'kurtosis_real': float(kurt_loss_real.detach()) if torch.is_tensor(kurt_loss_real) else 0.0,
            'kurtosis_imag': float(kurt_loss_imag.detach()) if torch.is_tensor(kurt_loss_imag) else 0.0,
            'anderson_darling_real': float(ad_loss_real.detach()) if torch.is_tensor(ad_loss_real) else 0.0,
            'anderson_darling_imag': float(ad_loss_imag.detach()) if torch.is_tensor(ad_loss_imag) else 0.0,
            'final_gaussianity': float(final_loss.detach())
        }
            
        return final_loss
    
    def _vectorized_skewness_loss(self, data: torch.Tensor, target: float = 0.0) -> torch.Tensor:
        """Vectorized skewness computation across batch dimension"""
        # data: [batch, H, W]
        batch_size = data.shape[0]
        
        # Flatten spatial dimensions for each batch sample
        flat_data = data.view(batch_size, -1)  # [batch, H*W]
        
        # Compute mean, std, and skewness for each batch sample
        mean = flat_data.mean(dim=1, keepdim=True)  # [batch, 1]
        centered = flat_data - mean  # [batch, H*W]
        
        # Compute moments
        moment2 = (centered ** 2).mean(dim=1)  # [batch] - variance
        moment3 = (centered ** 3).mean(dim=1)  # [batch] - third moment
        
        # Skewness = E[(X-μ)³] / σ³
        std = torch.sqrt(moment2 + 1e-8)  # Add epsilon for numerical stability
        skewness = moment3 / (std ** 3 + 1e-8)  # [batch]
        
        # L2 loss against target
        skew_loss = ((skewness - target) ** 2).mean()  # Scalar
        return skew_loss
    
    def _vectorized_kurtosis_loss(self, data: torch.Tensor, target: float = 3.0) -> torch.Tensor:
        """Vectorized kurtosis computation across batch dimension"""
        # data: [batch, H, W]
        batch_size = data.shape[0]
        
        # Flatten spatial dimensions for each batch sample
        flat_data = data.view(batch_size, -1)  # [batch, H*W]
        
        # Compute mean and moments
        mean = flat_data.mean(dim=1, keepdim=True)  # [batch, 1]
        centered = flat_data - mean  # [batch, H*W]
        
        moment2 = (centered ** 2).mean(dim=1)  # [batch] - variance
        moment4 = (centered ** 4).mean(dim=1)  # [batch] - fourth moment
        
        # Kurtosis = E[(X-μ)⁴] / σ⁴
        var = moment2 + 1e-8  # Add epsilon for numerical stability
        kurtosis = moment4 / (var ** 2 + 1e-8)  # [batch]
        
        # L2 loss against target (3.0 for normal distribution)
        kurt_loss = ((kurtosis - target) ** 2).mean()  # Scalar
        return kurt_loss
    
    def _vectorized_anderson_darling_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Vectorized Anderson-Darling test approximation"""
        # data: [batch, H, W]
        batch_size = data.shape[0]
        
        # Flatten and sort each batch sample
        flat_data = data.view(batch_size, -1)  # [batch, H*W]
        n = flat_data.shape[1]
        
        # Standardize each sample (subtract mean, divide by std)
        mean = flat_data.mean(dim=1, keepdim=True)  # [batch, 1]
        std = flat_data.std(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        standardized = (flat_data - mean) / std  # [batch, H*W]
        
        # Sort each batch sample
        sorted_data, _ = torch.sort(standardized, dim=1)  # [batch, H*W]
        
        # Compute standard normal CDF using torch.special.erf
        # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=data.device))
        cdf_values = 0.5 * (1 + torch.erf(sorted_data / sqrt_2))  # [batch, H*W]
        
        # Anderson-Darling statistic computation (vectorized)
        i = torch.arange(1, n + 1, device=data.device, dtype=torch.float32)  # [H*W]
        i = i.unsqueeze(0).expand(batch_size, -1)  # [batch, H*W]
        
        # A² = -n - (1/n) * Σ[(2i-1) * (ln(F(X_i)) + ln(1-F(X_{n+1-i})))]
        log_cdf = torch.log(cdf_values + 1e-8)  # [batch, H*W]
        log_1_minus_cdf = torch.log(1 - cdf_values + 1e-8)  # [batch, H*W]
        
        # Flip for the second term
        log_1_minus_cdf_flipped = torch.flip(log_1_minus_cdf, dims=[1])  # [batch, H*W]
        
        # Compute the sum term
        sum_term = ((2 * i - 1) * (log_cdf + log_1_minus_cdf_flipped)).sum(dim=1)  # [batch]
        
        # Anderson-Darling statistic
        A_squared = -n - (1.0 / n) * sum_term  # [batch]
        
        # Convert to loss (higher A² means less Gaussian, so we want to minimize A²)
        ad_loss = A_squared.mean()  # Scalar - average across batch
        return ad_loss
    
    def _bounded_skewness_loss(self, data: torch.Tensor, target: float = 0.0) -> torch.Tensor:
        """Bounded skewness computation with tanh normalization to [-1,1]"""
        raw_skew_loss = self._vectorized_skewness_loss(data, target)
        # Use tanh to bound extreme skewness values to [-1,1] range
        # Scale factor 2.0 gives good sensitivity around normal range
        return torch.tanh(raw_skew_loss / 2.0)
    
    def _bounded_kurtosis_loss(self, data: torch.Tensor, target: float = 3.0) -> torch.Tensor:
        """Bounded kurtosis computation with tanh normalization to [-1,1]"""
        raw_kurt_loss = self._vectorized_kurtosis_loss(data, target)
        # Use tanh to bound extreme kurtosis values to [-1,1] range
        # Scale factor 5.0 allows for wider range before saturation
        return torch.tanh(raw_kurt_loss / 5.0)
    
    def _bounded_anderson_darling_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Bounded Anderson-Darling with sigmoid normalization to [0,1]"""
        raw_ad_loss = self._vectorized_anderson_darling_loss(data)
        # Sigmoid normalization: maps typical AD values (0-10) to [0,1]
        # Scale factor 5.0: maps 0→0.5, 5→0.88, 10→0.99
        normalized_ad = torch.sigmoid(raw_ad_loss / 5.0)
        # Center around 0 by subtracting 0.5, giving range [-0.5, 0.5]
        return normalized_ad - 0.5
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor [3, H, W] to PIL Image (deprecated - use direct tensor processing)"""
        # Convert from [3, H, W] to [H, W, 3] and scale to 0-255
        image_np = tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype('uint8')
        
        from PIL import Image
        return Image.fromarray(image_np)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()

        total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    with autocast(device_type='cuda'):
                        loss = self.compute_loss(images, masks)
                else:
                    loss = self.compute_loss(images, masks)

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def estimate_training_time(self, dataset_size: int) -> Dict[str, float]:
        """Estimate training time for V100/H200"""
        batch_size = self.config["training"]["batch_size"]
        gradient_accumulation = self.config["training"]["gradient_accumulation"]
        max_epochs = self.config["training"]["max_epochs"]

        effective_batch_size = batch_size * gradient_accumulation
        steps_per_epoch = dataset_size // effective_batch_size
        total_steps = steps_per_epoch * max_epochs

        if self.gpu_type == "V100":
            # Conservative estimates - can run for days
            seconds_per_step = 0.5  # Slower due to memory constraints
            estimated_hours = (total_steps * seconds_per_step) / 3600

        elif self.gpu_type == "H200":
            # Must complete within 2 hours
            seconds_per_step = 0.2  # Faster due to better hardware
            estimated_hours = min((total_steps * seconds_per_step) / 3600, 2.0)
            
        elif self.gpu_type == "GTX1080Ti":
            # Consumer GPU - moderate speed
            seconds_per_step = 0.8  # Slower than H200, faster than V100
            estimated_hours = (total_steps * seconds_per_step) / 3600
            
        else:
            # Default estimate for unknown GPUs
            seconds_per_step = 1.0
            estimated_hours = (total_steps * seconds_per_step) / 3600

        return {
            "estimated_hours": estimated_hours,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "seconds_per_step": seconds_per_step,
        }

    def estimate_total_steps(self, dataset_size: int) -> int:
        """Estimate total training steps"""
        batch_size = self.config["training"]["batch_size"]
        gradient_accumulation = self.config["training"]["gradient_accumulation"]
        max_epochs = self.config["training"]["max_epochs"]

        effective_batch_size = batch_size * gradient_accumulation
        steps_per_epoch = dataset_size // effective_batch_size
        return steps_per_epoch * max_epochs

    def log_training_progress(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        loss: float,
        elapsed_time: float,
    ):
        """Log training progress"""
        progress = batch_idx / num_batches * 100
        samples_per_sec = (
            batch_idx * self.config["training"]["batch_size"] / elapsed_time
            if elapsed_time > 0
            else 0
        )

        logger.info(
            f"Epoch {epoch} [{batch_idx}/{num_batches} ({progress:.1f}%)] "
            f"Loss: {loss:.6f} | {samples_per_sec:.1f} samples/sec"
        )

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory_used_gb": 0.0, "memory_total_gb": 0.0}

        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3

        return {
            "device": self.device.type,
            "memory_used_gb": memory_used,
            "memory_total_gb": memory_total,
            "memory_cached_gb": memory_cached,
            "memory_utilization": (
                memory_used / memory_total if memory_total > 0 else 0.0
            ),
        }

    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config,
            "metrics": metrics,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def finalize_training(self):
        """Finalize training - generate plots and save data"""
        logger.info("Finalizing training...")
        self.loss_tracker.finalize()
        logger.info("Training finalization complete")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded training config: {config_path}")
    return config


def get_recommended_config(gpu_type: str, memory_gb: int = None) -> str:
    """Get recommended config file path based on hardware"""
    if gpu_type.upper() == "V100" or (memory_gb and memory_gb <= 16):
        return "configs/training/v100_config.yaml"
    elif gpu_type.upper() == "H200" or (memory_gb and memory_gb > 100):
        return "configs/training/h200_config.yaml"
    else:
        # Default to V100 config for unknown hardware
        return "configs/training/v100_config.yaml"
