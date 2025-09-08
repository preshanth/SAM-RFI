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
from datetime import datetime, timedelta

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.cuda.amp import GradScaler, autocast

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    logging.warning(f"PyTorch not available: {e}")

logger = logging.getLogger(__name__)


class GPUOptimizedTrainer:
    """Training pipeline optimized for V100/H200 constraints"""

    def __init__(self, config: Dict[str, Any]):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for training")

        self.config = config
        self.gpu_type = config["hardware"]["target_gpu"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.current_epoch = 0
        self.global_step = 0

        # Setup optimizations based on GPU type
        self.setup_memory_optimization()

        logger.info(f"Initialized trainer for {self.gpu_type}")

    def setup_memory_optimization(self):
        """Configure memory-efficient training based on GPU type"""
        if self.gpu_type == "V100":
            # V100: Memory-constrained but time-flexible
            self.enable_gradient_checkpointing = True
            self.use_mixed_precision = self.config.get("training", {}).get(
                "mixed_precision", True
            )
            self.compile_model = False  # May use extra memory
            logger.info("V100 optimizations: gradient checkpointing, mixed precision")

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

        total_loss = 0.0
        num_batches = len(dataloader)
        batch_size = self.config["training"]["batch_size"]
        gradient_accumulation = self.config["training"]["gradient_accumulation"]

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    loss = self.compute_loss(images, masks)
                    loss = loss / gradient_accumulation  # Scale loss for accumulation

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                loss = self.compute_loss(images, masks)
                loss = loss / gradient_accumulation
                loss.backward()

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

            # Logging
            if batch_idx % self.config["logging"]["log_every_n_steps"] == 0:
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
        """Compute loss using actual SAM2 model forward pass"""
        batch_size, channels, height, width = images.shape
        eps = 1e-6
        
        # Get SAM2 adapter for model and processor access
        from samrfi.adapters.sam2_adapter import SAM2Adapter
        
        # Access model and processor from training setup
        sam2_model = self.model  # This should be the loaded SAM2 model
        sam2_processor = getattr(self, 'processor', None)
        
        if sam2_processor is None:
            # Fallback: try to get processor from adapter
            raise RuntimeError("SAM2 processor not available. Check training setup.")
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i in range(batch_size):
            # Get single image and mask
            single_image = images[i]  # [3, H, W]
            single_mask = masks[i]    # [H, W]
            
            # Generate prompts from ground truth mask
            mask_np = single_mask.cpu().numpy()
            input_points, input_labels, input_boxes = self._generate_prompts_from_mask(mask_np)
            
            if input_points is None:
                # No RFI in this sample, skip
                continue
                
            # Prepare inputs for SAM2
            # Convert image from [3, H, W] to PIL format for processor
            image_pil = self._tensor_to_pil(single_image)
            
            # Process inputs through SAM2 processor
            inputs = sam2_processor(
                images=image_pil,
                input_points=[[input_points.tolist()]],  # Nested list format
                input_labels=[[input_labels.tolist()]],
                input_boxes=[[input_boxes]] if input_boxes else None,
                return_tensors="pt"
            )
            
            # Move inputs to device
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
            
            # Forward pass through SAM2
            with torch.set_grad_enabled(True):
                outputs = sam2_model(**inputs)
            
            # Extract predictions
            pred_masks = outputs.pred_masks  # [1, num_masks, H, W]
            iou_scores = outputs.iou_scores  # [1, num_masks]
            
            # Use best mask (highest IoU score)
            best_mask_idx = torch.argmax(iou_scores[0])
            predicted_mask = pred_masks[0, best_mask_idx]  # [H, W]
            predicted_score = iou_scores[0, best_mask_idx]  # scalar
            
            # Resize predicted mask to match ground truth if needed
            if predicted_mask.shape != single_mask.shape:
                predicted_mask = torch.nn.functional.interpolate(
                    predicted_mask.unsqueeze(0).unsqueeze(0),
                    size=single_mask.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # Apply sigmoid to get probabilities
            predicted_mask = torch.sigmoid(predicted_mask)
            predicted_score = torch.sigmoid(predicted_score)
            
            # Compute losses for this sample
            gt_mask = single_mask.float()
            
            # Binary Cross-Entropy Loss
            seg_loss = (-gt_mask * torch.log(predicted_mask + eps) - 
                       (1 - gt_mask) * torch.log(1 - predicted_mask + eps)).mean()
            
            # Score Loss (compare predicted IoU score with actual IoU)
            intersection = (gt_mask * (predicted_mask > 0.5)).sum()
            union = gt_mask.sum() + (predicted_mask > 0.5).sum() - intersection
            actual_iou = intersection / (union + eps)
            score_loss = torch.abs(predicted_score - actual_iou)
            
            # Combine losses
            sample_loss = seg_loss + 0.05 * score_loss
            total_loss = total_loss + sample_loss
        
        # Average loss across batch
        if batch_size > 0:
            total_loss = total_loss / batch_size
        
        return total_loss
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor [3, H, W] to PIL Image"""
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
                    with autocast():
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
