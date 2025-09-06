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
            self.use_mixed_precision = True
            self.compile_model = False  # May use extra memory
            logger.info("V100 optimizations: gradient checkpointing, mixed precision")

        elif self.gpu_type == "H200":
            # H200: Time-constrained but memory-rich
            self.enable_gradient_checkpointing = False
            self.use_mixed_precision = True
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
            self.use_mixed_precision = True
            self.compile_model = False
            logger.info(f"Default optimizations for {self.gpu_type}: gradient checkpointing, mixed precision")

        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()

    def setup_model(self, sam_adapter, dataset_size: int):
        """Setup model, optimizer, and scheduler"""
        self.model = sam_adapter.model

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
        """Compute training loss (placeholder - will be implemented with actual SAM2 training)"""
        # This is a placeholder - actual SAM2 training loss would be more complex
        # For now, return a dummy loss to test the training pipeline
        batch_size = images.shape[0]
        return torch.randn(1, device=self.device, requires_grad=True).mean()

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
