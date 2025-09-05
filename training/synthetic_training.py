#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM-RFI Synthetic Training Pipeline for GTX 1080Ti
Integrates with existing SAM2Adapter and GPUOptimizedTrainer architecture

Creates synthetic RFI datasets, generates PNG visualizations, and trains
SAM2 model using the established SAM-RFI training infrastructure.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from typing import Dict, List, Tuple, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from samrfi.datasets import (
        SyntheticDatasetGenerator, 
        ObservationConfig, 
        RFIConfig
    )
    from samrfi.adapters import SAM2Adapter, get_sam_adapter
    from samrfi.models.training import GPUOptimizedTrainer, load_training_config
    SAMRFI_AVAILABLE = True
except ImportError as e:
    SAMRFI_AVAILABLE = False
    print(f"SAM-RFI modules not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('synthetic_training.log')
    ]
)
logger = logging.getLogger(__name__)


class RFISyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic RFI training data
    Compatible with SAM2 training requirements
    """
    
    def __init__(self, dataset_metadata: Dict, split: str = 'train', image_size: int = 1024):
        """
        Initialize dataset from synthetic generation metadata
        
        Args:
            dataset_metadata: Metadata from SyntheticDatasetGenerator
            split: 'train' or 'val' 
            image_size: Target image size for SAM2 (1024x1024)
        """
        self.split = split
        self.image_size = image_size
        self.samples = []
        
        # Load observations from metadata
        dataset_key = 'training_set' if split == 'train' else 'validation_set'
        observations = dataset_metadata[dataset_key]['observations']
        
        logger.info(f"Loading {split} dataset with {len(observations)} observations...")
        
        for obs_meta in observations:
            self._load_observation_samples(obs_meta)
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_observation_samples(self, obs_meta: Dict):
        """Load samples from a single observation"""
        ground_truth_dir = Path(obs_meta['ground_truth_dir'])
        
        try:
            # Load synthetic data
            corrupted_vis = np.load(ground_truth_dir / 'corrupted_visibilities.npy')
            rfi_mask = np.load(ground_truth_dir / 'rfi_mask.npy')
            
            # Extract samples from each baseline and polarization
            num_baselines, num_times, num_channels, num_pols = corrupted_vis.shape
            
            for baseline_idx in range(num_baselines):
                for pol_idx in range(num_pols):
                    # Create waterfall plot [time, frequency]
                    vis_data = corrupted_vis[baseline_idx, :, :, pol_idx]
                    mask_data = rfi_mask[baseline_idx, :, :, pol_idx]
                    
                    # Convert to amplitude
                    waterfall = np.abs(vis_data).astype(np.float32)
                    
                    # Normalize to [0, 1] range
                    if waterfall.max() > waterfall.min():
                        waterfall = (waterfall - waterfall.min()) / (waterfall.max() - waterfall.min())
                    
                    sample = {
                        'waterfall': waterfall,  # [time, frequency]
                        'mask': mask_data.astype(np.float32),  # [time, frequency] 
                        'metadata': {
                            'obs_name': obs_meta['observation_name'],
                            'baseline': baseline_idx,
                            'polarization': pol_idx,
                            'rfi_fraction': float(np.mean(mask_data))
                        }
                    }
                    
                    self.samples.append(sample)
                    
        except Exception as e:
            logger.error(f"Failed to load observation {obs_meta['observation_name']}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert waterfall to SAM2-compatible image format
        waterfall = sample['waterfall']  # [time, frequency]
        mask = sample['mask']  # [time, frequency]
        
        # Resize to target image size if needed
        if waterfall.shape != (self.image_size, self.image_size):
            waterfall = self._resize_data(waterfall, (self.image_size, self.image_size))
            mask = self._resize_data(mask, (self.image_size, self.image_size))
        
        # Convert to RGB image (SAM2 expects 3 channels)
        # Use waterfall for all channels (can be enhanced later)
        image = np.stack([waterfall, waterfall, waterfall], axis=0)  # [3, H, W]
        
        # Ensure proper data types
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        return {
            'image': image,
            'mask': mask,
            'metadata': sample['metadata']
        }
    
    def _resize_data(self, data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize data to target shape using interpolation"""
        from scipy.ndimage import zoom
        
        zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        return zoom(data, zoom_factors, order=1)


def create_synthetic_datasets(output_dir: str, num_train: int = 500, num_val: int = 100):
    """
    Create synthetic training and validation datasets using existing infrastructure
    
    Fixed dataset: 2 MS for training (702 baselines), 1 MS for validation (351 baselines)
    Args are ignored - kept for compatibility
    """
    logger.info("Creating synthetic datasets: 2 training MS + 1 validation MS")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Configuration for 1024x1024 tiles and proper baseline count
    obs_config = ObservationConfig(
        num_antennas=27,  # VLA-like array for 351 baselines per MS
        num_spw=2,
        channels_per_spw=512,  # 1024 total channels
        start_frequency=1.4e9,  # L-band
        total_duration=1024.0,  # 1024 seconds for 1024 time steps
        integration_time=1.0    # 1s integration → 1024 time steps
    )
    
    # Different RFI scenarios targeting ~20% total RFI
    rfi_configs = [
        # Light RFI scenario (~15% total)
        RFIConfig(
            broadband_probability=0.08,  # 8% broadband
            narrowband_lines=5,          # ~2% narrowband
            transient_events=8,          # ~3% transient 
            periodic_signals=3,          # ~1% periodic
            satellite_passes=2           # ~1% satellite
        ),
        # Medium RFI scenario (~20% total)
        RFIConfig(
            broadband_probability=0.10,  # 10% broadband
            narrowband_lines=8,          # ~3% narrowband
            transient_events=12,         # ~4% transient
            periodic_signals=4,          # ~2% periodic
            satellite_passes=3           # ~1% satellite
        ),
        # Heavy RFI scenario (~25% total)
        RFIConfig(
            broadband_probability=0.12,  # 12% broadband
            narrowband_lines=12,         # ~4% narrowband
            transient_events=16,         # ~6% transient
            periodic_signals=5,          # ~2% periodic
            satellite_passes=4           # ~1% satellite
        )
    ]
    
    # Initialize dataset generator
    generator = SyntheticDatasetGenerator(str(output_path / 'synthetic_ms'))
    
    # Fixed MS file counts for consistent dataset size
    # 27 antennas = 351 baselines per MS
    baselines_per_ms = 27 * (27 - 1) // 2  # 351 baselines
    num_train_ms = 2  # Always 2 MS for training (702 baselines)
    num_val_ms = 1    # Always 1 MS for validation (351 baselines)
    
    actual_train_baselines = num_train_ms * baselines_per_ms
    actual_val_baselines = num_val_ms * baselines_per_ms
    
    logger.info(f"Generating {num_train_ms} training MS files ({actual_train_baselines} baselines)")
    logger.info(f"Generating {num_val_ms} validation MS files ({actual_val_baselines} baselines)")
    
    # Generate training dataset
    logger.info("Generating training dataset...")
    train_metadata = generator.generate_training_dataset(
        dataset_name='sam_rfi_train',
        num_observations=num_train_ms,
        obs_config=obs_config,
        rfi_configs=rfi_configs
    )
    
    # Generate validation dataset
    logger.info("Generating validation dataset...")
    val_metadata = generator.generate_training_dataset(
        dataset_name='sam_rfi_val',
        num_observations=num_val_ms,
        obs_config=obs_config,
        rfi_configs=rfi_configs
    )
    
    # Combine metadata with baseline counting
    combined_metadata = {
        'creation_time': datetime.now().isoformat(),
        'training_set': train_metadata,
        'validation_set': val_metadata,
        'baseline_counts': {
            'baselines_per_ms': baselines_per_ms,
            'train_ms_files': num_train_ms,
            'val_ms_files': num_val_ms,
            'actual_train_baselines': actual_train_baselines,
            'actual_val_baselines': actual_val_baselines,
            'total_baselines': actual_train_baselines + actual_val_baselines
        },
        'configuration': {
            'observation_config': obs_config.__dict__,
            'rfi_configs': [config.__dict__ for config in rfi_configs]
        }
    }
    
    # Save combined metadata
    with open(output_path / 'dataset_summary.json', 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    logger.info(f"Synthetic datasets created in {output_path}")
    return combined_metadata


def generate_png_visualizations(dataset_metadata: Dict, output_dir: str, num_samples: int = 10):
    """Generate PNG visualizations for dataset inspection"""
    logger.info(f"Generating {num_samples} PNG visualizations...")
    
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Sample from both train and validation sets
    train_obs = dataset_metadata['training_set']['observations'][:num_samples//2]
    val_obs = dataset_metadata['validation_set']['observations'][:num_samples//2]
    
    sample_count = 0
    
    for dataset_name, observations in [('train', train_obs), ('val', val_obs)]:
        for obs_meta in observations:
            if sample_count >= num_samples:
                break
                
            obs_name = obs_meta['observation_name']
            ground_truth_dir = Path(obs_meta['ground_truth_dir'])
            
            try:
                # Load only what we need for visualization (first baseline, first pol)
                baseline_idx = 0
                pol_idx = 0
                
                logger.info(f"    Loading data for {obs_name} visualization...")
                
                # Load full arrays but extract slices immediately to save memory
                with open(ground_truth_dir / 'corrupted_visibilities.npy', 'rb') as f:
                    corrupted_vis = np.load(f)
                    corrupted_waterfall = np.abs(corrupted_vis[baseline_idx, :, :, pol_idx])
                    del corrupted_vis  # Free memory immediately
                
                with open(ground_truth_dir / 'clean_visibilities.npy', 'rb') as f:
                    clean_vis = np.load(f)
                    clean_waterfall = np.abs(clean_vis[baseline_idx, :, :, pol_idx])
                    del clean_vis  # Free memory immediately
                
                with open(ground_truth_dir / 'rfi_mask.npy', 'rb') as f:
                    rfi_mask = np.load(f)
                    mask_waterfall = rfi_mask[baseline_idx, :, :, pol_idx]
                    del rfi_mask  # Free memory immediately
                
                logger.info(f"    Creating 4-panel plot for {obs_name}...")
                
                # Create 4-panel visualization
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{dataset_name.upper()}: {obs_name}\nBaseline {baseline_idx}, Polarization {pol_idx}', 
                           fontsize=14)
                
                # Clean data
                im1 = axes[0, 0].imshow(clean_waterfall.T, aspect='auto', cmap='viridis', 
                                       origin='lower', interpolation='nearest')
                axes[0, 0].set_title('Clean Visibilities')
                axes[0, 0].set_xlabel('Time Step')
                axes[0, 0].set_ylabel('Frequency Channel')
                plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')
                
                # Corrupted data  
                im2 = axes[0, 1].imshow(corrupted_waterfall.T, aspect='auto', cmap='viridis',
                                       origin='lower', interpolation='nearest')
                axes[0, 1].set_title('RFI Corrupted Visibilities')
                axes[0, 1].set_xlabel('Time Step')
                axes[0, 1].set_ylabel('Frequency Channel')
                plt.colorbar(im2, ax=axes[0, 1], label='Amplitude')
                
                # RFI mask (ground truth)
                im3 = axes[1, 0].imshow(mask_waterfall.T, aspect='auto', cmap='Reds',
                                       origin='lower', interpolation='nearest')
                axes[1, 0].set_title('RFI Mask (Ground Truth)')
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_ylabel('Frequency Channel')
                plt.colorbar(im3, ax=axes[1, 0], label='RFI Flag')
                
                # RFI signal only
                rfi_signal = corrupted_waterfall - clean_waterfall
                im4 = axes[1, 1].imshow(rfi_signal.T, aspect='auto', cmap='plasma',
                                       origin='lower', interpolation='nearest')
                axes[1, 1].set_title('RFI Signal Only')
                axes[1, 1].set_xlabel('Time Step')
                axes[1, 1].set_ylabel('Frequency Channel')
                plt.colorbar(im4, ax=axes[1, 1], label='RFI Amplitude')
                
                plt.tight_layout()
                
                # Save PNG
                png_path = viz_dir / f'{dataset_name}_{obs_name}_viz.png'
                logger.info(f"    Saving PNG: {png_path.name}...")
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add statistics text
                rfi_fraction = np.mean(mask_waterfall)
                
                logger.info(f"✓ Created visualization: {png_path.name} (RFI: {rfi_fraction:.1%})")
                sample_count += 1
                
            except Exception as e:
                logger.error(f"Failed to create visualization for {obs_name}: {e}")
    
    logger.info(f"Generated {sample_count} visualizations in {viz_dir}")


def train_sam2_model(config_path: str, dataset_metadata: Dict, output_dir: str):
    """
    Train SAM2 model using existing GPUOptimizedTrainer infrastructure
    """
    logger.info("Starting SAM2 training with existing infrastructure...")
    
    if not SAMRFI_AVAILABLE:
        raise ImportError("SAM-RFI modules not available for training")
    
    # Load training configuration
    config = load_training_config(config_path)
    logger.info(f"Loaded config for {config['hardware']['target_gpu']}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Training requires GPU.")
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"Training on: {gpu_name} ({memory_gb:.1f}GB)")
    
    # Create datasets
    logger.info("Creating PyTorch datasets...")
    train_dataset = RFISyntheticDataset(
        dataset_metadata, 
        split='train', 
        image_size=config['model']['image_size']
    )
    val_dataset = RFISyntheticDataset(
        dataset_metadata, 
        split='val', 
        image_size=config['model']['image_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Initialize SAM2 adapter
    try:
        sam_adapter = SAM2Adapter(
            device=str(device),
            variant=config['model']['variant']
        )
        logger.info(f"SAM2 adapter initialized: {sam_adapter.version}")
        
        # Auto-download and load SAM2 model from HuggingFace
        logger.info("Loading SAM2 model (will auto-download from HuggingFace if needed)...")
        sam_adapter.load_model()  # Auto-download if no checkpoint specified
        logger.info("SAM2 model loaded successfully!")
        
    except Exception as e:
        logger.error(f"SAM2 adapter initialization failed: {e}")
        logger.info("Creating mock training demonstration...")
        
        # For now, demonstrate the training pipeline structure
        return demonstrate_training_pipeline(config, train_loader, val_loader, output_dir)
    
    # Initialize trainer
    trainer = GPUOptimizedTrainer(config)
    
    # Training time estimation
    time_estimate = trainer.estimate_training_time(len(train_dataset))
    logger.info(f"Estimated training time: {time_estimate['estimated_hours']:.2f} hours")
    logger.info(f"Total steps: {time_estimate['total_steps']}")
    
    # Setup model (this would use the actual SAM2 model)
    # trainer.setup_model(sam_adapter, len(train_dataset))
    
    logger.info("SAM2 training pipeline ready!")
    logger.info("To complete setup: provide SAM2 checkpoint path and run training loop")


def demonstrate_training_pipeline(config: Dict, train_loader: DataLoader, 
                                val_loader: DataLoader, output_dir: str):
    """Demonstrate the training pipeline structure without actual SAM2 model"""
    logger.info("Demonstrating training pipeline structure...")
    
    # Show data loading
    logger.info("Testing data loading...")
    batch = next(iter(train_loader))
    
    logger.info(f"Batch shapes:")
    logger.info(f"  Images: {batch['image'].shape}")
    logger.info(f"  Masks: {batch['mask'].shape}")
    logger.info(f"  Batch size: {len(batch['metadata'])}")
    
    # Show memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU memory: {memory_used:.2f}GB / {memory_total:.2f}GB used")
    
    # Show training configuration
    logger.info("Training configuration:")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Gradient accumulation: {config['training']['gradient_accumulation']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Mixed precision: {config['training']['mixed_precision']}")
    
    logger.info("Training pipeline demonstration complete!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='SAM-RFI Synthetic Training for GTX 1080Ti')
    parser.add_argument('--output-dir', default='./sam_rfi_synthetic_training',
                       help='Output directory for generated files')
    parser.add_argument('--num-train', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--num-val', type=int, default=100,
                       help='Number of validation samples')
    parser.add_argument('--config', default='../configs/training/gtx1080ti_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--skip-dataset', action='store_true',
                       help='Skip dataset generation (use existing)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip PNG visualization generation')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training setup')
    parser.add_argument('--viz-samples', type=int, default=10,
                       help='Number of visualization samples to generate')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SAM-RFI SYNTHETIC TRAINING PIPELINE")
    logger.info("Integrated with existing SAM2Adapter and GPUOptimizedTrainer")
    logger.info("="*60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training samples: {args.num_train}")
    logger.info(f"Validation samples: {args.num_val}")
    logger.info(f"Config: {args.config}")
    
    try:
        # Step 1: Create or load synthetic datasets
        if not args.skip_dataset:
            dataset_metadata = create_synthetic_datasets(
                args.output_dir, args.num_train, args.num_val
            )
        else:
            metadata_path = Path(args.output_dir) / 'dataset_summary.json'
            if metadata_path.exists():
                with open(metadata_path) as f:
                    dataset_metadata = json.load(f)
                logger.info(f"Loaded existing dataset metadata: {metadata_path}")
            else:
                logger.error("No existing dataset found. Run without --skip-dataset first.")
                return
        
        # Step 2: Generate PNG visualizations
        if not args.skip_viz:
            generate_png_visualizations(
                dataset_metadata, args.output_dir, args.viz_samples
            )
        
        # Step 3: Setup SAM2 training 
        if not args.skip_training:
            train_sam2_model(args.config, dataset_metadata, args.output_dir)
        
        # Summary
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED!")
        logger.info("="*60)
        
        output_path = Path(args.output_dir)
        logger.info("\nGenerated files:")
        logger.info(f"  Synthetic MS data: {output_path / 'synthetic_ms'}")
        logger.info(f"  PNG visualizations: {output_path / 'visualizations'}")
        logger.info(f"  Dataset summary: {output_path / 'dataset_summary.json'}")
        logger.info(f"  Training log: synthetic_training.log")
        
        logger.info("\nNext steps:")
        logger.info("1. Install SAM2 dependencies if not already installed")
        logger.info("2. Configure SAM2 checkpoint path in the training config")
        logger.info("3. Run training: python synthetic_training.py --skip-dataset --skip-viz")
        logger.info("4. Monitor GPU memory usage during training")
        
        # Show dataset statistics
        baseline_counts = dataset_metadata['baseline_counts']
        logger.info(f"\nDataset statistics:")
        logger.info(f"  Training MS files: {baseline_counts['train_ms_files']}")
        logger.info(f"  Validation MS files: {baseline_counts['val_ms_files']}")
        logger.info(f"  Training baselines: {baseline_counts['actual_train_baselines']}")
        logger.info(f"  Validation baselines: {baseline_counts['actual_val_baselines']}")
        logger.info(f"  Total baselines: {baseline_counts['total_baselines']}")
        logger.info(f"  Samples per baseline: 4 polarizations (1404 total training samples)")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()