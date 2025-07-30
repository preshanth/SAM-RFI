"""
GTX 1080 Ti Training Test

Lightweight training test designed specifically for GTX 1080 Ti (11GB VRAM).
Tests the complete training pipeline with minimal resources.
"""

import sys
import numpy as np
from pathlib import Path
import tempfile
import time

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available: {e}")

try:
    from samrfi.adapters import get_sam_adapter, list_sam_versions
    from samrfi.datasets import RFIDatasetCreator
    from samrfi.models import load_training_config, GPUOptimizedTrainer
    SAMRFI_AVAILABLE = True
except ImportError as e:
    SAMRFI_AVAILABLE = False
    print(f"SAM-RFI modules not available: {e}")


class MockSAMDataset(Dataset):
    """Mock dataset that doesn't require actual SAM2 models"""
    
    def __init__(self, size=10, image_size=512):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create mock data
        image = torch.randn(3, self.image_size, self.image_size)
        mask = torch.randint(0, 2, (self.image_size, self.image_size), dtype=torch.float32)
        
        return {
            'image': image,
            'mask': mask,
            'metadata': {
                'sample_id': idx,
                'rfi_type': 'synthetic'
            }
        }


class MockSAMModel(nn.Module):
    """Mock SAM model for testing without real SAM2"""
    
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size
        
        # Simple CNN to simulate SAM processing
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.features(x)


class MockTrainer(GPUOptimizedTrainer):
    """Modified trainer that works without real SAM2"""
    
    def setup_model(self, mock_model, dataset_size: int):
        """Setup with mock model instead of SAM adapter"""
        self.model = mock_model.to(self.device)
        
        # Enable gradient checkpointing if possible
        if self.enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Setup optimizer  
        optimizer_config = self.config.get('optimizer', {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8)
        )
        
        # Setup scheduler
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('name') == 'cosine':
            total_steps = self.estimate_total_steps(dataset_size)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
    
    def compute_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute actual training loss with mock model"""
        predictions = self.model(images)
        
        # Resize predictions to match mask size
        if predictions.shape[-2:] != masks.shape[-2:]:
            predictions = torch.nn.functional.interpolate(
                predictions, size=masks.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Binary cross entropy loss
        predictions = predictions.squeeze(1)  # Remove channel dimension
        loss = torch.nn.functional.binary_cross_entropy(predictions, masks)
        
        return loss


def test_gpu_detection():
    """Test GPU detection and memory info"""
    print("🔍 Testing GPU detection...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    device_count = torch.cuda.device_count()
    print(f"✅ Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   Device {i}: {props.name} ({memory_gb:.1f}GB)")
        
        if "1080" in props.name.upper():
            print(f"   🎯 GTX 1080 Ti detected! Perfect for testing.")
    
    return True


def test_memory_usage():
    """Test memory allocation and cleanup"""
    print("\n💾 Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("❌ Skipping memory test - CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    # Clear cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    
    # Allocate some memory
    test_tensor = torch.randn(1000, 1000, device=device)
    allocated_memory = torch.cuda.memory_allocated() / 1024**3
    
    print(f"   Initial memory: {initial_memory:.2f}GB")
    print(f"   After allocation: {allocated_memory:.2f}GB")
    print(f"   Used: {allocated_memory - initial_memory:.2f}GB")
    
    # Clean up
    del test_tensor
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"   After cleanup: {final_memory:.2f}GB")
    
    return True


def test_mock_model():
    """Test mock SAM model creation and forward pass"""
    print("\n🤖 Testing mock SAM model...")
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("   Using CPU for model test")
    else:
        device = torch.device('cuda')
        print("   Using GPU for model test")
    
    # Create mock model
    model = MockSAMModel(image_size=512)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512, device=device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(test_input)
    forward_time = time.time() - start_time
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Forward pass time: {forward_time:.3f}s")
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory used: {memory_used:.2f}GB")
    
    return True


def test_training_loop():
    """Test actual training loop with mock data"""
    print("\n🏋️ Testing training loop...")
    
    if not SAMRFI_AVAILABLE or not TORCH_AVAILABLE:
        print("❌ Skipping training test - dependencies not available")
        return False
    
    # Load GTX 1080 Ti config
    config_path = Path(__file__).parent.parent / "configs/training/gtx1080ti_config.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    config = load_training_config(str(config_path))
    print(f"   Loaded config for {config['hardware']['target_gpu']}")
    
    # Create mock dataset
    train_dataset = MockSAMDataset(size=20, image_size=config['model']['image_size'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_dataset = MockSAMDataset(size=5, image_size=config['model']['image_size'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create trainer
    trainer = MockTrainer(config)
    
    # Create and setup mock model
    mock_model = MockSAMModel(image_size=config['model']['image_size'])
    trainer.setup_model(mock_model, len(train_dataset))
    
    print(f"   Training on: {trainer.device}")
    
    # Run a few training steps
    max_epochs = min(2, config['training']['max_epochs'])  # Limit for testing
    
    for epoch in range(max_epochs):
        print(f"\n   Epoch {epoch + 1}/{max_epochs}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"     Train Loss: {train_metrics['loss']:.4f}")
        print(f"     Epoch Time: {train_metrics['epoch_time']:.2f}s")
        print(f"     Samples/sec: {train_metrics['samples_per_second']:.1f}")
        
        # Validation
        val_metrics = trainer.validate(val_loader)
        print(f"     Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Memory info
        memory_info = trainer.get_memory_usage()
        if memory_info['device'] != 'cpu':
            print(f"     Memory: {memory_info['memory_used_gb']:.2f}GB / {memory_info['memory_total_gb']:.2f}GB ({memory_info['memory_utilization']:.1%})")
    
    print("   ✅ Training loop completed successfully!")
    return True


def test_time_estimation():
    """Test training time estimation"""
    print("\n⏱️ Testing time estimation...")
    
    if not SAMRFI_AVAILABLE:
        print("❌ Skipping time estimation - SAM-RFI not available")
        return False
    
    config_path = Path(__file__).parent.parent / "configs/training/gtx1080ti_config.yaml"
    config = load_training_config(str(config_path))
    
    trainer = MockTrainer(config)
    
    # Estimate for different dataset sizes
    dataset_sizes = [100, 500, 1000]
    
    for dataset_size in dataset_sizes:
        estimate = trainer.estimate_training_time(dataset_size)
        
        print(f"   Dataset size {dataset_size}:")
        print(f"     Estimated time: {estimate['estimated_hours']:.2f} hours")
        print(f"     Total steps: {estimate['total_steps']}")
        print(f"     Steps per epoch: {estimate['steps_per_epoch']}")
    
    return True


def run_all_tests():
    """Run all GTX 1080 Ti tests"""
    print("🚀 Running GTX 1080 Ti Training Tests")
    print("=" * 50)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Memory Usage", test_memory_usage), 
        ("Mock Model", test_mock_model),
        ("Training Loop", test_training_loop),
        ("Time Estimation", test_time_estimation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your GTX 1080 Ti is ready for SAM-RFI training.")
        print("\nNext steps:")
        print("- Install SAM2 dependencies for real model testing")
        print("- Try training with actual measurement set data")
        print("- Experiment with different batch sizes and image sizes")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()