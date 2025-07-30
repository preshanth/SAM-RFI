"""
Test SAM2 Adapter with Small Dataset

Test to verify SAM2 adapter works correctly with synthetic data.
"""

import sys
import numpy as np
from pathlib import Path
import tempfile
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from samrfi.adapters import get_sam_adapter, list_sam_versions
    from samrfi.datasets import RFIDatasetCreator
    SAMRFI_AVAILABLE = True
except ImportError as e:
    SAMRFI_AVAILABLE = False
    import warnings
    warnings.warn(f"SAM-RFI modules not available: {e}")


@pytest.mark.skipif(not SAMRFI_AVAILABLE, reason="SAM-RFI modules not available")
def test_sam_adapter_registry():
    """Test that SAM adapters are registered correctly"""
    versions = list_sam_versions()
    
    # Should have at least some SAM2 variants registered
    assert len(versions) > 0, "No SAM adapters registered"
    
    # Check for expected SAM2 variants
    expected_variants = ['sam2', 'sam2_small', 'sam2_large']
    available_variants = [v for v in expected_variants if v in versions]
    
    assert len(available_variants) > 0, f"No expected SAM2 variants found. Available: {versions}"
    
    print(f"✅ SAM adapter registry test passed. Available versions: {versions}")


@pytest.mark.skipif(not SAMRFI_AVAILABLE, reason="SAM-RFI modules not available")
def test_sam2_adapter_creation():
    """Test SAM2 adapter creation without loading model"""
    try:
        # Try to create a SAM2 adapter (without loading model)
        adapter = get_sam_adapter('sam2', device='cpu')
        
        # Test basic properties
        assert adapter.version.startswith('sam2'), f"Expected sam2 version, got {adapter.version}"
        assert adapter.input_size == 1024, f"Expected input size 1024, got {adapter.input_size}"
        assert not adapter.is_model_loaded(), "Model should not be loaded initially"
        
        # Test model info
        info = adapter.get_model_info()
        assert 'version' in info
        assert 'device' in info
        assert 'input_size' in info
        assert 'is_loaded' in info
        
        print(f"✅ SAM2 adapter creation test passed. Info: {info}")
        
    except ImportError as e:
        pytest.skip(f"SAM2 dependencies not available: {e}")


@pytest.mark.skipif(not SAMRFI_AVAILABLE, reason="SAM-RFI modules not available")
def test_image_validation():
    """Test image validation and preprocessing"""
    try:
        adapter = get_sam_adapter('sam2', device='cpu')
        
        # Test grayscale image conversion
        grayscale_image = np.random.rand(512, 512)
        rgb_image = adapter.validate_image(grayscale_image)
        
        assert rgb_image.shape == (512, 512, 3), f"Expected RGB shape, got {rgb_image.shape}"
        assert rgb_image.dtype == np.float32, f"Expected float32, got {rgb_image.dtype}"
        
        # Test single channel image
        single_channel = np.random.rand(256, 256, 1)
        rgb_single = adapter.validate_image(single_channel)
        assert rgb_single.shape == (256, 256, 3), f"Expected RGB shape, got {rgb_single.shape}"
        
        # Test RGB image (should pass through)
        rgb_input = np.random.rand(128, 128, 3).astype(np.float32) * 255
        rgb_output = adapter.validate_image(rgb_input)
        assert rgb_output.shape == (128, 128, 3), f"Expected same RGB shape, got {rgb_output.shape}"
        
        print("✅ Image validation test passed")
        
    except ImportError as e:
        pytest.skip(f"SAM2 dependencies not available: {e}")


@pytest.mark.skipif(not SAMRFI_AVAILABLE, reason="SAM-RFI modules not available")
def test_default_prompts():
    """Test default prompt generation for RFI detection"""
    try:
        adapter = get_sam_adapter('sam2', device='cpu')
        
        # Create synthetic RFI-like image
        image = np.random.rand(1024, 1024, 3).astype(np.float32) * 255
        
        # Add some high-intensity regions (simulating RFI)
        image[100:200, 100:300] = 200  # Bright region
        image[500:600, 700:800] = 180  # Another bright region
        
        prompts = adapter.create_default_prompts(image, num_points=5)
        
        assert 'points' in prompts
        assert 'bbox' in prompts
        assert 'point_labels' in prompts
        
        assert prompts['points'].shape[1] == 2, "Points should be (N, 2) format"
        assert len(prompts['point_labels']) == len(prompts['points']), "Labels should match points"
        assert prompts['bbox'].shape == (4,), "Bbox should be (4,) format"
        
        print(f"✅ Default prompts test passed. Generated {len(prompts['points'])} points")
        
    except ImportError as e:
        pytest.skip(f"SAM2 dependencies not available: {e}")


@pytest.mark.skipif(not SAMRFI_AVAILABLE, reason="SAM-RFI modules not available")
def test_synthetic_dataset_creation():
    """Test synthetic dataset creation"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            creator = RFIDatasetCreator(output_dir=temp_dir)
            
            # Create small dataset for testing
            dataset = creator.create_training_dataset(
                synthetic_samples=10,  # Small for testing
                train_split=0.8,
                image_size=256,  # Smaller for testing
                seed=42
            )
            
            assert 'train' in dataset
            assert 'validation' in dataset
            
            # Check dataset sizes
            train_size = len(dataset['train'])
            val_size = len(dataset['validation'])
            
            assert train_size > 0, "Training dataset should not be empty"
            assert val_size > 0, "Validation dataset should not be empty"
            assert train_size + val_size == 10, f"Total samples should be 10, got {train_size + val_size}"
            
            # Check sample structure
            sample = dataset['train'][0]
            assert 'image' in sample
            assert 'mask' in sample
            assert 'metadata' in sample
            
            # Check metadata structure
            metadata = sample['metadata']
            required_fields = ['source', 'sample_id', 'rfi_type', 'rfi_intensity', 'image_size']
            for field in required_fields:
                assert field in metadata, f"Missing metadata field: {field}"
            
            print(f"✅ Dataset creation test passed. Train: {train_size}, Val: {val_size}")
            
    except ImportError as e:
        pytest.skip(f"HuggingFace dependencies not available: {e}")


@pytest.mark.skipif(not SAMRFI_AVAILABLE, reason="SAM-RFI modules not available") 
def test_hardware_optimization_recommendations():
    """Test hardware optimization recommendations"""
    try:
        adapter = get_sam_adapter('sam2', device='cpu')
        
        # Test V100 recommendations (16GB memory)
        v100_recs = adapter.optimize_for_hardware(memory_gb=16)
        
        assert 'batch_size' in v100_recs
        assert 'gradient_accumulation' in v100_recs
        assert 'mixed_precision' in v100_recs
        assert v100_recs['mixed_precision'] == True, "Mixed precision should be enabled for V100"
        assert v100_recs['gradient_checkpointing'] == True, "Gradient checkpointing should be enabled for V100"
        
        # Test H200 recommendations (141GB memory, 2-hour limit)
        h200_recs = adapter.optimize_for_hardware(memory_gb=141, time_limit_hours=2)
        
        assert h200_recs['batch_size'] >= v100_recs['batch_size'], "H200 should allow larger batch size"
        assert h200_recs['gradient_checkpointing'] == False, "H200 should disable checkpointing for speed"
        
        print(f"✅ Hardware optimization test passed")
        print(f"   V100 recommendations: {v100_recs}")
        print(f"   H200 recommendations: {h200_recs}")
        
    except ImportError as e:
        pytest.skip(f"SAM2 dependencies not available: {e}")


def run_tests():
    """Run all tests manually (for when pytest is not available)"""
    print("Running SAM2 adapter tests...")
    
    try:
        test_sam_adapter_registry()
        test_sam2_adapter_creation()
        test_image_validation()
        test_default_prompts()
        test_synthetic_dataset_creation()
        test_hardware_optimization_recommendations()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()