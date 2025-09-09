"""
Pytest wrapper for end-to-end validation test

This provides a pytest interface for the comprehensive validation pipeline.
Use this for automated testing and CI/CD integration.

Usage:
    pytest tests/test_end_to_end_validation.py -v                    # Full test
    pytest tests/test_end_to_end_validation.py -v -k quick           # Quick test
    pytest tests/test_end_to_end_validation.py -v -s                 # With output
"""

import sys
from pathlib import Path
import pytest
import tempfile
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validate_end_to_end import EndToEndValidator

# Setup logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory(prefix="samrfi_test_") as temp_dir:
        yield temp_dir


@pytest.fixture
def validator_quick(temp_output_dir):
    """Create validator configured for quick testing"""
    return EndToEndValidator(temp_output_dir, quick_test=True)


@pytest.fixture
def validator_full(temp_output_dir):
    """Create validator configured for full testing"""
    return EndToEndValidator(temp_output_dir, quick_test=False)


class TestEndToEndValidation:
    """End-to-end validation test suite"""
    
    def test_synthetic_data_generation(self, validator_quick):
        """Test synthetic measurement set generation"""
        dataset_meta = validator_quick.step1_generate_synthetic_data()
        
        # Validate dataset metadata
        assert 'corrupted_ms' in dataset_meta
        assert 'ground_truth_dir' in dataset_meta
        assert 'rfi_statistics' in dataset_meta
        
        # Check RFI statistics are reasonable
        rfi_fraction = dataset_meta['rfi_statistics']['rfi_fraction']
        assert 0.001 <= rfi_fraction <= 0.2  # Between 0.1% and 20%
        
        # Check files were created
        corrupted_ms = Path(dataset_meta['corrupted_ms'])
        ground_truth_dir = Path(dataset_meta['ground_truth_dir'])
        
        assert corrupted_ms.exists(), f"Corrupted MS not created: {corrupted_ms}"
        assert ground_truth_dir.exists(), f"Ground truth dir not created: {ground_truth_dir}"
        
        # Check ground truth files
        assert (ground_truth_dir / 'corrupted_visibilities.npy').exists()
        assert (ground_truth_dir / 'rfi_mask.npy').exists()
    
    def test_memory_efficient_loading(self, validator_quick):
        """Test memory-efficient data loading"""
        # Generate test data first
        dataset_meta = validator_quick.step1_generate_synthetic_data()
        
        # Test data loading
        patches, masks, metadata = validator_quick.step2_load_and_process_data(dataset_meta)
        
        # Validate loaded data
        assert len(patches) > 0, "No training patches extracted"
        assert len(masks) > 0, "No training masks extracted"
        assert len(patches) == len(masks), "Patch and mask count mismatch"
        
        # Check data shapes and types
        assert patches.ndim == 3, f"Expected 3D patches, got {patches.ndim}D"
        assert masks.ndim == 3, f"Expected 3D masks, got {masks.ndim}D"
        assert masks.dtype == bool, f"Expected boolean masks, got {masks.dtype}"
        
        # Check patch dimensions match config
        expected_size = validator_quick.training_config['training']['patch_size']
        assert patches.shape[1] == expected_size, f"Patch height mismatch: {patches.shape[1]} vs {expected_size}"
        assert patches.shape[2] == expected_size, f"Patch width mismatch: {patches.shape[2]} vs {expected_size}"
    
    def test_model_training_simulation(self, validator_quick):
        """Test model training pipeline (simulated)"""
        # Generate data
        dataset_meta = validator_quick.step1_generate_synthetic_data()
        patches, masks, metadata = validator_quick.step2_load_and_process_data(dataset_meta)
        
        # Test training
        training_stats = validator_quick.step3_train_model(patches, masks)
        
        # Validate training statistics
        assert 'epochs' in training_stats
        assert 'losses' in training_stats
        assert 'accuracies' in training_stats
        assert 'memory_usage' in training_stats
        
        # Check we have data for all epochs
        num_epochs = validator_quick.training_config['training']['max_epochs']
        assert len(training_stats['epochs']) == num_epochs
        assert len(training_stats['losses']) == num_epochs
        assert len(training_stats['accuracies']) == num_epochs
        
        # Check losses are decreasing trend (allowing for noise)
        losses = training_stats['losses']
        assert losses[0] > losses[-1] * 0.8, "Loss should show decreasing trend"
        
        # Check accuracies are reasonable
        accuracies = training_stats['accuracies']
        assert all(0.0 <= acc <= 1.0 for acc in accuracies), "Accuracies should be between 0 and 1"
    
    def test_inference_and_metrics(self, validator_quick):
        """Test inference and metric calculation"""
        # Generate data and train
        dataset_meta = validator_quick.step1_generate_synthetic_data()
        patches, masks, metadata = validator_quick.step2_load_and_process_data(dataset_meta)
        training_stats = validator_quick.step3_train_model(patches, masks)
        
        # Test inference
        predictions, metrics = validator_quick.step4_run_inference(patches, masks)
        
        # Validate predictions
        assert len(predictions) > 0, "No predictions generated"
        assert predictions.shape == masks[:len(predictions)].shape, "Prediction shape mismatch"
        assert predictions.dtype == bool, f"Expected boolean predictions, got {predictions.dtype}"
        
        # Validate metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0.0 <= metrics[metric] <= 1.0, f"Invalid {metric}: {metrics[metric]}"
    
    @pytest.mark.slow
    def test_full_pipeline_quick(self, validator_quick):
        """Test complete pipeline with quick settings"""
        summary = validator_quick.run_full_validation()
        
        # Validate summary
        assert summary['validation_successful'], f"Validation failed: {summary.get('error', 'Unknown')}"
        assert 'synthetic_data' in summary
        assert 'training_patches' in summary
        assert 'training_stats' in summary
        assert 'inference_metrics' in summary
        
        # Check reasonable performance
        metrics = summary['inference_metrics']
        assert metrics['accuracy'] > 0.3, f"Accuracy too low: {metrics['accuracy']}"
        
        # Check training completed
        assert summary['training_patches'] > 0, "No training patches processed"
        
        # Check duration is reasonable (should complete in reasonable time)
        duration = summary['duration_seconds']
        assert duration < 600, f"Validation took too long: {duration} seconds"  # 10 minute limit
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_pipeline_with_gpu(self, validator_full):
        """Test complete pipeline with GPU if available"""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU test")
        
        # Run validation
        summary = validator_full.run_full_validation()
        
        # Validate GPU was used effectively
        assert summary['validation_successful'], "GPU validation failed"
        
        # Check memory usage was tracked
        training_stats = summary['training_stats']
        assert 'memory_usage' in training_stats
        assert len(training_stats['memory_usage']) > 0, "No memory usage recorded"
    
    def test_plot_generation(self, validator_quick, temp_output_dir):
        """Test plot generation doesn't crash"""
        # Generate minimal data for plotting
        dataset_meta = validator_quick.step1_generate_synthetic_data()
        patches, masks, metadata = validator_quick.step2_load_and_process_data(dataset_meta)
        training_stats = validator_quick.step3_train_model(patches, masks)
        predictions, metrics = validator_quick.step4_run_inference(patches, masks)
        
        # Test plot generation (should not raise exception)
        validator_quick.step5_create_plots(patches, masks, predictions, training_stats, metrics)
        
        # Check plot file was created
        plot_path = Path(temp_output_dir) / 'plots' / 'validation_results.png'
        assert plot_path.exists(), f"Plot file not created: {plot_path}"


# Test configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take several minutes)")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")


# Helper function for manual test runs
def run_quick_test():
    """Run quick validation test manually"""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        validator = EndToEndValidator(temp_dir, quick_test=True)
        summary = validator.run_full_validation()
        return summary


if __name__ == "__main__":
    # Allow running this file directly for manual testing
    print("Running quick validation test...")
    summary = run_quick_test()
    
    if summary['validation_successful']:
        print("✅ Manual test PASSED")
    else:
        print("❌ Manual test FAILED")
        print(f"Error: {summary.get('error', 'Unknown')}")