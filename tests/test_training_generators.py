"""
Unit tests for training data generators

Tests both CASA-based and pure Python training data generation approaches.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from samrfi.datasets.synthetic_ms_legacy import ObservationConfig, RFIConfig
from samrfi.datasets.generators import (
    CASATrainingGenerator, 
    PurePythonTrainingGenerator,
    RFIGenerator
)

# Test configurations
TEST_OBS_CONFIG = ObservationConfig(
    num_antennas=4,  # Small for fast testing
    num_spw=1,
    channels_per_spw=64,  # Small for fast testing
    start_frequency=1.0e9,
    channel_width=1e6,
    total_duration=120.0,  # 2 minutes
    integration_time=10.0,
    thermal_noise_sigma=1e-3
)

TEST_RFI_CONFIG = RFIConfig(
    broadband_probability=0.02,
    narrowband_lines=2,
    transient_events=2,
    periodic_signals=1,
    satellite_passes=1
)


class TestRFIGenerator:
    """Test the RFI generation component"""
    
    def test_rfi_generator_initialization(self):
        """Test RFI generator can be created"""
        generator = RFIGenerator(TEST_RFI_CONFIG, TEST_OBS_CONFIG)
        assert generator.rfi_config == TEST_RFI_CONFIG
        assert generator.obs_config == TEST_OBS_CONFIG
    
    def test_baseline_rfi_generation(self):
        """Test RFI generation for single baseline"""
        generator = RFIGenerator(TEST_RFI_CONFIG, TEST_OBS_CONFIG)
        
        # Test shape
        shape = (12, 64, 4)  # [ntime, nchan, npol]
        rfi_array, rfi_mask = generator.generate_baseline_rfi(shape, 0, 1)
        
        assert rfi_array.shape == shape
        assert rfi_mask.shape == shape
        assert rfi_array.dtype == np.complex64
        assert rfi_mask.dtype == bool
        
        # Check that some RFI was generated
        assert np.any(rfi_mask), "No RFI was generated"
        assert np.any(rfi_array != 0), "RFI array is all zeros"
        
        # Check RFI contamination level
        contamination = np.mean(rfi_mask)
        assert 0.05 < contamination < 0.5, f"RFI contamination {contamination:.3f} outside expected range"


class TestPurePythonGenerator:
    """Test pure Python training data generator"""
    
    def test_initialization(self):
        """Test pure Python generator can be created"""
        generator = PurePythonTrainingGenerator(TEST_OBS_CONFIG)
        assert generator.obs_config == TEST_OBS_CONFIG
        assert hasattr(generator, 'antenna_positions')
    
    def test_baseline_generation(self):
        """Test clean baseline data generation"""
        generator = PurePythonTrainingGenerator(TEST_OBS_CONFIG)
        
        shape = (12, 64, 4)  # [ntime, nchan, npol]
        baseline_data = generator.generate_clean_baseline(0, 1, shape)
        
        assert baseline_data.shape == shape
        assert baseline_data.dtype == np.complex64
        assert not np.all(baseline_data == 0), "Baseline data is all zeros"
    
    def test_training_arrays_generation(self):
        """Test full training array generation"""
        generator = PurePythonTrainingGenerator(TEST_OBS_CONFIG)
        
        # Test without RFI
        clean_vis, rfi_mask = generator.create_training_ms_arrays()
        
        expected_baselines = TEST_OBS_CONFIG.num_antennas * (TEST_OBS_CONFIG.num_antennas - 1) // 2
        expected_times = int(TEST_OBS_CONFIG.total_duration / TEST_OBS_CONFIG.integration_time)
        expected_channels = TEST_OBS_CONFIG.num_spw * TEST_OBS_CONFIG.channels_per_spw
        
        expected_shape = (expected_baselines, expected_times, expected_channels, 4)
        assert clean_vis.shape == expected_shape
        assert rfi_mask is None  # No RFI requested
        
        # Test with RFI
        corrupted_vis, rfi_mask = generator.create_training_ms_arrays(TEST_RFI_CONFIG)
        
        assert corrupted_vis.shape == expected_shape
        assert rfi_mask.shape == expected_shape
        assert np.any(rfi_mask), "No RFI flags generated"


@pytest.mark.skipif(
    not hasattr(pytest, "casa_available") or not pytest.casa_available,
    reason="CASA tools not available"
)
class TestCASAGenerator:
    """Test CASA-based training data generator"""
    
    def test_initialization(self):
        """Test CASA generator can be created"""
        try:
            generator = CASATrainingGenerator(TEST_OBS_CONFIG)
            assert generator.obs_config == TEST_OBS_CONFIG
        except ImportError:
            pytest.skip("CASA tools not available")
    
    def test_baseline_generation(self):
        """Test CASA baseline data generation"""
        try:
            generator = CASATrainingGenerator(TEST_OBS_CONFIG)
        except ImportError:
            pytest.skip("CASA tools not available")
        
        shape = (12, 64, 4)  # [ntime, nchan, npol]
        baseline_data = generator._generate_clean_baseline(0, 1, shape)
        
        assert baseline_data.shape == shape
        assert baseline_data.dtype == np.complex64
        assert not np.all(baseline_data == 0), "Baseline data is all zeros"
    
    def test_ms_creation(self):
        """Test full MS creation with RFI"""
        try:
            generator = CASATrainingGenerator(TEST_OBS_CONFIG)
        except ImportError:
            pytest.skip("CASA tools not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ms_path = Path(temp_dir) / "test.ms"
            
            # Create MS with RFI
            result_path = generator.create_training_ms(
                str(ms_path),
                rfi_config=TEST_RFI_CONFIG,
                include_rfi_flags=True
            )
            
            assert result_path == str(ms_path)
            assert ms_path.exists(), "MS was not created"
            
            # Check that plots were created
            plot_dir = Path(str(ms_path).replace('.ms', '_plots'))
            assert plot_dir.exists(), "Plot directory was not created"
            
            # Check for plot files (first 5 baselines × 4 polarizations)
            plot_files = list(plot_dir.glob("*.png"))
            assert len(plot_files) > 0, "No plot files were created"


class TestLegacyCompatibility:
    """Test backward compatibility with legacy SimulatedMS"""
    
    def test_simulated_ms_delegation(self):
        """Test that SimulatedMS properly delegates to CASATrainingGenerator"""
        try:
            from samrfi.datasets.simulated_ms import SimulatedMS
        except ImportError:
            pytest.skip("CASA tools not available")
        
        # Test that deprecation warning is issued
        with pytest.warns(None) as warning_list:
            sim_ms = SimulatedMS(TEST_OBS_CONFIG)
        
        # Check that warning contains deprecation message
        deprecation_warnings = [w for w in warning_list if "deprecated" in str(w.message).lower()]
        assert len(deprecation_warnings) > 0, "No deprecation warning issued"
        
        # Test that it has the expected interface
        assert hasattr(sim_ms, 'create_ms_with_rfi')
        assert hasattr(sim_ms, 'print_memory')


def test_generator_comparison():
    """Compare output between CASA and Pure Python generators"""
    
    # Test pure Python generator
    py_generator = PurePythonTrainingGenerator(TEST_OBS_CONFIG)
    py_vis, py_mask = py_generator.create_training_ms_arrays(TEST_RFI_CONFIG)
    
    # Basic shape and type checks
    expected_baselines = TEST_OBS_CONFIG.num_antennas * (TEST_OBS_CONFIG.num_antennas - 1) // 2
    expected_times = int(TEST_OBS_CONFIG.total_duration / TEST_OBS_CONFIG.integration_time)
    expected_channels = TEST_OBS_CONFIG.num_spw * TEST_OBS_CONFIG.channels_per_spw
    expected_shape = (expected_baselines, expected_times, expected_channels, 4)
    
    assert py_vis.shape == expected_shape
    assert py_mask.shape == expected_shape
    
    # Check RFI contamination levels are reasonable
    py_contamination = np.mean(py_mask)
    assert 0.05 < py_contamination < 0.5, f"Pure Python RFI contamination {py_contamination:.3f} outside expected range"
    
    print(f"Pure Python generator - RFI contamination: {py_contamination:.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])