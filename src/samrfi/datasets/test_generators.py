#!/usr/bin/env python3
"""
Training Data Generators Test

Tests the core functionality of the new generator architecture:
1. Create MS with 3 antennas using base class
2. Fill with Gaussian noise for each baseline
3. Inject different RFI types
4. Write RFI flags back to MS
5. Validate file exists, shapes, and basic data properties
"""

# import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging

import sys
from pathlib import Path

# Test direct imports without module structure
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import config classes directly
exec(open(Path(__file__).parent / 'synthetic_ms_legacy.py').read())

# Import generator classes directly  
exec(open(Path(__file__).parent / 'generators' / 'rfi_generator.py').read())
exec(open(Path(__file__).parent / 'generators' / 'python_generator.py').read())

try:
    exec(open(Path(__file__).parent / 'generators' / 'base.py').read())
    exec(open(Path(__file__).parent / 'generators' / 'casa_generator.py').read())
    CASA_AVAILABLE = True
except:
    CASA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGenerators:
    """Test training data generation pipeline"""

    def test_pure_python_generator(self):
        """Test pure Python generator end-to-end"""
        
        # Simple 3-antenna configuration
        obs_config = ObservationConfig(
            num_antennas=3,
            num_spw=1,
            channels_per_spw=32,
            start_frequency=1.0e9,
            channel_width=1e6,
            total_duration=60.0,  # 1 minute
            integration_time=10.0,
            thermal_noise_sigma=1e-3
        )
        
        # RFI configuration with different types
        rfi_config = RFIConfig(
            broadband_probability=0.02,
            narrowband_lines=3,
            transient_events=2,
            periodic_signals=1,
            satellite_passes=1
        )
        
        # Create generator
        generator = PurePythonTrainingGenerator(obs_config)
        
        # Generate training arrays
        vis_arrays, rfi_mask = generator.create_training_ms_arrays(rfi_config)
        
        # Validate shapes
        expected_baselines = 3 * 2 // 2  # 3 baselines for 3 antennas
        expected_times = int(60.0 / 10.0)  # 6 time steps
        expected_channels = 32
        expected_pols = 4
        expected_shape = (expected_baselines, expected_times, expected_channels, expected_pols)
        
        assert vis_arrays.shape == expected_shape, f"Expected {expected_shape}, got {vis_arrays.shape}"
        assert rfi_mask.shape == expected_shape, f"RFI mask shape mismatch"
        
        # Validate data properties
        assert vis_arrays.dtype == np.complex64, "Visibility data should be complex64"
        assert rfi_mask.dtype == bool, "RFI mask should be boolean"
        
        # Check that we have some data (not all zeros)
        assert not np.all(vis_arrays == 0), "Visibility data is all zeros"
        assert np.any(rfi_mask), "No RFI was injected"
        
        # Check RFI contamination level is reasonable
        contamination = np.mean(rfi_mask)
        assert 0.01 < contamination < 0.8, f"RFI contamination {contamination:.3f} outside reasonable range"
        
        # Check that flagged data has higher amplitude than unflagged
        flagged_data = vis_arrays[rfi_mask]
        unflagged_data = vis_arrays[~rfi_mask]
        
        flagged_power = np.mean(np.abs(flagged_data)**2)
        unflagged_power = np.mean(np.abs(unflagged_data)**2)
        
        assert flagged_power > unflagged_power, "RFI should have higher power than noise"
        
        logger.info(f"✅ Pure Python generator test passed")
        logger.info(f"   Shape: {vis_arrays.shape}")
        logger.info(f"   RFI contamination: {contamination:.1%}")
        logger.info(f"   RFI/noise power ratio: {flagged_power/unflagged_power:.1f}")

    # @pytest.mark.skipif(not CASA_AVAILABLE, reason="CASA tools not available")
    def test_casa_generator_ms_creation(self):
        """Test CASA generator creates actual MS file"""
        
        # Simple 3-antenna configuration
        obs_config = ObservationConfig(
            num_antennas=3,
            num_spw=1,
            channels_per_spw=32,
            start_frequency=1.0e9,
            channel_width=1e6,
            total_duration=60.0,
            integration_time=10.0,
            thermal_noise_sigma=1e-3
        )
        
        # RFI configuration
        rfi_config = RFIConfig(
            broadband_probability=0.02,
            narrowband_lines=2,
            transient_events=1,
            periodic_signals=1,
            satellite_passes=1
        )
        
        # Create generator
        generator = CASATrainingGenerator(obs_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ms_path = Path(temp_dir) / "test_3ant.ms"
            
            # Generate MS with RFI
            result_path = generator.create_training_ms(
                str(ms_path),
                rfi_config=rfi_config,
                include_rfi_flags=True
            )
            
            # Validate MS was created
            assert result_path == str(ms_path)
            assert ms_path.exists(), "MS file was not created"
            assert ms_path.is_dir(), "MS should be a directory"
            
            # Check for essential MS tables
            required_tables = ['ANTENNA', 'DATA_DESCRIPTION', 'FIELD', 'OBSERVATION', 
                             'POLARIZATION', 'SPECTRAL_WINDOW', 'table.dat']
            
            for table in required_tables:
                table_path = ms_path / table
                assert table_path.exists(), f"Required MS table {table} missing"
            
            # Check that plots were created
            plot_dir = Path(str(ms_path).replace('.ms', '_plots'))
            assert plot_dir.exists(), "Plot directory was not created"
            
            # Should have plots for first few baselines
            plot_files = list(plot_dir.glob("*.png"))
            assert len(plot_files) > 0, "No plot files were created"
            
            # Validate using CASA tools if available
            try:
                from casatools import table
                tb = table()
                tb.open(str(ms_path))
                
                nrows = tb.nrows()
                expected_baselines = 3 * 2 // 2  # 3 baselines
                expected_times = int(60.0 / 10.0)  # 6 time steps
                expected_rows = expected_baselines * expected_times
                
                assert nrows == expected_rows, f"Expected {expected_rows} rows, got {nrows}"
                
                # Check data and flag columns
                data_sample = tb.getcell("DATA", 0)
                flag_sample = tb.getcell("FLAG", 0)
                
                expected_data_shape = (4, 32)  # [npol, nchan]
                assert data_sample.shape == expected_data_shape, f"Data shape mismatch: {data_sample.shape}"
                assert flag_sample.shape == expected_data_shape, f"Flag shape mismatch: {flag_sample.shape}"
                
                # Check that some flags were set
                total_flags = np.sum(flag_sample)
                assert total_flags > 0, "No flags were written to MS"
                
                # Check that data is not all zeros
                assert not np.all(data_sample == 0), "MS data is all zeros"
                
                tb.close()
                
                logger.info(f"✅ CASA generator test passed")
                logger.info(f"   MS rows: {nrows}")
                logger.info(f"   Data shape: {data_sample.shape}")
                logger.info(f"   Flags set: {total_flags}/{flag_sample.size}")
                logger.info(f"   Plot files: {len(plot_files)}")
                
            except ImportError:
                logger.warning("Could not validate MS contents - CASA table tools unavailable")


if __name__ == "__main__":
    test = TestGenerators()
    
    print("Testing Pure Python generator...")
    test.test_pure_python_generator()
    
    if CASA_AVAILABLE:
        print("Testing CASA generator...")
        test.test_casa_generator_ms_creation()
    else:
        print("Skipping CASA generator test - CASA tools not available")
    
    print("✅ All tests completed!")