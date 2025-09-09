#!/usr/bin/env python3
"""
Phase 1: Test clean MS generation without RFI
Memory monitoring to ensure no leaks
"""

import sys
import numpy as np
from pathlib import Path
import psutil
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from samrfi.datasets import ObservationConfig, RFIConfig, SimulatedMS

def print_memory(label):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{label}: {mem_mb:.1f} MB")

def main():
    print_memory("START")
    
    # Exact config from synthetic_training.py
    obs_config = ObservationConfig(
        num_antennas=27,  # VLA-like array for 351 baselines per MS
        num_spw=2,
        channels_per_spw=512,  # 1024 total channels
        start_frequency=1.4e9,  # L-band
        total_duration=1024.0,  # 1024 seconds for 1024 time steps
        integration_time=1.0,   # 1s integration → 1024 time steps
        thermal_noise_sigma=1e-3  # Thermal noise level
    )
    print_memory("Config created")
    
    # Heavy RFI config for 25%+ occupancy
    rfi_config = RFIConfig(
        broadband_probability=0.25,  # 25% broadband probability  
        narrowband_lines=15,         # 15 persistent lines
        transient_events=8,          # 8 pulse/burst events
        periodic_signals=3,          # 3 periodic patterns
        satellite_passes=2           # 2 satellite passes
    )
    print_memory("RFI config created")
    
    print(f"Config: {obs_config.num_antennas} antennas, {obs_config.total_duration/obs_config.integration_time:.0f} times, {obs_config.num_spw * obs_config.channels_per_spw} channels")
    print(f"Expected baselines: {obs_config.num_antennas * (obs_config.num_antennas - 1) // 2}")
    
    # Test new SimulatedMS approach
    print("Creating MS with CASA + baseline RFI injection...")
    simulator = SimulatedMS(obs_config)
    print_memory("SimulatedMS created")
    
    simulator.create_ms_with_rfi("test_simulated.ms", rfi_config=rfi_config, include_rfi_flags=True)
    print_memory("MS creation complete")
    
    # Print baseline statistics summary
    print("\nBaseline Statistics Summary:")
    print("=" * 50)
    
    try:
        from casatools import table
        tb = table()
        tb.open("test_simulated.ms", nomodify=True)
        
        # Get sample of data for statistics
        sample_size = min(1000, tb.nrows())
        data_sample = []
        for i in range(0, tb.nrows(), max(1, tb.nrows() // sample_size)):
            data_sample.append(tb.getcell("DATA", i))
        
        tb.close()
        
        # Convert to numpy array and compute stats
        import numpy as np
        data_sample = np.array(data_sample)
        amplitudes = np.abs(data_sample)
        
        print(f"Data shape: {data_sample.shape}")
        print(f"Amplitude range: {amplitudes.min():.2e} to {amplitudes.max():.2e}")
        print(f"Amplitude mean: {amplitudes.mean():.2e}")
        print(f"Amplitude std: {amplitudes.std():.2e}")
        print(f"Non-zero fraction: {(amplitudes > 1e-10).mean():.3f}")
        
    except Exception as e:
        print(f"Could not compute statistics: {e}")
    
    print("Test complete: test_simulated.ms")

if __name__ == "__main__":
    main()