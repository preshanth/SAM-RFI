#!/usr/bin/env python3
"""
Test SimulatedMS: CASA-generated data + baseline-by-baseline RFI injection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from samrfi.datasets import ObservationConfig, RFIConfig, SimulatedMS

def main():
    # Same config as before
    obs_config = ObservationConfig(
        num_antennas=27,  # VLA-like array for 351 baselines per MS
        num_spw=2,
        channels_per_spw=512,  # 1024 total channels
        start_frequency=1.4e9,  # L-band
        total_duration=1024.0,  # 1024 seconds for 1024 time steps
        integration_time=1.0    # 1s integration → 1024 time steps
    )
    
    # Light RFI config
    rfi_config = RFIConfig(
        broadband_probability=0.01,
        narrowband_lines=2,
        transient_events=2,
        periodic_signals=1,
        satellite_passes=1
    )
    
    print(f"Config: {obs_config.num_antennas} antennas, {obs_config.total_duration/obs_config.integration_time:.0f} times, {obs_config.num_spw * obs_config.channels_per_spw} channels")
    print(f"Expected baselines: {obs_config.num_antennas * (obs_config.num_antennas - 1) // 2}")
    
    # Test 1: Clean MS (CASA data only)
    print("\n=== Test 1: Clean MS ===")
    simulator = SimulatedMS(obs_config)
    simulator.create_ms_with_rfi("test_casa_clean.ms", rfi_config=None)
    print("Clean MS created successfully")
    
    # Test 2: MS with RFI (CASA data + baseline RFI injection)
    print("\n=== Test 2: MS with RFI ===")
    simulator2 = SimulatedMS(obs_config)
    simulator2.create_ms_with_rfi("test_casa_rfi.ms", rfi_config=rfi_config, include_rfi_flags=True)
    print("RFI MS created successfully")
    
    print("\nBoth tests complete!")

if __name__ == "__main__":
    main()