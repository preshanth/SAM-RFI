"""
RFI Generator for creating realistic radio frequency interference patterns

Extracted from simulated_ms.py to provide reusable RFI generation for multiple
training data sources (CASA simulator, pure Python, etc.)
"""

import numpy as np
import logging
from typing import Tuple

from ..synthetic_ms_legacy import RFIConfig, ObservationConfig

logger = logging.getLogger(__name__)


class RFIGenerator:
    """
    Generate realistic RFI patterns for radio astronomy data
    
    Creates 5 types of RFI:
    1. Broadband RFI with polynomial frequency variation
    2. Narrowband persistent lines
    3. Transient pulses and repeated bursts  
    4. Periodic signals (time-varying patterns)
    5. Satellite RFI (frequency drifting over time)
    """

    def __init__(self, rfi_config: RFIConfig, obs_config: ObservationConfig):
        self.rfi_config = rfi_config
        self.obs_config = obs_config

    def generate_baseline_rfi(
        self, 
        shape: Tuple[int, int, int],  # [ntime, nchan, npol]
        ant1: int, 
        ant2: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic RFI for single baseline with heavy contamination (25%+)
        
        Args:
            shape: Data shape [ntime, nchan, npol]
            ant1: First antenna index
            ant2: Second antenna index
            
        Returns:
            Tuple of (rfi_array, rfi_mask) with same shape as input
        """
        
        ntime, nchan, npol = shape
        
        # Create RFI array and mask
        rfi_array = np.zeros(shape, dtype=np.complex64)
        rfi_mask = np.zeros(shape, dtype=bool)
        
        # RFI amplitudes: 10^3 to 10^6 times noise level from config
        base_noise_level = self.obs_config.thermal_noise_sigma
        
        # 1. Broadband RFI with polynomial frequency variation (15% occupancy)
        self._add_broadband_rfi(rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level)
        
        # 2. Narrowband persistent lines (increased for 30% total target)
        self._add_narrowband_rfi(rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level)
        
        # 3. Transient pulses and repeated bursts
        self._add_transient_rfi(rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level)
        
        # 4. Periodic signals (time-varying patterns)
        self._add_periodic_rfi(rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level)
        
        # 5. Satellite RFI (frequency drifting over time)
        self._add_satellite_rfi(rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level)
        
        logger.info(f"RFI occupancy: {np.mean(rfi_mask) * 100:.1f}% for baseline {ant1}-{ant2}")
        
        return rfi_array, rfi_mask

    def _add_broadband_rfi(self, rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level):
        """Add broadband RFI with polynomial frequency variation (15% occupancy, non-overlapping)"""
        
        if self.rfi_config.broadband_probability <= 0:
            return
            
        # Track occupied frequency ranges to prevent overlap
        occupied_freq_ranges = []
        
        # Calculate target broadband occupancy: 15% of total data
        target_broadband_occupancy = 0.15
        current_occupancy = 0.0
        n_broadband_events = np.random.randint(3, 6)  # 3-6 broadband events
        
        for event_idx in range(n_broadband_events):
            if current_occupancy >= target_broadband_occupancy:
                break
                
            # Find non-overlapping frequency range
            max_attempts = 20
            for attempt in range(max_attempts):
                freq_start = np.random.randint(0, nchan//2)
                freq_width = np.random.randint(nchan//16, nchan//8)  # 6.25% to 12.5%
                freq_end = min(freq_start + freq_width, nchan)
                
                # Check for overlap with existing events
                overlap = False
                for existing_start, existing_end in occupied_freq_ranges:
                    if not (freq_end <= existing_start or freq_start >= existing_end):
                        overlap = True
                        break
                
                if not overlap:
                    occupied_freq_ranges.append((freq_start, freq_end))
                    break
            else:
                # Could not find non-overlapping range, skip this event
                continue
            
            # Random time range (reduced coverage)
            time_start = np.random.randint(0, ntime//3)
            time_width = np.random.randint(ntime//6, ntime//2)  # Shorter duration
            time_end = min(time_start + time_width, ntime)
            
            # Calculate occupancy contribution
            event_occupancy = ((freq_end - freq_start) * (time_end - time_start)) / (nchan * ntime)
            current_occupancy += event_occupancy
            
            # Polynomial amplitude variation across frequency
            freq_indices = np.arange(freq_end - freq_start)
            # Parabolic/cubic envelope
            poly_coeffs = np.random.uniform(-0.5, 0.5, 4)  # Cubic polynomial
            normalized_freq = (freq_indices - freq_indices.mean()) / (freq_indices.std() + 1e-6)
            amplitude_envelope = np.polyval(poly_coeffs, normalized_freq)
            amplitude_envelope = np.abs(amplitude_envelope)  # Ensure positive
            
            # Normalize envelope to [0,1] range to prevent explosive amplitudes
            if amplitude_envelope.max() > amplitude_envelope.min():
                amplitude_envelope = (amplitude_envelope - amplitude_envelope.min()) / (amplitude_envelope.max() - amplitude_envelope.min())
            else:
                amplitude_envelope = np.ones_like(amplitude_envelope)
            
            # Scale to RFI amplitude range
            base_amplitude = np.random.uniform(10.0, 1000.0) * base_noise_level
            amplitude_envelope = base_amplitude * (1 + amplitude_envelope)
            
            # Apply to all times and polarizations in this event
            for t in range(time_start, time_end):
                for f_idx, f in enumerate(range(freq_start, freq_end)):
                    phase = np.random.uniform(0, 2*np.pi)
                    amp = amplitude_envelope[f_idx]
                    rfi_array[t, f, :] = amp * np.exp(1j * phase)
                    rfi_mask[t, f, :] = True

    def _add_narrowband_rfi(self, rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level):
        """Add narrowband persistent lines"""
        
        n_narrowband = getattr(self.rfi_config, 'narrowband_lines', 20)  # Increased from 15
        for _ in range(n_narrowband):
            freq_idx = np.random.randint(0, nchan)
            amplitude = np.random.uniform(50.0, 500.0) * base_noise_level
            phase = np.random.uniform(0, 2*np.pi)
            
            # Persistent across all time
            rfi_array[:, freq_idx, :] = amplitude * np.exp(1j * phase)
            rfi_mask[:, freq_idx, :] = True

    def _add_transient_rfi(self, rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level):
        """Add transient pulses and repeated bursts"""
        
        n_transients = getattr(self.rfi_config, 'transient_events', 12)  # Increased from 8
        for _ in range(n_transients):
            # Single pulse or repeated bursts
            is_repeated = np.random.random() < 0.5
            
            if is_repeated:
                # Repeated short bursts
                burst_duration = np.random.randint(2, 8)  # Short bursts
                burst_interval = np.random.randint(20, 50)  # Repeat interval
                n_bursts = min(5, ntime // burst_interval)
                
                for burst_idx in range(n_bursts):
                    t_start = burst_idx * burst_interval + np.random.randint(0, 10)
                    t_end = min(t_start + burst_duration, ntime)
                    
                    # Affect random frequency range
                    f_start = np.random.randint(0, nchan//2)
                    f_width = np.random.randint(nchan//20, nchan//5)
                    f_end = min(f_start + f_width, nchan)
                    
                    amplitude = np.random.uniform(20.0, 200.0) * base_noise_level
                    phase = np.random.uniform(0, 2*np.pi)
                    
                    rfi_array[t_start:t_end, f_start:f_end, :] = amplitude * np.exp(1j * phase)
                    rfi_mask[t_start:t_end, f_start:f_end, :] = True
            else:
                # Single pulse
                t_start = np.random.randint(0, ntime-10)
                t_duration = np.random.randint(1, 5)  # Very short
                t_end = min(t_start + t_duration, ntime)
                
                # Affect broad frequency range
                f_start = np.random.randint(0, nchan//4)
                f_width = np.random.randint(nchan//8, nchan//2)
                f_end = min(f_start + f_width, nchan)
                
                amplitude = np.random.uniform(100.0, 800.0) * base_noise_level
                phase = np.random.uniform(0, 2*np.pi)
                
                rfi_array[t_start:t_end, f_start:f_end, :] = amplitude * np.exp(1j * phase)
                rfi_mask[t_start:t_end, f_start:f_end, :] = True

    def _add_periodic_rfi(self, rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level):
        """Add periodic signals (time-varying patterns)"""
        
        n_periodic = getattr(self.rfi_config, 'periodic_signals', 3)
        for _ in range(n_periodic):
            # Periodic pulse train
            period = np.random.randint(10, 30)  # Period in time steps
            duty_cycle = np.random.uniform(0.1, 0.3)  # 10-30% duty cycle
            pulse_width = int(period * duty_cycle)
            
            # Fixed frequency
            freq_idx = np.random.randint(0, nchan)
            amplitude = np.random.uniform(30.0, 300.0) * base_noise_level
            
            for t in range(0, ntime, period):
                t_end = min(t + pulse_width, ntime)
                phase = np.random.uniform(0, 2*np.pi)
                rfi_array[t:t_end, freq_idx, :] = amplitude * np.exp(1j * phase)
                rfi_mask[t:t_end, freq_idx, :] = True

    def _add_satellite_rfi(self, rfi_array, rfi_mask, ntime, nchan, npol, base_noise_level):
        """Add satellite RFI (frequency drifting over time)"""
        
        n_satellites = getattr(self.rfi_config, 'satellite_passes', 4)  # Increased from 2
        for _ in range(n_satellites):
            # Linear frequency drift
            t_start = np.random.randint(0, ntime//4)
            t_duration = np.random.randint(ntime//8, ntime//2)
            t_end = min(t_start + t_duration, ntime)
            
            f_start = np.random.randint(0, nchan//2)
            f_end = np.random.randint(nchan//2, nchan)
            
            amplitude = np.random.uniform(40.0, 400.0) * base_noise_level
            
            # Linear drift from f_start to f_end over time
            for t_idx, t in enumerate(range(t_start, t_end)):
                progress = t_idx / max(1, (t_end - t_start - 1))
                current_freq = int(f_start + progress * (f_end - f_start))
                if 0 <= current_freq < nchan:
                    phase = np.random.uniform(0, 2*np.pi)
                    rfi_array[t, current_freq, :] = amplitude * np.exp(1j * phase)
                    rfi_mask[t, current_freq, :] = True