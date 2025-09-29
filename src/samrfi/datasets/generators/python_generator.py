"""
Pure Python training data generator

Fallback implementation that doesn't require CASA tools.
Generates synthetic visibility data using pure Python.
"""

import numpy as np
import logging
from typing import Tuple

from ..synthetic_ms_legacy import ObservationConfig
from .rfi_generator import RFIGenerator

logger = logging.getLogger(__name__)


class PurePythonTrainingGenerator:
    """
    Generate training data using pure Python (no CASA dependency)
    
    This is a simplified fallback for environments where CASA is not available.
    Creates basic visibility structure with synthetic antenna positions.
    """

    def __init__(self, obs_config: ObservationConfig):
        self.obs_config = obs_config
        
        # Generate synthetic antenna positions
        self.antenna_positions = self._generate_antenna_positions()
        
        logger.info(f"Pure Python generator: {obs_config.num_antennas} antennas")

    def _generate_antenna_positions(self) -> np.ndarray:
        """Generate realistic antenna positions"""
        # Simple spiral array pattern (from synthetic_ms.py)
        positions = np.zeros((self.obs_config.num_antennas, 3))

        for i in range(self.obs_config.num_antennas):
            if i == 0:
                # Reference antenna at origin
                positions[i] = [0, 0, 0]
            else:
                # Spiral pattern with increasing radius
                angle = 2 * np.pi * i / self.obs_config.num_antennas
                radius = 100 * (1 + i / 4)  # meters, increasing outward
                positions[i] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0,  # Assume flat array
                ]

        return positions

    def generate_clean_baseline(self, ant1: int, ant2: int, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate clean baseline data using pure Python
        
        Args:
            ant1: First antenna index
            ant2: Second antenna index
            shape: Data shape [ntime, nchan, npol]
            
        Returns:
            Clean complex visibility data
        """
        ntime, nchan, npol = shape
        
        # Baseline vector
        baseline_vec = self.antenna_positions[ant2] - self.antenna_positions[ant1]
        baseline_length = np.linalg.norm(baseline_vec[:2])  # Ignore z for simplicity

        # Vectorized generation for this baseline
        noise_level = 0.1
        
        # Generate noise for all times/frequencies at once
        real_noise = np.random.normal(0, noise_level, shape)
        imag_noise = np.random.normal(0, noise_level, shape)
        
        # Simple source model (adjust amplitude by baseline length)
        source_amplitude = 1.0 / (1.0 + baseline_length / 1000.0)
        
        # Add polarization structure
        pol_amplitudes = np.array([
            source_amplitude,
            0.1 * source_amplitude, 
            0.1 * source_amplitude,
            source_amplitude,
        ])
        
        # Create source signal
        source_signal = np.zeros(shape, dtype=np.complex64)
        for pol_idx in range(npol):
            phase = np.random.uniform(0, 2*np.pi)
            source_signal[:, :, pol_idx] = pol_amplitudes[pol_idx] * np.exp(1j * phase)
        
        # Combine source + noise
        baseline_vis = source_signal + real_noise + 1j * imag_noise
        
        logger.debug(f"Generated baseline {ant1}-{ant2}: baseline_length={baseline_length:.1f}m, source_amp={source_amplitude:.3f}")
        
        return baseline_vis

    def create_training_ms_arrays(self, rfi_config=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full training arrays (alternative to MS files)
        
        Returns:
            Tuple of (corrupted_visibilities, rfi_mask) as numpy arrays
        """
        # Calculate dimensions
        num_baselines = self.obs_config.num_antennas * (self.obs_config.num_antennas - 1) // 2
        num_times = int(self.obs_config.total_duration / self.obs_config.integration_time)
        total_channels = self.obs_config.num_spw * self.obs_config.channels_per_spw
        
        # Initialize arrays
        vis_shape = (num_baselines, num_times, total_channels, 4)
        visibilities = np.zeros(vis_shape, dtype=np.complex64)
        rfi_mask = np.zeros(vis_shape, dtype=bool) if rfi_config else None
        
        # Generate data for each baseline
        baseline_idx = 0
        for ant1 in range(self.obs_config.num_antennas):
            for ant2 in range(ant1 + 1, self.obs_config.num_antennas):
                
                # Generate clean baseline
                clean_baseline = self.generate_clean_baseline(
                    ant1, ant2, (num_times, total_channels, 4)
                )
                visibilities[baseline_idx] = clean_baseline
                
                # Add RFI if requested
                if rfi_config:
                    rfi_generator = RFIGenerator(rfi_config, self.obs_config)
                    rfi_array, baseline_rfi_mask = rfi_generator.generate_baseline_rfi(
                        clean_baseline.shape, ant1, ant2
                    )
                    visibilities[baseline_idx] += rfi_array
                    rfi_mask[baseline_idx] = baseline_rfi_mask
                
                baseline_idx += 1
                
                if baseline_idx % 50 == 0:
                    logger.info(f"Generated {baseline_idx}/{num_baselines} baselines")
        
        logger.info(f"Generated training arrays: {vis_shape}")
        if rfi_config:
            logger.info(f"RFI contamination: {np.mean(rfi_mask) * 100:.1f}%")
        
        return visibilities, rfi_mask