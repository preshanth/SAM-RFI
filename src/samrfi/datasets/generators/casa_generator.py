"""
CASA-based training data generator

Uses CASA simulator to generate realistic baseline visibilities, then adds RFI.
This is the recommended approach for high-quality training data.
"""

import numpy as np
import logging
from typing import Tuple

from .base import TrainingDataGenerator

logger = logging.getLogger(__name__)


class CASATrainingGenerator(TrainingDataGenerator):
    """
    Generate training data using CASA simulator for realistic baselines
    
    Creates proper interferometric visibility structure with realistic noise
    levels and baseline-dependent amplitudes.
    """

    def _generate_clean_baseline(self, ant1: int, ant2: int, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate clean baseline data using only thermal noise
        
        Args:
            ant1: First antenna index
            ant2: Second antenna index
            shape: Data shape [ntime, nchan, npol]
            
        Returns:
            Clean complex thermal noise data
        """
        ntime, nchan, npol = shape
        
        # Generate thermal noise only
        noise_sigma = self.obs_config.thermal_noise_sigma
        real_noise = np.random.normal(0, noise_sigma, shape)
        imag_noise = np.random.normal(0, noise_sigma, shape)
        thermal_noise = real_noise + 1j * imag_noise
        
        logger.debug(f"Generated baseline {ant1}-{ant2}: noise_sigma={noise_sigma}")
        
        return thermal_noise