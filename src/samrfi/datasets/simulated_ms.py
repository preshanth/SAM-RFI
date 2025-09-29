"""
Simulated MS with CASA-generated data and baseline-by-baseline RFI injection

Legacy wrapper around new CASATrainingGenerator. This class is maintained for
backward compatibility but delegates to the new generator architecture.
"""

import numpy as np
from typing import Optional
from pathlib import Path
import logging

from .synthetic_ms_legacy import ObservationConfig, RFIConfig
from .generators import CASATrainingGenerator

logger = logging.getLogger(__name__)


class SimulatedMS:
    """
    Legacy wrapper for CASA-based training data generation
    
    Delegates to CASATrainingGenerator for backward compatibility.
    New code should use CASATrainingGenerator directly.
    """

    def __init__(self, obs_config: ObservationConfig):
        self.obs_config = obs_config
        self._generator = CASATrainingGenerator(obs_config)
        
        logger.warning(
            "SimulatedMS is deprecated. Use CASATrainingGenerator directly: "
            "from samrfi.datasets.generators import CASATrainingGenerator"
        )

    def print_memory(self, label: str):
        """Print current memory usage"""
        self._generator.print_memory(label)

    def create_ms_with_rfi(
        self,
        ms_path: str,
        rfi_config: Optional[RFIConfig] = None,
        include_rfi_flags: bool = True,
        save_training_data: bool = False,
        training_data_dir: Optional[str] = None,
    ) -> None:
        """
        Create MS using CASA simulator + add RFI baseline-by-baseline
        
        Delegates to CASATrainingGenerator for backward compatibility.
        
        Args:
            ms_path: Output measurement set path
            rfi_config: RFI configuration for corruption (None = clean MS)
            include_rfi_flags: Whether to flag RFI-corrupted data
            save_training_data: Whether to save .npy files for training
            training_data_dir: Directory to save training .npy files
        """
        # Delegate to new CASATrainingGenerator
        return self._generator.create_training_ms(
            ms_path=ms_path,
            rfi_config=rfi_config,
            include_rfi_flags=include_rfi_flags,
            save_training_data=save_training_data,
            training_data_dir=training_data_dir
        )