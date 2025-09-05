"""
Synthetic Measurement Set Generator

Creates realistic measurement sets with controllable RFI for training and benchmarking.
Produces proper CASA-compatible MS format that can be read by aoflagger, CASA flaggers, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Removed python-casacore dependency - now using CASA tools/tasks in ms_writer.py

logger = logging.getLogger(__name__)


@dataclass
class ObservationConfig:
    """Configuration for synthetic observation"""

    # Array configuration
    num_antennas: int = 8
    antenna_diameter: float = 25.0  # meters
    array_name: str = "SYNTHETIC_VLA"

    # Frequency setup
    num_spw: int = 4
    channels_per_spw: int = 256
    start_frequency: float = 1.0e9  # Hz (L-band)
    channel_width: float = 1e6  # Hz

    # Time setup
    start_time: str = "2024-01-01T00:00:00"
    integration_time: float = 10.0  # seconds
    total_duration: float = 3600.0  # seconds (1 hour)

    # Source/field configuration
    source_name: str = "SYNTHETIC_SOURCE"
    source_ra: float = 0.0  # radians
    source_dec: float = 0.7854  # radians (45 degrees)

    # Data configuration
    polarizations: List[str] = None  # ['XX', 'XY', 'YX', 'YY']

    def __post_init__(self):
        if self.polarizations is None:
            self.polarizations = ["XX", "XY", "YX", "YY"]


@dataclass
class RFIConfig:
    """Configuration for RFI injection"""

    # Broadband RFI
    broadband_probability: float = 0.02  # Fraction of time-freq cells
    broadband_amplitude_range: Tuple[float, float] = (5.0, 20.0)  # Times noise

    # Narrowband RFI
    narrowband_lines: int = 5  # Number of persistent frequency lines
    narrowband_amplitude_range: Tuple[float, float] = (10.0, 50.0)
    narrowband_width_range: Tuple[int, int] = (1, 5)  # Channels

    # Transient RFI
    transient_events: int = 10  # Number of transient bursts
    transient_duration_range: Tuple[float, float] = (10.0, 60.0)  # Seconds
    transient_amplitude_range: Tuple[float, float] = (20.0, 100.0)

    # Periodic RFI
    periodic_signals: int = 3  # Number of periodic interferers
    periodic_period_range: Tuple[float, float] = (60.0, 300.0)  # Seconds
    periodic_duty_cycle_range: Tuple[float, float] = (0.1, 0.3)  # Fraction on

    # Satellite RFI
    satellite_passes: int = 2  # Number of satellite passes
    satellite_duration_range: Tuple[float, float] = (300.0, 600.0)  # Seconds
    satellite_drift_rate: float = 1e3  # Hz/second frequency drift


class SyntheticVisibilityGenerator:
    """Generate realistic synthetic visibility data"""

    def __init__(self, obs_config: ObservationConfig, rfi_config: RFIConfig):
        self.obs_config = obs_config
        self.rfi_config = rfi_config

        # Calculate derived parameters
        self.num_times = int(obs_config.total_duration / obs_config.integration_time)
        self.num_baselines = (
            obs_config.num_antennas * (obs_config.num_antennas - 1) // 2
        )
        self.total_channels = obs_config.num_spw * obs_config.channels_per_spw

        # Generate antenna positions (rough VLA-like array)
        self.antenna_positions = self._generate_antenna_positions()

        # Generate frequency and time grids
        self.frequencies = self._generate_frequency_grid()
        self.times = self._generate_time_grid()

        logger.info(
            f"Synthetic MS config: {obs_config.num_antennas} antennas, "
            f"{self.num_times} times, {self.total_channels} channels"
        )

    def _generate_antenna_positions(self) -> np.ndarray:
        """Generate realistic antenna positions"""
        # Simple spiral array pattern
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

    def _generate_frequency_grid(self) -> np.ndarray:
        """Generate frequency grid for all SPWs"""
        frequencies = []

        for spw in range(self.obs_config.num_spw):
            spw_start = (
                self.obs_config.start_frequency
                + spw * self.obs_config.channels_per_spw * self.obs_config.channel_width
            )

            spw_freqs = np.linspace(
                spw_start,
                spw_start
                + self.obs_config.channels_per_spw * self.obs_config.channel_width,
                self.obs_config.channels_per_spw,
            )
            frequencies.extend(spw_freqs)

        return np.array(frequencies)

    def _generate_time_grid(self) -> np.ndarray:
        """Generate time grid"""
        start_time = datetime.fromisoformat(
            self.obs_config.start_time.replace("T", " ")
        )
        times = []

        for i in range(self.num_times):
            time = start_time + timedelta(seconds=i * self.obs_config.integration_time)
            # Convert to MJD seconds
            mjd_seconds = (time - datetime(1858, 11, 17)).total_seconds()
            times.append(mjd_seconds)

        return np.array(times)

    def generate_clean_visibilities(self) -> np.ndarray:
        """
        Generate clean (RFI-free) synthetic visibilities

        Returns:
            Complex visibility data [baseline, time, freq, pol]
        """
        logger.info("Generating clean synthetic visibilities...")

        # Initialize visibility array
        vis_shape = (self.num_baselines, self.num_times, self.total_channels, 4)
        visibilities = np.zeros(vis_shape, dtype=np.complex64)

        # Generate baseline mapping
        baseline_idx = 0
        total_baselines = self.num_baselines
        logger.info(f"Generating data for {total_baselines} baselines x {self.num_times} times x {self.total_channels} channels...")
        
        for ant1 in range(self.obs_config.num_antennas):
            for ant2 in range(ant1 + 1, self.obs_config.num_antennas):
                # Progress reporting every 50 baselines
                if baseline_idx % 50 == 0:
                    progress = (baseline_idx / total_baselines) * 100
                    logger.info(f"  Processing baseline {baseline_idx}/{total_baselines} ({progress:.1f}%)")
                
                # Simple point source model
                # In real interferometry, this would include proper uv-plane sampling

                # Baseline vector
                baseline_vec = (
                    self.antenna_positions[ant2] - self.antenna_positions[ant1]
                )
                baseline_length = np.linalg.norm(
                    baseline_vec[:2]
                )  # Ignore z for simplicity

                # Vectorized generation for this baseline (much faster)
                noise_level = 0.1
                
                # Generate noise for all times/frequencies at once
                real_noise = np.random.normal(0, noise_level, (self.num_times, self.total_channels, 4))
                imag_noise = np.random.normal(0, noise_level, (self.num_times, self.total_channels, 4))
                
                # Simple source model (adjust amplitude by baseline length)
                source_amplitude = 1.0 / (1.0 + baseline_length / 1000.0)
                
                # Add polarization structure
                pol_amplitudes = np.array([
                    source_amplitude,
                    0.1 * source_amplitude, 
                    0.1 * source_amplitude,
                    source_amplitude,
                ])
                
                # Broadcast source model to all times/frequencies
                visibilities[baseline_idx, :, :, :] = pol_amplitudes + real_noise + 1j * imag_noise

                baseline_idx += 1

        logger.info("Clean visibilities generated")
        return visibilities

    def inject_rfi(
        self, clean_visibilities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject various types of RFI into clean visibilities

        Args:
            clean_visibilities: Clean visibility data

        Returns:
            Tuple of (corrupted_visibilities, rfi_mask)
        """
        logger.info("Injecting RFI patterns...")
        logger.info(f"  Data shape: {clean_visibilities.shape}")
        logger.info(f"  RFI types: broadband, narrowband, transient, periodic, satellite")

        corrupted_vis = clean_visibilities.copy()
        rfi_mask = np.zeros(clean_visibilities.shape, dtype=bool)

        # 1. Broadband RFI (random cells) - vectorized
        logger.info("  Adding broadband RFI...")
        broadband_mask = (
            np.random.random(corrupted_vis.shape)
            < self.rfi_config.broadband_probability
        )
        # Only generate complex values where needed
        affected_cells = np.sum(broadband_mask)
        if affected_cells > 0:
            amplitudes = np.random.uniform(
                *self.rfi_config.broadband_amplitude_range, size=affected_cells
            )
            complex_vals = (amplitudes * 
                           (np.random.random(affected_cells) + 1j * np.random.random(affected_cells)))
            corrupted_vis[broadband_mask] += complex_vals
        rfi_mask |= broadband_mask

        # 2. Narrowband persistent RFI - optimized
        logger.info(f"  Adding {self.rfi_config.narrowband_lines} narrowband lines...")
        for i in range(self.rfi_config.narrowband_lines):
            if i % 2 == 0 or i == self.rfi_config.narrowband_lines - 1:
                logger.info(f"    Narrowband line {i+1}/{self.rfi_config.narrowband_lines}")
            # Random frequency channel range
            width = np.random.randint(*self.rfi_config.narrowband_width_range)
            start_chan = np.random.randint(0, self.total_channels - width)
            end_chan = start_chan + width

            amplitude = np.random.uniform(*self.rfi_config.narrowband_amplitude_range)

            # Only generate random values for affected channels
            affected_shape = (corrupted_vis.shape[0], corrupted_vis.shape[1], width, corrupted_vis.shape[3])
            complex_vals = amplitude * (np.random.random(affected_shape) + 1j * np.random.random(affected_shape))
            
            # Apply directly to affected channels
            corrupted_vis[:, :, start_chan:end_chan, :] += complex_vals
            rfi_mask[:, :, start_chan:end_chan, :] = True

        # 3. Transient RFI events - optimized
        logger.info(f"  Adding {self.rfi_config.transient_events} transient events...")
        for i in range(self.rfi_config.transient_events):
            if i % 4 == 0 or i == self.rfi_config.transient_events - 1:
                logger.info(f"    Transient event {i+1}/{self.rfi_config.transient_events}")
            # Random time range
            duration_samples = int(
                np.random.uniform(*self.rfi_config.transient_duration_range)
                / self.obs_config.integration_time
            )
            start_time = np.random.randint(0, max(1, self.num_times - duration_samples))
            end_time = start_time + duration_samples

            # Random frequency range
            freq_width = np.random.randint(50, 200)  # channels
            start_freq = np.random.randint(0, max(1, self.total_channels - freq_width))
            end_freq = start_freq + freq_width

            amplitude = np.random.uniform(*self.rfi_config.transient_amplitude_range)

            # Only generate random values for affected region
            affected_shape = (corrupted_vis.shape[0], duration_samples, freq_width, corrupted_vis.shape[3])
            complex_vals = amplitude * (np.random.random(affected_shape) + 1j * np.random.random(affected_shape))
            
            # Apply directly to affected region
            corrupted_vis[:, start_time:end_time, start_freq:end_freq, :] += complex_vals
            rfi_mask[:, start_time:end_time, start_freq:end_freq, :] = True

        # 4. Periodic RFI
        logger.info(f"  Adding {self.rfi_config.periodic_signals} periodic signals...")
        for i in range(self.rfi_config.periodic_signals):
            logger.info(f"    Periodic signal {i+1}/{self.rfi_config.periodic_signals}")
            period = np.random.uniform(*self.rfi_config.periodic_period_range)
            duty_cycle = np.random.uniform(*self.rfi_config.periodic_duty_cycle_range)

            period_samples = int(period / self.obs_config.integration_time)
            on_samples = int(period_samples * duty_cycle)

            amplitude = np.random.uniform(*self.rfi_config.broadband_amplitude_range)

            # Create periodic pattern
            periodic_pattern = np.zeros(self.num_times)
            for t in range(0, self.num_times, period_samples):
                end_on = min(t + on_samples, self.num_times)
                periodic_pattern[t:end_on] = 1.0

            # Apply to random frequency range
            freq_range = min(np.random.randint(20, 100), self.total_channels - 1)
            start_freq = np.random.randint(0, max(1, self.total_channels - freq_range))

            for baseline_idx in range(self.num_baselines):
                for freq_idx in range(start_freq, start_freq + freq_range):
                    mask = periodic_pattern > 0
                    corrupted_vis[baseline_idx, mask, freq_idx, :] += amplitude * (
                        np.random.random(4) + 1j * np.random.random(4)
                    )
                    rfi_mask[baseline_idx, mask, freq_idx, :] = True

        # 5. Satellite RFI (frequency drifting)
        logger.info(f"  Adding {self.rfi_config.satellite_passes} satellite passes...")
        for i in range(self.rfi_config.satellite_passes):
            logger.info(f"    Satellite pass {i+1}/{self.rfi_config.satellite_passes}")
            duration = np.random.uniform(*self.rfi_config.satellite_duration_range)
            duration_samples = int(duration / self.obs_config.integration_time)
            start_time = np.random.randint(0, max(1, self.num_times - duration_samples))

            # Drifting frequency
            start_freq_idx = np.random.randint(10, max(11, self.total_channels - 10))
            drift_rate = (
                self.rfi_config.satellite_drift_rate / self.obs_config.channel_width
            )  # channels per second

            amplitude = np.random.uniform(*self.rfi_config.transient_amplitude_range)

            for t in range(duration_samples):
                if start_time + t >= self.num_times:
                    break

                # Calculate current frequency based on drift
                current_freq_idx = int(
                    start_freq_idx + drift_rate * t * self.obs_config.integration_time
                )
                if 0 <= current_freq_idx < self.total_channels:
                    # Satellite affects all baselines at this time/freq
                    corrupted_vis[
                        :, start_time + t, current_freq_idx, :
                    ] += amplitude * (np.random.random(4) + 1j * np.random.random(4))
                    rfi_mask[:, start_time + t, current_freq_idx, :] = True

        logger.info(
            f"RFI injection complete. {np.sum(rfi_mask) / rfi_mask.size * 100:.1f}% data flagged"
        )

        return corrupted_vis, rfi_mask
