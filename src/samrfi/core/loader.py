"""
Memory-Efficient Measurement Set Loader

Handles lazy loading, chunked processing, and memory management for large measurement sets.
Uses python-casacore for efficient data access.
"""

import os
import psutil
import numpy as np
from typing import Optional, List, Tuple, Dict, Iterator, Union
from pathlib import Path
import logging
from dataclasses import dataclass

CASACORE_AVAILABLE = False
pt = None

logger = logging.getLogger(__name__)


@dataclass
class MSMetadata:
    """Measurement set metadata"""

    ms_path: str
    num_antennas: int
    num_spw: int
    num_fields: int
    channels_per_spw: np.ndarray
    time_range: Tuple[float, float]
    baseline_count: int
    data_size_gb: float


class MemoryMonitor:
    """Monitor system memory usage"""

    @staticmethod
    def get_available_memory_gb() -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)

    @staticmethod
    def get_memory_usage_gb() -> float:
        """Get current process memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    @classmethod
    def calculate_safe_batch_size(
        cls,
        total_baselines: int,
        channels: int,
        time_steps: int,
        polarizations: int = 4,
        dtype_bytes: int = 8,
        safety_factor: float = 0.3,
    ) -> int:
        """
        Calculate safe batch size based on available memory

        Args:
            total_baselines: Total number of baselines
            channels: Number of frequency channels
            time_steps: Number of time steps
            polarizations: Number of polarizations (default: 4)
            dtype_bytes: Bytes per data element (8 for complex64)
            safety_factor: Fraction of available memory to use

        Returns:
            Safe number of baselines to load at once
        """
        available_gb = cls.get_available_memory_gb()
        safe_memory_gb = available_gb * safety_factor

        # Calculate memory per baseline in GB
        memory_per_baseline = (channels * time_steps * polarizations * dtype_bytes) / (
            1024**3
        )

        # Calculate safe batch size
        batch_size = max(1, int(safe_memory_gb / memory_per_baseline))
        batch_size = min(batch_size, total_baselines)

        logger.info(
            f"Memory: {available_gb:.1f}GB available, using {safe_memory_gb:.1f}GB"
        )
        logger.info(
            f"Batch size: {batch_size} baselines ({memory_per_baseline:.3f}GB each)"
        )

        return batch_size


class MSLoader:
    """
    Memory-efficient measurement set loader with chunked processing
    """

    def __init__(self, ms_path: str, work_dir: Optional[str] = None):
        """
        Initialize MS loader

        Args:
            ms_path: Path to measurement set
            work_dir: Working directory for cache/temp files
        """
        if not CASACORE_AVAILABLE:
            raise ImportError("python-casacore is required but not available")

        self.ms_path = Path(ms_path)
        if not self.ms_path.exists():
            raise FileNotFoundError(f"Measurement set not found: {ms_path}")

        # Setup working directory
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path.cwd() / "samrfi_data"
        self.work_dir.mkdir(exist_ok=True)

        # Initialize metadata
        self.metadata = self._scan_measurement_set()
        self.memory_monitor = MemoryMonitor()

        # Current state
        self._table = None
        self._current_selection = None

    def _scan_measurement_set(self) -> MSMetadata:
        """Scan measurement set to gather metadata"""
        logger.info(f"Scanning measurement set: {self.ms_path}")

        with pt.table(str(self.ms_path), readonly=True) as ms_table:
            # Antenna info
            with pt.table(str(self.ms_path / "ANTENNA"), readonly=True) as ant_table:
                num_antennas = ant_table.nrows()

            # Spectral window info
            with pt.table(
                str(self.ms_path / "SPECTRAL_WINDOW"), readonly=True
            ) as spw_table:
                num_spw = spw_table.nrows()
                channels_per_spw = spw_table.getcol("NUM_CHAN")

            # Field info
            with pt.table(str(self.ms_path / "FIELD"), readonly=True) as field_table:
                num_fields = field_table.nrows()

            # Time range
            times = ms_table.getcol("TIME")
            time_range = (times.min(), times.max())

            # Data size estimation
            sample_row = ms_table.getcol("DATA", startrow=0, nrow=1)
            data_shape = sample_row.shape
            total_rows = ms_table.nrows()
            data_size_gb = (total_rows * np.prod(data_shape) * 8) / (
                1024**3
            )  # Complex64

            baseline_count = num_antennas * (num_antennas - 1) // 2

        metadata = MSMetadata(
            ms_path=str(self.ms_path),
            num_antennas=num_antennas,
            num_spw=num_spw,
            num_fields=num_fields,
            channels_per_spw=channels_per_spw,
            time_range=time_range,
            baseline_count=baseline_count,
            data_size_gb=data_size_gb,
        )

        logger.info(
            f"MS Info: {num_antennas} antennas, {num_spw} SPWs, {data_size_gb:.1f}GB"
        )
        return metadata

    def open_table(self, readonly: bool = True) -> None:
        """Open measurement set table"""
        if self._table is None:
            self._table = pt.table(str(self.ms_path), readonly=readonly)
            logger.debug("Opened measurement set table")

    def close_table(self) -> None:
        """Close measurement set table"""
        if self._table is not None:
            self._table.close()
            self._table = None
            logger.debug("Closed measurement set table")

    def get_baseline_iterator(
        self,
        ant_limit: Optional[int] = None,
        spw_list: Optional[List[int]] = None,
        field_id: int = 0,
        batch_size: Optional[int] = None,
    ) -> Iterator[Dict]:
        """
        Get iterator for processing baselines in memory-efficient chunks

        Args:
            ant_limit: Limit number of antennas (for testing)
            spw_list: List of SPW IDs to process
            field_id: Field ID to process
            batch_size: Number of baselines per batch (auto if None)

        Yields:
            Dictionary with baseline data and metadata
        """
        self.open_table()

        # Filter SPWs with same channel count
        if spw_list is None:
            ref_channels = self.metadata.channels_per_spw[0]
            spw_list = [
                i
                for i, ch in enumerate(self.metadata.channels_per_spw)
                if ch == ref_channels
            ]

        num_antennas = ant_limit or self.metadata.num_antennas

        # Calculate batch size if not provided
        if batch_size is None:
            # Estimate data size per baseline
            ref_channels = self.metadata.channels_per_spw[0]
            # Get rough time estimate
            time_sample = self._table.query(
                f"ANTENNA1==0 && ANTENNA2==1 && DATA_DESC_ID==0"
            )
            time_steps = len(time_sample.getcol("TIME"))
            time_sample.close()

            batch_size = self.memory_monitor.calculate_safe_batch_size(
                total_baselines=(num_antennas * (num_antennas - 1)) // 2,
                channels=len(spw_list) * ref_channels,
                time_steps=time_steps,
            )

        logger.info(
            f"Processing {num_antennas} antennas in batches of {batch_size} baselines"
        )

        # Generate baselines
        baselines = [
            (i, j) for i in range(num_antennas) for j in range(i + 1, num_antennas)
        ]

        # Process in batches
        for batch_start in range(0, len(baselines), batch_size):
            batch_end = min(batch_start + batch_size, len(baselines))
            batch_baselines = baselines[batch_start:batch_end]

            logger.info(
                f"Loading batch {batch_start//batch_size + 1}: "
                f"baselines {batch_start}-{batch_end-1}"
            )

            batch_data = self._load_baseline_batch(batch_baselines, spw_list, field_id)

            yield {
                "batch_id": batch_start // batch_size,
                "baselines": batch_baselines,
                "spw_list": spw_list,
                "data": batch_data,
                "metadata": {
                    "channels_per_spw": self.metadata.channels_per_spw[spw_list],
                    "time_steps": batch_data.shape[-1] if len(batch_data) > 0 else 0,
                    "memory_usage_gb": self.memory_monitor.get_memory_usage_gb(),
                },
            }

    def _load_baseline_batch(
        self, baselines: List[Tuple[int, int]], spw_list: List[int], field_id: int
    ) -> np.ndarray:
        """
        Load a batch of baselines efficiently

        Args:
            baselines: List of (ant1, ant2) tuples
            spw_list: List of SPW IDs
            field_id: Field ID

        Returns:
            Data array [baselines, polarizations, channels, time]
        """
        if not baselines:
            return np.array([])

        ref_channels = self.metadata.channels_per_spw[spw_list[0]]
        total_channels = len(spw_list) * ref_channels

        batch_data = []

        for ant1, ant2 in baselines:
            # Initialize combined data for this baseline
            combined_data = None

            for spw_idx, spw_id in enumerate(spw_list):
                # Query for this baseline and SPW
                query = f"ANTENNA1=={ant1} && ANTENNA2=={ant2} && DATA_DESC_ID=={spw_id} && FIELD_ID=={field_id}"
                subtable = self._table.query(query)

                if subtable.nrows() > 0:
                    spw_data = subtable.getcol("DATA")  # [pol, chan, time]

                    if combined_data is None:
                        # Initialize combined array on first SPW
                        time_steps = spw_data.shape[2]
                        combined_data = np.zeros(
                            (4, total_channels, time_steps), dtype=np.complex64
                        )

                    # Place SPW data in combined array
                    start_chan = spw_idx * ref_channels
                    end_chan = (spw_idx + 1) * ref_channels
                    combined_data[:, start_chan:end_chan, :] = spw_data

                subtable.close()

            if combined_data is not None:
                batch_data.append(combined_data)
            else:
                # Create empty data if no data found
                logger.warning(f"No data found for baseline {ant1}-{ant2}")
                batch_data.append(np.zeros((4, total_channels, 1), dtype=np.complex64))

        return np.stack(batch_data) if batch_data else np.array([])

    def load_flags_batch(
        self, baselines: List[Tuple[int, int]], spw_list: List[int], field_id: int = 0
    ) -> np.ndarray:
        """
        Load flags for a batch of baselines

        Args:
            baselines: List of (ant1, ant2) tuples
            spw_list: List of SPW IDs
            field_id: Field ID

        Returns:
            Flag array [baselines, polarizations, channels, time]
        """
        self.open_table()

        ref_channels = self.metadata.channels_per_spw[spw_list[0]]
        total_channels = len(spw_list) * ref_channels

        batch_flags = []

        for ant1, ant2 in baselines:
            combined_flags = None

            for spw_idx, spw_id in enumerate(spw_list):
                query = f"ANTENNA1=={ant1} && ANTENNA2=={ant2} && DATA_DESC_ID=={spw_id} && FIELD_ID=={field_id}"
                subtable = self._table.query(query)

                if subtable.nrows() > 0:
                    spw_flags = subtable.getcol("FLAG")

                    if combined_flags is None:
                        time_steps = spw_flags.shape[2]
                        combined_flags = np.zeros(
                            (4, total_channels, time_steps), dtype=bool
                        )

                    start_chan = spw_idx * ref_channels
                    end_chan = (spw_idx + 1) * ref_channels
                    combined_flags[:, start_chan:end_chan, :] = spw_flags

                subtable.close()

            if combined_flags is not None:
                batch_flags.append(combined_flags)
            else:
                batch_flags.append(np.zeros((4, total_channels, 1), dtype=bool))

        return np.stack(batch_flags) if batch_flags else np.array([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_table()
