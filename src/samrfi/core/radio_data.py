"""
Measurement Set Data Loading and Management

Modern implementation of RadioRFI functionality with support for:
- Measurement set loading and processing
- Incremental flagging operations
- CASA table integration
- Multi-antenna/baseline data handling
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
import logging
from tqdm import tqdm

CASA_AVAILABLE = False
casacore_tables = None
table = None

logger = logging.getLogger(__name__)


class RadioData:
    """
    Modern measurement set data loader with incremental flagging support

    Features:
    - Efficient measurement set loading
    - Incremental flagging (preserves existing flags)
    - Multi-baseline processing
    - Spectral window handling
    - Memory-efficient data access
    """

    def __init__(self, ms_path: Optional[str] = None, work_dir: Optional[str] = None):
        """
        Initialize RadioData loader

        Args:
            ms_path: Path to measurement set
            work_dir: Working directory for outputs (default: current + 'samrfi_data')
        """
        if not CASA_AVAILABLE:
            raise ImportError("python-casacore is required but not available")

        # Core data storage
        self.ms_path = ms_path
        self.rfi_antenna_data = None
        self.flags = None
        self.ms_flags = None  # Original flags from MS

        # MS metadata
        self.num_antennas = None
        self.num_spw = None
        self.channels_per_spw = None
        self.antenna_baseline_map = None
        self.spw = None
        self.time_steps = None

        # CASA table handles
        self.tb = None
        self.tb_antenna = None
        self.tb_spw = None

        # Working directory setup
        if work_dir:
            work_dir = Path(work_dir)
        else:
            work_dir = Path.cwd() / "samrfi_data"

        work_dir.mkdir(exist_ok=True)
        self.work_dir = work_dir

        # Create logs directory
        logs_dir = work_dir.parent / "casalogs"
        logs_dir.mkdir(exist_ok=True)

        if ms_path:
            self._open_measurement_set(ms_path)

    def _open_measurement_set(self, ms_path: str) -> None:
        """Open measurement set and read metadata"""
        ms_path = Path(ms_path)
        if not ms_path.exists():
            raise FileNotFoundError(f"Measurement set not found: {ms_path}")

        self.ms_path = str(ms_path)
        logger.info(f"Opening measurement set: {self.ms_path}")

        # Read antenna table
        self.tb_antenna = table()
        self.tb_antenna.open(f"{self.ms_path}/ANTENNA")
        self.num_antennas = self.tb_antenna.nrows()
        self.tb_antenna.close()

        # Read spectral window table
        self.tb_spw = table()
        self.tb_spw.open(f"{self.ms_path}/SPECTRAL_WINDOW")
        self.num_spw = self.tb_spw.nrows()
        self.channels_per_spw = self.tb_spw.getcol("NUM_CHAN")
        self.tb_spw.close()

        # Open main table
        self.tb = table()
        self.tb.open(self.ms_path, nomodify=False)

        logger.info(f"MS Info: {self.num_antennas} antennas, {self.num_spw} SPWs")
        logger.info(f"Channels per SPW: {self.channels_per_spw}")

    def load_data(self, mode: str = "DATA", ant_i: Optional[int] = None) -> np.ndarray:
        """
        Load visibility data from measurement set (matches legacy RadioRFI.load interface)

        Args:
            mode: Column to load ('DATA', 'CORRECTED_DATA', 'FLAG')
            ant_i: Number of antennas to process (matches legacy interface)

        Returns:
            Loaded data array [baselines, polarizations, channels, time]
        """
        if not self.tb:
            raise RuntimeError("No measurement set opened")

        # Get time steps from first baseline
        test_query = self.tb.query(
            f"DATA_DESC_ID=={0} && ANTENNA1=={0} && ANTENNA2=={1}"
        )
        self.time_steps = len(test_query.getcol("TIME"))
        test_query.close()

        # Filter SPWs with same channel count (legacy behavior)
        spw_array = list(range(self.num_spw))
        same_spw_array = []
        same_channels_per_spw_array = []

        reference_channels = self.channels_per_spw[0]
        for spw, spw_numchan in zip(spw_array, self.channels_per_spw):
            if spw_numchan == reference_channels:
                same_spw_array.append(spw)
                same_channels_per_spw_array.append(spw_numchan)

        self.spw = same_spw_array
        init_chan = same_channels_per_spw_array[0]
        same_num_spw = len(same_spw_array)

        # Set antenna range (legacy behavior)
        num_antennas_to_process = ant_i if ant_i else self.num_antennas
        self.num_antennas_i = num_antennas_to_process

        logger.info(
            f"Loading {mode} data for {num_antennas_to_process} antennas, {same_num_spw} SPWs"
        )
        print(f"\nLoading data...")

        rfi_list = []
        antenna_baseline_map = []

        # Process all baselines (matches legacy nested loop structure)
        for i in tqdm(range(num_antennas_to_process)):
            for j in tqdm(range(i + 1, self.num_antennas), leave=False):
                # Initialize combined data array [pols, channels, time]
                data_type = "complex128" if mode != "FLAG" else "bool"
                combined_data = np.zeros(
                    [4, same_num_spw * init_chan, self.time_steps], dtype=data_type
                )

                # Load data for each SPW and combine
                for spw_spec, spw, num_chan in zip(
                    same_spw_array, range(same_num_spw), same_channels_per_spw_array
                ):
                    subtable = self.tb.query(
                        f"DATA_DESC_ID=={spw_spec} && ANTENNA1=={i} && ANTENNA2=={j}"
                    )

                    if subtable.nrows() > 0:
                        spw_data = subtable.getcol(mode)
                        # Place SPW data in combined array
                        combined_data[
                            :, spw * init_chan : (spw + 1) * init_chan, :
                        ] += spw_data

                    subtable.close()

                rfi_list.append(combined_data)
                antenna_baseline_map.append((i, j))

        self.antenna_baseline_map = antenna_baseline_map
        self.channels_per_spw = same_channels_per_spw_array

        # Store data based on mode (matches legacy behavior)
        if mode == "DATA":
            self.rfi_antenna_data_complex = np.stack(rfi_list)
            self.rfi_antenna_data = np.abs(self.rfi_antenna_data_complex)
            print(f"Data shape: {self.rfi_antenna_data.shape}")
            logger.info(f"Loaded data shape: {self.rfi_antenna_data.shape}")
            return self.rfi_antenna_data
        elif mode == "FLAG":
            self.ms_flags = np.stack(rfi_list)
            print(f"Flags shape: {self.ms_flags.shape}")
            logger.info(f"Loaded flags shape: {self.ms_flags.shape}")
            return self.ms_flags
        else:
            data_array = np.stack(rfi_list)
            print(f"{mode} shape: {data_array.shape}")
            logger.info(f"Loaded {mode} shape: {data_array.shape}")
            return data_array

    def update_flags(self, new_flags: np.ndarray, mode: str = "replace") -> None:
        """
        Update internal flags with incremental support

        Args:
            new_flags: New flag array [baselines, pols, channels, time]
            mode: 'replace' or 'combine' (OR with existing flags)
        """
        if mode == "combine" and self.flags is not None:
            # Incremental flagging: combine with existing flags
            self.flags = np.logical_or(self.flags, new_flags)
            logger.info("Combined new flags with existing flags (incremental)")
        else:
            # Replace existing flags
            self.flags = new_flags.copy()
            logger.info("Replaced flags")

        logger.info(
            f"Flag statistics: {np.sum(self.flags)}/{self.flags.size} flagged "
            f"({100*np.sum(self.flags)/self.flags.size:.1f}%)"
        )

    def save_flags(self, backup: bool = True) -> None:
        """
        Save flags back to measurement set

        Args:
            backup: Whether to backup original flags first
        """
        if self.flags is None:
            raise RuntimeError("No flags to save")

        logger.info("Saving flags to measurement set...")

        # Backup original flags if requested
        if backup and self.ms_flags is None:
            logger.info("Backing up original flags...")
            self.load_data(mode="FLAG")

        # Write flags for each baseline and SPW
        for baseline_idx, (ant1, ant2) in enumerate(
            tqdm(self.antenna_baseline_map, desc="Saving flags")
        ):
            baseline_flags = self.flags[baseline_idx, :, :, :]

            for spw_idx, spw_id in enumerate(self.spw):
                # Extract flags for this SPW
                start_chan = spw_idx * self.channels_per_spw[0]
                end_chan = (spw_idx + 1) * self.channels_per_spw[0]
                spw_flags = baseline_flags[:, start_chan:end_chan, :]

                # Update measurement set
                subtable = self.tb.query(
                    f"DATA_DESC_ID=={spw_id} && ANTENNA1=={ant1} && ANTENNA2=={ant2}"
                )
                if subtable.nrows() > 0:
                    subtable.putcol("FLAG", spw_flags)
                subtable.close()

        logger.info("Flags saved successfully")

    def restore_flags(self) -> None:
        """Restore original flags from backup"""
        if self.ms_flags is None:
            raise RuntimeError("No backup flags available")

        self.flags = self.ms_flags.copy()
        self.save_flags(backup=False)
        logger.info("Restored original flags")

    def get_waterfall_data(
        self, baseline: int = 0, polarization: int = 0
    ) -> np.ndarray:
        """
        Get waterfall plot data for visualization

        Args:
            baseline: Baseline index
            polarization: Polarization index (0=XX, 1=XY, 2=YX, 3=YY)

        Returns:
            Waterfall data [channels, time]
        """
        if self.rfi_antenna_data is None:
            raise RuntimeError("No data loaded")

        return self.rfi_antenna_data[baseline, polarization, :, :]

    def get_flag_statistics(self) -> Dict[str, float]:
        """Get comprehensive flag statistics"""
        if self.flags is None:
            return {"error": "No flags available"}

        total_points = self.flags.size
        flagged_points = np.sum(self.flags)

        # Per-baseline statistics
        baseline_stats = []
        for i, (ant1, ant2) in enumerate(self.antenna_baseline_map):
            baseline_flags = self.flags[i]
            baseline_total = baseline_flags.size
            baseline_flagged = np.sum(baseline_flags)
            baseline_stats.append(
                {
                    "baseline": f"{ant1}-{ant2}",
                    "flagged_fraction": baseline_flagged / baseline_total,
                }
            )

        return {
            "total_points": int(total_points),
            "flagged_points": int(flagged_points),
            "flagged_fraction": flagged_points / total_points,
            "baseline_statistics": baseline_stats,
        }

    def close(self) -> None:
        """Close measurement set connection"""
        if self.tb:
            self.tb.close()
            self.tb = None
        logger.info("Closed measurement set")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
