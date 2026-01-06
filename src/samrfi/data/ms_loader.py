"""
CASA Measurement Set Loader for RFI Analysis.

This module provides simplified data loading from CASA measurement sets,
extracting complex visibilities and flags for radio frequency interference
(RFI) detection and analysis.

Classes
-------
MSLoader
    Load and manipulate complex visibilities from CASA measurement sets.

Examples
--------
Load measurement set and extract visibilities:

>>> from samrfi.data import MSLoader
>>> loader = MSLoader('observation.ms')
>>> data = loader.load(num_antennas=5, mode='DATA')
>>> print(data.shape)
(10, 4, 1024, 60)  # (baselines, pols, channels, times)

Load and update flags:

>>> flags = loader.load_flags()
>>> # ... modify flags ...
>>> loader.save_flags(flags)
>>> loader.close()

Notes
-----
Requires CASA installation. Install with:
    pip install samrfi[casa]

See also: https://casadocs.readthedocs.io/
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

try:
    from casatools import table
except Exception as e:
    raise ImportError(
        "MSLoader requires CASA to be properly installed and configured.\n"
        "Install with: pip install samrfi[casa]\n"
        "See: https://casadocs.readthedocs.io/\n"
        f"Original error: {e}"
    ) from e


class MSLoader:
    """
    Load complex visibilities from CASA measurement sets.

    Provides clean interface for loading data, flags, and metadata from
    CASA measurement sets (MS) for RFI analysis. Handles multiple spectral
    windows (SPWs), baselines, polarizations, and time samples.

    Parameters
    ----------
    ms_path : str or Path
        Path to CASA measurement set directory.

    Attributes
    ----------
    ms_path : str
        Path to measurement set.
    num_antennas : int
        Total number of antennas in the measurement set.
    num_spw : int
        Number of spectral windows.
    channels_per_spw : NDArray[np.int32]
        Number of channels in each spectral window.
    num_times : int
        Number of time samples.
    data : NDArray[np.complex128] or None
        Loaded visibility data, shape (baselines, pols, channels, times).
    flags : NDArray[np.bool_] or None
        Loaded flag data, same shape as data.
    antenna_baseline_map : List[Tuple[int, int]] or None
        List of (antenna1, antenna2) pairs for each loaded baseline.
    spw_list : List[int] or None
        List of spectral window indices that were loaded.
    tb : table
        CASA table object for the main measurement set table.

    Examples
    --------
    Basic usage:

    >>> loader = MSLoader('observation.ms')
    >>> data = loader.load(num_antennas=5, mode='DATA')
    >>> print(data.shape)
    (10, 4, 1024, 60)  # (baselines, pols, channels, times)

    Load single baseline:

    >>> baseline_data = loader.load_single_baseline(ant1=0, ant2=1, pol_idx=0)
    >>> print(baseline_data.shape)
    (1024, 60)  # (channels, times)

    Work with flags:

    >>> flags = loader.load_flags()
    >>> # Modify flags...
    >>> loader.save_flags(flags)
    >>> loader.close()

    Access magnitude:

    >>> magnitude = loader.magnitude  # Compute from complex data
    """

    def __init__(self, ms_path: str | Path) -> None:
        """
        Initialize MS loader and read metadata.

        Parameters
        ----------
        ms_path : str or Path
            Path to CASA measurement set directory.

        Raises
        ------
        FileNotFoundError
            If measurement set path does not exist.
        RuntimeError
            If CASA table operations fail.
        """
        self.ms_path = str(ms_path)

        # Open MS and read metadata
        tb = table()

        # Number of antennas
        tb.open(self.ms_path + "/ANTENNA")
        self.num_antennas = tb.nrows()
        tb.close()

        # Number of spectral windows and channels
        tb.open(self.ms_path + "/SPECTRAL_WINDOW")
        self.num_spw = tb.nrows()
        self.channels_per_spw = tb.getcol("NUM_CHAN")
        tb.close()

        # Main table
        self.tb = table()
        self.tb.open(self.ms_path, nomodify=False)

        # Get number of time samples
        subtable = self.tb.query("DATA_DESC_ID==0 && ANTENNA1==0 && ANTENNA2==1")
        self.num_times = len(subtable.getcol("TIME"))
        subtable.close()

        # Storage for loaded data
        self.data = None
        self.flags = None
        self.antenna_baseline_map = None
        self.spw_list = None

    def load(
        self, num_antennas: Optional[int] = None, mode: str = "DATA"
    ) -> NDArray[np.complex128]:
        """
        Load complex visibilities from measurement set.

        Loads visibility data for specified antennas across all spectral windows
        that have matching channel counts. Combines multiple SPWs into a single
        frequency axis.

        Parameters
        ----------
        num_antennas : int, optional
            Number of antennas to load from the measurement set. If None,
            loads all antennas. Default is None.
        mode : str, default='DATA'
            Name of the data column to load. Common options:
            - 'DATA': Raw visibility data
            - 'CORRECTED_DATA': Calibrated visibility data
            - 'MODEL_DATA': Model visibility data

        Returns
        -------
        NDArray[np.complex128]
            Complex visibility data with shape (num_baselines, num_pols,
            num_channels, num_times). The data is stored in the `self.data`
            attribute and also returned.

        Raises
        ------
        ValueError
            If specified data column does not exist in measurement set.

        Notes
        -----
        - Only loads spectral windows with matching channel counts
        - Number of baselines = num_antennas * (num_antennas - 1) / 2
        - Polarizations are typically [XX, XY, YX, YY] for full-pol data
        - Updates `self.antenna_baseline_map` with loaded baseline pairs

        Examples
        --------
        >>> loader = MSLoader('observation.ms')
        >>> data = loader.load(num_antennas=10, mode='DATA')
        >>> print(f"Loaded {data.shape[0]} baselines")
        Loaded 45 baselines
        """
        if num_antennas is None:
            num_antennas = self.num_antennas

        # Filter to SPWs with same number of channels
        same_spw_list = []
        same_channels_list = []

        for spw, num_chan in enumerate(self.channels_per_spw):
            if num_chan == self.channels_per_spw[0]:
                same_spw_list.append(spw)
                same_channels_list.append(num_chan)

        num_channels = same_channels_list[0]
        num_spw = len(same_spw_list)
        total_channels = num_spw * num_channels

        # Load baselines
        data_list = []
        baseline_map = []

        print(f"\nLoading {mode} from {self.ms_path}...")
        print(f"  Antennas: {num_antennas}/{self.num_antennas}")
        print(f"  SPWs: {num_spw} ({num_channels} channels each = {total_channels} total)")
        print(f"  Times: {self.num_times}")

        for i in tqdm(range(num_antennas), desc="Antenna 1"):
            for j in range(i + 1, self.num_antennas):
                # Allocate array for this baseline
                baseline_data = np.zeros([4, total_channels, self.num_times], dtype="complex128")

                # Check if this baseline has any data
                has_data = False

                # Load all SPWs for this baseline
                for spw_idx, spw in enumerate(same_spw_list):
                    subtable = self.tb.query(
                        f"DATA_DESC_ID=={spw} && ANTENNA1=={i} && ANTENNA2=={j}"
                    )

                    # Skip if no data for this baseline/SPW
                    if subtable.nrows() == 0:
                        subtable.close()
                        continue

                    has_data = True

                    # Extract data for this SPW
                    spw_data = subtable.getcol(mode)

                    # Place in combined array
                    start_ch = spw_idx * num_channels
                    end_ch = (spw_idx + 1) * num_channels
                    baseline_data[:, start_ch:end_ch, :] = spw_data

                    subtable.close()

                # Only add baseline if it has data
                if has_data:
                    data_list.append(baseline_data)
                    baseline_map.append((i, j))

        # Stack all baselines
        self.data = np.stack(data_list)  # Shape: (baselines, pols, channels, times)
        self.antenna_baseline_map = baseline_map
        self.spw_list = same_spw_list
        self.channels_per_spw_list = same_channels_list

        print(f"  Loaded shape: {self.data.shape}")

        return self.data

    def load_single_baseline(
        self, ant1: int = 0, ant2: int = 1, pol_idx: int = 0, mode: str = "DATA"
    ) -> NDArray[np.complex128]:
        """
        Load single baseline and single polarization.

        Convenience method for loading data from one antenna pair and one
        polarization. Useful for quick inspection or testing.

        Parameters
        ----------
        ant1 : int, default=0
            Index of first antenna in baseline.
        ant2 : int, default=1
            Index of second antenna in baseline.
        pol_idx : int, default=0
            Polarization index to load:
            - 0: XX (horizontal-horizontal)
            - 1: XY (horizontal-vertical)
            - 2: YX (vertical-horizontal)
            - 3: YY (vertical-vertical)
        mode : str, default='DATA'
            Name of the data column to load ('DATA', 'CORRECTED_DATA', etc.).

        Returns
        -------
        NDArray[np.complex128]
            Complex visibility data with shape (total_channels, num_times).

        Raises
        ------
        ValueError
            If no data exists for the specified baseline or if antenna
            indices are invalid.

        Examples
        --------
        >>> loader = MSLoader('observation.ms')
        >>> baseline = loader.load_single_baseline(ant1=0, ant2=1, pol_idx=0)
        >>> print(baseline.shape)
        (1024, 60)  # (channels, times)
        """
        # Filter to SPWs with same number of channels
        same_spw_list = []
        same_channels_list = []

        for spw, num_chan in enumerate(self.channels_per_spw):
            if num_chan == self.channels_per_spw[0]:
                same_spw_list.append(spw)
                same_channels_list.append(num_chan)

        num_channels = same_channels_list[0]
        num_spw = len(same_spw_list)
        total_channels = num_spw * num_channels

        print(f"\nLoading single baseline from {self.ms_path}...")
        print(f"  Baseline: {ant1}-{ant2}, Pol: {pol_idx}")
        print(f"  SPWs: {num_spw} ({num_channels} channels each = {total_channels} total)")
        print(f"  Times: {self.num_times}")

        # Allocate array for this baseline
        baseline_data = np.zeros([total_channels, self.num_times], dtype="complex128")

        # Load all SPWs for this baseline
        for spw_idx, spw in enumerate(same_spw_list):
            subtable = self.tb.query(f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}")

            if subtable.nrows() == 0:
                subtable.close()
                raise ValueError(f"No data for baseline {ant1}-{ant2} in SPW {spw}")

            # Extract data for this SPW, single pol
            spw_data = subtable.getcol(mode)  # Shape: (pols, channels, times)
            spw_data_pol = spw_data[pol_idx, :, :]  # Shape: (channels, times)

            # Place in combined array
            start_ch = spw_idx * num_channels
            end_ch = (spw_idx + 1) * num_channels
            baseline_data[start_ch:end_ch, :] = spw_data_pol

            subtable.close()

        print(f"  Loaded shape: {baseline_data.shape}")

        return baseline_data

    def load_flags(self) -> NDArray[np.bool_]:
        """
        Load existing flags from measurement set.

        Loads flag data for all baselines that were previously loaded with
        the `load()` method. Flag array matches the shape of the visibility
        data.

        Returns
        -------
        NDArray[np.bool_]
            Boolean flag array with shape (num_baselines, num_pols,
            num_channels, num_times). True indicates flagged (bad) data.

        Raises
        ------
        ValueError
            If `load()` has not been called first to establish baseline map.

        Notes
        -----
        - Flags are loaded for the same baselines and SPWs as the visibility data
        - True indicates flagged (bad) data that should be excluded
        - False indicates unflagged (good) data

        Examples
        --------
        >>> loader = MSLoader('observation.ms')
        >>> data = loader.load(num_antennas=5)
        >>> flags = loader.load_flags()
        >>> print(f"Flagged fraction: {flags.mean():.2%}")
        Flagged fraction: 12.50%
        """
        if self.antenna_baseline_map is None:
            raise ValueError("Must call load() first to establish baseline map")

        print("\nLoading flags from MS...")

        flags_list = []
        num_channels = self.channels_per_spw_list[0]
        num_spw = len(self.spw_list)
        total_channels = num_spw * num_channels

        for ant1, ant2 in tqdm(self.antenna_baseline_map, desc="Baselines"):
            baseline_flags = np.zeros([4, total_channels, self.num_times], dtype=bool)

            for spw_idx, spw in enumerate(self.spw_list):
                subtable = self.tb.query(
                    f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}"
                )

                spw_flags = subtable.getcol("FLAG")

                start_ch = spw_idx * num_channels
                end_ch = (spw_idx + 1) * num_channels
                baseline_flags[:, start_ch:end_ch, :] = spw_flags

                subtable.close()

            flags_list.append(baseline_flags)

        self.flags = np.stack(flags_list)
        print(f"  Loaded flags shape: {self.flags.shape}")

        return self.flags

    def save_flags(self, flags: NDArray[np.bool_]) -> None:
        """
        Write flags back to measurement set.

        Updates the FLAG column in the measurement set with new flag values.
        Flags are written for all baselines that were loaded with `load()`.

        Parameters
        ----------
        flags : NDArray[np.bool_]
            Boolean flag array with shape (num_baselines, num_pols,
            num_channels, num_times). Must match the shape of loaded data.

        Raises
        ------
        ValueError
            If `load()` has not been called first to establish baseline map,
            or if flag array shape doesn't match loaded data.

        Notes
        -----
        - Flags are written to the FLAG column in the measurement set
        - This operation modifies the measurement set on disk
        - True indicates flagged (bad) data

        Examples
        --------
        >>> loader = MSLoader('observation.ms')
        >>> data = loader.load(num_antennas=5)
        >>> flags = loader.load_flags()
        >>> # Apply RFI detection to create new flags
        >>> new_flags = detect_rfi(data)
        >>> loader.save_flags(new_flags)
        >>> loader.close()
        """
        if self.antenna_baseline_map is None:
            raise ValueError("Must call load() first to establish baseline map")

        print("\nSaving flags to MS...")

        num_channels = self.channels_per_spw_list[0]

        for baseline_idx, (ant1, ant2) in enumerate(
            tqdm(self.antenna_baseline_map, desc="Baselines")
        ):
            baseline_flags = flags[baseline_idx]

            for spw_idx, spw in enumerate(self.spw_list):
                # Extract flags for this SPW
                start_ch = spw_idx * num_channels
                end_ch = (spw_idx + 1) * num_channels
                spw_flags = baseline_flags[:, start_ch:end_ch, :]

                # Write to MS
                subtable = self.tb.query(
                    f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}"
                )
                subtable.putcol("FLAG", spw_flags)
                subtable.close()

        print("  Flags saved successfully")

    def close(self) -> None:
        """
        Close the measurement set table.

        Releases CASA table resources. Should be called when finished working
        with the measurement set to avoid file locking issues.

        Examples
        --------
        >>> loader = MSLoader('observation.ms')
        >>> data = loader.load()
        >>> # ... process data ...
        >>> loader.close()
        """
        if hasattr(self, "tb"):
            self.tb.close()

    def __del__(self) -> None:
        """
        Ensure measurement set is closed on object deletion.

        Automatically called by Python garbage collector. Ensures that the
        CASA table is properly closed even if `close()` was not called
        explicitly.
        """
        self.close()

    @property
    def magnitude(self) -> NDArray[np.float64]:
        """
        Get magnitude of complex visibilities.

        Computes the absolute value (magnitude) of the complex visibility data.
        Useful for visualization and analysis that doesn't require phase
        information.

        Returns
        -------
        NDArray[np.float64]
            Magnitude array with same shape as data (num_baselines, num_pols,
            num_channels, num_times).

        Raises
        ------
        ValueError
            If `load()` has not been called first.

        Examples
        --------
        >>> loader = MSLoader('observation.ms')
        >>> data = loader.load(num_antennas=5)
        >>> mag = loader.magnitude
        >>> print(f"Mean magnitude: {mag.mean():.3e}")
        Mean magnitude: 1.234e-03
        """
        if self.data is None:
            raise ValueError("Must call load() first")
        return np.abs(self.data)
