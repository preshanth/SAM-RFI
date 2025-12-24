"""
MS Loader - Load CASA measurement sets for RFI analysis

Clean rewrite of RadioRFI functionality, focused on data loading only.
"""

import numpy as np
from casatools import table
from tqdm import tqdm


class MSLoader:
    """
    Load complex visibilities from CASA measurement sets.

    Simplified interface:
    >>> loader = MSLoader('observation.ms')
    >>> loader.load(num_antennas=5, mode='DATA')
    >>> data = loader.data  # Shape: (baselines, pols, channels, times)
    >>> flags = loader.load_flags()  # Load existing flags
    """

    def __init__(self, ms_path):
        """
        Initialize MS loader.

        Args:
            ms_path: Path to measurement set
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

    def load(self, num_antennas=None, mode="DATA"):
        """
        Load complex visibilities from MS.

        Args:
            num_antennas: Number of antennas to load (default: all)
            mode: Column to load ('DATA', 'CORRECTED_DATA', etc.)

        Returns:
            Loaded data shape: (num_baselines, num_pols, num_channels, num_times)
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

                # Load all SPWs for this baseline
                for spw_idx, spw in enumerate(same_spw_list):
                    subtable = self.tb.query(
                        f"DATA_DESC_ID=={spw} && ANTENNA1=={i} && ANTENNA2=={j}"
                    )

                    # Extract data for this SPW
                    spw_data = subtable.getcol(mode)

                    # Place in combined array
                    start_ch = spw_idx * num_channels
                    end_ch = (spw_idx + 1) * num_channels
                    baseline_data[:, start_ch:end_ch, :] = spw_data

                    subtable.close()

                data_list.append(baseline_data)
                baseline_map.append((i, j))

        # Stack all baselines
        self.data = np.stack(data_list)  # Shape: (baselines, pols, channels, times)
        self.antenna_baseline_map = baseline_map
        self.spw_list = same_spw_list
        self.channels_per_spw_list = same_channels_list

        print(f"  Loaded shape: {self.data.shape}")

        return self.data

    def load_flags(self):
        """
        Load existing flags from MS.

        Returns:
            Flags shape: (num_baselines, num_pols, num_channels, num_times)
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

    def save_flags(self, flags):
        """
        Write flags back to MS.

        Args:
            flags: Flag array shape (num_baselines, num_pols, num_channels, num_times)
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

    def close(self):
        """Close the measurement set."""
        if hasattr(self, "tb"):
            self.tb.close()

    def __del__(self):
        """Ensure MS is closed on deletion."""
        self.close()

    @property
    def magnitude(self):
        """Get magnitude of complex visibilities."""
        if self.data is None:
            raise ValueError("Must call load() first")
        return np.abs(self.data)
