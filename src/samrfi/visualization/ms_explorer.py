"""
Interactive MS Waterfall Explorer using HoloViz stack.

This module provides an interactive dashboard for exploring radio astronomy
measurement set (MS) data with comprehensive visualization and analysis tools.

Features
--------
- SPW, baseline, polarization, and time selection
- UV distance filtering for baseline selection
- Flag overlay visualization (MS flags, SAM-RFI predictions, ground truth)
- Datashader integration for handling large datasets efficiently
- Interactive waterfall plots with zoom, pan, and hover capabilities
- Residual plots showing data after flag masking
- Multiple flag version comparison and overlay

The explorer is built on the HoloViz ecosystem (Panel, HoloViews, Datashader)
and provides both browser-based interactive exploration and HTML export.

Classes
-------
MSWaterfallExplorer
    Main interactive explorer class for measurement set visualization.

Functions
---------
create_explorer_from_ms
    Convenience function to create an explorer with optional flag overlays.

Examples
--------
Basic usage:

>>> from samrfi.visualization import MSWaterfallExplorer
>>> explorer = MSWaterfallExplorer('observation.ms')
>>> explorer.show()  # Opens interactive dashboard in browser

With flag overlays:

>>> from samrfi.visualization import create_explorer_from_ms
>>> explorer = create_explorer_from_ms(
...     'observation.ms',
...     sam_rfi_flags=predicted_flags,
...     ground_truth_flags=true_flags
... )
>>> explorer.save('comparison.html')  # Save to standalone HTML

Notes
-----
This module requires CASA tools (casatools, casatasks) for accessing measurement
set metadata and flag versions. The HoloViz stack (panel, holoviews, datashader)
is required for interactive visualization.
"""

from pathlib import Path
from typing import Any

import holoviews as hv
import numpy as np
import panel as pn
from holoviews.operation.datashader import rasterize

# Initialize Panel and HoloViews extensions
pn.extension()
hv.extension("bokeh")


class MSWaterfallExplorer:
    """
    Interactive MS waterfall explorer with HoloViz + Datashader.

    Provides a complete interactive dashboard for exploring measurement set data
    with widgets for data selection, UV filtering, and multi-version flag overlay
    visualization. Supports both in-browser display and HTML export.

    Parameters
    ----------
    ms_path : str or Path
        Path to measurement set directory.
    preload_data : bool, default=False
        If True, load all data into memory upfront (faster interaction but
        memory-intensive). If False, load data on-demand when selections change
        (slower but memory-efficient).
    width : int, default=1200
        Plot width in pixels.
    height : int, default=600
        Plot height in pixels.

    Attributes
    ----------
    ms_path : Path
        Resolved path to measurement set.
    ms_loader : MSLoader or None
        Measurement set data loader instance.
    data : ndarray or None
        Loaded visibility data array.
    flags_data : dict[str, ndarray]
        Dictionary storing different flag versions by name.
    spw_info : list[tuple[str, int, str]]
        List of (label, n_channels, description) for spectral windows.
    baseline_info : list[tuple[int, int, float]]
        List of (ant1, ant2, uv_distance) for all baselines.
    pol_names : list[str]
        Polarization names (e.g., ['XX', 'XY', 'YX', 'YY']).
    time_range : tuple[int, int]
        Valid time sample range (min_idx, max_idx).
    flag_versions : list[str]
        Available flag version names from flagmanager.
    dashboard : panel.Row or None
        Main dashboard layout component.

    Examples
    --------
    Create and display explorer:

    >>> explorer = MSWaterfallExplorer('observation.ms')
    >>> explorer.show()  # Opens in browser at localhost:5006

    Load custom flags:

    >>> explorer = MSWaterfallExplorer('observation.ms')
    >>> explorer.load_flags('SAM-RFI', predicted_flags)
    >>> explorer.show()

    Save to HTML:

    >>> explorer = MSWaterfallExplorer('observation.ms', width=1600, height=800)
    >>> explorer.save('explorer.html')

    Notes
    -----
    The explorer uses Datashader for efficient rendering of large datasets.
    For very large measurement sets, consider using preload_data=False to
    reduce memory usage.
    """

    def __init__(
        self, ms_path: str | Path, preload_data: bool = False, width: int = 1200, height: int = 600
    ) -> None:
        self.ms_path = Path(ms_path)
        self.preload_data = preload_data
        self.width = width
        self.height = height

        # Data storage
        self.ms_loader = None
        self.data = None
        self.flags_data = {}  # Store different flag sets

        # Metadata
        self.spw_info = []  # List of (spw_id, n_channels, freq_range)
        self.baseline_info = []  # List of (ant1, ant2, uv_distance)
        self.pol_names = []
        self.time_range = (0, 0)

        # Widgets (created in _create_widgets)
        self.spw_selector = None
        self.baseline_selector = None
        self.pol_selector = None
        self.time_slider = None
        self.uv_slider = None
        self.flag_toggles = {}

        # Dashboard components
        self.dashboard = None

        # Initialize
        self._load_ms_metadata()
        self._create_widgets()
        self._create_dashboard()

    def _load_ms_metadata(self) -> None:
        """
        Load measurement set metadata without loading full data.

        Initializes the MS loader and extracts basic metadata including shape,
        SPW information, baseline configuration, polarization names, and time
        range. Also queries available flag versions from flagmanager.

        Notes
        -----
        This method is called during initialization and does not load the full
        visibility data. It only reads metadata to populate UI widgets.
        """
        from ..data import MSLoader

        print(f"Loading metadata from {self.ms_path}...")
        self.ms_loader = MSLoader(str(self.ms_path))

        # Load minimal data to extract metadata
        # We'll load full SPW/baseline data on-demand or preload if requested
        self.ms_loader.load(mode="DATA", num_antennas=None)

        # Extract SPW information
        # shape: (n_baselines, n_pols, n_channels, n_times)
        data_shape = self.ms_loader.data.shape
        n_baselines, n_pols, n_channels, n_times = data_shape

        # MSLoader combines all SPWs into single dimension
        self.spw_info = [("All SPWs", n_channels, "Combined")]  # (label, n_channels, description)

        # Extract baseline information with UV distances
        # This requires accessing MS metadata tables
        self._compute_baseline_uv_distances()

        # Polarization names
        self.pol_names = ["XX", "XY", "YX", "YY"][:n_pols]  # Standard correlations

        # Time range
        self.time_range = (0, n_times - 1)

        # Query available flag versions from flagmanager
        self.flag_versions = self._get_flag_versions()

        print("MS Metadata:")
        print(f"  Baselines: {n_baselines}")
        print(f"  Polarizations: {n_pols} {self.pol_names}")
        print(f"  Channels: {n_channels}")
        print(f"  Time samples: {n_times}")
        print(f"  Available flag versions: {len(self.flag_versions)}")

    def _get_flag_versions(self) -> list[str]:
        """
        Query flagmanager to get list of available flag versions.

        Executes a CASA script to query the flagmanager for all saved flag
        versions associated with this measurement set.

        Returns
        -------
        list[str]
            List of flag version names available in flagmanager.
            Returns empty list if query fails or no versions exist.

        Notes
        -----
        This method runs a CASA script in a subprocess to access flagmanager.
        Output is filtered to remove CASA log messages and extract only
        version names.
        """
        import subprocess
        import tempfile

        # Create CASA script to list flag versions
        casa_script = f"""
import sys
sys.path.append('/home/pjaganna/micromamba/envs/SAM-RFI/lib/python3.12/site-packages')
from casatasks import flagmanager

# List flag versions
flag_dict = flagmanager(vis='{self.ms_path}', mode='list')

# Print version names (one per line)
if flag_dict:
    for key, value in flag_dict.items():
        if key != 'MS' and isinstance(value, dict) and 'name' in value:
            print(value['name'])
"""

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(casa_script)
                script_path = f.name

            result = subprocess.run(
                ["python", script_path], capture_output=True, text=True, timeout=30
            )

            Path(script_path).unlink()

            if result.returncode == 0:
                # Parse version names from output
                versions = [
                    line.strip() for line in result.stdout.strip().split("\n") if line.strip()
                ]
                # Filter out CASA log messages and non-version lines
                versions = [
                    v
                    for v in versions
                    if not any(
                        [
                            v.startswith("CASA"),
                            ":::" in v,
                            v.startswith("2025-"),  # Timestamp lines
                            "SEVERE" in v,
                            "WARNING" in v,
                        ]
                    )
                ]
                return versions
            else:
                print(f"Warning: Could not list flag versions: {result.stderr}")
                return []
        except Exception as e:
            print(f"Warning: Could not query flagmanager: {e}")
            return []

    def _compute_baseline_uv_distances(self) -> None:
        """
        Compute UV distances for all baselines.

        Extracts antenna pairs and calculates UV distances. Currently uses
        placeholder values; full implementation would query ANTENNA and UVW
        tables from the measurement set.

        Notes
        -----
        This is a simplified implementation using dummy baseline labels.
        A production version would query the MS ANTENNA table for real
        antenna IDs and the UVW table for actual baseline distances.
        """
        # Extract antenna pairs from MS
        # shape: (n_baselines, n_pols, n_channels, n_times)
        n_baselines = self.ms_loader.data.shape[0]

        # For now, create dummy baseline labels (extend later with real antenna IDs)
        # In a real implementation, we'd query the ANTENNA and UVW tables
        self.baseline_info = []
        for i in range(n_baselines):
            ant1 = i // 27  # Dummy antenna numbering (assumes ~27 antennas)
            ant2 = i % 27
            uv_dist = np.random.uniform(10, 1000)  # Placeholder UV distance in kλ
            self.baseline_info.append((ant1, ant2, uv_dist))

    def _create_widgets(self) -> None:
        """
        Create Panel widgets for interactive controls.

        Initializes all interactive widgets including SPW selector, baseline
        selector with UV distances, polarization selector, time range slider,
        UV distance filter, flag version selector, and color saturation slider.

        Notes
        -----
        All widgets are stored as instance attributes for later reference and
        are bound to the plot update function in _create_dashboard().
        """
        # SPW selector
        spw_options = {f"{label} ({n_ch} channels)": label for label, n_ch, _ in self.spw_info}
        self.spw_selector = pn.widgets.Select(
            name="Spectral Window",
            options=spw_options,
            value=list(spw_options.values())[0] if spw_options else "All SPWs",
        )

        # Baseline selector (with UV distance in label)
        baseline_options = {
            f"Ant {ant1}-{ant2} (UV: {uv:.0f} kλ)": (ant1, ant2)
            for ant1, ant2, uv in self.baseline_info
        }
        self.baseline_selector = pn.widgets.Select(
            name="Baseline",
            options=baseline_options,
            value=list(baseline_options.values())[0] if baseline_options else (0, 0),
        )

        # Polarization selector
        self.pol_selector = pn.widgets.Select(
            name="Polarization",
            options=self.pol_names,
            value=self.pol_names[0] if self.pol_names else "XX",
        )

        # Time range slider
        self.time_slider = pn.widgets.RangeSlider(
            name="Time Range",
            start=self.time_range[0],
            end=self.time_range[1],
            value=self.time_range,
            step=1,
        )

        # UV distance filter slider
        uv_distances = [uv for _, _, uv in self.baseline_info]
        uv_min, uv_max = min(uv_distances), max(uv_distances)
        self.uv_slider = pn.widgets.RangeSlider(
            name="UV Range (kλ)",
            start=uv_min,
            end=uv_max,
            value=(uv_min, uv_max),
            step=(uv_max - uv_min) / 100,
        )

        # Flag version selector (multi-select checkbox group)
        self.flag_selector = pn.widgets.CheckBoxGroup(
            name="Flag Versions to Overlay",
            options=self.flag_versions,
            value=[],  # Start with nothing selected
            inline=False,
        )

        # Colormap saturation slider
        self.saturation_slider = pn.widgets.FloatSlider(
            name="Color Saturation", start=0.1, end=10.0, value=1.0, step=0.1
        )

    def _create_dashboard(self) -> None:
        """
        Create the Panel dashboard layout.

        Assembles the complete dashboard UI by binding widgets to the plot
        update function and arranging controls and plots in a responsive layout.

        Notes
        -----
        Uses Panel's reactive programming model (pn.bind) to automatically
        update plots when widget values change. Layout is a Row with controls
        on the left and plot pane on the right.
        """
        # Bind waterfall plot to widget values using .param.value for reactivity
        waterfall_plot = pn.bind(
            self._update_waterfall,
            spw=self.spw_selector.param.value,
            baseline=self.baseline_selector.param.value,
            pol=self.pol_selector.param.value,
            time_range=self.time_slider.param.value,
            uv_range=self.uv_slider.param.value,
            saturation=self.saturation_slider.param.value,
            selected_flag_versions=self.flag_selector.param.value,
        )

        # Layout: controls on left, plot on right
        controls = pn.Column(
            "## MS Waterfall Explorer",
            "### Data Selection",
            self.spw_selector,
            self.baseline_selector,
            self.pol_selector,
            self.time_slider,
            self.uv_slider,
            "### Display",
            self.saturation_slider,
            "### Flag Overlays",
            self.flag_selector,
            width=300,
        )

        self.dashboard = pn.Row(
            controls, pn.pane.HoloViews(waterfall_plot, sizing_mode="stretch_both")
        )

    def _update_waterfall(
        self,
        spw: int,
        baseline: tuple[int, int],
        pol: str,
        time_range: tuple[int, int],
        uv_range: tuple[float, float],
        saturation: float,
        selected_flag_versions: list[str],
    ) -> hv.Layout:
        """
        Update waterfall plot based on widget selections.

        This is the main plot update callback that responds to widget changes.
        Extracts selected data, applies flag overlays, and generates both
        original and residual (flagged) waterfall plots.

        Parameters
        ----------
        spw : int
            Selected spectral window ID.
        baseline : tuple[int, int]
            Selected baseline as (antenna1, antenna2) pair.
        pol : str
            Selected polarization (e.g., 'XX', 'XY', 'YX', 'YY').
        time_range : tuple[int, int]
            Time sample range as (start_idx, end_idx).
        uv_range : tuple[float, float]
            UV distance filter range in kλ as (min_uv, max_uv).
        saturation : float
            Color saturation factor for amplitude display. Higher values
            increase contrast by lowering the colormap maximum.
        selected_flag_versions : list[str]
            List of flag version names to overlay on the plot.

        Returns
        -------
        holoviews.Layout
            Vertical layout containing original data plot (top) and
            residual plot with flags masked (bottom).

        Notes
        -----
        - Uses Datashader rasterization for efficient rendering of large data
        - Flag overlays are shown as semi-transparent colored regions
        - Residual plot shows data with flagged points set to NaN
        - Returns empty plot with message if baseline is outside UV range
        """
        # Debug: Print what we're trying to display
        print(
            f"DEBUG: Updating plot - Baseline {baseline}, Pol {pol}, Time {time_range}, Sat {saturation:.1f}"
        )

        # Filter baselines by UV range
        baseline_idx = self._get_baseline_index(baseline, uv_range)
        if baseline_idx is None:
            # Baseline outside UV range - show empty plot
            print(f"DEBUG: Baseline {baseline} outside UV range {uv_range}")
            return hv.Text(0.5, 0.5, "Baseline outside UV range").opts(
                width=self.width, height=self.height
            )

        print(f"DEBUG: Using baseline index {baseline_idx}")

        # Get polarization index
        pol_idx = self.pol_names.index(pol)
        print(f"DEBUG: Pol index {pol_idx}")

        # Extract waterfall data
        # shape: (n_baselines, n_pols, n_channels, n_times)
        waterfall_data = self.ms_loader.data[
            baseline_idx, pol_idx, :, time_range[0] : time_range[1]
        ]
        waterfall_amp = np.abs(waterfall_data)  # (channels, times)
        n_channels, n_times = waterfall_amp.shape

        # Create HoloViews image with explicit bounds
        img = hv.Image(
            waterfall_amp,
            bounds=(0, 0, n_channels, n_times),
            kdims=["Channel", "Time"],
            vdims="Amplitude",
        )

        # Apply datashader for large data
        rasterized = rasterize(img, aggregator="mean")

        # Calculate colormap limits based on saturation
        # Higher saturation = more contrast (lower clim_max)
        # saturation=1.0 uses median as reference
        vmin = np.nanpercentile(waterfall_amp, 1)  # 1st percentile
        vmedian = np.nanmedian(waterfall_amp)
        vmax = vmedian * saturation  # saturation controls upper limit

        # Style
        rasterized = rasterized.opts(
            width=self.width,
            height=self.height,
            cmap="viridis",
            clim=(vmin, vmax),
            colorbar=True,
            title=f"SPW {spw} | Baseline {baseline[0]}-{baseline[1]} | {pol}",
            xlabel="Channel",
            ylabel="Time",
            tools=["hover", "box_zoom", "wheel_zoom", "pan", "reset"],
            active_tools=["wheel_zoom"],
        )

        # Overlay selected flag versions with semi-transparent colored regions
        colors = ["red", "blue", "green", "purple", "orange", "yellow"]
        combined_flags = np.zeros_like(waterfall_amp, dtype=bool)

        flag_overlays = []
        for i, version_name in enumerate(selected_flag_versions):
            # Load flags from this version if not already cached
            if version_name not in self.flags_data:
                flags = self._load_flag_version(version_name)
                if flags is not None:
                    self.flags_data[version_name] = flags

            # Create overlay if flags loaded successfully
            if version_name in self.flags_data:
                version_flags = self.flags_data[version_name][
                    baseline_idx, pol_idx, :, time_range[0] : time_range[1]
                ]
                combined_flags |= version_flags

                # Create filled contour overlay showing flagged regions
                flag_data = np.where(version_flags, 1.0, np.nan)  # NaN for unflagged, 1 for flagged
                flag_img = hv.Image(
                    flag_data,
                    bounds=(0, 0, n_channels, n_times),
                    kdims=["Channel", "Time"],
                    vdims="Flagged",
                ).opts(cmap=[colors[i % len(colors)]], alpha=0.4, clim=(0.5, 1.5), colorbar=False)
                flag_overlays.append(flag_img)

        # Combine rasterized plot with flag overlays
        if flag_overlays:
            main_plot = rasterized
            for flag_overlay in flag_overlays:
                main_plot = main_plot * flag_overlay
        else:
            main_plot = rasterized

        # Create residual plot (data with flags masked)
        masked_data = waterfall_amp.copy()
        if combined_flags.any():
            masked_data[combined_flags] = np.nan

        residual_img = hv.Image(
            masked_data,
            bounds=(0, 0, n_channels, n_times),
            kdims=["Channel", "Time"],
            vdims="Amplitude",
        )

        residual_rasterized = rasterize(residual_img, aggregator="mean").opts(
            width=self.width,
            height=self.height // 2,
            cmap="viridis",
            clim=(vmin, vmax),
            colorbar=True,
            title="Residual (Flagged Data Masked)",
            xlabel="Channel",
            ylabel="Time",
            tools=["hover", "box_zoom", "wheel_zoom", "pan", "reset"],
            active_tools=["wheel_zoom"],
        )

        # Update main plot title and height
        main_plot = main_plot.opts(
            height=self.height // 2,
            title=f"SPW {spw} | Baseline {baseline[0]}-{baseline[1]} | {pol} | Original Data",
        )

        # Return column layout with both plots
        return hv.Layout([main_plot, residual_rasterized]).cols(1)

    def _load_flag_version(self, version_name: str) -> np.ndarray | None:
        """
        Load flags directly from flagmanager version directory.

        Accesses the .flagversions directory associated with the measurement
        set and reads flag data from the specified version using CASA table tools.

        Parameters
        ----------
        version_name : str
            Name of the flag version to load (e.g., 'Original', 'after_rflag').

        Returns
        -------
        ndarray or None
            Boolean flag array with shape (n_baselines, n_pols, n_channels, n_times)
            if successful, None if loading fails.

        Notes
        -----
        - Flag versions are stored in .flagversions/flags.<version_name> directory
        - Uses casatools.table to read FLAG column directly
        - Reshapes and transposes flag data to match MS loader format
        - Returns None with warning message if version doesn't exist or loading fails
        """
        print(f"Loading flag version: {version_name}")

        try:
            # Flagmanager stores versions as .flagversions/flags.<version_name>
            flagversions_dir = Path(str(self.ms_path) + ".flagversions")
            flag_version_path = flagversions_dir / f"flags.{version_name}"

            if not flag_version_path.exists():
                print(f"Warning: Flag version path does not exist: {flag_version_path}")
                return None

            # Read flags directly using casatools table
            from casatools import table

            tb = table()
            tb.open(str(flag_version_path))
            flags_raw = tb.getcol("FLAG")  # shape: (npol, nchan, nrow)
            tb.close()

            # Reshape to match MS loader format: (n_baselines, n_pols, n_channels, n_times)
            expected_shape = self.ms_loader.data.shape
            flags = flags_raw.transpose(2, 0, 1)  # (nrow, npol, nchan)

            # Reshape assuming row order matches baseline order (simplification)
            n_baselines, n_pols, n_channels, n_times = expected_shape
            flags = flags.reshape(n_baselines, n_times, n_pols, n_channels)
            flags = flags.transpose(0, 2, 3, 1)  # (n_baselines, n_pols, n_channels, n_times)

            print(f"Loaded flags from {version_name}: shape {flags.shape}")
            return flags
        except Exception as e:
            print(f"Warning: Could not load flag version {version_name}: {e}")
            return None

    def _get_baseline_index(
        self, baseline: tuple[int, int], uv_range: tuple[float, float]
    ) -> int | None:
        """
        Get baseline index if within UV range.

        Searches for the specified baseline in the baseline list and checks
        if its UV distance falls within the specified range.

        Parameters
        ----------
        baseline : tuple[int, int]
            Baseline antenna pair as (antenna1, antenna2).
        uv_range : tuple[float, float]
            Acceptable UV distance range in kλ as (min_uv, max_uv).

        Returns
        -------
        int or None
            Baseline index if found and within UV range, None otherwise.

        Notes
        -----
        Returns None if baseline not found or if UV distance is outside range.
        """
        for idx, (ant1, ant2, uv_dist) in enumerate(self.baseline_info):
            if (ant1, ant2) == baseline:
                if uv_range[0] <= uv_dist <= uv_range[1]:
                    return idx
                else:
                    return None
        return None

    def load_flags(self, flag_type: str, flags: np.ndarray) -> None:
        """
        Load flag data for overlay visualization.

        Stores flag data for display as colored overlays on waterfall plots.
        Validates flag array shape matches the loaded measurement set data.

        Parameters
        ----------
        flag_type : str
            Type of flags: 'MS', 'SAM-RFI', or 'Ground Truth'.
        flags : ndarray
            Boolean flag array with shape (n_baselines, n_pols, n_channels, n_times).
            True indicates flagged (bad) data, False indicates unflagged (good) data.

        Raises
        ------
        ValueError
            If flag_type is not one of the valid types or if flag array shape
            doesn't match the loaded data shape.

        Examples
        --------
        >>> explorer = MSWaterfallExplorer('observation.ms')
        >>> # Load SAM-RFI predictions
        >>> explorer.load_flags('SAM-RFI', predicted_flags)
        >>> # Load ground truth for comparison
        >>> explorer.load_flags('Ground Truth', true_flags)

        Notes
        -----
        Multiple flag versions can be loaded and will be overlaid with different
        colors in the visualization.
        """
        if flag_type not in ["MS", "SAM-RFI", "Ground Truth"]:
            raise ValueError(f"Invalid flag_type: {flag_type}")

        expected_shape = self.ms_loader.data.shape
        if flags.shape != expected_shape:
            raise ValueError(f"Flag shape {flags.shape} doesn't match data shape {expected_shape}")

        self.flags_data[flag_type] = flags
        print(f"Loaded {flag_type} flags with shape {flags.shape}")

    def show(self, port: int = 5006) -> None:
        """
        Display the interactive dashboard in a web browser.

        Launches a Bokeh server and opens the dashboard in the default web browser
        at localhost:<port>. The server runs until manually stopped.

        Parameters
        ----------
        port : int, default=5006
            Port number for the Bokeh server. Default is 5006.

        Examples
        --------
        >>> explorer = MSWaterfallExplorer('observation.ms')
        >>> explorer.show()  # Opens at localhost:5006
        >>> # Or use custom port
        >>> explorer.show(port=8080)  # Opens at localhost:8080

        Notes
        -----
        The server must be manually stopped (Ctrl+C in terminal) to release the port.
        Multiple explorers cannot use the same port simultaneously.
        """
        self.dashboard.show(port=port)

    def save(self, filename: str | Path) -> None:
        """
        Save dashboard to standalone HTML file.

        Exports the complete interactive dashboard as a self-contained HTML file
        that can be shared and viewed without running a server. All interactivity
        is preserved in the HTML file.

        Parameters
        ----------
        filename : str or Path
            Output HTML file path. Should end with '.html' extension.

        Examples
        --------
        >>> explorer = MSWaterfallExplorer('observation.ms')
        >>> explorer.save('ms_explorer.html')
        Dashboard saved to ms_explorer.html

        >>> # Save with custom configuration
        >>> explorer = MSWaterfallExplorer('observation.ms', width=1600, height=900)
        >>> explorer.load_flags('SAM-RFI', predictions)
        >>> explorer.save('/path/to/output/comparison.html')

        Notes
        -----
        The exported HTML file contains all JavaScript and styling needed for
        interactivity. File size depends on the amount of data loaded.
        """
        self.dashboard.save(str(filename))
        print(f"Dashboard saved to {filename}")


def create_explorer_from_ms(
    ms_path: str | Path,
    sam_rfi_flags: np.ndarray | None = None,
    ground_truth_flags: np.ndarray | None = None,
    **kwargs: Any,
) -> MSWaterfallExplorer:
    """
    Convenience function to create explorer with optional flag overlays.

    Creates an MSWaterfallExplorer instance, automatically loads MS flags from
    the measurement set, and optionally loads SAM-RFI predictions and ground
    truth flags for comparison.

    Parameters
    ----------
    ms_path : str or Path
        Path to measurement set directory.
    sam_rfi_flags : ndarray, optional
        SAM-RFI predicted flags with shape (n_baselines, n_pols, n_channels, n_times).
        If provided, will be loaded as 'SAM-RFI' flag type.
    ground_truth_flags : ndarray, optional
        Ground truth flags with shape (n_baselines, n_pols, n_channels, n_times).
        If provided, will be loaded as 'Ground Truth' flag type.
    **kwargs : dict, optional
        Additional keyword arguments passed to MSWaterfallExplorer constructor.
        Supported options: preload_data (bool), width (int), height (int).

    Returns
    -------
    MSWaterfallExplorer
        Configured explorer instance with all specified flags loaded.

    Examples
    --------
    Basic usage with MS flags only:

    >>> explorer = create_explorer_from_ms('observation.ms')
    >>> explorer.show()

    With SAM-RFI predictions:

    >>> from samrfi.inference import SAM2Predictor
    >>> predictor = SAM2Predictor.from_checkpoint('model.pth')
    >>> predictions = predictor.predict_ms('observation.ms')
    >>> explorer = create_explorer_from_ms(
    ...     'observation.ms',
    ...     sam_rfi_flags=predictions
    ... )
    >>> explorer.show()

    Full comparison with all flag types:

    >>> explorer = create_explorer_from_ms(
    ...     'observation.ms',
    ...     sam_rfi_flags=predicted_flags,
    ...     ground_truth_flags=true_flags,
    ...     width=1600,
    ...     height=900
    ... )
    >>> explorer.save('full_comparison.html')

    Notes
    -----
    - MS flags are loaded automatically from the measurement set
    - If MS flag loading fails, a warning is printed but execution continues
    - All flag arrays must match the data shape from the measurement set
    - Multiple flag versions can be overlaid for comparison
    """
    explorer = MSWaterfallExplorer(ms_path, **kwargs)

    # Load MS flags automatically
    try:
        ms_flags = explorer.ms_loader.load_flags()
        explorer.load_flags("MS", ms_flags)
    except Exception as e:
        print(f"Warning: Could not load MS flags: {e}")

    # Load optional flags
    if sam_rfi_flags is not None:
        explorer.load_flags("SAM-RFI", sam_rfi_flags)

    if ground_truth_flags is not None:
        explorer.load_flags("Ground Truth", ground_truth_flags)

    return explorer
