"""
Interactive MS Waterfall Explorer using HoloViz stack.

Provides an interactive dashboard for exploring measurement set data with:
- SPW, baseline, polarization, time selection
- UV distance filtering
- Flag overlay visualization (MS flags, SAM-RFI predictions, ground truth)
- Datashader for large data handling

Usage:
    from samrfi.visualization import MSWaterfallExplorer

    explorer = MSWaterfallExplorer('observation.ms')
    explorer.show()  # Opens in browser

    # Or save to HTML
    explorer.save('explorer.html')
"""

from pathlib import Path

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

    Provides widgets for selecting SPW, baseline, polarization, time range,
    and UV distance filtering. Displays waterfall plot with optional flag overlays.

    Parameters
    ----------
    ms_path : str or Path
        Path to measurement set
    preload_data : bool, default=False
        If True, load all data into memory upfront (faster interaction but memory-intensive)
        If False, load data on-demand when selections change (slower but memory-efficient)
    width : int, default=1200
        Plot width in pixels
    height : int, default=600
        Plot height in pixels
    """

    def __init__(
        self, ms_path: str | Path, preload_data: bool = False, width: int = 1200, height: int = 600
    ):
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

    def _load_ms_metadata(self):
        """Load MS metadata without loading full data."""
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

    def _get_flag_versions(self):
        """Query flagmanager to get list of available flag versions."""
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

    def _compute_baseline_uv_distances(self):
        """Compute UV distances for all baselines."""
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

    def _create_widgets(self):
        """Create Panel widgets for interactive controls."""
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

    def _create_dashboard(self):
        """Create the Panel dashboard layout."""
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
    ):
        """Update waterfall plot based on widget selections."""
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
        """Load flags directly from flagmanager version directory."""
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
        """Get baseline index if within UV range."""
        for idx, (ant1, ant2, uv_dist) in enumerate(self.baseline_info):
            if (ant1, ant2) == baseline:
                if uv_range[0] <= uv_dist <= uv_range[1]:
                    return idx
                else:
                    return None
        return None

    def load_flags(self, flag_type: str, flags: np.ndarray):
        """
        Load flag data for overlay.

        Parameters
        ----------
        flag_type : str
            Type of flags: 'MS', 'SAM-RFI', 'Ground Truth'
        flags : np.ndarray
            Boolean flag array with shape (n_baselines, n_pols, n_channels, n_times)
        """
        if flag_type not in ["MS", "SAM-RFI", "Ground Truth"]:
            raise ValueError(f"Invalid flag_type: {flag_type}")

        expected_shape = self.ms_loader.data.shape
        if flags.shape != expected_shape:
            raise ValueError(f"Flag shape {flags.shape} doesn't match data shape {expected_shape}")

        self.flags_data[flag_type] = flags
        print(f"Loaded {flag_type} flags with shape {flags.shape}")

    def show(self, port: int = 5006):
        """
        Display the dashboard in a browser.

        Parameters
        ----------
        port : int, default=5006
            Port number for Bokeh server
        """
        self.dashboard.show(port=port)

    def save(self, filename: str | Path):
        """
        Save dashboard to standalone HTML file.

        Parameters
        ----------
        filename : str or Path
            Output HTML file path
        """
        self.dashboard.save(str(filename))
        print(f"Dashboard saved to {filename}")


def create_explorer_from_ms(
    ms_path: str | Path,
    sam_rfi_flags: np.ndarray | None = None,
    ground_truth_flags: np.ndarray | None = None,
    **kwargs,
) -> MSWaterfallExplorer:
    """
    Convenience function to create explorer with optional flag overlays.

    Parameters
    ----------
    ms_path : str or Path
        Path to measurement set
    sam_rfi_flags : np.ndarray, optional
        SAM-RFI predicted flags
    ground_truth_flags : np.ndarray, optional
        Ground truth flags
    **kwargs
        Additional arguments passed to MSWaterfallExplorer

    Returns
    -------
    MSWaterfallExplorer
        Configured explorer instance

    Examples
    --------
    >>> explorer = create_explorer_from_ms('observation.ms')
    >>> explorer.show()

    >>> # With flag overlays
    >>> explorer = create_explorer_from_ms(
    ...     'observation.ms',
    ...     sam_rfi_flags=predicted_flags,
    ...     ground_truth_flags=true_flags
    ... )
    >>> explorer.save('comparison.html')
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
