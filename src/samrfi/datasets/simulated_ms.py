"""
Simulated MS with CASA-generated data and baseline-by-baseline RFI injection

Uses CASA simulator to create MS with realistic data, then adds RFI
by reading/corrupting/writing each baseline individually to minimize memory usage.
"""

import numpy as np
from typing import Optional
from pathlib import Path
import logging
import shutil
import os
import psutil

try:
    from casatools import simulator, table, measures, quanta, ctsys
    from casatasks import flagdata
    from casatasks.private import simutil
    CASA_AVAILABLE = True
except ImportError as e:
    CASA_AVAILABLE = False
    logging.warning(f"CASA tools not available: {e}")

from .synthetic_ms import ObservationConfig, RFIConfig

logger = logging.getLogger(__name__)


class SimulatedMS:
    """
    Creates measurement sets using CASA simulator + baseline RFI injection
    """

    def __init__(self, obs_config: ObservationConfig):
        if not CASA_AVAILABLE:
            raise ImportError("CASA tools/tasks required for MS creation")

        self.obs_config = obs_config
        
        # Initialize CASA tools
        self.sm = simulator()
        self.tb = table()
        self.me = measures()
        self.qa = quanta()
        self.mysu = simutil.simutil()

    def __del__(self):
        """Destructor to cleanup CASA tools"""
        try:
            if hasattr(self, 'sm'):
                self.sm.close()
                self.sm.done()
            if hasattr(self, 'tb'):
                self.tb.close()
                self.tb.done()
            if hasattr(self, 'me'):
                self.me.done()
            if hasattr(self, 'qa'):
                self.qa.done()
            import gc
            gc.collect()
        except:
            pass

    def print_memory(self, label: str):
        """Print current memory usage"""
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"{label}: {mem_mb:.1f} MB")

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
        
        Args:
            ms_path: Output measurement set path
            rfi_config: RFI configuration for corruption (None = clean MS)
            include_rfi_flags: Whether to flag RFI-corrupted data
            save_training_data: Whether to save .npy files for training
            training_data_dir: Directory to save training .npy files
        """
        ms_path = Path(ms_path)
        
        # Remove existing MS
        if ms_path.exists():
            logger.info(f"Removing existing MS: {ms_path}")
            shutil.rmtree(ms_path)

        self.print_memory("Starting MS creation")
        
        # Step 1: Let CASA create MS with realistic data
        logger.info(f"Creating CASA-generated MS: {ms_path}")
        self._create_casa_ms(str(ms_path))
        self.print_memory("CASA MS created")
        
        # Step 2: Add RFI baseline-by-baseline (if requested)
        if rfi_config is not None:
            logger.info("Adding RFI baseline-by-baseline")
            self._add_rfi_to_ms(str(ms_path), rfi_config, include_rfi_flags)
            self.print_memory("RFI injection complete")
        
        # Step 3: Save training data (.npy files) if requested
        if save_training_data:
            logger.info("Extracting training data from MS")
            self._save_training_data(str(ms_path), training_data_dir)
            self.print_memory("Training data extraction complete")
        
        logger.info(f"Simulated MS created: {ms_path}")

    def _create_casa_ms(self, ms_path: str):
        """Create MS using CASA simulator with realistic data"""
        
        # Remove any existing MS
        os.system(f'rm -rf {ms_path}')

        # Open the simulator
        self.sm.open(ms=ms_path)

        # Get antenna configuration
        antenna_config = self._get_antenna_config()
        x, y, z, d, an, telname = antenna_config
        
        # Update obs_config with actual antenna count
        actual_num_antennas = len(x)
        if actual_num_antennas != self.obs_config.num_antennas:
            logger.info(f"Updating antenna count from {self.obs_config.num_antennas} to {actual_num_antennas}")
            self.obs_config.num_antennas = actual_num_antennas
        
        # Set antenna configuration
        self.sm.setconfig(
            telescopename=telname,
            x=x, y=y, z=z,
            dishdiameter=d,
            mount=['alt-az'] * len(x),
            antname=an,
            coordsystem='global',
            referencelocation=self.me.observatory(telname)
        )

        # Set polarization feed
        self.sm.setfeed(mode='perfect X Y', pol=[''])

        # Set spectral window
        num_channels = self.obs_config.num_spw * self.obs_config.channels_per_spw
        self.sm.setspwindow(
            spwname="SPW0",
            freq=f'{self.obs_config.start_frequency}Hz',
            deltafreq=f'{self.obs_config.channel_width}Hz', 
            freqresolution=f'{self.obs_config.channel_width}Hz',
            nchannels=num_channels,
            stokes='XX XY YX YY'
        )

        # Set field/source
        ra_str = f"{getattr(self.obs_config, 'source_ra', 0.0) * 180 / np.pi}deg"  
        dec_str = f"{getattr(self.obs_config, 'source_dec', 0.7854) * 180 / np.pi}deg"
        
        self.sm.setfield(
            sourcename=getattr(self.obs_config, 'source_name', 'SYNTHETIC_SOURCE'),
            sourcedirection=self.me.direction('J2000', ra_str, dec_str)
        )

        # Set limits and no autocorrelations
        self.sm.setlimits(shadowlimit=0.01, elevationlimit='10deg')
        self.sm.setauto(autocorrwt=0.0)

        # Set timing
        ref_time = getattr(self.obs_config, 'start_time', '2024-01-01T00:00:00').replace('T', '/')
        self.sm.settimes(
            integrationtime=f'{self.obs_config.integration_time}s',
            usehourangle=False,
            referencetime=self.me.epoch('UTC', ref_time)
        )

        # Create observation - CASA will populate with data
        obs_duration = self.obs_config.total_duration
        self.sm.observe(
            sourcename=getattr(self.obs_config, 'source_name', 'SYNTHETIC_SOURCE'),
            spwname='SPW0',
            starttime='0s',
            stoptime=f'{obs_duration}s'
        )

        # Close and destroy simulator
        self.sm.close()
        self.sm.done()

        # Unflag everything initially
        flagdata(vis=ms_path, mode='unflag')
        
        logger.info("CASA-generated MS created")

    def _add_rfi_to_ms(self, ms_path: str, rfi_config: RFIConfig, include_flags: bool):
        """Add RFI to MS baseline-by-baseline - THE WORKING VERSION"""
        
        # Get MS dimensions
        self.tb.open(ms_path, nomodify=True)
        try:
            nrows = self.tb.nrows()
            # Get first row to determine data shape
            sample_data = self.tb.getcell("DATA", 0)
            npols, nchans = sample_data.shape
            self.tb.close()
        except Exception as e:
            self.tb.close()
            raise e
            
        # Calculate baselines and times
        num_antennas = self.obs_config.num_antennas
        num_baselines = num_antennas * (num_antennas - 1) // 2
        ntimes = nrows // num_baselines
        
        logger.info(f"MS dimensions: {num_baselines} baselines, {ntimes} times, {nchans} channels, {npols} pols")
        
        # Create plot directory
        plot_dir = Path(str(ms_path).replace('.ms', '_plots'))
        plot_dir.mkdir(exist_ok=True)
        
        # Process each baseline
        baseline_idx = 0
        for ant1 in range(num_antennas):
            for ant2 in range(ant1 + 1, num_antennas):
                
                # Process this baseline
                self._corrupt_baseline(
                    ms_path, baseline_idx, ant1, ant2,
                    rfi_config, include_flags,
                    ntimes, nchans, npols, plot_dir
                )
                
                baseline_idx += 1
                
                if baseline_idx % 50 == 0:
                    logger.info(f"Processed {baseline_idx}/{num_baselines} baselines")
                    self.print_memory(f"Baseline {baseline_idx}")

    def _corrupt_baseline(
        self,
        ms_path: str,
        baseline_idx: int,
        ant1: int,
        ant2: int,
        rfi_config: RFIConfig,
        include_flags: bool,
        ntimes: int,
        nchans: int,
        npols: int,
        plot_dir: Path
    ):
        """Read baseline, add RFI, write back, and plot (if first few baselines)"""
        
        # Calculate row range for this baseline
        start_row = baseline_idx * ntimes
        end_row = start_row + ntimes
        
        # Open MS for reading/writing
        self.tb.open(ms_path, nomodify=False)
        
        try:
            # Read baseline data
            baseline_data = []
            for row in range(start_row, end_row):
                row_data = self.tb.getcell("DATA", row)  # [npol, nchan]
                baseline_data.append(row_data.T)  # Convert to [nchan, npol]
            
            baseline_vis = np.array(baseline_data)  # [ntime, nchan, npol]
            
            # Add Gaussian thermal noise
            noise_sigma = self.obs_config.thermal_noise_sigma
            real_noise = np.random.normal(0, noise_sigma, baseline_vis.shape)
            imag_noise = np.random.normal(0, noise_sigma, baseline_vis.shape)
            thermal_noise = real_noise + 1j * imag_noise
            baseline_vis += thermal_noise
            
            clean_baseline = baseline_vis.copy()  # Save clean for plotting
            
            # Generate RFI for this baseline only
            rfi_array, rfi_mask = self._generate_baseline_rfi(
                baseline_vis.shape, ant1, ant2, rfi_config
            )
            
            # Add RFI to baseline (in-place)
            baseline_vis += rfi_array
            
            # Write corrupted data back
            for t, row in enumerate(range(start_row, end_row)):
                # Convert back to CASA format [npol, nchan]
                casa_data = baseline_vis[t].T
                self.tb.putcell("DATA", row, casa_data)
                
                # Write flags if requested
                if include_flags and rfi_mask is not None:
                    casa_flags = rfi_mask[t].T  # [nchan, npol] -> [npol, nchan]
                    self.tb.putcell("FLAG", row, casa_flags)
            
        finally:
            self.tb.close()
        
        # Create SAM tile plot for first few baselines
        if baseline_idx < 5:  # Plot first 5 baselines only
            self._create_baseline_sam_plots(
                clean_baseline, baseline_vis, rfi_mask, 
                baseline_idx, plot_dir
            )
        
        # Immediate cleanup
        del baseline_vis, clean_baseline, rfi_array, rfi_mask
        import gc
        gc.collect()

    def _generate_baseline_rfi(
        self, 
        shape,  # [ntime, nchan, npol]
        ant1: int, 
        ant2: int, 
        rfi_config: RFIConfig
    ):
        """Generate realistic RFI for single baseline with heavy contamination (25%+)"""
        
        ntime, nchan, npol = shape
        
        # Create RFI array and mask
        rfi_array = np.zeros(shape, dtype=np.complex64)
        rfi_mask = np.zeros(shape, dtype=bool)
        
        # RFI amplitudes: 10^3 to 10^6 times noise level from config
        base_noise_level = self.obs_config.thermal_noise_sigma
        
        # 1. Broadband RFI with polynomial frequency variation (25% occupancy)
        if rfi_config.broadband_probability > 0:
            # Generate broadband events covering portions of the band
            n_broadband_events = np.random.randint(3, 8)  # Multiple events
            
            for _ in range(n_broadband_events):
                # Random frequency range (portion of band)
                freq_start = np.random.randint(0, nchan//2)
                freq_width = np.random.randint(nchan//8, nchan//3)  # 12.5% to 33% of band
                freq_end = min(freq_start + freq_width, nchan)
                
                # Random time range
                time_start = np.random.randint(0, ntime//2)
                time_width = np.random.randint(ntime//4, ntime)
                time_end = min(time_start + time_width, ntime)
                
                # Polynomial amplitude variation across frequency
                freq_indices = np.arange(freq_end - freq_start)
                # Parabolic/cubic envelope
                poly_coeffs = np.random.uniform(-0.5, 0.5, 4)  # Cubic polynomial
                normalized_freq = (freq_indices - freq_indices.mean()) / (freq_indices.std() + 1e-6)
                amplitude_envelope = np.polyval(poly_coeffs, normalized_freq)
                amplitude_envelope = np.abs(amplitude_envelope)  # Ensure positive
                
                # Normalize envelope to [0,1] range to prevent explosive amplitudes
                if amplitude_envelope.max() > amplitude_envelope.min():
                    amplitude_envelope = (amplitude_envelope - amplitude_envelope.min()) / (amplitude_envelope.max() - amplitude_envelope.min())
                else:
                    amplitude_envelope = np.ones_like(amplitude_envelope)
                
                # Scale to RFI amplitude range
                base_amplitude = np.random.uniform(10.0, 1000.0) * base_noise_level
                amplitude_envelope = base_amplitude * (1 + amplitude_envelope)
                
                # Apply to all times and polarizations in this event
                for t in range(time_start, time_end):
                    for f_idx, f in enumerate(range(freq_start, freq_end)):
                        phase = np.random.uniform(0, 2*np.pi)
                        amp = amplitude_envelope[f_idx]
                        rfi_array[t, f, :] = amp * np.exp(1j * phase)
                        rfi_mask[t, f, :] = True
        
        # 2. Narrowband persistent lines (10-20 lines)
        n_narrowband = getattr(rfi_config, 'narrowband_lines', 15)
        for _ in range(n_narrowband):
            freq_idx = np.random.randint(0, nchan)
            amplitude = np.random.uniform(50.0, 500.0) * base_noise_level
            phase = np.random.uniform(0, 2*np.pi)
            
            # Persistent across all time
            rfi_array[:, freq_idx, :] = amplitude * np.exp(1j * phase)
            rfi_mask[:, freq_idx, :] = True
        
        # 3. Transient pulses and repeated bursts
        n_transients = getattr(rfi_config, 'transient_events', 8)
        for _ in range(n_transients):
            # Single pulse or repeated bursts
            is_repeated = np.random.random() < 0.5
            
            if is_repeated:
                # Repeated short bursts
                burst_duration = np.random.randint(2, 8)  # Short bursts
                burst_interval = np.random.randint(20, 50)  # Repeat interval
                n_bursts = min(5, ntime // burst_interval)
                
                for burst_idx in range(n_bursts):
                    t_start = burst_idx * burst_interval + np.random.randint(0, 10)
                    t_end = min(t_start + burst_duration, ntime)
                    
                    # Affect random frequency range
                    f_start = np.random.randint(0, nchan//2)
                    f_width = np.random.randint(nchan//20, nchan//5)
                    f_end = min(f_start + f_width, nchan)
                    
                    amplitude = np.random.uniform(20.0, 200.0) * base_noise_level
                    phase = np.random.uniform(0, 2*np.pi)
                    
                    rfi_array[t_start:t_end, f_start:f_end, :] = amplitude * np.exp(1j * phase)
                    rfi_mask[t_start:t_end, f_start:f_end, :] = True
            else:
                # Single pulse
                t_start = np.random.randint(0, ntime-10)
                t_duration = np.random.randint(1, 5)  # Very short
                t_end = min(t_start + t_duration, ntime)
                
                # Affect broad frequency range
                f_start = np.random.randint(0, nchan//4)
                f_width = np.random.randint(nchan//8, nchan//2)
                f_end = min(f_start + f_width, nchan)
                
                amplitude = np.random.uniform(100.0, 800.0) * base_noise_level
                phase = np.random.uniform(0, 2*np.pi)
                
                rfi_array[t_start:t_end, f_start:f_end, :] = amplitude * np.exp(1j * phase)
                rfi_mask[t_start:t_end, f_start:f_end, :] = True
        
        # 4. Periodic signals (time-varying patterns)
        n_periodic = getattr(rfi_config, 'periodic_signals', 3)
        for _ in range(n_periodic):
            # Periodic pulse train
            period = np.random.randint(10, 30)  # Period in time steps
            duty_cycle = np.random.uniform(0.1, 0.3)  # 10-30% duty cycle
            pulse_width = int(period * duty_cycle)
            
            # Fixed frequency
            freq_idx = np.random.randint(0, nchan)
            amplitude = np.random.uniform(30.0, 300.0) * base_noise_level
            
            for t in range(0, ntime, period):
                t_end = min(t + pulse_width, ntime)
                phase = np.random.uniform(0, 2*np.pi)
                rfi_array[t:t_end, freq_idx, :] = amplitude * np.exp(1j * phase)
                rfi_mask[t:t_end, freq_idx, :] = True
        
        # 5. Satellite RFI (frequency drifting over time)
        n_satellites = getattr(rfi_config, 'satellite_passes', 2)
        for _ in range(n_satellites):
            # Linear frequency drift
            t_start = np.random.randint(0, ntime//4)
            t_duration = np.random.randint(ntime//8, ntime//2)
            t_end = min(t_start + t_duration, ntime)
            
            f_start = np.random.randint(0, nchan//2)
            f_end = np.random.randint(nchan//2, nchan)
            
            amplitude = np.random.uniform(40.0, 400.0) * base_noise_level
            
            # Linear drift from f_start to f_end over time
            for t_idx, t in enumerate(range(t_start, t_end)):
                progress = t_idx / max(1, (t_end - t_start - 1))
                current_freq = int(f_start + progress * (f_end - f_start))
                if 0 <= current_freq < nchan:
                    phase = np.random.uniform(0, 2*np.pi)
                    rfi_array[t, current_freq, :] = amplitude * np.exp(1j * phase)
                    rfi_mask[t, current_freq, :] = True
        
        logger.info(f"RFI occupancy: {np.mean(rfi_mask) * 100:.1f}% for baseline {ant1}-{ant2}")
        
        return rfi_array, rfi_mask

    def _create_baseline_sam_plots(self, clean_baseline, corrupted_baseline, rfi_mask, baseline_idx, plot_dir):
        """Create 1024x1024 SAM tile plots for this baseline"""
        
        import matplotlib.pyplot as plt
        
        ntime, nchan, npols = clean_baseline.shape
        
        for pol_idx in range(npols):
            # Extract data for this polarization [time, freq] - what SAM sees
            clean_tile = clean_baseline[:, :, pol_idx]
            corrupted_tile = corrupted_baseline[:, :, pol_idx]  
            rfi_tile = rfi_mask[:, :, pol_idx] if rfi_mask is not None else np.zeros_like(clean_tile, dtype=bool)
            
            # Create 4-panel plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Baseline {baseline_idx}, Pol {pol_idx} - SAM Input Tile ({ntime}x{nchan})')
            
            # Show amplitude (log scale to see noise structure)
            clean_amp = np.abs(clean_tile)
            im1 = axes[0,0].imshow(np.log10(clean_amp + 1e-10), 
                                 aspect='auto', cmap='viridis', origin='lower')
            axes[0,0].set_title('Clean Data (log amp)')
            axes[0,0].set_xlabel('Frequency')
            axes[0,0].set_ylabel('Time')
            plt.colorbar(im1, ax=axes[0,0])
            
            corrupted_amp = np.abs(corrupted_tile) 
            im2 = axes[0,1].imshow(np.log10(corrupted_amp + 1e-10), 
                                 aspect='auto', cmap='viridis', origin='lower')
            axes[0,1].set_title('Corrupted Data (log amp)')
            axes[0,1].set_xlabel('Frequency')
            axes[0,1].set_ylabel('Time')
            plt.colorbar(im2, ax=axes[0,1])
            
            # RFI mask
            im3 = axes[1,0].imshow(rfi_tile.astype(float), 
                                 aspect='auto', cmap='Reds', origin='lower')
            axes[1,0].set_title('RFI Mask')
            axes[1,0].set_xlabel('Frequency')
            axes[1,0].set_ylabel('Time')
            plt.colorbar(im3, ax=axes[1,0])
            
            # RFI only (difference)
            rfi_only = corrupted_amp - clean_amp
            im4 = axes[1,1].imshow(np.log10(np.abs(rfi_only) + 1e-10), 
                                 aspect='auto', cmap='plasma', origin='lower')
            axes[1,1].set_title('RFI Only (log amp)')
            axes[1,1].set_xlabel('Frequency')
            axes[1,1].set_ylabel('Time')
            plt.colorbar(im4, ax=axes[1,1])
            
            plt.tight_layout()
            
            # Save plot
            plot_file = plot_dir / f'sam_tile_baseline_{baseline_idx:03d}_pol_{pol_idx}.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created plot: {plot_file}")

    def _get_antenna_config(self):
        """Get antenna configuration - borrowed from MSWriter"""
        array_configs = {
            "VLA": "vla.d.cfg",
            "SYNTHETIC_VLA": "vla.d.cfg", 
            "EVLA": "vla.d.cfg",
        }
        
        array_name = self.obs_config.array_name.upper()
        config_file = array_configs.get(array_name, "vla.d.cfg")
        
        try:
            antennalist = os.path.join(ctsys.resolve("alma/simmos"), config_file)
            (x, y, z, d, an, an2, telname, obspos) = self.mysu.readantenna(antennalist)
            logger.info(f"Loaded antenna config: {config_file} with {len(x)} antennas")
            return x, y, z, d, an, telname
        except Exception as e:
            logger.warning(f"Failed to load {config_file}, falling back to VLA: {e}")
            antennalist = os.path.join(ctsys.resolve("alma/simmos"), "vla.d.cfg")
            (x, y, z, d, an, an2, telname, obspos) = self.mysu.readantenna(antennalist)
            return x, y, z, d, an, telname

    def _save_training_data(self, ms_path: str, training_data_dir: str):
        """
        Extract and save training data from MS in .npy format
        
        Saves:
        - corrupted_visibilities.npy: [num_baselines, num_times, num_channels, num_pols]
        - rfi_mask.npy: [num_baselines, num_times, num_channels, num_pols]
        """
        if training_data_dir is None:
            training_data_dir = str(Path(ms_path).parent)
            
        training_dir = Path(training_data_dir)
        training_dir.mkdir(exist_ok=True, parents=True)
        
        # Get MS dimensions
        self.tb.open(ms_path, nomodify=True)
        try:
            nrows = self.tb.nrows()
            sample_data = self.tb.getcell("DATA", 0)
            npols, nchans = sample_data.shape
            self.tb.close()
        except Exception as e:
            self.tb.close()
            raise e
            
        # Calculate baselines and times
        num_antennas = self.obs_config.num_antennas
        num_baselines = num_antennas * (num_antennas - 1) // 2
        ntimes = nrows // num_baselines
        
        logger.info(f"Extracting training data: {num_baselines} baselines, {ntimes} times, {nchans} channels, {npols} pols")
        
        # Pre-allocate arrays
        corrupted_vis = np.zeros((num_baselines, ntimes, nchans, npols), dtype=np.complex64)
        rfi_mask = np.zeros((num_baselines, ntimes, nchans, npols), dtype=bool)
        
        # Extract data baseline-by-baseline
        self.tb.open(ms_path, nomodify=True)
        try:
            baseline_idx = 0
            for ant1 in range(num_antennas):
                for ant2 in range(ant1 + 1, num_antennas):
                    # Calculate row range for this baseline
                    start_row = baseline_idx * ntimes
                    end_row = start_row + ntimes
                    
                    # Read baseline data
                    for t, row in enumerate(range(start_row, end_row)):
                        # Data: [npol, nchan] -> [nchan, npol] -> store as [ntimes, nchans, npols]
                        row_data = self.tb.getcell("DATA", row).T  # [nchan, npol]
                        corrupted_vis[baseline_idx, t, :, :] = row_data
                        
                        # Flags: [npol, nchan] -> [nchan, npol] -> store as [ntimes, nchans, npols]
                        row_flags = self.tb.getcell("FLAG", row).T  # [nchan, npol]
                        rfi_mask[baseline_idx, t, :, :] = row_flags
                    
                    baseline_idx += 1
                    
                    if baseline_idx % 50 == 0:
                        logger.info(f"Extracted {baseline_idx}/{num_baselines} baselines")
        finally:
            self.tb.close()
        
        # Save arrays
        corrupted_path = training_dir / 'corrupted_visibilities.npy'
        mask_path = training_dir / 'rfi_mask.npy'
        
        np.save(corrupted_path, corrupted_vis)
        np.save(mask_path, rfi_mask)
        
        logger.info(f"Saved training data:")
        logger.info(f"  Corrupted visibilities: {corrupted_path}")
        logger.info(f"  RFI mask: {mask_path}")
        logger.info(f"  Shape: {corrupted_vis.shape}")