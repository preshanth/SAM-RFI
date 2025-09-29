"""
Base class for training data generation with RFI injection

Provides abstract interface for generating measurement sets with RFI from various
data sources (CASA simulator, pure Python, etc.)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
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

from ..synthetic_ms_legacy import ObservationConfig, RFIConfig
from .rfi_generator import RFIGenerator

logger = logging.getLogger(__name__)


class TrainingDataGenerator(ABC):
    """
    Abstract base class for generating training measurement sets with RFI
    
    Coordinates MS creation, RFI injection, and optional training data export.
    Concrete implementations provide different clean data generation strategies.
    """

    def __init__(self, obs_config: ObservationConfig):
        self.obs_config = obs_config
        
        # Initialize CASA tools if available
        if CASA_AVAILABLE:
            self.sm = simulator()
            self.tb = table()
            self.me = measures()
            self.qa = quanta()
            self.mysu = simutil.simutil()
        else:
            logger.warning("CASA tools not available - some functionality may be limited")

    def __del__(self):
        """Destructor to cleanup CASA tools"""
        if not CASA_AVAILABLE:
            return
            
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
        logger.info(f"{label}: {mem_mb:.1f} MB")

    def create_training_ms(
        self,
        ms_path: str,
        rfi_config: Optional[RFIConfig] = None,
        include_rfi_flags: bool = True,
        save_training_data: bool = False,
        training_data_dir: Optional[str] = None,
    ) -> str:
        """
        Create training MS with optional RFI injection
        
        Args:
            ms_path: Output measurement set path
            rfi_config: RFI configuration for corruption (None = clean MS)
            include_rfi_flags: Whether to flag RFI-corrupted data
            save_training_data: Whether to save .npy files for training
            training_data_dir: Directory to save training .npy files
            
        Returns:
            Path to created MS
        """
        ms_path = Path(ms_path)
        
        # Remove existing MS
        if ms_path.exists():
            logger.info(f"Removing existing MS: {ms_path}")
            shutil.rmtree(ms_path)

        self.print_memory("Starting MS creation")
        
        # Step 1: Create MS structure
        logger.info(f"Creating MS structure: {ms_path}")
        self._create_ms_structure(str(ms_path))
        self.print_memory("MS structure created")
        
        # Step 2: Fill with clean data using concrete implementation
        logger.info("Filling MS with clean data")
        self._fill_ms_with_clean_data(str(ms_path))
        self.print_memory("Clean data filling complete")
        
        # Step 3: Add RFI baseline-by-baseline (if requested)
        if rfi_config is not None:
            logger.info("Adding RFI baseline-by-baseline")
            self._add_rfi_to_ms(str(ms_path), rfi_config, include_rfi_flags)
            self.print_memory("RFI injection complete")
        
        # Step 4: Save training data (.npy files) if requested
        if save_training_data:
            logger.info("Extracting training data from MS")
            self._save_training_data(str(ms_path), training_data_dir)
            self.print_memory("Training data extraction complete")
        
        logger.info(f"Training MS created: {ms_path}")
        return str(ms_path)

    @abstractmethod
    def _generate_clean_baseline(self, ant1: int, ant2: int, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate clean baseline data for specific antenna pair
        
        Args:
            ant1: First antenna index
            ant2: Second antenna index  
            shape: Data shape [ntime, nchan, npol]
            
        Returns:
            Clean complex visibility data with shape [ntime, nchan, npol]
        """
        pass

    def _create_ms_structure(self, ms_path: str):
        """Create MS structure using CASA simulator (common implementation)"""
        
        if not CASA_AVAILABLE:
            raise ImportError("CASA tools required for MS creation")
        
        # Remove any existing MS
        os.system(f'rm -rf {ms_path}')

        # Open the simulator
        self.sm.open(ms=ms_path)

        # Get antenna configuration
        antenna_config = self._get_antenna_config()
        x, y, z, d, an, telname = antenna_config
        
        # Slice to requested antenna count if needed
        if self.obs_config.num_antennas < len(x):
            x = x[:self.obs_config.num_antennas]
            y = y[:self.obs_config.num_antennas] 
            z = z[:self.obs_config.num_antennas]
            d = d[:self.obs_config.num_antennas]
            an = an[:self.obs_config.num_antennas]
            logger.info(f"Using first {self.obs_config.num_antennas} antennas from {len(antenna_config[0])}-antenna config")
        else:
            # Update config if we have fewer antennas than requested
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

        # Create observation
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
        try:
            flagdata(vis=ms_path, mode='unflag')
            logger.info("Unflagged MS data")
        except Exception as e:
            logger.warning(f"Failed to unflag: {e}")
        
        logger.info("MS structure created")

    def _fill_ms_with_clean_data(self, ms_path: str):
        """Fill MS with clean data using concrete implementation strategy"""
        
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
        
        logger.info(f"Filling MS: {num_baselines} baselines, {ntimes} times, {nchans} channels, {npols} pols")
        
        # Fill each baseline using concrete implementation
        baseline_idx = 0
        for ant1 in range(num_antennas):
            for ant2 in range(ant1 + 1, num_antennas):
                
                # Generate clean data for this baseline
                baseline_data = self._generate_clean_baseline(ant1, ant2, (ntimes, nchans, npols))
                
                # Write to MS
                self._write_baseline_to_ms(ms_path, baseline_idx, baseline_data, ntimes)
                
                baseline_idx += 1
                
                if baseline_idx % 50 == 0:
                    logger.info(f"Filled {baseline_idx}/{num_baselines} baselines")

    def _write_baseline_to_ms(self, ms_path: str, baseline_idx: int, baseline_data: np.ndarray, ntimes: int):
        """Write baseline data to MS"""
        
        # Calculate row range for this baseline
        start_row = baseline_idx * ntimes
        end_row = start_row + ntimes
        
        # Open MS for writing
        self.tb.open(ms_path, nomodify=False)
        
        try:
            # Write baseline data
            for t, row in enumerate(range(start_row, end_row)):
                # Convert [ntime, nchan, npol] -> [npol, nchan] for CASA
                casa_data = baseline_data[t].T
                self.tb.putcell("DATA", row, casa_data)
        finally:
            self.tb.close()

    def _add_rfi_to_ms(self, ms_path: str, rfi_config: RFIConfig, include_flags: bool):
        """Add RFI to MS baseline-by-baseline using RFIGenerator"""
        
        # Create RFI generator
        rfi_generator = RFIGenerator(rfi_config, self.obs_config)
        
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
        
        logger.info(f"Adding RFI: {num_baselines} baselines, {ntimes} times, {nchans} channels, {npols} pols")
        
        # Create plot directory
        plot_dir = Path(str(ms_path).replace('.ms', '_plots'))
        plot_dir.mkdir(exist_ok=True)
        
        # Process each baseline
        baseline_idx = 0
        for ant1 in range(num_antennas):
            for ant2 in range(ant1 + 1, num_antennas):
                
                # Corrupt this baseline
                self._corrupt_baseline(
                    ms_path, baseline_idx, ant1, ant2,
                    rfi_generator, include_flags,
                    ntimes, nchans, npols, plot_dir
                )
                
                baseline_idx += 1
                
                if baseline_idx % 50 == 0:
                    logger.info(f"Processed {baseline_idx}/{num_baselines} baselines")

    def _corrupt_baseline(
        self,
        ms_path: str,
        baseline_idx: int,
        ant1: int,
        ant2: int,
        rfi_generator: RFIGenerator,
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
            clean_baseline = baseline_vis.copy()  # Save clean for plotting
            
            # Generate RFI for this baseline
            rfi_array, rfi_mask = rfi_generator.generate_baseline_rfi(
                baseline_vis.shape, ant1, ant2
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

    def _get_antenna_config(self):
        """Get antenna configuration"""
        
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
        """Extract and save training data from MS in .npy format"""
        
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