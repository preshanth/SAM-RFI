"""
CASA Measurement Set Writer

Creates proper CASA-compatible measurement sets from synthetic data.
Compatible with aoflagger, CASA flaggers (tfcrop, rflag), and other tools.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
import shutil
import os

# Import required CASA tools/tasks
try:
    from casatools import simulator, table, measures, quanta, ctsys
    from casatasks import flagdata
    from casatasks.private import simutil
    
    CASA_AVAILABLE = True
except ImportError as e:
    CASA_AVAILABLE = False
    logging.warning(f"CASA tools not available: {e}")

from .synthetic_ms_legacy import ObservationConfig, SyntheticVisibilityGenerator

logger = logging.getLogger(__name__)


class MSWriter:
    """
    Write synthetic data to proper CASA measurement set format using casatools/casatasks
    """

    def __init__(self, obs_config: ObservationConfig):
        if not CASA_AVAILABLE:
            raise ImportError("CASA tools/tasks required for MS writing")

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
            # mysu doesn't have done() method
            import gc
            gc.collect()
        except:
            pass  # Ignore cleanup errors

    def create_measurement_set(
        self,
        ms_path: str,
        visibilities: np.ndarray,
        rfi_mask: Optional[np.ndarray] = None,
        include_rfi_flags: bool = True,
    ) -> None:
        """
        Create complete measurement set with all required tables

        Args:
            ms_path: Output measurement set path
            visibilities: Visibility data [baseline, time, freq, pol]
            rfi_mask: Optional RFI mask for ground truth flags
            include_rfi_flags: Whether to include RFI flags in FLAG column
        """
        ms_path = Path(ms_path)

        # Remove existing MS
        if ms_path.exists():
            logger.warning(f"Removing existing MS: {ms_path}")
            shutil.rmtree(ms_path)

        logger.info(f"Creating measurement set: {ms_path}")

        # Validate visibilities shape
        shape = visibilities.shape
        if len(shape) != 4:
            raise ValueError(f"visibilities must be 4D, got shape: {shape}")
            
        # Assume shape is [baseline, time, freq, pol]
        num_baselines, num_times, num_channels, num_pols = shape
        logger.info(f"Visibilities shape: baselines={num_baselines}, times={num_times}, channels={num_channels}, pols={num_pols}")

        # Create the MS frame using CASA simulator (this may update obs_config.num_antennas)
        original_num_antennas = self.obs_config.num_antennas
        self._create_ms_frame(str(ms_path), num_times, num_channels)
        
        # Check if antenna count changed and regenerate data if needed
        if self.obs_config.num_antennas != original_num_antennas:
            logger.info("Antenna count changed, regenerating synthetic data...")
            from .synthetic_ms import SyntheticVisibilityGenerator, RFIConfig
            
            # Create a simple RFI config for regeneration
            rfi_config = RFIConfig(
                broadband_probability=0.02,
                narrowband_lines=2,
                transient_events=2
            )
            
            generator = SyntheticVisibilityGenerator(self.obs_config, rfi_config)
            clean_vis = generator.generate_clean_visibilities()
            visibilities, rfi_mask = generator.inject_rfi(clean_vis)
            
            logger.info(f"Regenerated visibilities shape: {visibilities.shape}")
        
        # Fill with visibility data
        self._fill_visibility_data(str(ms_path), visibilities, rfi_mask, include_rfi_flags)

        logger.info(f"Measurement set created successfully: {ms_path}")

    def _create_ms_frame(self, ms_path: str, num_times: int, num_channels: int) -> None:
        """Create empty MS frame with proper structure using CASA simulator"""
        
        # Remove any existing MS
        os.system(f'rm -rf {ms_path}')

        # Open the simulator
        self.sm.open(ms=ms_path)

        # Get antenna configuration from array name
        antenna_config = self._get_antenna_config()
        x, y, z, d, an, telname = antenna_config
        
        # If requested antennas < available antennas, slice to requested count
        if self.obs_config.num_antennas < len(x):
            x = x[:self.obs_config.num_antennas]
            y = y[:self.obs_config.num_antennas] 
            z = z[:self.obs_config.num_antennas]
            d = d[:self.obs_config.num_antennas]
            an = an[:self.obs_config.num_antennas]
            logger.info(f"Using first {self.obs_config.num_antennas} antennas from {len(antenna_config[0])}-antenna {telname} config")
        else:
            # Keep all antennas and update config if we have fewer than requested
            actual_num_antennas = len(x)
            if actual_num_antennas != self.obs_config.num_antennas:
                logger.info(f"Updating antenna count from {self.obs_config.num_antennas} to {actual_num_antennas} (from {telname} config)")
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
        total_bandwidth = num_channels * self.obs_config.channel_width
        self.sm.setspwindow(
            spwname="SPW0",
            freq=f'{self.obs_config.start_frequency}Hz',
            deltafreq=f'{self.obs_config.channel_width}Hz', 
            freqresolution=f'{self.obs_config.channel_width}Hz',
            nchannels=num_channels,
            stokes='XX XY YX YY'
        )

        # Set field/source
        ra_str = f"{self.obs_config.source_ra * 180 / np.pi}deg"  
        dec_str = f"{self.obs_config.source_dec * 180 / np.pi}deg"
        
        self.sm.setfield(
            sourcename=self.obs_config.source_name,
            sourcedirection=self.me.direction('J2000', ra_str, dec_str)
        )

        # Set limits
        self.sm.setlimits(shadowlimit=0.01, elevationlimit='10deg')

        # No autocorrelations
        self.sm.setauto(autocorrwt=0.0)

        # Set timing
        ref_time = self.obs_config.start_time.replace('T', '/')
        self.sm.settimes(
            integrationtime=f'{self.obs_config.integration_time}s',
            usehourangle=False,
            referencetime=self.me.epoch('UTC', ref_time)
        )

        # Calculate observation duration
        obs_duration = num_times * self.obs_config.integration_time
        
        # Create single scan observation
        self.sm.observe(
            sourcename=self.obs_config.source_name,
            spwname='SPW0',
            starttime='0s',
            stoptime=f'{obs_duration}s'
        )

        # Close and destroy simulator
        self.sm.close()
        self.sm.done()

        # Unflag everything initially
        flagdata(vis=ms_path, mode='unflag')

        logger.info("MS frame created with CASA simulator")

    def _fill_visibility_data(
        self, 
        ms_path: str, 
        visibilities: np.ndarray, 
        rfi_mask: Optional[np.ndarray],
        include_rfi_flags: bool
    ) -> None:
        """Fill the MS with actual visibility data"""
        
        num_baselines, num_times, num_channels, num_pols = visibilities.shape
        logger.info(f"DEBUG: Input visibilities shape: {visibilities.shape}")
        logger.info(f"DEBUG: num_baselines={num_baselines}, num_times={num_times}, num_channels={num_channels}, num_pols={num_pols}")
        
        # Open MS table for writing
        self.tb.open(ms_path, nomodify=False)
        
        try:
            # Get number of rows in MS
            nrows = self.tb.nrows()
            expected_rows = num_baselines * num_times
            
            logger.info(f"DEBUG: MS has {nrows} rows")
            logger.info(f"DEBUG: Expected rows = {num_baselines} baselines × {num_times} times = {expected_rows}")
            
            if nrows != expected_rows:
                logger.warning(f"Row count mismatch: expected {expected_rows}, got {nrows}")
            
            # Check DATA column shape
            data_desc = self.tb.getcoldesc("DATA")
            logger.info(f"DEBUG: DATA column description: {data_desc}")
            
            # CASA format is [npol, nchan, nbaseline*ntime]
            logger.info(f"DEBUG: Creating data arrays for CASA format [npol={num_pols}, nchan={num_channels}, nrows={nrows}]")
            data_array = np.zeros((num_pols, num_channels, nrows), dtype=np.complex64)
            flag_array = np.zeros((num_pols, num_channels, nrows), dtype=bool)
            weight_array = np.ones((num_pols, nrows), dtype=np.float32)
            sigma_array = np.ones((num_pols, nrows), dtype=np.float32) * 0.1
            
            logger.info(f"DEBUG: Created data_array shape: {data_array.shape}")
            logger.info(f"DEBUG: Created flag_array shape: {flag_array.shape}")
            
            # Fill data in CASA format [npol, nchan, nrows]
            row = 0
            rows_filled = 0
            for baseline_idx in range(min(num_baselines, nrows)):
                for time_idx in range(min(num_times, nrows // num_baselines if num_baselines > 0 else 0)):
                    if row >= nrows:
                        logger.info(f"DEBUG: Stopping at row {row}, reached MS limit")
                        break
                        
                    # Get visibility data for this baseline/time [freq, pol]
                    if baseline_idx < visibilities.shape[0] and time_idx < visibilities.shape[1]:
                        vis_data = visibilities[baseline_idx, time_idx, :, :]  # [nchan, npol]
                        
                        # Fill CASA format [npol, nchan, row] - vectorized
                        data_array[:, :, row] = vis_data.T  # Transpose [chan,pol] -> [pol,chan]
                        
                        # Handle flags if provided - vectorized
                        if rfi_mask is not None and include_rfi_flags and baseline_idx < rfi_mask.shape[0] and time_idx < rfi_mask.shape[1]:
                            flag_data = rfi_mask[baseline_idx, time_idx, :, :]  # [nchan, npol]
                            flag_array[:, :, row] = flag_data.T  # Transpose [chan,pol] -> [pol,chan]
                        
                        rows_filled += 1
                    
                    row += 1
                
                if row >= nrows:
                    break
            
            logger.info(f"DEBUG: Filled {rows_filled} rows out of {nrows} total MS rows")
            logger.info(f"DEBUG: Final data_array shape before putcol: {data_array.shape}")
            logger.info(f"DEBUG: Expected format: [npol={num_pols}, nchan={num_channels}, nrows={nrows}]")
            
            # Write data to MS
            logger.info("DEBUG: About to write DATA column with shape: {data_array.shape}")
            self.tb.putcol("DATA", data_array)
            logger.info("DEBUG: DATA column written successfully")
            
            logger.info("DEBUG: About to write FLAG column with shape: {flag_array.shape}")
            self.tb.putcol("FLAG", flag_array) 
            logger.info("DEBUG: FLAG column written successfully")
            
            logger.info("DEBUG: About to write WEIGHT column with shape: {weight_array.shape}")
            self.tb.putcol("WEIGHT", weight_array)
            logger.info("DEBUG: WEIGHT column written successfully")
            
            logger.info("DEBUG: About to write SIGMA column with shape: {sigma_array.shape}")
            self.tb.putcol("SIGMA", sigma_array)
            logger.info("DEBUG: SIGMA column written successfully")
            
            # Set FLAG_ROW based on whether entire rows are flagged [nrows]
            flag_row = np.all(flag_array, axis=(0, 1))  # All pols and chans for each row
            logger.info(f"DEBUG: About to write FLAG_ROW column with shape: {flag_row.shape}")
            self.tb.putcol("FLAG_ROW", flag_row)
            logger.info("DEBUG: FLAG_ROW column written successfully")
            
            logger.info("DEBUG: All columns written successfully")
            logger.info("Visibility data written to MS")
            
        finally:
            self.tb.close()

    def _get_antenna_config(self) -> Tuple[List, List, List, List, List, str]:
        """Get antenna configuration based on array name"""
        
        # Map array names to CASA antenna configuration files
        array_configs = {
            "VLA": "vla.d.cfg",
            "SYNTHETIC_VLA": "vla.d.cfg", 
            "EVLA": "vla.d.cfg",
            "MEERKAT": "meerkat.cfg" if os.path.exists(os.path.join(ctsys.resolve("alma/simmos"), "meerkat.cfg")) else "vla.d.cfg",
            "ALMA": "alma.out20.cfg",
            "ATCA": "atca.cfg" if os.path.exists(os.path.join(ctsys.resolve("alma/simmos"), "atca.cfg")) else "vla.d.cfg"
        }
        
        # Default to VLA if array name not recognized
        array_name = self.obs_config.array_name.upper()
        config_file = array_configs.get(array_name, "vla.d.cfg")
        
        try:
            antennalist = os.path.join(ctsys.resolve("alma/simmos"), config_file)
            (x, y, z, d, an, an2, telname, obspos) = self.mysu.readantenna(antennalist)
            
            logger.info(f"Loaded antenna config: {config_file} with {len(x)} antennas")
            return x, y, z, d, an, telname
            
        except Exception as e:
            # Fallback to VLA if specific config fails
            logger.warning(f"Failed to load {config_file}, falling back to VLA: {e}")
            antennalist = os.path.join(ctsys.resolve("alma/simmos"), "vla.d.cfg")
            (x, y, z, d, an, an2, telname, obspos) = self.mysu.readantenna(antennalist)
            return x, y, z, d, an, telname

    def _generate_antenna_positions(self) -> np.ndarray:
        """Generate antenna positions for the array"""
        generator = SyntheticVisibilityGenerator(self.obs_config, None)
        return generator._generate_antenna_positions()

    def validate_measurement_set(self, ms_path: str) -> bool:
        """Validate the created measurement set"""
        try:
            self.tb.open(ms_path, nomodify=True)
            nrows = self.tb.nrows()
            self.tb.close()
            
            expected_rows = (
                self.obs_config.num_antennas * (self.obs_config.num_antennas - 1) // 2 *
                int(self.obs_config.total_duration / self.obs_config.integration_time)
            )
            
            if nrows == expected_rows:
                logger.info(f"✓ MS validation passed: {nrows} rows")
                return True
            else:
                logger.warning(f"Row count mismatch: expected {expected_rows}, got {nrows}")
                return False
                
        except Exception as e:
            logger.error(f"MS validation failed: {e}")
            return False