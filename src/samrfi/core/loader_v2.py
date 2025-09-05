"""
Memory-Efficient Measurement Set Loader v2
Uses casatools table and msmetadata for efficient MS access.
Based on legacy samrfi/radiorfi.py patterns.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy import stats

try:
    from casatools import table, msmetadata
    CASATOOLS_AVAILABLE = True
except ImportError:
    CASATOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BaselineInfo:
    """Information for a single baseline"""
    ant1: int
    ant2: int
    baseline_id: int


class MSMetadataExtractor:
    """Per-field metadata extraction using casatools.msmetadata"""
    
    def __init__(self, ms_path: str, field_id: int = 0):
        if not CASATOOLS_AVAILABLE:
            raise ImportError("casatools required but not available")
            
        self.ms_path = str(ms_path)
        self.field_id = field_id
        self.msmd = msmetadata()
        self.tb = table()
        self._is_open = False
        
    def open(self):
        """Open MS for metadata access"""
        if not self._is_open:
            self.msmd.open(self.ms_path)
            self._is_open = True
            
    def get_basic_info(self) -> Dict[str, Any]:
        """Get comprehensive MS metadata for specified field"""
        self.open()
        
        try:
            # Basic antenna info
            num_antennas = self.msmd.nantennas()
            
            # Generate baseline pairs (ant1 < ant2, following legacy pattern)
            baseline_pairs = []
            baseline_id = 0
            for i in range(num_antennas):
                for j in range(i + 1, num_antennas):
                    baseline_pairs.append(BaselineInfo(i, j, baseline_id))
                    baseline_id += 1
            
            num_baselines = len(baseline_pairs)
            
            # SPW information with shape checking
            nspw = self.msmd.nspw()
            spw_info = {}
            
            for spw_id in range(nspw):
                nchan = self.msmd.nchan(spw_id)
                chanfreqs = self.msmd.chanfreqs(spw_id)
                spw_info[spw_id] = {
                    'nchan': nchan,
                    'chanfreqs': chanfreqs
                }
            
            # Find compatible SPW groups by checking actual data shapes
            compatible_spws = self._find_compatible_spw_group(spw_info)
            
            if compatible_spws:
                target_nchan = spw_info[compatible_spws[0]]['nchan']
                total_channels = len(compatible_spws) * target_nchan
            else:
                target_nchan = 0
                total_channels = 0
            
            # Time information for this field
            time_info = self._get_field_time_info()
            
            return {
                'num_antennas': num_antennas,
                'num_baselines': num_baselines,
                'baseline_pairs': baseline_pairs,
                'spw_info': spw_info,
                'compatible_spws': compatible_spws,  # Renamed from same_nchan_spws
                'total_channels': total_channels,
                'time_info': time_info,
                'field_id': self.field_id,
                'target_nchan': target_nchan
            }
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            raise RuntimeError(f"MS metadata extraction failed: {e}")
    
    def _get_field_time_info(self) -> Dict[str, Any]:
        """Get time information for the specified field"""
        try:
            # Get times for this field
            times = self.msmd.timesforfield(self.field_id)
            if len(times) == 0:
                raise ValueError(f"No data found for field {self.field_id}")
                
            return {
                'start_time': float(np.min(times)),
                'end_time': float(np.max(times)),
                'ntime': len(np.unique(times)),
                'times': times
            }
        except Exception as e:
            logger.warning(f"Could not get field time info: {e}")
            return {
                'start_time': 0.0,
                'end_time': 0.0,
                'ntime': 0,
                'times': np.array([])
            }
    
    def _find_compatible_spw_group(self, spw_info: Dict) -> List[int]:
        """
        Find the largest group of SPWs with compatible shapes using baseband information.
        Groups SPWs by baseband and channel count, returns group with most total channels.
        """
        if not spw_info:
            return []
        
        logger.info("Finding compatible SPW groups using baseband information...")
        
        try:
            # Group SPWs by baseband and channel count
            baseband_groups = {}  # (baseband, nchan) -> [spw_ids]
            
            for spw_id in spw_info.keys():
                try:
                    # Get baseband for this SPW
                    baseband = self.msmd.baseband(spw_id)
                    nchan = spw_info[spw_id]['nchan']
                    
                    group_key = (baseband, nchan)
                    if group_key not in baseband_groups:
                        baseband_groups[group_key] = []
                    baseband_groups[group_key].append(spw_id)
                    
                    logger.info(f"SPW {spw_id}: baseband {baseband}, {nchan} channels")
                    
                except Exception as e:
                    logger.warning(f"Could not get baseband for SPW {spw_id}: {e}")
                    # If baseband info not available, group by channel count only
                    nchan = spw_info[spw_id]['nchan']
                    group_key = ('unknown', nchan)
                    if group_key not in baseband_groups:
                        baseband_groups[group_key] = []
                    baseband_groups[group_key].append(spw_id)
            
            if not baseband_groups:
                logger.warning("No SPW groups found")
                return []
            
            # Find the group with the most total channels
            best_group = []
            max_total_channels = 0
            best_group_key = None
            
            for group_key, spw_list in baseband_groups.items():
                baseband, nchan = group_key
                total_channels = len(spw_list) * nchan
                
                logger.info(f"Baseband {baseband}, {nchan} chan/SPW: {len(spw_list)} SPWs, {total_channels} total channels")
                
                if total_channels > max_total_channels:
                    max_total_channels = total_channels
                    best_group = spw_list
                    best_group_key = group_key
            
            logger.info(f"Selected baseband group {best_group_key}: SPWs {best_group} ({max_total_channels} total channels)")
            
            # Log skipped groups
            if len(baseband_groups) > 1:
                skipped_spws = []
                for group_key, spw_list in baseband_groups.items():
                    if group_key != best_group_key:
                        skipped_spws.extend(spw_list)
                        baseband, nchan = group_key
                        logger.info(f"Skipped baseband {baseband} group: SPWs {spw_list}")
                logger.info(f"Total skipped SPWs: {skipped_spws}")
            
            # Validate shapes within selected baseband (as fallback check)
            if not self._validate_spw_group_shapes(sorted(best_group)):
                raise RuntimeError(f"Selected SPW group has incompatible data shapes. Consider manual SPW selection.")
            
            return sorted(best_group)
            
        except Exception as e:
            logger.error(f"Error finding baseband SPW groups: {e}")
            logger.error("Consider using manual SPW selection or checking data consistency")
            raise RuntimeError(f"SPW grouping failed: {e}")
    
    def _validate_spw_group_shapes(self, spw_list: List[int]) -> bool:
        """
        Validate that SPWs in a group have compatible data shapes.
        Quick check to catch shape mismatches within a baseband.
        """
        if len(spw_list) <= 1:
            return True
        
        try:
            # Check a few SPWs to validate shape consistency
            tb_temp = table()
            tb_temp.open(self.ms_path, nomodify=True)
            
            reference_shape = None
            sample_spws = spw_list[:min(3, len(spw_list))]  # Check first 3 SPWs
            
            for spw_id in sample_spws:
                query_str = f'DATA_DESC_ID=={spw_id} && ANTENNA1==0 && ANTENNA2==1 && FIELD_ID=={self.field_id}'
                subtable = tb_temp.query(query_str)
                
                if subtable.nrows() > 0:
                    data_sample = subtable.getcol('DATA')
                    current_shape = (data_sample.shape[1], data_sample.shape[2])  # (nchan, ntime)
                    
                    if reference_shape is None:
                        reference_shape = current_shape
                    elif reference_shape != current_shape:
                        logger.error(f"Shape mismatch in baseband group: SPW {sample_spws[0]} has {reference_shape}, SPW {spw_id} has {current_shape}")
                        tb_temp.close()
                        return False
                
                subtable.close()
            
            tb_temp.close()
            logger.info(f"Shape validation passed for baseband group: {reference_shape}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not validate shapes: {e}")
            return True  # Assume valid if we can't check
    
    def get_flagging_stats(self) -> Dict[str, Any]:
        """Get pre-flagging statistics"""
        # Placeholder for pre-flagging stats
        # Will be implemented when we have data loading working
        return {
            'total_data_points': 0,
            'pre_flagged_percentage': 0.0
        }
    
    def close(self):
        """Close metadata connection"""
        if self._is_open:
            self.msmd.close()
            self._is_open = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MSLoader:
    """Memory-aware baseline loader using casatools.table"""
    
    def __init__(self, ms_path: str, field_id: int = 0):
        if not CASATOOLS_AVAILABLE:
            raise ImportError("casatools required but not available")
            
        self.ms_path = str(ms_path)
        self.field_id = field_id
        self.tb = table()
        self._metadata = None
        self._baseband_info = None
        self._is_open = False
        
    def open(self):
        """Open MS table for data access"""
        if not self._is_open:
            self.tb.open(self.ms_path, nomodify=True)
            self._is_open = True
    
    def load_baseline_data(self, ant1: int, ant2: int, spw_group_id: int = None) -> Dict[str, Any]:
        """
        Load data for a single baseline, specific SPW group, following legacy pattern
        
        Args:
            ant1: First antenna (must be < ant2)
            ant2: Second antenna
            spw_group_id: Specific SPW group to process (None = auto-select first available)
            
        Returns:
            dict with keys: 'data', 'existing_flags', 'metadata', 'stats'
        """
        if ant1 >= ant2:
            raise ValueError(f"ant1 ({ant1}) must be < ant2 ({ant2})")
            
        self.open()
        
        # Get SPW group info if not cached
        if self._baseband_info is None:
            self._baseband_info = self._get_scan_based_spw_groups()
        
        # Get SPWs for specified or first available group
        if spw_group_id is not None:
            if spw_group_id not in self._baseband_info:
                raise ValueError(f"SPW group {spw_group_id} not found. Available: {list(self._baseband_info.keys())}")
            compatible_spws = self._baseband_info[spw_group_id]
        else:
            # Auto-select first available group
            if not self._baseband_info:
                raise RuntimeError("No SPW groups found in MS")
            first_group = list(self._baseband_info.keys())[0]
            compatible_spws = self._baseband_info[first_group]
            spw_group_id = first_group
            logger.info(f"Auto-selected SPW group {spw_group_id}")
        
        if not compatible_spws:
            raise RuntimeError(f"No SPWs found for group {spw_group_id}")
        
        # Calculate channels for this SPW group
        # Assume all SPWs in group have same channel count
        with MSMetadataExtractor(self.ms_path, self.field_id) as meta:
            spw_info = meta.get_basic_info()['spw_info']
            target_nchan = spw_info[compatible_spws[0]]['nchan']
            total_channels = len(compatible_spws) * target_nchan
        
        try:
            # Load data from all matching SPWs (following legacy pattern)
            spw_data_list = []
            spw_flags_list = []
            spw_metadata = {}
            
            logger.info(f"Loading baseline {ant1}-{ant2}, SPWs: {compatible_spws}")
            
            for spw_idx, spw_id in enumerate(compatible_spws):
                # Query following legacy pattern: DATA_DESC_ID, ANTENNA1, ANTENNA2, FIELD_ID
                query_str = f'DATA_DESC_ID=={spw_id} && ANTENNA1=={ant1} && ANTENNA2=={ant2} && FIELD_ID=={self.field_id}'
                
                try:
                    subtable = self.tb.query(query_str)
                    if subtable.nrows() == 0:
                        logger.warning(f"No data for baseline {ant1}-{ant2}, SPW {spw_id}, field {self.field_id}")
                        continue
                        
                    # Get data and flags - casatools format: [npol, nchan, ntime]
                    spw_data = subtable.getcol('DATA')
                    spw_flags = subtable.getcol('FLAG')
                    
                    logger.info(f"SPW {spw_id} data shape: {spw_data.shape}")
                    
                    spw_data_list.append(spw_data)
                    spw_flags_list.append(spw_flags)
                    
                    # Store metadata for flag writing
                    spw_metadata[spw_id] = {
                        'start_chan': spw_idx * target_nchan,
                        'end_chan': (spw_idx + 1) * target_nchan,
                        'nchan': target_nchan
                    }
                    
                    subtable.close()
                    
                except Exception as e:
                    logger.error(f"Failed to load SPW {spw_id}: {e}")
                    continue
            
            if not spw_data_list:
                raise RuntimeError(f"No data loaded for baseline {ant1}-{ant2}")
            
            # Concatenate SPWs following legacy pattern
            combined_data = self._concatenate_spws(spw_data_list, target_nchan)
            combined_flags = self._concatenate_spws(spw_flags_list, target_nchan)
            
            logger.info(f"Combined data shape: {combined_data.shape}")
            
            # Calculate pre-flagging statistics
            stats = self._calculate_baseline_stats(combined_data, combined_flags, ant1, ant2)
            
            return {
                'data': combined_data,
                'existing_flags': combined_flags,
                'metadata': {
                    'spw_metadata': spw_metadata,
                    'ant1': ant1,
                    'ant2': ant2,
                    'field_id': self.field_id,
                    'spw_group_id': spw_group_id,
                    'total_channels': total_channels,
                    'target_nchan': target_nchan
                },
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error loading baseline data: {e}")
            raise RuntimeError(f"Failed to load baseline {ant1}-{ant2}: {e}")
    
    def _concatenate_spws(self, spw_data_list: List[np.ndarray], target_nchan: int) -> np.ndarray:
        """Concatenate SPW data following legacy pattern"""
        if not spw_data_list:
            raise ValueError("No SPW data to concatenate")
        
        # Get shape from first SPW
        first_shape = spw_data_list[0].shape  # [npol, nchan, ntime]
        npol, nchan, ntime = first_shape
        
        if nchan != target_nchan:
            raise ValueError(f"Channel count mismatch: expected {target_nchan}, got {nchan}")
        
        # Create combined array
        total_channels = len(spw_data_list) * target_nchan
        combined_data = np.zeros((npol, total_channels, ntime), dtype=spw_data_list[0].dtype)
        
        # Fill data following legacy pattern: spw*nchan:(spw+1)*nchan
        for spw_idx, spw_data in enumerate(spw_data_list):
            start_chan = spw_idx * target_nchan
            end_chan = (spw_idx + 1) * target_nchan
            combined_data[:, start_chan:end_chan, :] = spw_data
        
        return combined_data
    
    def apply_existing_flags_as_zeros(self, data: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Apply existing flags by zeroing flagged data"""
        if data.shape != flags.shape:
            raise ValueError(f"Data and flags shape mismatch: {data.shape} vs {flags.shape}")
        
        # Zero out flagged data (flags=True means flagged)
        masked_data = data.copy()
        masked_data[flags] = 0.0
        
        return masked_data
    
    def generate_1024x1024_tiles(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Generate 1024x1024 tiles for SAM processing
        
        Args:
            data: [npol, channels, time] array
            
        Returns:
            List of 4 tiles (one per polarization), each [1024, 1024]
        """
        npol, channels, ntime = data.shape
        
        # Pad/truncate to 1024 channels and 1024 time steps
        padded_data = self._pad_to_1024x1024(data)
        
        # Create tiles per polarization
        tiles = []
        for pol in range(npol):
            # Take magnitude for complex data
            if np.iscomplexobj(padded_data):
                tile = np.abs(padded_data[pol, :, :])
            else:
                tile = padded_data[pol, :, :]
            
            tiles.append(tile)
        
        return tiles
    
    def _pad_to_1024x1024(self, data: np.ndarray) -> np.ndarray:
        """Pad or truncate data to 1024x1024 following zero-pad strategy"""
        npol, channels, ntime = data.shape
        
        # Handle channel dimension
        if channels < 1024:
            # Zero-pad channels
            pad_channels = 1024 - channels
            channel_padded = np.pad(data, ((0, 0), (0, pad_channels), (0, 0)), mode='constant')
        elif channels > 1024:
            # Truncate channels (could also chunk into multiple tiles)
            channel_padded = data[:, :1024, :]
            logger.warning(f"Truncating {channels} channels to 1024")
        else:
            channel_padded = data
        
        # Handle time dimension
        _, _, ntime = channel_padded.shape
        if ntime < 1024:
            # Zero-pad time
            pad_time = 1024 - ntime
            time_padded = np.pad(channel_padded, ((0, 0), (0, 0), (0, pad_time)), mode='constant')
        elif ntime > 1024:
            # Truncate time
            time_padded = channel_padded[:, :, :1024]
            logger.warning(f"Truncating {ntime} time steps to 1024")
        else:
            time_padded = channel_padded
        
        return time_padded
    
    def _get_scan_based_spw_groups(self) -> Dict[int, List[int]]:
        """Get SPW groups based on scan temporal structure"""
        try:
            with MSMetadataExtractor(self.ms_path, self.field_id) as meta:
                # Get all scans for this field
                scan_numbers = meta.msmd.scannumbers()
                logger.info(f"Found {len(scan_numbers)} scans: {scan_numbers}")
                
                # Collect SPWs used in each scan
                scan_to_spws = {}
                for scan_num in scan_numbers:
                    try:
                        spws_in_scan = meta.msmd.spwsforscan(scan_num)
                        scan_to_spws[scan_num] = sorted(spws_in_scan)
                        logger.info(f"Scan {scan_num}: SPWs {sorted(spws_in_scan)}")
                    except Exception as e:
                        logger.warning(f"Could not get SPWs for scan {scan_num}: {e}")
                
                # Group scans by their SPW sets
                spw_groups = {}  # spw_set_key -> [scan_numbers]
                spw_set_to_spws = {}  # spw_set_key -> [spw_list]
                
                for scan_num, spw_list in scan_to_spws.items():
                    spw_key = tuple(sorted(spw_list))  # Use tuple as dict key
                    
                    if spw_key not in spw_groups:
                        spw_groups[spw_key] = []
                        spw_set_to_spws[spw_key] = list(spw_list)
                    
                    spw_groups[spw_key].append(scan_num)
                
                # Convert to ordered groups (0, 1, 2, ...) by chronological order
                ordered_groups = {}
                group_id = 0
                
                # Sort SPW sets by their first scan number (chronological order)
                sorted_spw_sets = sorted(spw_groups.items(), key=lambda x: min(x[1]))
                
                for spw_key, scan_list in sorted_spw_sets:
                    spw_list = spw_set_to_spws[spw_key]
                    ordered_groups[group_id] = sorted(spw_list)
                    
                    logger.info(f"SPW Group {group_id}: SPWs {sorted(spw_list)} (scans {sorted(scan_list)})")
                    group_id += 1
                
                logger.info(f"Created {len(ordered_groups)} temporal SPW groups")
                return ordered_groups
                
        except Exception as e:
            logger.error(f"Failed to get scan-based SPW groups: {e}")
            raise RuntimeError(f"Could not determine scan structure: {e}")
    
    def get_available_spw_groups(self) -> List[int]:
        """Get list of available SPW group IDs"""
        if self._baseband_info is None:
            self._baseband_info = self._get_scan_based_spw_groups()
        return list(self._baseband_info.keys())
    
    def _calculate_baseline_stats(self, data: np.ndarray, flags: np.ndarray, 
                                 ant1: int, ant2: int) -> Dict[str, Any]:
        """Calculate pre-flagging statistics for baseline"""
        baseline_stats = {
            'ant1': ant1,
            'ant2': ant2,
            'per_pol_stats': {}
        }
        
        npol = data.shape[0]
        
        for pol in range(npol):
            pol_data = data[pol]
            pol_flags = flags[pol]
            
            # Get unflagged data
            unflagged_data = pol_data[~pol_flags]
            
            if len(unflagged_data) > 0:
                # Handle complex data
                if np.iscomplexobj(unflagged_data):
                    # Use magnitude for statistics
                    unflagged_real = np.abs(unflagged_data)
                else:
                    unflagged_real = unflagged_data.real
                
                pol_stats = {
                    'std': float(np.std(unflagged_real)),
                    'skewness': float(stats.skew(unflagged_real.flatten())),
                    'kurtosis': float(stats.kurtosis(unflagged_real.flatten())),
                    'pct_flagged': float(np.sum(pol_flags) / pol_flags.size * 100),
                    'total_points': int(pol_flags.size),
                    'flagged_points': int(np.sum(pol_flags))
                }
            else:
                # All data flagged
                pol_stats = {
                    'std': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'pct_flagged': 100.0,
                    'total_points': int(pol_flags.size),
                    'flagged_points': int(pol_flags.size)
                }
            
            baseline_stats['per_pol_stats'][pol] = pol_stats
        
        return baseline_stats
    
    def close(self):
        """Close table connection"""
        if self._is_open:
            self.tb.close()
            self._is_open = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()