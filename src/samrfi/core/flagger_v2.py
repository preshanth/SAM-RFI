"""
Measurement Set Flagger v2
Uses casatools table for flag writing and statistics tracking.
Based on legacy samrfi/radiorfi.py patterns with iterative flagging support.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy import stats
import pandas as pd

try:
    from casatools import table
    CASATOOLS_AVAILABLE = True
except ImportError:
    CASATOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FlaggingMetrics:
    """Container for flagging performance metrics"""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    baseline: int
    polarization: int
    antenna1: int
    antenna2: int


class StatisticsTracker:
    """Calculate comprehensive flagging statistics"""
    
    @staticmethod
    def calculate_data_stats(data: np.ndarray, flags: np.ndarray) -> Dict[str, float]:
        """Calculate statistical properties of unflagged data"""
        if data.shape != flags.shape:
            raise ValueError(f"Data and flags shape mismatch: {data.shape} vs {flags.shape}")
        
        unflagged_data = data[~flags]
        
        if len(unflagged_data) == 0:
            return {
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'pct_flagged': 100.0,
                'total_points': int(flags.size),
                'flagged_points': int(flags.size)
            }
        
        # Handle complex data
        if np.iscomplexobj(unflagged_data):
            unflagged_real = np.abs(unflagged_data)
        else:
            unflagged_real = unflagged_data.real
        
        return {
            'std': float(np.std(unflagged_real)),
            'skewness': float(stats.skew(unflagged_real.flatten())),
            'kurtosis': float(stats.kurtosis(unflagged_real.flatten())),
            'pct_flagged': float(np.sum(flags) / flags.size * 100),
            'total_points': int(flags.size),
            'flagged_points': int(np.sum(flags))
        }
    
    @staticmethod
    def calculate_flagging_metrics(predicted_flags: np.ndarray, 
                                  ground_truth_flags: np.ndarray,
                                  baseline_id: int, pol: int,
                                  ant1: int, ant2: int) -> FlaggingMetrics:
        """
        Calculate flagging performance metrics (TP/TN/FP/FN, precision, recall, F1)
        Following legacy metricscalculator.py patterns
        """
        if predicted_flags.shape != ground_truth_flags.shape:
            raise ValueError("Predicted and ground truth flags must have same shape")
        
        # Convert to int for calculations
        pred = predicted_flags.astype(int).flatten()
        truth = ground_truth_flags.astype(int).flatten()
        
        # Calculate confusion matrix elements
        tp = int(np.sum((pred == 1) & (truth == 1)))
        tn = int(np.sum((pred == 0) & (truth == 0)))
        fp = int(np.sum((pred == 1) & (truth == 0)))
        fn = int(np.sum((pred == 0) & (truth == 1)))
        
        # Calculate performance metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return FlaggingMetrics(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=false_positive_rate,
            baseline=baseline_id,
            polarization=pol,
            antenna1=ant1,
            antenna2=ant2
        )
    
    @staticmethod
    def calculate_per_antenna_stats(baseline_stats: List[Dict], 
                                   antenna_pairs: List[Tuple[int, int]]) -> Dict[int, Dict]:
        """Aggregate statistics per antenna from baseline data"""
        antenna_stats = {}
        
        for ant_pair in antenna_pairs:
            ant1, ant2 = ant_pair
            
            # Initialize antenna entries
            for ant_id in [ant1, ant2]:
                if ant_id not in antenna_stats:
                    antenna_stats[ant_id] = {
                        'baselines_count': 0,
                        'total_flagged_pct': [],
                        'std_values': [],
                        'skewness_values': [],
                        'kurtosis_values': []
                    }
        
        # Aggregate statistics
        for baseline_stat in baseline_stats:
            ant1 = baseline_stat['ant1']
            ant2 = baseline_stat['ant2']
            
            for ant_id in [ant1, ant2]:
                antenna_stats[ant_id]['baselines_count'] += 1
                
                # Aggregate across polarizations
                for pol_stats in baseline_stat['per_pol_stats'].values():
                    antenna_stats[ant_id]['total_flagged_pct'].append(pol_stats['pct_flagged'])
                    antenna_stats[ant_id]['std_values'].append(pol_stats['std'])
                    antenna_stats[ant_id]['skewness_values'].append(pol_stats['skewness'])
                    antenna_stats[ant_id]['kurtosis_values'].append(pol_stats['kurtosis'])
        
        # Calculate aggregated values
        for ant_id in antenna_stats:
            stats_dict = antenna_stats[ant_id]
            stats_dict['mean_flagged_pct'] = float(np.mean(stats_dict['total_flagged_pct']))
            stats_dict['mean_std'] = float(np.mean(stats_dict['std_values']))
            stats_dict['mean_skewness'] = float(np.mean(stats_dict['skewness_values']))
            stats_dict['mean_kurtosis'] = float(np.mean(stats_dict['kurtosis_values']))
        
        return antenna_stats


class MSFlagger:
    """Simple flag writer with statistics tracking using casatools.table"""
    
    def __init__(self, ms_path: str, field_id: int = 0):
        if not CASATOOLS_AVAILABLE:
            raise ImportError("casatools required but not available")
            
        self.ms_path = str(ms_path)
        self.field_id = field_id
        self.tb = table()
        self._is_open = False
        self.stats_tracker = StatisticsTracker()
        
    def open(self):
        """Open MS table for flag writing"""
        if not self._is_open:
            self.tb.open(self.ms_path, nomodify=False)
            self._is_open = True
    
    def get_current_flags(self, ant1: int, ant2: int, spw_metadata: Dict) -> np.ndarray:
        """Read current flag state from MS for a baseline"""
        if ant1 >= ant2:
            raise ValueError(f"ant1 ({ant1}) must be < ant2 ({ant2})")
            
        self.open()
        
        spw_flags_list = []
        
        try:
            for spw_id, meta in spw_metadata.items():
                query_str = f'DATA_DESC_ID=={spw_id} && ANTENNA1=={ant1} && ANTENNA2=={ant2} && FIELD_ID=={self.field_id}'
                
                subtable = self.tb.query(query_str)
                if subtable.nrows() == 0:
                    logger.warning(f"No data for baseline {ant1}-{ant2}, SPW {spw_id}")
                    continue
                
                spw_flags = subtable.getcol('FLAG')  # [npol, nchan, ntime]
                spw_flags_list.append(spw_flags)
                subtable.close()
            
            if not spw_flags_list:
                raise RuntimeError(f"No flag data found for baseline {ant1}-{ant2}")
            
            # Concatenate SPWs
            combined_flags = self._concatenate_spw_flags(spw_flags_list)
            return combined_flags
            
        except Exception as e:
            logger.error(f"Error reading current flags: {e}")
            raise RuntimeError(f"Failed to read flags for baseline {ant1}-{ant2}: {e}")
    
    def write_baseline_flags(self, ant1: int, ant2: int, 
                           new_flags: np.ndarray,
                           metadata: Dict,
                           mode: str = 'combine') -> Dict[str, Any]:
        """
        Write flags for a baseline, converting from 1024x1024 tiles back to SPW format
        
        Args:
            ant1, ant2: Antenna pair
            new_flags: SAM output flags [npol, 1024, ntime] or [npol, channels, ntime]
            metadata: Contains SPW mapping information  
            mode: 'combine' (OR with existing), 'replace' (overwrite)
            
        Returns:
            Post-flagging statistics
        """
        if ant1 >= ant2:
            raise ValueError(f"ant1 ({ant1}) must be < ant2 ({ant2})")
        
        if mode not in ['combine', 'replace']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'combine' or 'replace'")
            
        self.open()
        
        spw_metadata = metadata['spw_metadata']
        total_channels = metadata.get('total_channels', 0)
        target_nchan = metadata.get('target_nchan', 0)
        
        logger.info(f"Writing flags for baseline {ant1}-{ant2}, mode: {mode}")
        logger.info(f"New flags shape: {new_flags.shape}")
        
        try:
            # Convert 1024x1024 tiles back to original channel/time dimensions
            processed_flags = self._convert_tiles_to_original_shape(
                new_flags, total_channels, metadata
            )
            
            # Split back to individual SPWs and write
            for spw_id, spw_meta in spw_metadata.items():
                start_chan = spw_meta['start_chan']
                end_chan = spw_meta['end_chan']
                
                # Extract flags for this SPW
                spw_flags = processed_flags[:, start_chan:end_chan, :]
                
                # Get current flags if combining
                if mode == 'combine':
                    current_flags = self._get_current_spw_flags(ant1, ant2, spw_id)
                    if current_flags is not None:
                        # Combine: OR operation (True = flagged)
                        spw_flags = spw_flags | current_flags
                
                # Write flags to MS
                self._write_spw_flags(ant1, ant2, spw_id, spw_flags)
            
            # Calculate post-flagging statistics
            post_stats = self._calculate_post_flagging_stats(
                ant1, ant2, processed_flags, metadata
            )
            
            logger.info(f"Successfully wrote flags for baseline {ant1}-{ant2}")
            return post_stats
            
        except Exception as e:
            logger.error(f"Error writing baseline flags: {e}")
            raise RuntimeError(f"Failed to write flags for baseline {ant1}-{ant2}: {e}")
    
    def _convert_tiles_to_original_shape(self, tile_flags: np.ndarray, 
                                       total_channels: int, 
                                       metadata: Dict) -> np.ndarray:
        """Convert 1024x1024 tiles back to original [npol, channels, ntime] shape"""
        npol = tile_flags.shape[0]
        
        # If tiles are 1024x1024, extract the relevant portion
        if tile_flags.shape[1] == 1024 and tile_flags.shape[2] == 1024:
            # Extract up to total_channels and original time steps
            # For now, we'll take the first total_channels and assume time matches
            original_flags = tile_flags[:, :total_channels, :]
        else:
            # Flags are already in correct shape
            original_flags = tile_flags
        
        return original_flags
    
    def _get_current_spw_flags(self, ant1: int, ant2: int, spw_id: int) -> Optional[np.ndarray]:
        """Get current flags for a specific SPW"""
        try:
            query_str = f'DATA_DESC_ID=={spw_id} && ANTENNA1=={ant1} && ANTENNA2=={ant2} && FIELD_ID=={self.field_id}'
            subtable = self.tb.query(query_str)
            
            if subtable.nrows() == 0:
                return None
            
            current_flags = subtable.getcol('FLAG')
            subtable.close()
            return current_flags
            
        except Exception as e:
            logger.warning(f"Could not read current flags for SPW {spw_id}: {e}")
            return None
    
    def _write_spw_flags(self, ant1: int, ant2: int, spw_id: int, flags: np.ndarray):
        """Write flags to a specific SPW following legacy pattern"""
        query_str = f'DATA_DESC_ID=={spw_id} && ANTENNA1=={ant1} && ANTENNA2=={ant2} && FIELD_ID=={self.field_id}'
        
        subtable = self.tb.query(query_str)
        if subtable.nrows() == 0:
            logger.warning(f"No rows to update for SPW {spw_id}, baseline {ant1}-{ant2}")
            return
        
        logger.info(f"Writing flags for SPW {spw_id}, shape: {flags.shape}")
        
        # Write flags (following legacy putcol pattern)
        subtable.putcol('FLAG', flags)
        subtable.close()
        
        logger.info(f"Successfully wrote flags for SPW {spw_id}")
    
    def _concatenate_spw_flags(self, spw_flags_list: List[np.ndarray]) -> np.ndarray:
        """Concatenate flags from multiple SPWs"""
        if not spw_flags_list:
            raise ValueError("No SPW flags to concatenate")
        
        # Get shape info
        first_shape = spw_flags_list[0].shape  # [npol, nchan, ntime]
        npol, nchan, ntime = first_shape
        
        # Create combined array
        total_channels = len(spw_flags_list) * nchan
        combined_flags = np.zeros((npol, total_channels, ntime), dtype=bool)
        
        # Fill flags
        for spw_idx, spw_flags in enumerate(spw_flags_list):
            start_chan = spw_idx * nchan
            end_chan = (spw_idx + 1) * nchan
            combined_flags[:, start_chan:end_chan, :] = spw_flags
        
        return combined_flags
    
    def _calculate_post_flagging_stats(self, ant1: int, ant2: int, 
                                     flags: np.ndarray, 
                                     metadata: Dict) -> Dict[str, Any]:
        """Calculate post-flagging statistics"""
        stats = {
            'ant1': ant1,
            'ant2': ant2,
            'field_id': self.field_id,
            'post_flagging_stats': {}
        }
        
        npol = flags.shape[0]
        
        for pol in range(npol):
            pol_flags = flags[pol]
            
            pol_stats = {
                'pct_flagged': float(np.sum(pol_flags) / pol_flags.size * 100),
                'total_points': int(pol_flags.size),
                'flagged_points': int(np.sum(pol_flags))
            }
            
            stats['post_flagging_stats'][pol] = pol_stats
        
        return stats
    
    def backup_flags(self, backup_name: str) -> bool:
        """
        Backup current FLAG column (placeholder implementation)
        In practice, this could copy FLAG to FLAG_BACKUP or save to external file
        """
        logger.info(f"Flag backup requested: {backup_name}")
        logger.warning("Flag backup not implemented - consider manual MS backup")
        return True
    
    def generate_flagging_report(self, pre_stats: Dict, post_stats: Dict, 
                               ground_truth_flags: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Generate comprehensive flagging report"""
        report_data = []
        
        ant1 = pre_stats.get('ant1', 0)
        ant2 = pre_stats.get('ant2', 0)
        
        for pol in pre_stats.get('per_pol_stats', {}):
            pre_pol = pre_stats['per_pol_stats'][pol]
            post_pol = post_stats.get('post_flagging_stats', {}).get(pol, {})
            
            row_data = {
                'antenna1': ant1,
                'antenna2': ant2,
                'baseline': f"{ant1}-{ant2}",
                'polarization': pol,
                'pre_pct_flagged': pre_pol.get('pct_flagged', 0.0),
                'post_pct_flagged': post_pol.get('pct_flagged', 0.0),
                'std': pre_pol.get('std', 0.0),
                'skewness': pre_pol.get('skewness', 0.0),
                'kurtosis': pre_pol.get('kurtosis', 0.0),
            }
            
            # Add performance metrics if ground truth available
            if ground_truth_flags is not None:
                # This would require predicted flags - placeholder for now
                row_data.update({
                    'true_positives': 0,
                    'true_negatives': 0, 
                    'false_positives': 0,
                    'false_negatives': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'false_positive_rate': 0.0
                })
            
            report_data.append(row_data)
        
        return pd.DataFrame(report_data)
    
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