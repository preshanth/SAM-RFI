"""
MS Flag Applicator

Orchestrates the complete pipeline: MS loading → tiling → SAM inference → flag application.
Uses existing MSLoader/MSFlagger infrastructure with new tiling and inference components.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from .loader import MSLoader, MSMetadataExtractor  
from .flagger import MSFlagger, StatisticsTracker
from .tiling import TilingProcessor
from .inference import SAMInferenceEngine

logger = logging.getLogger(__name__)


class MSFlagApplicator:
    """Complete MS flag application pipeline using SAM2"""
    
    def __init__(self, ms_path: str, 
                 sam_device: str = "cuda",
                 sam_variant: str = "large", 
                 sam_model_path: Optional[str] = None,
                 field_id: int = 0):
        """
        Initialize flag applicator
        
        Args:
            ms_path: Path to measurement set
            sam_device: Device for SAM2 model (cuda/cpu)
            sam_variant: SAM2 variant (tiny, small, base_plus, large)
            sam_model_path: Path to trained model (None = untrained SAM2)
            field_id: Field ID to process
        """
        self.ms_path = Path(ms_path)
        if not self.ms_path.exists():
            raise FileNotFoundError(f"MS not found: {ms_path}")
        
        # Initialize components
        self.loader = MSLoader(str(ms_path), field_id=field_id)
        self.flagger = MSFlagger(str(ms_path))
        self.tiling = TilingProcessor(patch_size=256)
        self.inference = SAMInferenceEngine(
            device=sam_device, 
            variant=sam_variant,
            local_model_path=sam_model_path
        )
        self.stats = StatisticsTracker()
        
        # Get MS metadata
        self.metadata_extractor = MSMetadataExtractor(str(ms_path))
        self.field_id = field_id
        
        logger.info(f"Initialized MSFlagApplicator for {ms_path}")
        logger.info(f"SAM2 config: {sam_variant} on {sam_device}")
        
    def process_all_baselines(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Process all baselines in the MS
        
        Args:
            dry_run: If True, don't actually write flags
            
        Returns:
            Summary statistics and processing info
        """
        baselines = self.loader.get_baselines()
        spw_groups = self.loader.get_spw_groups()
        
        total_baselines = len(baselines) * len(spw_groups)
        processed_count = 0
        
        logger.info(f"Processing {len(baselines)} baselines × {len(spw_groups)} SPW groups "
                   f"= {total_baselines} total combinations")
        
        processing_stats = {
            'total_combinations': total_baselines,
            'processed_count': 0,
            'failed_count': 0,
            'baseline_results': []
        }
        
        for spw_group_id in spw_groups:
            for ant1, ant2 in baselines:
                try:
                    result = self.process_baseline(ant1, ant2, spw_group_id, dry_run=dry_run)
                    processing_stats['baseline_results'].append(result)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{total_baselines} baseline combinations")
                        
                except Exception as e:
                    logger.error(f"Failed to process baseline {ant1}-{ant2} SPW group {spw_group_id}: {e}")
                    processing_stats['failed_count'] += 1
        
        processing_stats['processed_count'] = processed_count
        
        # Generate summary report
        if not dry_run:
            summary = self.stats.get_summary_statistics()
            processing_stats['flagging_summary'] = summary
        
        logger.info(f"Processing complete: {processed_count}/{total_baselines} successful")
        return processing_stats
    
    def process_baseline(self, ant1: int, ant2: int, spw_group_id: int, 
                        dry_run: bool = False) -> Dict[str, Any]:
        """
        Process a single baseline
        
        Args:
            ant1, ant2: Antenna indices
            spw_group_id: SPW group identifier
            dry_run: If True, don't write flags
            
        Returns:
            Processing results for this baseline
        """
        logger.debug(f"Processing baseline {ant1}-{ant2}, SPW group {spw_group_id}")
        
        # 1. Load baseline data (existing infrastructure)
        baseline_data = self.loader.load_baseline_data(ant1, ant2, spw_group_id)
        data = baseline_data['data']  # [time, freq, pol]
        existing_flags = baseline_data['existing_flags']
        metadata = baseline_data['metadata']
        
        # Apply existing flags as zeros
        masked_data = self.loader.apply_existing_flags_as_zeros(data, existing_flags)
        
        # 2. Generate 1024x1024 tiles per polarization  
        tiles_per_pol = []
        new_flags_per_pol = []
        
        for pol_idx in range(masked_data.shape[2]):  # Loop over polarizations
            pol_data = masked_data[:, :, pol_idx]  # [time, freq]
            
            # Generate 1024x1024 tile (existing method)
            tile_1024 = self.loader.generate_1024x1024_tiles(pol_data)  # [1024, 1024]
            
            # Convert to 3-channel format for SAM2 (simulate RGB channels)
            # For untrained SAM, duplicate channels
            tile_3ch = np.stack([tile_1024, tile_1024, tile_1024], axis=0)  # [3, 1024, 1024]
            
            # 3. Split into 256x256 patches (new tiling infrastructure)
            tiling_result = self.tiling.create_patches(tile_3ch)  # Input: [3, 1024, 1024]
            patches = tiling_result['patches']  # [3, 4, 256, 256] - 3 channels, 4 patches
            tiling_metadata = tiling_result['metadata']
            
            # Reshape for inference: [4, 3, 256, 256] - 4 patches of 3 channels each
            patches_for_inference = patches.transpose(1, 0, 2, 3)
            
            # 4. Run SAM2 inference on patches (new inference engine)
            patch_masks = self.inference.predict_patches(patches_for_inference)  # List of 4 masks
            
            # 5. Compute union of patch masks
            if patch_masks:
                union_mask_patches = self.inference.compute_union_mask(patch_masks)  # [256, 256] union of 4
            else:
                union_mask_patches = np.zeros((256, 256), dtype=np.float32)
            
            # Stack patches back for reconstruction: [4, 256, 256] -> [1, 4, 256, 256]
            mask_patches_stacked = np.stack(patch_masks, axis=0)[np.newaxis]  # [1, 4, 256, 256]
            
            # 6. Reconstruct full 1024x1024 mask
            full_mask = self.tiling.reconstruct_from_patches(mask_patches_stacked, tiling_metadata)
            full_mask = full_mask.squeeze()  # [1024, 1024]
            
            # Convert to boolean flags
            flags_1024 = full_mask > 0.5
            
            tiles_per_pol.append(tile_1024)
            new_flags_per_pol.append(flags_1024)
        
        # 7. Write flags using existing infrastructure
        if not dry_run:
            # Combine polarization flags - flag if ANY polarization is flagged
            combined_flags = np.any(np.stack(new_flags_per_pol, axis=2), axis=2)  # [1024, 1024]
            
            # Write flags (existing method)
            flag_result = self.flagger.write_baseline_flags(
                ant1, ant2, combined_flags, metadata, mode="combine"
            )
        else:
            flag_result = {"dry_run": True}
        
        # 8. Track statistics
        baseline_stats = {
            'ant1': ant1,
            'ant2': ant2, 
            'spw_group_id': spw_group_id,
            'data_shape': data.shape,
            'flags_applied': not dry_run,
            'n_polarizations': len(new_flags_per_pol),
            'flag_fraction': np.mean([flags.mean() for flags in new_flags_per_pol])
        }
        
        if not dry_run:
            self.stats.add_baseline_stats(baseline_stats)
        
        return baseline_stats
    
    def get_ms_info(self) -> Dict[str, Any]:
        """Get information about the measurement set"""
        try:
            antenna_info = self.metadata_extractor.get_antenna_info()
            field_info = self.metadata_extractor.get_field_info() 
            spw_info = self.metadata_extractor.get_spw_info()
            
            return {
                'ms_path': str(self.ms_path),
                'n_antennas': len(antenna_info),
                'n_baselines': len(self.loader.get_baselines()),
                'n_spw_groups': len(self.loader.get_spw_groups()),
                'field_id': self.field_id,
                'antenna_info': antenna_info,
                'field_info': field_info,
                'spw_info': spw_info
            }
        except Exception as e:
            logger.error(f"Error getting MS info: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate processing report"""
        ms_info = self.get_ms_info()
        model_info = self.inference.get_model_info()
        stats = self.stats.get_summary_statistics()
        
        report = f"""
SAM-RFI Processing Report
========================

Measurement Set: {self.ms_path}
Field ID: {self.field_id}
Antennas: {ms_info.get('n_antennas', 'unknown')}
Baselines: {ms_info.get('n_baselines', 'unknown')}
SPW Groups: {ms_info.get('n_spw_groups', 'unknown')}

SAM2 Model: {model_info.get('variant', 'unknown')} on {model_info.get('device', 'unknown')}
Model Status: {model_info.get('status', 'unknown')}

Processing Results:
------------------
{self.flagger.generate_flagging_report()}

Statistics Summary:
------------------
{stats}
"""
        return report