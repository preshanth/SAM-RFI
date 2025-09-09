#!/usr/bin/env python3
"""
Test program for new MSLoader and MSFlagger classes
Validates functionality against expectations from legacy code
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

import numpy as np
import logging
from pathlib import Path
import argparse

from samrfi.core import MSMetadataExtractor, MSLoader, MSFlagger, StatisticsTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_metadata_extraction(ms_path: str, field_id: int = 0):
    """Test metadata extraction functionality"""
    logger.info("=" * 60)
    logger.info("TESTING METADATA EXTRACTION")
    logger.info("=" * 60)
    
    try:
        with MSMetadataExtractor(ms_path, field_id) as meta:
            basic_info = meta.get_basic_info()
            
            logger.info(f"MS Path: {ms_path}")
            logger.info(f"Field ID: {field_id}")
            logger.info(f"Number of antennas: {basic_info['num_antennas']}")
            logger.info(f"Number of baselines: {basic_info['num_baselines']}")
            logger.info(f"SPW information:")
            
            for spw_id, spw_info in basic_info['spw_info'].items():
                logger.info(f"  SPW {spw_id}: {spw_info['nchan']} channels")
            
            logger.info(f"Compatible SPWs: {basic_info['compatible_spws']}")
            logger.info(f"Total channels (concatenated): {basic_info['total_channels']}")
            logger.info(f"Time info: {basic_info['time_info']['ntime']} time steps")
            
            # Show first few baselines
            logger.info("First 5 baselines:")
            for i, baseline in enumerate(basic_info['baseline_pairs'][:5]):
                logger.info(f"  Baseline {i}: {baseline.ant1}-{baseline.ant2}")
            
            return basic_info
            
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return None


def test_baseline_loading(ms_path: str, field_id: int = 0, test_baseline: int = 0):
    """Test baseline data loading with multi-baseband support"""
    logger.info("=" * 60)
    logger.info("TESTING BASELINE LOADING")
    logger.info("=" * 60)
    
    try:
        # Get metadata first to know valid baselines
        with MSMetadataExtractor(ms_path, field_id) as meta:
            basic_info = meta.get_basic_info()
        
        if test_baseline >= len(basic_info['baseline_pairs']):
            test_baseline = 0
        
        baseline_info = basic_info['baseline_pairs'][test_baseline]
        ant1, ant2 = baseline_info.ant1, baseline_info.ant2
        
        logger.info(f"Testing baseline {test_baseline}: {ant1}-{ant2}")
        
        with MSLoader(ms_path, field_id) as loader:
            # Get available SPW groups
            available_groups = loader.get_available_spw_groups()
            logger.info(f"Available SPW groups: {available_groups}")
            
            # Test each SPW group
            all_tiles = []
            all_metadata = []
            
            for group_id in available_groups:
                logger.info(f"Loading SPW group {group_id} for baseline {ant1}-{ant2}")
                
                baseline_data = loader.load_baseline_data(ant1, ant2, group_id)
                
                data = baseline_data['data']
                flags = baseline_data['existing_flags']
                metadata = baseline_data['metadata']
                stats = baseline_data['stats']
                
                logger.info(f"SPW Group {group_id} - Data shape: {data.shape}")
                logger.info(f"SPW Group {group_id} - Flags shape: {flags.shape}")
                logger.info(f"SPW Group {group_id} - Total channels: {metadata['total_channels']}")
                
                logger.info(f"SPW Group {group_id} - Pre-flagging statistics:")
                for pol, pol_stats in stats['per_pol_stats'].items():
                    logger.info(f"  Pol {pol}: {pol_stats['pct_flagged']:.1f}% flagged, "
                              f"std={pol_stats['std']:.3f}, skew={pol_stats['skewness']:.3f}")
                
                # Test applying existing flags as zeros
                logger.info(f"Testing flag masking for SPW group {group_id}...")
                masked_data = loader.apply_existing_flags_as_zeros(data, flags)
                
                # Check that flagged data is zeroed
                flagged_points = np.sum(flags)
                zero_points = np.sum(masked_data == 0.0)
                logger.info(f"Flagged points: {flagged_points}, Zero points after masking: {zero_points}")
                
                # Test tile generation
                logger.info(f"Testing 1024x1024 tile generation for SPW group {group_id}...")
                tiles = loader.generate_1024x1024_tiles(masked_data)
                logger.info(f"Generated {len(tiles)} tiles for SPW group {group_id}")
                for i, tile in enumerate(tiles):
                    logger.info(f"  Tile {i} (pol {i}): shape {tile.shape}, dtype {tile.dtype}")
                    logger.info(f"    Range: [{np.min(tile):.3f}, {np.max(tile):.3f}]")
                
                all_tiles.append((group_id, tiles))
                all_metadata.append((group_id, metadata))
            
            logger.info(f"Successfully processed {len(available_groups)} SPW groups for baseline {ant1}-{ant2}")
            return all_tiles, all_metadata
            
    except Exception as e:
        logger.error(f"Baseline loading failed: {e}")
        return None, None


def test_flag_writing(ms_path: str, field_id: int = 0, test_baseline: int = 0):
    """Test flag writing functionality"""
    logger.info("=" * 60)
    logger.info("TESTING FLAG WRITING")
    logger.info("=" * 60)
    
    try:
        # Load baseline data first
        with MSMetadataExtractor(ms_path, field_id) as meta:
            basic_info = meta.get_basic_info()
        
        if test_baseline >= len(basic_info['baseline_pairs']):
            test_baseline = 0
        
        baseline_info = basic_info['baseline_pairs'][test_baseline]
        ant1, ant2 = baseline_info.ant1, baseline_info.ant2
        
        logger.info(f"Testing flag writing for baseline {test_baseline}: {ant1}-{ant2}")
        
        with MSLoader(ms_path, field_id) as loader:
            baseline_data = loader.load_baseline_data(ant1, ant2)
        
        data_shape = baseline_data['data'].shape
        metadata = baseline_data['metadata']
        
        # Create some dummy new flags (simulate SAM output)
        logger.info("Creating simulated SAM flag output...")
        npol, channels, ntime = data_shape
        
        # Simulate flagging ~5% of data randomly
        dummy_flags = np.random.random((npol, channels, ntime)) < 0.05
        logger.info(f"Simulated flags shape: {dummy_flags.shape}")
        logger.info(f"Simulated flagging percentage: {np.sum(dummy_flags) / dummy_flags.size * 100:.2f}%")
        
        # Test reading current flags
        with MSFlagger(ms_path, field_id) as flagger:
            logger.info("Reading current flags from MS...")
            current_flags = flagger.get_current_flags(ant1, ant2, metadata['spw_metadata'])
            logger.info(f"Current flags shape: {current_flags.shape}")
            logger.info(f"Current flagging percentage: {np.sum(current_flags) / current_flags.size * 100:.2f}%")
            
            # Test combining flags
            logger.info("Testing flag combining...")
            combined_flags = dummy_flags | current_flags
            logger.info(f"Combined flagging percentage: {np.sum(combined_flags) / combined_flags.size * 100:.2f}%")
            
            # Test flag writing (in dry-run mode - we won't actually write)
            logger.info("Would write flags to MS (dry-run mode)")
            logger.info(f"Would combine {np.sum(dummy_flags)} new flags with {np.sum(current_flags)} existing flags")
            logger.info(f"Result would be {np.sum(combined_flags)} total flagged points")
            
            # Test statistics calculation
            logger.info("Testing statistics calculation...")
            pre_stats = baseline_data['stats']
            
            # Simulate post-flagging stats
            post_stats = {
                'ant1': ant1,
                'ant2': ant2,
                'field_id': field_id,
                'post_flagging_stats': {}
            }
            
            for pol in range(npol):
                pol_combined_flags = combined_flags[pol]
                post_stats['post_flagging_stats'][pol] = {
                    'pct_flagged': float(np.sum(pol_combined_flags) / pol_combined_flags.size * 100),
                    'total_points': int(pol_combined_flags.size),
                    'flagged_points': int(np.sum(pol_combined_flags))
                }
            
            # Generate report
            report = flagger.generate_flagging_report(pre_stats, post_stats)
            logger.info("Flagging report:")
            logger.info("\n" + str(report))
            
            return True
            
    except Exception as e:
        logger.error(f"Flag writing test failed: {e}")
        return False


def test_statistics_tracking():
    """Test statistics tracking functionality"""
    logger.info("=" * 60)
    logger.info("TESTING STATISTICS TRACKING")
    logger.info("=" * 60)
    
    try:
        # Create dummy data for testing
        logger.info("Creating test data...")
        np.random.seed(42)  # For reproducible results
        
        # Simulate complex visibility data
        data = np.random.normal(0, 1, (4, 100, 50)) + 1j * np.random.normal(0, 1, (4, 100, 50))
        flags = np.random.random((4, 100, 50)) < 0.1  # 10% flagged
        
        logger.info(f"Test data shape: {data.shape}")
        logger.info(f"Test flags shape: {flags.shape}")
        
        # Test data statistics calculation
        tracker = StatisticsTracker()
        
        for pol in range(4):
            stats = tracker.calculate_data_stats(data[pol], flags[pol])
            logger.info(f"Pol {pol} stats: {stats}")
        
        # Test flagging metrics calculation
        logger.info("Testing flagging metrics...")
        
        # Create dummy ground truth and predicted flags
        ground_truth = np.random.random((100, 50)) < 0.15  # 15% true RFI
        predicted = np.random.random((100, 50)) < 0.12     # 12% predicted RFI
        
        metrics = tracker.calculate_flagging_metrics(
            predicted, ground_truth, baseline_id=0, pol=0, ant1=0, ant2=1
        )
        
        logger.info(f"Flagging metrics:")
        logger.info(f"  True Positives: {metrics.true_positives}")
        logger.info(f"  True Negatives: {metrics.true_negatives}")
        logger.info(f"  False Positives: {metrics.false_positives}")
        logger.info(f"  False Negatives: {metrics.false_negatives}")
        logger.info(f"  Precision: {metrics.precision:.3f}")
        logger.info(f"  Recall: {metrics.recall:.3f}")
        logger.info(f"  F1 Score: {metrics.f1_score:.3f}")
        logger.info(f"  False Positive Rate: {metrics.false_positive_rate:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Statistics tracking test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test new MSLoader and MSFlagger classes")
    parser.add_argument("ms_path", nargs="?", help="Path to measurement set")
    parser.add_argument("--field-id", type=int, default=0, help="Field ID to test")
    parser.add_argument("--baseline", type=int, default=0, help="Baseline index to test")
    parser.add_argument("--stats-only", action="store_true", help="Only test statistics tracking")
    
    args = parser.parse_args()
    
    # Always test statistics
    logger.info("Starting SAM-RFI Loader/Flagger Test Suite")
    
    success_count = 0
    total_tests = 1
    
    if test_statistics_tracking():
        success_count += 1
    
    if not args.stats_only:
        if not args.ms_path:
            logger.error("MS path required for MS-based tests")
            logger.info("Use --stats-only to test only statistics functionality")
            sys.exit(1)
        
        if not Path(args.ms_path).exists():
            logger.error(f"Measurement set not found: {args.ms_path}")
            sys.exit(1)
        
        # Test with actual MS
        total_tests = 4
        
        if test_metadata_extraction(args.ms_path, args.field_id):
            success_count += 1
        
        all_tiles, all_metadata = test_baseline_loading(args.ms_path, args.field_id, args.baseline)
        if all_tiles is not None:
            success_count += 1
        
        if test_flag_writing(args.ms_path, args.field_id, args.baseline):
            success_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("All tests PASSED! 🎉")
        sys.exit(0)
    else:
        logger.error(f"{total_tests - success_count} tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()