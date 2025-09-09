#!/usr/bin/env python3
"""
Test SAM-RFI Flag Application Pipeline

Tests the complete pipeline with untrained SAM2 model.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from samrfi.core import MSFlagApplicator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_pipeline(ms_path: str, n_baselines: int = 2):
    """Test the flag application pipeline"""
    
    logger.info("="*60)
    logger.info("SAM-RFI Flag Application Pipeline Test")
    logger.info("="*60)
    
    try:
        # Initialize flag applicator with untrained SAM2
        applicator = MSFlagApplicator(
            ms_path=ms_path,
            sam_device="cuda",  # Use GPU if available
            sam_variant="tiny",  # Use smallest model for testing
            sam_model_path=None,  # Untrained model
            field_id=0
        )
        
        logger.info("Flag applicator initialized successfully")
        
        # Get MS information
        ms_info = applicator.get_ms_info()
        logger.info(f"MS Info:")
        logger.info(f"  Path: {ms_info['ms_path']}")
        logger.info(f"  Antennas: {ms_info['n_antennas']}")
        logger.info(f"  Baselines: {ms_info['n_baselines']}")
        logger.info(f"  SPW Groups: {ms_info['n_spw_groups']}")
        
        # Test single baseline first
        baselines = applicator.loader.get_baselines()
        spw_groups = applicator.loader.get_spw_groups()
        
        if not baselines or not spw_groups:
            logger.error("No baselines or SPW groups found")
            return False
        
        # Process first few baselines in dry run mode
        logger.info(f"Testing with {min(n_baselines, len(baselines))} baselines (dry run)")
        
        for i in range(min(n_baselines, len(baselines))):
            ant1, ant2 = baselines[i]
            spw_group = spw_groups[0]  # Use first SPW group
            
            logger.info(f"Processing baseline {ant1}-{ant2}, SPW group {spw_group}")
            
            result = applicator.process_baseline(
                ant1, ant2, spw_group, dry_run=True
            )
            
            logger.info(f"  Data shape: {result['data_shape']}")
            logger.info(f"  Polarizations: {result['n_polarizations']}")  
            logger.info(f"  Flag fraction: {result['flag_fraction']:.3f}")
        
        logger.info("Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM-RFI flag application pipeline")
    parser.add_argument("ms_path", help="Path to measurement set")
    parser.add_argument("--n_baselines", type=int, default=2, 
                       help="Number of baselines to test (default: 2)")
    
    args = parser.parse_args()
    
    if not Path(args.ms_path).exists():
        logger.error(f"Measurement set not found: {args.ms_path}")
        return 1
    
    success = test_pipeline(args.ms_path, args.n_baselines)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())