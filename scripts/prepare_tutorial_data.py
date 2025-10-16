#!/usr/bin/env python
"""
Prepare VLA P-band tutorial data for SAM-RFI comparison

This script automates the initial CASA processing steps from the tutorial:
1. Download data (if needed)
2. Import SDM to MS with importasdm
3. Apply initial flags
4. Flag dead antennas
5. Apply Hanning smoothing

Output: 3C129_pband.ms ready for comparison script

Usage:
    python prepare_tutorial_data.py --data-dir ./tutorial_data --output ./processed_data
"""

import argparse
import sys
from pathlib import Path

try:
    from casatasks import importasdm, flagdata, hanningsmooth
    CASA_AVAILABLE = True
except ImportError:
    print("ERROR: CASA tasks not available. Run this script within CASA.")
    CASA_AVAILABLE = False

# Import download function from comparison script
import os
sys.path.insert(0, str(Path(__file__).parent))
from compare_flagging_methods import download_tutorial_data


def prepare_tutorial_ms(data_dir, output_dir, skip_download=False):
    """
    Prepare tutorial MS following VLA P-band guide

    Args:
        data_dir: Directory containing downloaded data
        output_dir: Directory for processed MS
        skip_download: If True, assume data already downloaded
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("VLA P-BAND TUTORIAL DATA PREPARATION")
    print("="*70)

    # Step 1: Download data if needed
    sdm_path = data_dir / "imag-test-copy.57080.956837025464"

    if not sdm_path.exists() and not skip_download:
        print("\n[Step 1/6] Downloading tutorial data...")
        sdm_path = download_tutorial_data(output_dir=data_dir, force=False)
    elif sdm_path.exists():
        print(f"\n[Step 1/6] Using existing data at: {sdm_path}")
    else:
        raise FileNotFoundError(f"SDM data not found at {sdm_path}. Run with --download first.")

    # Step 2: Import SDM to MS
    ms_raw = output_dir / "3C129.ms"
    importflags_file = output_dir / "importflags.txt"

    print(f"\n[Step 2/6] Importing SDM to MS...")
    print(f"  Input:  {sdm_path}")
    print(f"  Output: {ms_raw}")

    if ms_raw.exists():
        print(f"  Removing existing MS: {ms_raw}")
        import shutil
        shutil.rmtree(ms_raw)

    importasdm(
        asdm=str(sdm_path),
        vis=str(ms_raw),
        savecmds=True,
        outfile=str(importflags_file)
    )
    print(f"  ✓ MS created: {ms_raw}")

    # Step 3: Apply initial flags
    print(f"\n[Step 3/6] Applying initial flags...")

    # Add clip zeros and shadow flags to importflags.txt
    with open(importflags_file, 'a') as f:
        f.write("\nmode='clip' clipzeros=True\n")
        f.write("mode='shadow' tolerance=0.0\n")

    flagdata(
        vis=str(ms_raw),
        mode='list',
        inpfile=str(importflags_file),
        action='apply',
        reason='any',
        flagbackup=True
    )
    print("  ✓ Initial flags applied")

    # Step 4: Flag dead antennas and setup scans
    print(f"\n[Step 4/6] Flagging dead antennas and setup scans...")

    # Flag ea19 (dead antenna)
    flagdata(vis=str(ms_raw), mode='manual', antenna='ea19')
    print("  ✓ Flagged ea19 (dead antenna)")

    # Flag setup scans
    flagdata(vis=str(ms_raw), mode='manual', scan='1~2')
    print("  ✓ Flagged setup scans 1-2")

    # Step 5: Hanning smoothing
    ms_hanning = output_dir / "3C129_pband.ms"

    print(f"\n[Step 5/6] Applying Hanning smoothing...")
    print(f"  Input:  {ms_raw}")
    print(f"  Output: {ms_hanning}")
    print("  This may take several minutes...")

    if ms_hanning.exists():
        print(f"  Removing existing MS: {ms_hanning}")
        import shutil
        shutil.rmtree(ms_hanning)

    hanningsmooth(
        vis=str(ms_raw),
        outputvis=str(ms_hanning),
        datacolumn='data',
        spw='0~15'  # First 16 spectral windows
    )
    print(f"  ✓ Hanning-smoothed MS created: {ms_hanning}")

    # Step 6: Summary
    print(f"\n[Step 6/6] Summary")
    print("="*70)
    print(f"✓ Tutorial data prepared successfully!")
    print(f"\nOutput MS: {ms_hanning}")
    print(f"\nThis MS is ready for the comparison script:")
    print(f"\n  python compare_flagging_methods.py \\")
    print(f"      --ms {ms_hanning} \\")
    print(f"      --model /path/to/sam2_model.pth \\")
    print(f"      --output ./comparison_results/")
    print("="*70)

    return ms_hanning


def main():
    parser = argparse.ArgumentParser(
        description='Prepare VLA P-band tutorial data for SAM-RFI comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the CASA processing steps from the VLA P-band guide:
  1. Download data (optional)
  2. Import SDM
  3. Apply initial flags
  4. Flag dead antennas
  5. Hanning smoothing

Examples:
  # Download and process
  python prepare_tutorial_data.py --download --data-dir ./data --output ./processed

  # Process existing downloaded data
  python prepare_tutorial_data.py --data-dir ./data --output ./processed --skip-download
        """
    )

    parser.add_argument('--download', action='store_true',
                        help='Download tutorial data before processing')
    parser.add_argument('--data-dir', default='./tutorial_data',
                        help='Directory containing (or for) SDM data')
    parser.add_argument('--output', default='./processed_data',
                        help='Output directory for processed MS')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, use existing data')

    args = parser.parse_args()

    if not CASA_AVAILABLE:
        print("ERROR: This script must be run within CASA")
        print("Start CASA and run: execfile('prepare_tutorial_data.py')")
        sys.exit(1)

    # Download if requested
    if args.download:
        print("Downloading tutorial data...")
        download_tutorial_data(output_dir=args.data_dir, force=False)

    # Prepare MS
    try:
        ms_path = prepare_tutorial_ms(
            data_dir=args.data_dir,
            output_dir=args.output,
            skip_download=args.skip_download or args.download
        )
        print(f"\n✓ Success! MS ready at: {ms_path}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
