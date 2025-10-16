#!/usr/bin/env python
"""
Check MS Compatibility with SAM-RFI Patch Size

Quick utility to verify if MS dimensions work with target patch_size
"""

import argparse
import sys
from pathlib import Path

try:
    from samrfi.data import check_ms_compatibility
except ImportError:
    print("ERROR: SAM-RFI not installed")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Check MS compatibility with SAM-RFI patch size',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('ms', help='Path to measurement set')
    parser.add_argument('--patch-size', type=int, default=1024,
                       help='Patch size (default: 1024)')

    args = parser.parse_args()

    if not Path(args.ms).exists():
        print(f"ERROR: MS not found: {args.ms}")
        sys.exit(1)

    print(f"\nChecking MS compatibility...")
    print(f"  MS: {args.ms}")
    print(f"  Patch size: {args.patch_size}")
    print("="*70)

    try:
        info = check_ms_compatibility(args.ms, args.patch_size)

        print(f"\nMS Dimensions:")
        print(f"  Channels: {info['channels']}")
        print(f"  Times:    {info['times']}")

        print(f"\nCompatibility:")
        print(f"  Channels divisible by {args.patch_size}: {info['channels_divisible']}")
        print(f"  Times divisible by {args.patch_size}:    {info['times_divisible']}")

        if info['fully_compatible']:
            print(f"\n✓ FULLY COMPATIBLE")
            print("  No padding required!")
        else:
            print(f"\n⚠ PADDING REQUIRED:")
            if not info['channels_divisible']:
                print(f"    Channels: +{info['padding_required']['channels']} "
                      f"({info['padding_required']['channels']}/{info['channels']*100:.1f}%)")
            if not info['times_divisible']:
                print(f"    Times:    +{info['padding_required']['times']} "
                      f"({info['padding_required']['times']}/{info['times']*100:.1f}%)")

        print(f"\n{info['recommendation']}")
        print("="*70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
