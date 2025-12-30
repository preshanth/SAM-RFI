#!/usr/bin/env python3
"""
Extract single baseline from MS to numpy array.

Usage:
    python scripts/extract_baseline_to_npy.py \
        --ms /path/to/observation.ms \
        --output baseline_data.npy \
        --ant1 0 --ant2 1 --pol 0
"""

import argparse

import numpy as np

from samrfi.data import MSLoader


def main():
    parser = argparse.ArgumentParser(description="Extract single baseline to .npy")
    parser.add_argument("--ms", required=True, help="Path to measurement set")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--ant1", type=int, default=0, help="First antenna")
    parser.add_argument("--ant2", type=int, default=1, help="Second antenna")
    parser.add_argument("--pol", type=int, default=0, help="Polarization (0=XX, 1=XY, 2=YX, 3=YY)")
    parser.add_argument("--mode", default="DATA", help="MS column (DATA, CORRECTED_DATA)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Extract Single Baseline to NumPy")
    print("=" * 60)
    print(f"MS: {args.ms}")
    print(f"Baseline: {args.ant1}-{args.ant2}")
    print(f"Polarization: {args.pol}")
    print(f"Output: {args.output}")

    # Load baseline
    loader = MSLoader(args.ms)
    baseline_data = loader.load_single_baseline(
        ant1=args.ant1, ant2=args.ant2, pol_idx=args.pol, mode=args.mode
    )

    # Save
    np.save(args.output, baseline_data)

    print("\n" + "=" * 60)
    print(f"Saved: {args.output}")
    print(f"Shape: {baseline_data.shape}")
    print(f"Dtype: {baseline_data.dtype}")
    print("=" * 60)


if __name__ == "__main__":
    main()
