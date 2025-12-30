#!/usr/bin/env python3
"""
Clear all flags from a measurement set.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from casatools import table


def clear_flags(ms_path):
    """Clear all flags in MS (set to False)."""
    print(f"Clearing flags in {ms_path}...")

    tb = table()
    tb.open(ms_path, nomodify=False)

    # Get FLAG column shape
    flags = tb.getcol("FLAG")
    print(f"  FLAG shape: {flags.shape}")
    print(f"  Currently flagged: {flags.sum() / flags.size * 100:.2f}%")

    # Set all to False
    flags[:] = False
    tb.putcol("FLAG", flags)

    # Verify
    flags_check = tb.getcol("FLAG")
    print(f"  After clearing: {flags_check.sum() / flags_check.size * 100:.2f}%")

    tb.close()
    print("  Done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clear_ms_flags.py <ms_path>")
        sys.exit(1)

    clear_flags(sys.argv[1])
