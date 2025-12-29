"""
Statistical analysis for RFI flagging quality assessment.

Compute descriptive statistics and flagging fidelity metrics.
"""

import numpy as np


def compute_mad(data):
    """Median Absolute Deviation (MAD)."""
    median = np.median(data)
    return np.median(np.abs(data - median))


def compute_statistics(data, flags=None):
    """
    Compute statistics on data, optionally with flagging.

    Args:
        data: Complex or real array
        flags: Boolean mask (True = flagged)

    Returns:
        dict with keys: mean, median, std, mad, count, flagged_fraction
    """
    # Use magnitude for complex data
    if np.iscomplexobj(data):
        data = np.abs(data)

    # Get unflagged data
    if flags is not None:
        clean_data = data[~flags]
        flagged_fraction = np.sum(flags) / flags.size
    else:
        clean_data = data.ravel()
        flagged_fraction = 0.0

    if len(clean_data) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'mad': np.nan,
            'count': 0,
            'flagged_fraction': 1.0
        }

    return {
        'mean': float(np.mean(clean_data)),
        'median': float(np.median(clean_data)),
        'std': float(np.std(clean_data)),
        'mad': float(compute_mad(clean_data)),
        'count': len(clean_data),
        'flagged_fraction': float(flagged_fraction)
    }


def compute_ffi(data, flags):
    """
    Flagging Fidelity Index (FFI).

    Measures quality of flagging by comparing statistics before/after.
    Higher FFI = better flagging (clean data preserved, RFI removed).

    Args:
        data: Complex or real array
        flags: Boolean mask (True = flagged)

    Returns:
        dict with keys: ffi, mad_reduction, std_reduction
    """
    stats_before = compute_statistics(data, flags=None)
    stats_after = compute_statistics(data, flags=flags)

    # MAD reduction (should decrease if RFI removed)
    mad_reduction = 1.0 - (stats_after['mad'] / stats_before['mad'])

    # STD reduction
    std_reduction = 1.0 - (stats_after['std'] / stats_before['std'])

    # FFI: Combined metric (weighted average)
    # Penalize over-flagging (flagged_fraction)
    # Reward noise reduction (mad_reduction, std_reduction)
    flagged_penalty = stats_after['flagged_fraction']
    ffi = (0.5 * mad_reduction + 0.5 * std_reduction) * (1.0 - 0.5 * flagged_penalty)

    return {
        'ffi': float(ffi),
        'mad_reduction': float(mad_reduction),
        'std_reduction': float(std_reduction),
        'flagged_fraction': float(flagged_penalty)
    }


def print_statistics_comparison(data, flags):
    """
    Print before/after statistics and FFI.

    Args:
        data: Complex or real array
        flags: Boolean mask
    """
    stats_before = compute_statistics(data, flags=None)
    stats_after = compute_statistics(data, flags=flags)
    ffi_metrics = compute_ffi(data, flags)

    print("\n" + "="*60)
    print("Statistics Comparison (Before/After Flagging)")
    print("="*60)

    print(f"\nBefore Flagging:")
    print(f"  Mean:   {stats_before['mean']:.4e}")
    print(f"  Median: {stats_before['median']:.4e}")
    print(f"  Std:    {stats_before['std']:.4e}")
    print(f"  MAD:    {stats_before['mad']:.4e}")
    print(f"  Count:  {stats_before['count']}")

    print(f"\nAfter Flagging ({stats_after['flagged_fraction']*100:.2f}% flagged):")
    print(f"  Mean:   {stats_after['mean']:.4e}")
    print(f"  Median: {stats_after['median']:.4e}")
    print(f"  Std:    {stats_after['std']:.4e}")
    print(f"  MAD:    {stats_after['mad']:.4e}")
    print(f"  Count:  {stats_after['count']}")

    print(f"\nFlagging Fidelity Index (FFI):")
    print(f"  FFI:            {ffi_metrics['ffi']:.4f}")
    print(f"  MAD Reduction:  {ffi_metrics['mad_reduction']:.4f}")
    print(f"  STD Reduction:  {ffi_metrics['std_reduction']:.4f}")
