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
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "mad": np.nan,
            "count": 0,
            "flagged_fraction": 1.0,
        }

    return {
        "mean": float(np.mean(clean_data)),
        "median": float(np.median(clean_data)),
        "std": float(np.std(clean_data)),
        "mad": float(compute_mad(clean_data)),
        "count": len(clean_data),
        "flagged_fraction": float(flagged_fraction),
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

    # Handle edge case: all flagged
    if np.isnan(stats_after["mad"]) or np.isnan(stats_after["std"]):
        return {"ffi": 0.0, "mad_reduction": 0.0, "std_reduction": 0.0, "flagged_fraction": 1.0}

    # MAD reduction (should decrease if RFI removed)
    mad_reduction = 1.0 - (stats_after["mad"] / stats_before["mad"])

    # STD reduction
    std_reduction = 1.0 - (stats_after["std"] / stats_before["std"])

    # FFI: Combined metric (weighted average)
    # Penalize over-flagging (flagged_fraction)
    # Reward noise reduction (mad_reduction, std_reduction)
    flagged_penalty = stats_after["flagged_fraction"]
    ffi = (0.5 * mad_reduction + 0.5 * std_reduction) * (1.0 - 0.5 * flagged_penalty)

    return {
        "ffi": float(ffi),
        "mad_reduction": float(mad_reduction),
        "std_reduction": float(std_reduction),
        "flagged_fraction": float(flagged_penalty),
    }


def compute_calcquality(data, flags, reference_data=None):
    """
    Compute calcquality metric from paper (lower is better).

    Components:
    - a: Sensitivity (max deviation ~3σ for Gaussian)
    - b: Mean shift (normalized mean difference)
    - c: Std shift (normalized std difference)
    - d: Overflagging penalty (>70% only)

    Args:
        data: Complex or real array
        flags: Boolean mask (True = flagged)
        reference_data: Optional baseline (if None, uses pre-flag stats)

    Returns:
        dict: {
            'calcquality': float (combined score),
            'sensitivity': float (component a),
            'mean_shift': float (component b),
            'std_shift': float (component c),
            'overflagging_penalty': float (component d),
            'flagged_pct': float,
            'components': dict (debug values)
        }
    """
    # Convert complex → magnitude
    if np.iscomplexobj(data):
        data = np.abs(data)

    # Reference statistics
    if reference_data is not None:
        if np.iscomplexobj(reference_data):
            reference_data = np.abs(reference_data)
        ref_stats = compute_statistics(reference_data, flags=None)
        ref_data = reference_data.ravel()
    else:
        ref_stats = compute_statistics(data, flags=None)
        ref_data = data.ravel()

    # Flagged statistics
    flag_stats = compute_statistics(data, flags=flags)

    rmean = ref_stats["mean"]
    rstd = ref_stats["std"]
    fmean = flag_stats["mean"]
    fstd = flag_stats["std"]
    pflag = flag_stats["flagged_fraction"] * 100

    # Edge case: all flagged or invalid
    if np.isnan(fmean) or np.isnan(fstd) or rstd < 1e-10:
        return {
            "calcquality": np.inf,
            "sensitivity": np.inf,
            "mean_shift": np.inf,
            "std_shift": np.inf,
            "overflagging_penalty": np.inf,
            "flagged_pct": float(pflag),
            "components": {},
        }

    # Max deviation
    rmax = np.max(ref_data)
    maxdev = (rmax - rmean) / rstd
    fdiff = fmean - rmean
    sdiff = fstd - rstd

    # Four components
    a = abs(abs(maxdev) - 3)  # Sensitivity
    b = abs(fdiff) / rstd - 1  # Mean shift
    c = abs(sdiff) / rstd  # Std shift
    d = max(0, (pflag - 70) / 10)  # Overflagging

    # Euclidean norm
    calcquality = np.sqrt(a**2 + b**2 + c**2 + d**2)

    return {
        "calcquality": float(calcquality),
        "sensitivity": float(a),
        "mean_shift": float(b),
        "std_shift": float(c),
        "overflagging_penalty": float(d),
        "flagged_pct": float(pflag),
        "components": {
            "rmean": float(rmean),
            "rstd": float(rstd),
            "fmean": float(fmean),
            "fstd": float(fstd),
            "rmax": float(rmax),
            "maxdev": float(maxdev),
            "fdiff": float(fdiff),
            "sdiff": float(sdiff),
        },
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

    print("\n" + "=" * 60)
    print("Statistics Comparison (Before/After Flagging)")
    print("=" * 60)

    print("\nBefore Flagging:")
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

    print("\nFlagging Fidelity Index (FFI):")
    print(f"  FFI:            {ffi_metrics['ffi']:.4f}")
    print(f"  MAD Reduction:  {ffi_metrics['mad_reduction']:.4f}")
    print(f"  STD Reduction:  {ffi_metrics['std_reduction']:.4f}")
