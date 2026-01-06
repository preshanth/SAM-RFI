"""
Statistical analysis for RFI flagging quality assessment.

This module provides statistical measures and quality metrics for evaluating
RFI flagging performance. It includes:

- Descriptive statistics (mean, median, std, MAD)
- Flagging Fidelity Index (FFI) - measures quality of flagging decisions
- CalcQuality metric - comprehensive flagging assessment from literature
- Statistical comparison utilities

All functions handle complex visibility data by using magnitude for computations.
"""

from typing import Dict, Optional, Union

import numpy as np

# Type alias for input data
ArrayLike = Union[np.ndarray, complex]


def compute_mad(data: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation (MAD).

    MAD is a robust measure of statistical dispersion that is less sensitive
    to outliers than standard deviation. Formula: MAD = median(|X - median(X)|)

    Parameters
    ----------
    data : np.ndarray
        Input data array (real-valued).

    Returns
    -------
    float
        Median absolute deviation.

    Notes
    -----
    MAD provides a robust alternative to standard deviation for data with
    outliers. For Gaussian data: σ ≈ 1.4826 * MAD

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 100])  # Contains outlier
    >>> compute_mad(data)
    1.5
    >>> np.std(data)  # Standard deviation heavily influenced by outlier
    40.5...
    """
    median = np.median(data)
    return float(np.median(np.abs(data - median)))


def compute_statistics(
    data: np.ndarray, flags: Optional[np.ndarray] = None
) -> Dict[str, Union[float, int]]:
    """
    Compute descriptive statistics on data, optionally with flagging.

    Computes mean, median, standard deviation, MAD, count, and flagging
    fraction. For complex data, uses magnitude.

    Parameters
    ----------
    data : np.ndarray
        Complex or real array to analyze.
    flags : np.ndarray, optional
        Boolean mask where True = flagged (excluded from statistics).
        If None, all data is used.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean': Mean of unflagged data
        - 'median': Median of unflagged data
        - 'std': Standard deviation of unflagged data
        - 'mad': Median absolute deviation of unflagged data
        - 'count': Number of unflagged samples
        - 'flagged_fraction': Fraction of data flagged (0.0 if flags=None)

        Returns NaN for statistics if all data is flagged.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> stats = compute_statistics(data)
    >>> stats['mean']
    3.0
    >>> stats['flagged_fraction']
    0.0

    >>> # With flagging
    >>> flags = np.array([False, False, False, True, True])
    >>> stats = compute_statistics(data, flags)
    >>> stats['mean']
    2.0
    >>> stats['count']
    3
    >>> stats['flagged_fraction']
    0.4

    >>> # Complex data
    >>> complex_data = np.array([1+2j, 3+4j, 5+6j])
    >>> stats = compute_statistics(complex_data)
    >>> abs(stats['mean'] - np.mean(np.abs(complex_data))) < 1e-10
    True
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


def compute_ffi(data: np.ndarray, flags: np.ndarray) -> Dict[str, float]:
    """
    Compute Flagging Fidelity Index (FFI).

    FFI measures quality of flagging by comparing statistics before and after
    flagging. Higher FFI indicates better flagging (clean data preserved,
    RFI removed). Formula combines MAD/STD reduction with over-flagging penalty.

    Parameters
    ----------
    data : np.ndarray
        Complex or real visibility data.
    flags : np.ndarray
        Boolean mask where True = flagged.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ffi': Overall flagging fidelity index [0, 1]
        - 'mad_reduction': Reduction in MAD after flagging [0, 1]
        - 'std_reduction': Reduction in std after flagging [0, 1]
        - 'flagged_fraction': Fraction of data flagged [0, 1]

    Notes
    -----
    FFI formula: (0.5*mad_reduction + 0.5*std_reduction) * (1 - 0.5*flagged_fraction)

    Good flagging should:
    - Reduce MAD and STD (remove outliers/RFI)
    - Minimize flagged_fraction (preserve clean data)

    Examples
    --------
    >>> # Clean data - minimal flagging
    >>> data = np.random.randn(1000)
    >>> flags = np.zeros(1000, dtype=bool)
    >>> flags[:10] = True  # Flag 1%
    >>> ffi = compute_ffi(data, flags)
    >>> ffi['flagged_fraction']
    0.01

    >>> # Data with RFI - good flagging removes outliers
    >>> data = np.random.randn(1000)
    >>> data[100:200] = 10.0  # Inject RFI
    >>> flags = np.zeros(1000, dtype=bool)
    >>> flags[100:200] = True  # Flag RFI
    >>> ffi = compute_ffi(data, flags)
    >>> ffi['mad_reduction'] > 0  # Should reduce MAD
    True
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


def compute_calcquality(
    data: np.ndarray, flags: np.ndarray, reference_data: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute calcquality metric from literature (lower is better).

    CalcQuality is a comprehensive flagging assessment metric with four
    components. Used in radio astronomy for evaluating flagging algorithms.

    Parameters
    ----------
    data : np.ndarray
        Complex or real visibility data to assess.
    flags : np.ndarray
        Boolean mask where True = flagged.
    reference_data : np.ndarray, optional
        Optional baseline data for comparison.
        If None, uses pre-flagging statistics as reference.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'calcquality': Combined score (Euclidean norm of components)
        - 'sensitivity': Component a - deviation from 3σ Gaussian behavior
        - 'mean_shift': Component b - normalized mean difference
        - 'std_shift': Component c - normalized std difference
        - 'overflagging_penalty': Component d - penalty for >70% flagging
        - 'flagged_pct': Percentage of data flagged
        - 'components': Dict of intermediate calculation values

    Notes
    -----
    Four components:
    - a (sensitivity): |abs(max_deviation) - 3| (expect ~3σ for Gaussian)
    - b (mean_shift): |mean_diff| / ref_std - 1
    - c (std_shift): |std_diff| / ref_std
    - d (overflagging): max(0, (flagged_pct - 70) / 10)

    CalcQuality = sqrt(a² + b² + c² + d²)

    Lower values indicate better flagging. Returns np.inf if all data flagged.

    References
    ----------
    Offringa et al., "Post-correlation radio frequency interference
    classification methods", MNRAS, 2010.

    Examples
    --------
    >>> # Clean Gaussian data
    >>> data = np.random.randn(10000)
    >>> flags = np.zeros(10000, dtype=bool)
    >>> cq = compute_calcquality(data, flags)
    >>> cq['calcquality'] < 5  # Should be low for clean data
    True

    >>> # Heavy flagging penalty
    >>> flags = np.ones(10000, dtype=bool)
    >>> flags[:1000] = False  # 90% flagged
    >>> cq = compute_calcquality(data, flags)
    >>> cq['overflagging_penalty'] > 0  # Penalty for >70% flagging
    True
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


def print_statistics_comparison(data: np.ndarray, flags: np.ndarray) -> None:
    """
    Print formatted before/after statistics and FFI comparison.

    Convenience function for displaying flagging impact on data statistics
    and quality metrics.

    Parameters
    ----------
    data : np.ndarray
        Complex or real visibility data.
    flags : np.ndarray
        Boolean mask where True = flagged.

    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> data[100:200] = 10.0  # Add RFI
    >>> flags = np.zeros(1000, dtype=bool)
    >>> flags[100:200] = True  # Flag RFI
    >>> print_statistics_comparison(data, flags)  # doctest: +SKIP
    ============================================================
    Statistics Comparison (Before/After Flagging)
    ============================================================
    <BLANKLINE>
    Before Flagging:
      Mean:   ...
      Median: ...
      Std:    ...
      MAD:    ...
      Count:  1000
    <BLANKLINE>
    After Flagging (10.00% flagged):
      Mean:   ...
      Median: ...
      Std:    ...
      MAD:    ...
      Count:  900
    <BLANKLINE>
    Flagging Fidelity Index (FFI):
      FFI:            ...
      MAD Reduction:  ...
      STD Reduction:  ...
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
