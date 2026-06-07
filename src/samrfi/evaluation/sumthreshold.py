"""
Classical RFI flagging baselines for benchmarking the DINO segmenter.

- mad_flag: per-pixel robust threshold (the trivial floor; gets ~0% of
  sub-noise RFI by construction).
- sumthreshold: faithful Offringa et al. (2010) SumThreshold, run 1D along
  both time and frequency. This is the STRUCTURE-AWARE classical baseline --
  it sums along lines, so it is the fair competitor for buried RFI, not a
  naive threshold.

Both operate directly on a 2D amplitude array (channels x time), no MS / CASA
dependency. AOFlagger is the reference implementation but is a heavy C++
dependency; this Python SumThreshold is the standard literature substitute.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d


def _sigma_normalise(amp: np.ndarray) -> np.ndarray:
    """Background-subtract (per-channel median) and scale to robust sigma units."""
    bg = np.median(amp, axis=1, keepdims=True)  # per-channel (row) background
    resid = amp - bg
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = 1.4826 * mad + 1e-9
    return resid / sigma


def mad_flag(amp: np.ndarray, sigma_thresh: float = 5.0) -> np.ndarray:
    """Per-pixel robust threshold at sigma_thresh * robust-sigma."""
    x = _sigma_normalise(amp)
    return x > sigma_thresh


def _sumthreshold_1d(x: np.ndarray, flags: np.ndarray, M: int, chi: float, axis: int):
    """One SumThreshold pass with window length M along `axis`.

    Flag a window if the mean of its currently-unflagged samples exceeds chi
    (in sigma units). Vectorised with a length-M box filter; qualifying
    windows are spread back over their full extent.
    """
    vals = np.where(flags, 0.0, x)
    cnt = (~flags).astype(np.float64)
    kern = np.ones(M)
    ssum = convolve1d(vals, kern, axis=axis, mode="constant")
    scnt = convolve1d(cnt, kern, axis=axis, mode="constant")
    mean = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), 0.0)
    qualifies = (mean > chi).astype(np.float64)
    spread = convolve1d(qualifies, kern, axis=axis, mode="constant") > 0
    return flags | spread


def sumthreshold(
    amp: np.ndarray,
    chi_1: float = 6.0,
    rho: float = 1.5,
    windows=(1, 2, 4, 8, 16, 32, 64),
) -> np.ndarray:
    """Offringa-2010 SumThreshold over both axes.

    chi_1: base threshold (sigma) for window M=1; longer windows use
        chi_M = chi_1 * rho**(-log2 M), so faint extended RFI is caught by
        summation. Returns a boolean RFI mask (channels x time).
    """
    x = _sigma_normalise(amp)
    flags = np.zeros(amp.shape, dtype=bool)
    for M in windows:
        chi_M = chi_1 * rho ** (-np.log2(M))
        # frequency direction (axis 0) then time direction (axis 1)
        flags = _sumthreshold_1d(x, flags, M, chi_M, axis=0)
        flags = _sumthreshold_1d(x, flags, M, chi_M, axis=1)
    return flags
