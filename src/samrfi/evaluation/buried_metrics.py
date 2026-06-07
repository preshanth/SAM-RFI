"""
Supervised metrics for the buried-RFI benchmark.

The load-bearing measurement is recall as a function of LOCAL signal-to-noise:
for RFI pixels binned by |z| relative to the noise sigma, what fraction does
each method recover? Classical flaggers collapse below ~1 sigma; the question
is whether the learned global features hold. Pooled over a set of patches.
"""

from __future__ import annotations

import numpy as np

# local-SNR bin edges in robust-sigma units (|z| - background) / sigma
SNR_EDGES = np.array([-np.inf, 0.0, 0.5, 1.0, 2.0, 4.0, np.inf])
SNR_LABELS = ["<0", "0-0.5", "0.5-1", "1-2", "2-4", ">4"]


def _local_snr(z: np.ndarray) -> np.ndarray:
    amp = np.abs(z)
    bg = np.median(amp, axis=1, keepdims=True)
    resid = amp - bg
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = 1.4826 * mad + 1e-9
    return resid / sigma


def overall_scores(pred: np.ndarray, gt: np.ndarray):
    inter = (pred & gt).sum()
    dice = 2 * inter / max(1, pred.sum() + gt.sum())
    recall = inter / max(1, gt.sum())
    precision = inter / max(1, pred.sum())
    return dict(dice=float(dice), recall=float(recall), precision=float(precision))


def recall_vs_snr(preds, zs, gts):
    """Pooled recall per local-SNR bin across a set of patches.

    preds/gts: lists of boolean (H,W). zs: list of complex (H,W).
    Returns (labels, recall_per_bin, count_per_bin).
    """
    hit = np.zeros(len(SNR_LABELS))
    tot = np.zeros(len(SNR_LABELS))
    for pred, z, gt in zip(preds, zs, gts):
        snr = _local_snr(z)
        bins = np.digitize(snr[gt], SNR_EDGES) - 1
        bins = np.clip(bins, 0, len(SNR_LABELS) - 1)
        p = pred[gt]
        for b in range(len(SNR_LABELS)):
            sel = bins == b
            tot[b] += sel.sum()
            hit[b] += (p[sel]).sum()
    recall = np.divide(hit, tot, out=np.zeros_like(hit), where=tot > 0)
    return SNR_LABELS, recall, tot.astype(int)
