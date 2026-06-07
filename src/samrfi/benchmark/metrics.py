"""
Scoring helpers — thin wrappers over rfi_toolbox.evaluation.

Gate 0 scores flags against ground truth (segmentation metrics). Both gates
score downstream flagging quality (FFI / calcquality / statistics). Gate 1, with
no ground truth, additionally scores agreement between a flagger and a chosen
reference flagger using the same segmentation metrics.

These are pure-numpy (no CASA), so this module imports cleanly anywhere.
"""

from __future__ import annotations

import numpy as np


def _flat_bool(flags):
    return np.asarray(flags).reshape(-1).astype(bool)


def score_vs_gt(flags, ground_truth):
    """Segmentation metrics (iou, precision, recall, f1, dice) vs ground truth."""
    from rfi_toolbox.evaluation import evaluate_segmentation

    return evaluate_segmentation(_flat_bool(flags), _flat_bool(ground_truth))


def score_agreement(flags, reference_flags):
    """Segmentation metrics treating another flagger's output as the reference.

    Used in Gate 1 where there is no ground truth: how closely does each flagger
    agree with the reference (e.g. rflag)? Not a quality measure on its own --
    read alongside the downstream metrics.
    """
    from rfi_toolbox.evaluation import evaluate_segmentation

    return evaluate_segmentation(_flat_bool(flags), _flat_bool(reference_flags))


def score_downstream(data, flags):
    """Downstream flagging-quality metrics on the (complex) visibility data.

    Combines Flagging Fidelity Index, calcquality, and basic statistics. `data`
    is the complex visibility array (same shape as flags); the underlying metrics
    take its magnitude.
    """
    from rfi_toolbox.evaluation import compute_calcquality, compute_ffi, compute_statistics

    d = np.asarray(data).reshape(-1)
    f = _flat_bool(flags)
    out = {}
    out.update({f"ffi_{k}": v for k, v in compute_ffi(d, f).items()})
    calcq = compute_calcquality(d, f)
    out.update({f"calcq_{k}": v for k, v in calcq.items() if k != "components"})
    out["flag_fraction"] = float(f.mean())
    stats = compute_statistics(d, f)
    out.update({f"stat_{k}": v for k, v in stats.items() if k != "count"})
    return out


def results_to_rows(per_flagger: dict) -> list[dict]:
    """Flatten {flagger: {category: {metric: value}}} into long-format CSV rows."""
    rows = []
    for flagger, categories in per_flagger.items():
        for category, metrics in categories.items():
            for metric, value in metrics.items():
                rows.append(
                    {
                        "flagger": flagger,
                        "category": category,
                        "metric": metric,
                        "value": _to_native(value),
                    }
                )
    return rows


def _to_native(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def write_csv(rows: list[dict], path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["flagger", "category", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj, path) -> None:
    import json

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
