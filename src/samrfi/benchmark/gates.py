"""
The two benchmark gates.

Gate 0: synthetic RFI injected into a template MS (ground truth known).
Gate 1: a real MS (no ground truth; agreement vs a reference flagger).

Both reuse flaggers.run_all_flaggers and the metrics helpers, and write a single
long-format CSV (one row per flagger x metric) plus a JSON dump.
"""

from __future__ import annotations

from pathlib import Path

from .flaggers import run_all_flaggers
from .metrics import (
    results_to_rows,
    score_agreement,
    score_downstream,
    score_vs_gt,
    write_csv,
    write_json,
)


def _read_ms_data(ms_path, num_antennas=None):
    """Load complex visibilities and dimensions metadata for one MS."""
    from rfi_toolbox.io import MSLoader

    loader = MSLoader(str(ms_path))
    metadata = loader.get_metadata(num_antennas=num_antennas)
    loader.load(mode="DATA", num_antennas=num_antennas)
    data = loader.data
    loader.close()
    return data, metadata


def run_gate0(
    ms_path,
    config_path,
    output_dir,
    model_path=None,
    num_antennas=None,
    classical=None,
    sam_kwargs=None,
):
    """Gate 0: inject synthetic RFI, run flaggers, score every one against GT.

    Args:
        ms_path: Template MS to inject synthetic data into.
        config_path: Data config YAML with a `synthetic` section.
        output_dir: Where to write results.csv / results.json / ground_truth.npy.
        model_path: SAM-RFI checkpoint (None skips SAM).
        num_antennas: Limit antennas.
        classical: Subset of classical flaggers (default: all).
        sam_kwargs: Extra kwargs for the SAM runner (threshold, patch_size, ...).

    Returns:
        per_flagger dict {flagger: {"segmentation": {...}, "downstream": {...}}}.
    """
    import numpy as np

    from samrfi.config import ConfigLoader
    from .synthetic import generate_and_inject

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ConfigLoader.load_data(str(config_path))

    _, metadata = _read_ms_data(ms_path, num_antennas=num_antennas)
    synthetic_data, ground_truth = generate_and_inject(
        config,
        ms_path,
        metadata["baseline_map"],
        metadata["total_channels"],
        metadata["num_times"],
    )
    np.save(output_dir / "ground_truth.npy", ground_truth)

    flags = run_all_flaggers(
        ms_path, model_path=model_path, classical=classical,
        sam_kwargs=sam_kwargs, num_antennas=num_antennas,
    )

    per_flagger = {}
    for name, fl in flags.items():
        per_flagger[name] = {
            "segmentation": score_vs_gt(fl, ground_truth),
            "downstream": score_downstream(synthetic_data, fl),
        }

    _emit(per_flagger, output_dir, gate="gate0",
          extra={"gt_rfi_percent": float(ground_truth.mean() * 100)})
    return per_flagger


def run_gate1(
    ms_path,
    output_dir,
    model_path=None,
    reference="rflag",
    num_antennas=None,
    classical=None,
    sam_kwargs=None,
):
    """Gate 1: run flaggers on a real MS, score agreement vs a reference + downstream.

    Args:
        ms_path: Real measurement set (no ground truth).
        output_dir: Where to write results.csv / results.json.
        model_path: SAM-RFI checkpoint (None skips SAM).
        reference: Flagger name used as the agreement reference (default 'rflag').
        num_antennas: Limit antennas.
        classical: Subset of classical flaggers (default: all).
        sam_kwargs: Extra kwargs for the SAM runner.

    Returns:
        per_flagger dict {flagger: {"agreement_vs_<ref>": {...}, "downstream": {...}}}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, _ = _read_ms_data(ms_path, num_antennas=num_antennas)

    flags = run_all_flaggers(
        ms_path, model_path=model_path, classical=classical,
        sam_kwargs=sam_kwargs, num_antennas=num_antennas,
    )
    if reference not in flags:
        raise ValueError(
            f"Reference flagger '{reference}' not among run flaggers {list(flags)}. "
            "Pass it via `classical` (or ensure the model ran for reference='sam')."
        )
    ref_flags = flags[reference]

    per_flagger = {}
    for name, fl in flags.items():
        per_flagger[name] = {
            f"agreement_vs_{reference}": score_agreement(fl, ref_flags),
            "downstream": score_downstream(data, fl),
        }

    _emit(per_flagger, output_dir, gate="gate1", extra={"reference": reference})
    return per_flagger


def _emit(per_flagger, output_dir, gate, extra=None):
    """Write CSV + JSON and print a compact summary table."""
    rows = results_to_rows(per_flagger)
    write_csv(rows, output_dir / "results.csv")
    write_json({"gate": gate, "extra": extra or {}, "results": per_flagger},
               output_dir / "results.json")

    print(f"\n=== {gate} results ({output_dir}) ===")
    for flagger, categories in per_flagger.items():
        headline = []
        for cat in ("segmentation", f"agreement_vs_{(extra or {}).get('reference')}"):
            if cat in categories and "f1" in categories[cat]:
                headline.append(f"{cat.split('_')[0]} F1={categories[cat]['f1']:.3f}")
        ds = categories.get("downstream", {})
        if "flag_fraction" in ds:
            headline.append(f"flagged={ds['flag_fraction'] * 100:.1f}%")
        if "ffi_ffi" in ds:
            headline.append(f"FFI={ds['ffi_ffi']:.3f}")
        print(f"  {flagger:14s} " + "  ".join(headline))
    print(f"  -> results.csv, results.json written to {output_dir}")
