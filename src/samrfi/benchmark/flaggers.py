"""
Uniform flagger interface + flag isolation.

Every flagger starts from the same clean flag state, writes the FLAG column,
and has its result read back as a (baselines, pols, channels, times) boolean
array. Isolation uses CASA flagmanager versions so flaggers never see each
other's flags -- the pattern proven in scripts/benchmark_synthetic.py.

casatasks / RFIPredictor are imported lazily so this module imports without CASA
or a GPU.
"""

from __future__ import annotations

ORIGINAL_VERSION = "samrfi_benchmark_original"

# Classical flagdata pipelines. Each value is the ordered list of flagdata modes
# applied from a clean baseline; the combined entry mirrors a standard
# tfcrop-then-rflag pass.
CLASSICAL_FLAGGERS: dict[str, list[str]] = {
    "tfcrop": ["tfcrop"],
    "rflag": ["rflag"],
    "tfcrop+rflag": ["tfcrop", "rflag"],
}


def snapshot_clean_state(ms_path) -> None:
    """Save the current (assumed clean) flag column as the restore point.

    Call once before running any flagger. Overwrites a previous snapshot of the
    same name so re-runs are idempotent.
    """
    from casatasks import flagmanager

    try:
        flagmanager(vis=str(ms_path), mode="delete", versionname=ORIGINAL_VERSION)
    except Exception:
        pass  # No prior snapshot; fine.
    flagmanager(vis=str(ms_path), mode="save", versionname=ORIGINAL_VERSION)


def _restore_clean(ms_path) -> None:
    from casatasks import flagmanager

    flagmanager(vis=str(ms_path), mode="restore", versionname=ORIGINAL_VERSION)


def read_flags(ms_path, num_antennas=None):
    """Read the current FLAG column as (baselines, pols, channels, times) bool."""
    from rfi_toolbox.io import MSLoader

    loader = MSLoader(str(ms_path))
    loader.load(mode="DATA", num_antennas=num_antennas)
    flags = loader.load_flags()
    loader.close()
    return flags


def run_classical(ms_path, modes, num_antennas=None):
    """Run one classical flagdata pipeline from clean state; return its flags."""
    from casatasks import flagdata

    _restore_clean(ms_path)
    for mode in modes:
        flagdata(vis=str(ms_path), mode=mode, datacolumn="data", action="apply")
    flags = read_flags(ms_path, num_antennas=num_antennas)
    _restore_clean(ms_path)
    return flags


def run_sam(ms_path, model_path, device="cuda", patch_size=1024,
            stretch=None, threshold=None, num_antennas=None):
    """Run SAM-RFI from clean state (per-baseline, low memory); return its flags."""
    from samrfi.inference import RFIPredictor

    _restore_clean(ms_path)
    predictor = RFIPredictor(model_path=model_path, device=device)
    predictor.predict_ms_per_baseline(
        ms_path=str(ms_path),
        patch_size=patch_size,
        stretch=stretch,
        save_flags=True,
        threshold=threshold,
    )
    flags = read_flags(ms_path, num_antennas=num_antennas)
    _restore_clean(ms_path)
    return flags


def run_all_flaggers(
    ms_path,
    model_path=None,
    classical=None,
    sam_kwargs=None,
    num_antennas=None,
):
    """Run the requested flaggers on one MS and return {name: flag_array}.

    Args:
        ms_path: Path to the MS (flags are written and restored, not left dirty).
        model_path: SAM-RFI checkpoint; if None, SAM is skipped.
        classical: Iterable of CLASSICAL_FLAGGERS keys (default: all of them).
        sam_kwargs: Extra kwargs forwarded to run_sam (device, patch_size, ...).
        num_antennas: Limit antennas loaded when reading flags.

    Returns:
        Ordered dict {flagger_name: (baselines, pols, channels, times) bool array}.
    """
    classical = list(CLASSICAL_FLAGGERS) if classical is None else list(classical)
    sam_kwargs = dict(sam_kwargs or {})

    snapshot_clean_state(ms_path)
    results: dict = {}

    for name in classical:
        if name not in CLASSICAL_FLAGGERS:
            raise ValueError(f"Unknown classical flagger '{name}'. "
                             f"Options: {list(CLASSICAL_FLAGGERS)}")
        results[name] = run_classical(
            ms_path, CLASSICAL_FLAGGERS[name], num_antennas=num_antennas
        )

    if model_path is not None:
        results["sam"] = run_sam(
            ms_path, model_path, num_antennas=num_antennas, **sam_kwargs
        )

    return results
