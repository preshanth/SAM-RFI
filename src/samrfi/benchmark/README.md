# samrfi.benchmark — two-gate flagger harness

Trustworthy comparison of SAM-RFI against classical flaggers (`tfcrop`, `rflag`,
`tfcrop+rflag`) for the paper. Refactored from the original
`scripts/benchmark_synthetic.py` into a reusable core so both gates share one
flagger-orchestration path and emit one long-format CSV (`flagger, category,
metric, value`).

## The two gates

- **Gate 0 — `run_gate0` / `scripts/benchmark_synthetic.py`**
  Inject synthetic RFI into a template MS (ground truth known). Every flagger is
  scored against the exact mask with segmentation metrics (IoU/precision/recall/
  F1/dice) **and** downstream metrics (FFI, calcquality, flag fraction).

- **Gate 1 — `run_gate1` / `scripts/benchmark_real.py`**
  A real MS with no ground truth. Every flagger is scored by **agreement against
  a reference flagger** (default `rflag`) plus the same downstream metrics.
  Agreement is not a quality measure on its own — read it next to FFI/calcquality
  and, ideally, a post-flag image (see below).

## Run

```bash
# Gate 0 (self-contained: make a template MS first if you don't have one)
python scripts/create_template_ms.py --output template.ms      # if needed
python scripts/benchmark_synthetic.py template.ms \
    --config configs/validation.yaml \
    --model polarimetric/sam-rfi/large \
    --output ./benchmark_gate0

# Gate 1
python scripts/benchmark_real.py /path/to/observation.ms \
    --model polarimetric/sam-rfi/large \
    --reference rflag \
    --output ./benchmark_gate1
```

Omit `--model` to compare only the classical flaggers.

## Design notes

- **Flag isolation:** each flagger starts from a saved clean snapshot
  (`flagmanager` version), writes the FLAG column, has its flags read back, then
  the snapshot is restored before the next flagger. Flaggers never see each
  other's flags.
- **Gate 1 baseline state:** the "clean" snapshot is the MS's *current* flag
  state. If the MS already carries online flags, every flagger starts from those
  (usually what you want). To compare from scratch, `flagdata(mode='unflag')`
  first.
- **Metrics are reused** from `rfi_toolbox.evaluation` (segmentation + FFI +
  calcquality); this package only orchestrates.
- **Heavy deps are lazy:** `casatasks`, `casatools`/`MSLoader`, and the SAM model
  import only inside the functions that use them, so `import samrfi.benchmark`
  works without CASA or a GPU.

## Not yet here (future, for the paper)

- AOFlagger and a U-Net baseline (the flagger interface in `flaggers.py` is the
  place to add them).
- Image-domain downstream metric (post-flag `tclean` dynamic range / RMS) — the
  current downstream metrics are visibility-domain only.

## Status

Written and `py_compile`-clean, but **not yet run against CASA** (no CASA/GPU in
the authoring environment). Validate end-to-end on the deployment box: Gate 0 on
a small simulated MS first, then Gate 1 on a real observation.
