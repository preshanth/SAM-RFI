"""
Flagger benchmark harness for SAM-RFI.

Two gates behind a shared flagger-orchestration core:

- Gate 0 (run_gate0): synthetic RFI injected into a template MS, so ground-truth
  masks are known. Scores every flagger (SAM + classical) against GT with
  segmentation metrics plus downstream flagging-quality metrics.
- Gate 1 (run_gate1): a real MS with no ground truth. Scores flaggers by
  pairwise agreement against a chosen reference flagger plus the same downstream
  metrics.

CASA (casatasks/casatools) and the SAM model are only imported inside the
functions that need them, so `import samrfi.benchmark` works without them.
"""

from .gates import run_gate0, run_gate1

__all__ = ["run_gate0", "run_gate1"]
