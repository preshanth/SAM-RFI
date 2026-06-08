"""On-device synthetic batch streaming for DINO segmenter training.

Wraps the torch-backend ``RFISimulatorTorch`` (rfi_toolbox) to generate
full-plane complex visibility samples directly on the training device, so the
GPU stays hot without disk shards or CPU dataloader workers. Each sample is one
polarization of one generated plane plus its full-truth mask. ``baseline_frac``
is varied per plane (the simulator draws U(0,1) when it is None), giving a mix
of fringe densities.

A fixed-seed validation set is materialised once via :func:`make_val_set` and
reused, so val metrics are comparable across steps and runs.
"""

from __future__ import annotations

import numpy as np
import torch

from rfi_toolbox.core.simulator_torch import RFISimulatorTorch

from .dino_segmenter import complex_to_channels


class SimBatchStream:
    """Infinite stream of (input, mask, amplitude) batches generated on-device."""

    def __init__(
        self,
        size: int = 512,
        batch_size: int = 4,
        input_mode: str = "realimag",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        detect_floor: float = 1.0,
        baseline_frac=None,
        pols=("RR", "LL"),
    ):
        self.size = size
        self.batch_size = batch_size
        self.input_mode = input_mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.baseline_frac = baseline_frac
        self.pols = tuple(pols)
        self.sim = RFISimulatorTorch(size, size, device=device, dtype=dtype)
        self.sim.detect_floor = detect_floor

    def _one(self):
        pol = self.pols[np.random.randint(0, len(self.pols))]
        plane, mask = self.sim.generate_rfi(baseline_frac=self.baseline_frac)
        return plane[pol], mask.to(self.dtype)

    def next_batch(self):
        zs, ms = [], []
        for _ in range(self.batch_size):
            z, m = self._one()
            zs.append(z)
            ms.append(m)
        z = torch.stack(zs, 0)  # (B, H, W) complex
        m = torch.stack(ms, 0).unsqueeze(1)  # (B, 1, H, W)
        x = complex_to_channels(z, self.input_mode)  # (B, 3, H, W)
        amp = z.abs().unsqueeze(1)  # (B, 1, H, W) raw |z|
        return x, m, amp


def make_val_set(stream: SimBatchStream, n_batches: int, seed: int = 1234):
    """Materialise a deterministic validation set (list of (x, m, amp))."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return [stream.next_batch() for _ in range(n_batches)]
    finally:
        np.random.set_state(rng_state)
