"""
Inspect a single sample from rfi_toolbox.core.simulator.RFISimulator.

Unlike the flat-stamp data_generation path, RFISimulator produces complex
RFI (power * complex-Gaussian) over a wide power range, so amplitude has
internal structure, phase is physical, and low-power events are buried in
the noise. Renders log-amplitude, physical phase, mask, and the
amplitude histogram (noise vs RFI separability).

Usage:
    pixi run python scripts/inspect_simulator.py \
        --out experiments/sample_inspection/simulator.png
"""

import argparse

import numpy as np
from rfi_toolbox.core.simulator import RFISimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pol", default="RR")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(args.seed)
    sim = RFISimulator(time_bins=args.bins, freq_bins=args.bins)
    tf_plane, mask = sim.generate_rfi()
    viz = tf_plane[args.pol]

    amp = np.abs(viz)
    logamp = np.log10(amp + 1e-9)
    phase = np.angle(viz)
    occ = mask.mean()
    print(f"occupancy={occ:.3f}  amp range=[{amp.min():.2e},{amp.max():.2e}]")

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    im0 = ax[0, 0].imshow(logamp, aspect="auto", cmap="viridis", origin="lower")
    ax[0, 0].set_title(f"log10 |{args.pol}|")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    im1 = ax[0, 1].imshow(phase, aspect="auto", cmap="twilight", origin="lower")
    ax[0, 1].set_title("phase (physical)")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
    ax[0, 2].imshow(mask, aspect="auto", cmap="gray", origin="lower")
    ax[0, 2].set_title(f"mask (occ={occ:.1%})")

    # amplitude histogram: noise vs RFI overlap
    ax[1, 0].hist(np.log10(amp[~mask] + 1e-9).ravel(), bins=120, alpha=0.6,
                  label="noise", color="C0", density=True)
    ax[1, 0].hist(np.log10(amp[mask] + 1e-9).ravel(), bins=120, alpha=0.6,
                  label="RFI", color="C3", density=True)
    ax[1, 0].set_xlabel("log10 amplitude")
    ax[1, 0].set_ylabel("density")
    ax[1, 0].set_title("noise vs RFI amplitude (overlap = buried RFI)")
    ax[1, 0].legend()

    ax[1, 1].imshow(logamp, aspect="auto", cmap="gray", origin="lower")
    ax[1, 1].imshow(np.ma.masked_where(~mask, mask), aspect="auto",
                    cmap="autumn", alpha=0.4, origin="lower")
    ax[1, 1].set_title("log amp + mask overlay")

    # fraction of RFI pixels below the noise 99th percentile = unrecoverable by threshold
    noise_p99 = np.percentile(amp[~mask], 99)
    buried = (amp[mask] < noise_p99).mean()
    ax[1, 2].axis("off")
    ax[1, 2].text(0.05, 0.5,
                  f"RFI pixels below noise p99:\n{buried:.1%}\n\n"
                  f"(threshold at noise p99 misses\nthis fraction of RFI)",
                  fontsize=14, va="center")
    print(f"RFI buried below noise p99: {buried:.1%}")

    for a in ax.ravel()[:5]:
        if a is not ax[1, 0]:
            a.set_xlabel("freq")
            a.set_ylabel("time")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
