"""
Inspect a single synthetic RFI sample, driven entirely by a data-gen config.

Every generation parameter (geometry, physical scales, RFI counts, bandpass,
polarization) is read from the YAML via rfi_toolbox's ConfigLoader -- nothing
is hand-entered here. Produces the amplitude+gradient+phase 3-channel
representation a vision model would receive, rendered in log scale, plus the
exact ground-truth mask.

Usage:
    pixi run python scripts/inspect_sample.py \
        --config configs/inspect_single.yaml \
        --out experiments/sample_inspection/sample.png
"""

import argparse

import numpy as np
from rfi_toolbox.config.loader import ConfigLoader
from rfi_toolbox.data_generation.synthetic_generator import SyntheticDataGenerator


def generate_from_config(config, seed):
    """Generate one complex sample using only parameters from the config."""
    synth = config.get("synthetic")
    gen = SyntheticDataGenerator(config=config)

    rfi_config = gen._parse_rfi_config(synth)

    np.random.seed(seed)
    waterfall, exact_mask, params = gen._generate_single_sample(
        num_channels=synth["num_channels"],
        num_times=synth["num_times"],
        noise_level=synth["noise_mjy"],
        rfi_power_min=synth["rfi_power_min"],
        rfi_power_max=synth["rfi_power_max"],
        rfi_config=rfi_config,
        enable_bandpass=synth.get("enable_bandpass_rolloff", False),
        bandpass_order=synth.get("bandpass_polynomial_order", 8),
        num_polarizations=synth.get("num_polarizations", 1),
        pol_corr=synth.get("polarization_correlation", 0.8),
        synth_config=synth,
    )
    # (1, npol, nc, nt) -> baseline 0, pol 0
    return waterfall[0, 0], exact_mask[0, 0], params


def three_channel(viz):
    """amplitude + gradient(amplitude) + phase, all in log-amplitude space."""
    amp = np.abs(viz)
    logamp = np.log10(amp + 1e-3)  # log scale: 1 mJy noise to 1e7 mJy RFI
    gy, gx = np.gradient(logamp)
    grad = np.hypot(gx, gy)
    phase = np.angle(viz)  # NOTE: uniform random on synthetic data (no info)

    def norm(x):
        lo, hi = np.percentile(x, [1, 99])
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    return logamp, norm(logamp), norm(grad), norm(phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="data-gen YAML config path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = ConfigLoader.load_data(args.config)
    viz, mask, params = generate_from_config(config, args.seed)
    occ = mask.mean()
    print(f"config={args.config}  occupancy={occ:.3f}  rfi_components={len(params)}")

    logamp, r, g, b = three_channel(viz)
    rgb = np.stack([r, g, b], axis=-1)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    im0 = ax[0, 0].imshow(logamp, aspect="auto", cmap="viridis", origin="lower")
    ax[0, 0].set_title("log10 amplitude (mJy)")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    ax[0, 1].imshow(g, aspect="auto", cmap="viridis", origin="lower")
    ax[0, 1].set_title("|grad(log amp)|  (edges)")
    ax[0, 2].imshow(b, aspect="auto", cmap="twilight", origin="lower")
    ax[0, 2].set_title("phase (random on synth)")
    ax[1, 0].imshow(rgb, aspect="auto", origin="lower")
    ax[1, 0].set_title("RGB = amp / grad / phase")
    ax[1, 1].imshow(mask, aspect="auto", cmap="gray", origin="lower")
    ax[1, 1].set_title(f"exact mask (occ={occ:.1%})")
    ax[1, 2].imshow(r, aspect="auto", cmap="gray", origin="lower")
    ax[1, 2].imshow(
        np.ma.masked_where(~mask, mask),
        aspect="auto",
        cmap="autumn",
        alpha=0.5,
        origin="lower",
    )
    ax[1, 2].set_title("log amp + mask overlay")
    for a in ax.ravel():
        a.set_xlabel("time")
        a.set_ylabel("channel")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
