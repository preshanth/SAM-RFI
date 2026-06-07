"""
Stage 1a: can a frozen-DINO + DPT head recover BURIED RFI from amplitude
structure alone, generalising to held-out samples?

Uses rfi_toolbox.core.simulator.RFISimulator (complex RFI over a 10-order
power range; ~half the RFI sits below the noise floor). No phase modelling:
amplitude input only. The load-bearing metric is recall split by
buried (|z| < noise p99) vs bright RFI pixels, measured on a held-out val
set -- generalisation, not memorisation. A threshold can only get the bright
pixels; anything > 0 on the buried set is structure the global features
recovered.

Usage:
    pixi run python scripts/dino_buried_test.py \
        --model-id facebook/dinov2-small --input-mode amplitude \
        --bins 448 --n-train 12 --n-val 4 --steps 250 \
        --out experiments/dino_buried/result.png
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from rfi_toolbox.core.simulator import RFISimulator


def _load(name, path, patch_rel=None):
    src = open(path).read()
    if patch_rel:
        src = src.replace(patch_rel[0], patch_rel[1])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seg = _load("dino_segmenter", os.path.join(HERE, "src/samrfi/training/dino_segmenter.py"))
sys.modules["dino_segmenter"] = seg
trn = _load(
    "dino_trainer",
    os.path.join(HERE, "src/samrfi/training/dino_trainer.py"),
    ("from .dino_segmenter import", "from dino_segmenter import"),
)


def gen(bins, n, seed, pol="RR"):
    zs, ms = [], []
    for i in range(n):
        np.random.seed(seed + i)
        sim = RFISimulator(time_bins=bins, freq_bins=bins)
        tf, mask = sim.generate_rfi()
        zs.append(torch.as_tensor(tf[pol]))
        ms.append(torch.as_tensor(mask.astype("float32")))
    return zs, ms


@torch.no_grad()
def buried_split_metrics(model, zs, ms, mode, device):
    """Recall on buried vs bright RFI pixels, pooled over the given set."""
    model.eval()
    tot = {"buried_hit": 0, "buried_tot": 0, "bright_hit": 0, "bright_tot": 0,
           "inter": 0.0, "psum": 0.0, "gsum": 0.0}
    for z, m in zip(zs, ms):
        x = seg.complex_to_channels(z.unsqueeze(0), mode).to(device)
        pred = (torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy()
        gt = m.numpy().astype(bool)
        amp = np.abs(z.numpy())
        noise_p99 = np.percentile(amp[~gt], 99)
        buried = gt & (amp < noise_p99)
        bright = gt & (amp >= noise_p99)
        tot["buried_hit"] += (pred & buried).sum()
        tot["buried_tot"] += buried.sum()
        tot["bright_hit"] += (pred & bright).sum()
        tot["bright_tot"] += bright.sum()
        tot["inter"] += (pred & gt).sum()
        tot["psum"] += pred.sum()
        tot["gsum"] += gt.sum()
    buried_recall = tot["buried_hit"] / max(1, tot["buried_tot"])
    bright_recall = tot["bright_hit"] / max(1, tot["bright_tot"])
    dice = 2 * tot["inter"] / max(1, tot["psum"] + tot["gsum"])
    return dice, bright_recall, buried_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="facebook/dinov2-small")
    ap.add_argument("--input-mode", default="amplitude", choices=["amplitude", "realimag"])
    ap.add_argument("--bins", type=int, default=448)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="experiments/dino_buried/result.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model_id}  mode={args.input_mode}")

    ztr, mtr = gen(args.bins, args.n_train, seed=0)
    zva, mva = gen(args.bins, args.n_val, seed=1000)
    occ = np.mean([m.mean().item() for m in mtr])

    # report how buried this data is (the difficulty)
    bfrac = []
    for z, m in zip(ztr, mtr):
        gt = m.numpy().astype(bool)
        amp = np.abs(z.numpy())
        p99 = np.percentile(amp[~gt], 99)
        bfrac.append((amp[gt] < p99).mean())
    print(f"train occ={occ:.1%}  RFI below noise p99: {np.mean(bfrac):.1%}")

    ds = trn.ComplexPatchDataset(ztr, mtr, input_mode=args.input_mode)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = seg.DINOSegmenter(model_id=args.model_id, pretrained=True)
    print(f"trainable head: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    trn.fit(model, loader, device=device, steps=args.steps, lr=args.lr, log_interval=50)

    for split, zs, ms in [("TRAIN", ztr, mtr), ("VAL", zva, mva)]:
        d, br, bu = buried_split_metrics(model, zs, ms, args.input_mode, device)
        print(f"{split:5s}  dice={d:.3f}  bright_recall={br:.3f}  buried_recall={bu:.3f}")

    # figure on val
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    k = min(args.n_val, 4)
    fig, ax = plt.subplots(3, k, figsize=(4 * k, 9))
    if k == 1:
        ax = ax.reshape(3, 1)
    for j in range(k):
        z, m = zva[j], mva[j]
        x = seg.complex_to_channels(z.unsqueeze(0), args.input_mode).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        ax[0, j].imshow(np.log10(np.abs(z.numpy()) + 1e-9), aspect="auto", cmap="viridis", origin="lower")
        ax[0, j].set_title(f"log|z| #{j}")
        ax[1, j].imshow(m.numpy(), aspect="auto", cmap="gray", origin="lower")
        ax[1, j].set_title("GT mask")
        ax[2, j].imshow(prob, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=1)
        ax[2, j].set_title("pred prob (val)")
    fig.suptitle(f"Buried-RFI test (val): {args.model_id} {args.input_mode}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
