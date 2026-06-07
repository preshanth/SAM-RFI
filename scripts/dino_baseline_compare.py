"""
Buried-RFI benchmark: DINO segmenter vs classical SumThreshold vs MAD.

Trains the frozen-DINO + DPT head (amplitude mode) on RFISimulator data, then
on a held-out val set compares it against SumThreshold (structure-aware
classical) and a MAD per-pixel threshold. Reports overall dice/precision/
recall, the recall-vs-local-SNR curve (the load-bearing figure), and the
paper's calcquality metric for each method.

Usage:
    pixi run python scripts/dino_baseline_compare.py \
        --model-id facebook/dinov2-small --bins 448 \
        --n-train 12 --n-val 6 --steps 250 \
        --out experiments/dino_baseline/compare.png
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from rfi_toolbox.core.simulator import RFISimulator
from rfi_toolbox.evaluation.statistics import compute_calcquality


def _load(name, path, patch=None):
    src = open(path).read()
    if patch:
        src = src.replace(*patch)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seg = _load("dino_segmenter", f"{HERE}/src/samrfi/training/dino_segmenter.py")
sys.modules["dino_segmenter"] = seg
trn = _load("dino_trainer", f"{HERE}/src/samrfi/training/dino_trainer.py",
            ("from .dino_segmenter import", "from dino_segmenter import"))
st = _load("sumthreshold", f"{HERE}/src/samrfi/evaluation/sumthreshold.py")
bm = _load("buried_metrics", f"{HERE}/src/samrfi/evaluation/buried_metrics.py")


def gen(bins, n, seed, pol="RR"):
    zs, ms = [], []
    for i in range(n):
        np.random.seed(seed + i)
        sim = RFISimulator(time_bins=bins, freq_bins=bins)
        tf, mask = sim.generate_rfi()
        zs.append(torch.as_tensor(tf[pol]))
        ms.append(mask.astype(bool))
    return zs, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="facebook/dinov2-small")
    ap.add_argument("--bins", type=int, default=448)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chi1", type=float, default=6.0, help="SumThreshold base sigma")
    ap.add_argument("--out", default="experiments/dino_baseline/compare.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model_id}")

    ztr, mtr = gen(args.bins, args.n_train, 0)
    zva, mva = gen(args.bins, args.n_val, 1000)

    # train DINO (amplitude mode)
    ds = trn.ComplexPatchDataset(ztr, [m.astype("float32") for m in mtr], input_mode="amplitude")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
    model = seg.DINOSegmenter(model_id=args.model_id, pretrained=True)
    trn.fit(model, loader, device=device, steps=args.steps, lr=args.lr, log_interval=50)
    model.eval()

    # predictions on val
    dino_pred, st_pred, mad_pred = [], [], []
    for z in zva:
        amp = np.abs(z.numpy())
        x = seg.complex_to_channels(z.unsqueeze(0), "amplitude").to(device)
        with torch.no_grad():
            dino_pred.append((torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy())
        st_pred.append(st.sumthreshold(amp, chi_1=args.chi1))
        mad_pred.append(st.mad_flag(amp, sigma_thresh=5.0))

    methods = {"DINO": dino_pred, "SumThreshold": st_pred, "MAD": mad_pred}
    print("\n=== overall (val) ===")
    for name, preds in methods.items():
        sc = [bm.overall_scores(p, g) for p, g in zip(preds, mva)]
        cq = np.mean([
            compute_calcquality(z.numpy(), p)["calcquality"]
            for z, p in zip(zva, preds)
        ])
        m = {k: np.mean([s[k] for s in sc]) for k in ["dice", "recall", "precision"]}
        print(f"  {name:12s} dice={m['dice']:.3f} recall={m['recall']:.3f} "
              f"precision={m['precision']:.3f} calcquality={cq:.3f}")

    print("\n=== recall vs local SNR (val) ===")
    curves = {}
    for name, preds in methods.items():
        labels, rec, cnt = bm.recall_vs_snr(preds, zva, mva)
        curves[name] = rec
        print(f"  {name:12s} " + "  ".join(f"{l}:{r:.2f}" for l, r in zip(labels, rec)))
    print("  " + " " * 12 + "counts: " + "  ".join(f"{l}:{c}" for l, c in zip(labels, cnt)))

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8))
    axc = fig.add_subplot(2, 3, 1)
    for name, rec in curves.items():
        axc.plot(labels, rec, marker="o", label=name)
    axc.set_xlabel("local SNR bin (sigma)")
    axc.set_ylabel("recall")
    axc.set_title("Recall vs local SNR (buried = low bins)")
    axc.axvspan(-0.5, 2.5, color="orange", alpha=0.12)
    axc.text(0.5, 0.05, "buried", color="darkorange")
    axc.set_ylim(0, 1.02)
    axc.legend()

    # example masks for val[0]
    z0, g0 = zva[0], mva[0]
    axc2 = fig.add_subplot(2, 3, 2)
    axc2.imshow(np.log10(np.abs(z0.numpy()) + 1e-9), aspect="auto", cmap="viridis", origin="lower")
    axc2.set_title("log|z| (val 0)")
    axc3 = fig.add_subplot(2, 3, 3)
    axc3.imshow(g0, aspect="auto", cmap="gray", origin="lower")
    axc3.set_title("GT mask")
    for k, (name, preds) in enumerate(methods.items()):
        ax = fig.add_subplot(2, 3, 4 + k)
        ax.imshow(preds[0], aspect="auto", cmap="magma", origin="lower")
        ax.set_title(f"{name} pred")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
