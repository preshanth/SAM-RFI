"""
Domain-gap test: apply the synthetic-trained DINO segmenter to a REAL MS.

Trains the frozen-DINO + DPT head on RFISimulator synthetic data (amplitude
mode), then predicts on a single baseline waterfall loaded from a real
Measurement Set via rfi_toolbox MSLoader. Renders the real log-amplitude
waterfall, the DINO RFI mask, and the MS's pre-existing flags for reference
(those flags are prior flagging, NOT ground truth).

Per-sample amplitude standardisation is what lets a model trained at synthetic
scales transfer to the real dynamic range.

Usage:
    pixi run python scripts/dino_predict_ms.py \
        --ms /home/pjaganna/Data/measurement_sets/SNR_G55_10s.ms \
        --ant1 0 --ant2 5 --pol 0 --steps 250 \
        --ckpt experiments/dino_ms/decoder.pt \
        --out experiments/dino_ms/predict.png
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from rfi_toolbox.core.simulator import RFISimulator
from rfi_toolbox.io.ms_loader import MSLoader


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


def crop16(a):
    """Crop trailing rows/cols so both dims are multiples of 16."""
    h, w = a.shape[-2], a.shape[-1]
    return a[..., : h - h % 16, : w - w % 16]


def gen(bins, n, seed):
    zs, ms = [], []
    for i in range(n):
        np.random.seed(seed + i)
        sim = RFISimulator(time_bins=bins, freq_bins=bins)
        tf, mask = sim.generate_rfi()
        zs.append(torch.as_tensor(tf["RR"]))
        ms.append(mask.astype("float32"))
    return zs, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", required=True)
    ap.add_argument("--ant1", type=int, default=0)
    ap.add_argument("--ant2", type=int, default=5)
    ap.add_argument("--pol", type=int, default=0)
    ap.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", default="experiments/dino_ms/decoder.pt")
    ap.add_argument("--flag-ms", default=None,
                    help="optional pre-flagged MS (e.g. tfcrop+rflag) to overlay its FLAG column")
    ap.add_argument("--out", default="experiments/dino_ms/predict.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = seg.DINOSegmenter(model_id=args.model_id, pretrained=True)

    # train on synthetic (or load cached decoder)
    if os.path.exists(args.ckpt):
        model.decoder.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"loaded decoder from {args.ckpt}")
    else:
        ztr, mtr = gen(448, 12, 0)
        ds = trn.ComplexPatchDataset(ztr, mtr, input_mode="amplitude")
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        trn.fit(model, loader, device=device, steps=args.steps, lr=args.lr, log_interval=50)
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        torch.save(model.decoder.state_dict(), args.ckpt)
        print(f"saved decoder to {args.ckpt}")
    model.eval()

    # load real baseline
    L = MSLoader(args.ms)
    wf = L.load_single_baseline(ant1=args.ant1, ant2=args.ant2, pol_idx=args.pol)
    flags = None
    try:
        allflags = L.load_flags()  # may be (baselines,pols,chan,times) or similar
        flags = np.asarray(allflags)
    except Exception as e:
        print(f"(no flags loaded: {e})")

    z = torch.as_tensor(crop16(wf))
    amp = np.abs(z.numpy())
    print(f"real waterfall {tuple(z.shape)}  amp median {np.median(amp):.4g} max {amp.max():.4g}")

    x = seg.complex_to_channels(z.unsqueeze(0), "amplitude").to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pred = prob > 0.5
    print(f"DINO flagged fraction: {pred.mean():.1%}")

    st_pred = st.sumthreshold(amp, chi_1=6.0)
    print(f"SumThreshold flagged fraction: {st_pred.mean():.1%}")
    agree = (pred & st_pred).sum() / max(1, (pred | st_pred).sum())
    print(f"DINO/SumThreshold IoU: {agree:.3f}")

    tf_flag = None
    if args.flag_ms:
        Lf = MSLoader(args.flag_ms)
        f = Lf.load_single_baseline(ant1=args.ant1, ant2=args.ant2, pol_idx=args.pol, mode="FLAG")
        tf_flag = crop16(np.asarray(f).astype(bool))
        print(f"tfcrop+rflag flagged fraction: {tf_flag.mean():.1%}")
        print(f"DINO/tfcrop+rflag IoU: {(pred & tf_flag).sum() / max(1,(pred|tf_flag).sum()):.3f}")
        print(f"SumThr/tfcrop+rflag IoU: {(st_pred & tf_flag).sum() / max(1,(st_pred|tf_flag).sum()):.3f}")

    # render (transpose so freq on x, time on y -> tall image readable)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logamp = np.log10(amp + 1e-6)
    panels = 5 if tf_flag is not None else 4
    fig, ax = plt.subplots(1, panels, figsize=(4.5 * panels, 8), sharey=True)
    ax[0].imshow(logamp.T, aspect="auto", cmap="viridis", origin="lower")
    ax[0].set_title("real log|V| (baseline)")
    ax[1].imshow(prob.T, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=1)
    ax[1].set_title(f"DINO RFI prob ({pred.mean():.0%} flagged)")
    ax[2].imshow(logamp.T, aspect="auto", cmap="gray", origin="lower")
    ax[2].imshow(np.ma.masked_where(~pred.T, pred.T), aspect="auto",
                 cmap="autumn", alpha=0.5, origin="lower")
    ax[2].set_title("DINO mask")
    ax[3].imshow(logamp.T, aspect="auto", cmap="gray", origin="lower")
    ax[3].imshow(np.ma.masked_where(~st_pred.T, st_pred.T), aspect="auto",
                 cmap="winter", alpha=0.5, origin="lower")
    ax[3].set_title(f"SumThreshold mask ({st_pred.mean():.0%} flagged)")
    if tf_flag is not None:
        ax[4].imshow(logamp.T, aspect="auto", cmap="gray", origin="lower")
        ax[4].imshow(np.ma.masked_where(~tf_flag.T, tf_flag.T), aspect="auto",
                     cmap="cool", alpha=0.5, origin="lower")
        ax[4].set_title(f"tfcrop+rflag mask ({tf_flag.mean():.0%} flagged)")
    for a in ax:
        a.set_xlabel("channel")
    ax[0].set_ylabel("time")
    fig.suptitle(f"{os.path.basename(args.ms)}  bl {args.ant1}-{args.ant2} pol {args.pol}  ({args.model_id.split('/')[-1]}, synth-trained)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
