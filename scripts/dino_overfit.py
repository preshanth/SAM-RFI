"""
DINO segmenter go/no-go: overfit a single batch of synthetic RFI patches.

If a frozen DINO backbone + small DPT head can drive Dice -> ~1.0 on a handful
of patches, the features grip RFI and the architecture is validated. If it
cannot overfit even one batch, the domain gap needs the LoRA / continued-
pretraining ladder.

Loads modules standalone (avoids the heavy top-level samrfi package import).

Usage:
    pixi run python scripts/dino_overfit.py \
        --config configs/dino_overfit.yaml \
        --model-id facebook/dinov2-small \
        --steps 300 --out experiments/dino_overfit/result.png
"""

import argparse
import importlib.util
import os

import numpy as np
import torch
from rfi_toolbox.config.loader import ConfigLoader
from rfi_toolbox.data_generation.synthetic_generator import SyntheticDataGenerator


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seg = _load("dino_segmenter", os.path.join(HERE, "src/samrfi/training/dino_segmenter.py"))
# dino_trainer imports dino_segmenter via relative import; load as a package member
import sys  # noqa: E402

sys.modules["dino_segmenter"] = seg
trn_spec = importlib.util.spec_from_file_location(
    "dino_trainer", os.path.join(HERE, "src/samrfi/training/dino_trainer.py")
)
# patch the relative import to our standalone module
trn_src = open(os.path.join(HERE, "src/samrfi/training/dino_trainer.py")).read()
trn_src = trn_src.replace("from .dino_segmenter import", "from dino_segmenter import")
trn = importlib.util.module_from_spec(trn_spec)
exec(compile(trn_src, "dino_trainer.py", "exec"), trn.__dict__)


def gen_patches(config_path, n, seed):
    config = ConfigLoader.load_data(config_path)
    synth = config.get("synthetic")
    gen = SyntheticDataGenerator(config=config)
    rfi_config = gen._parse_rfi_config(synth)
    zs, ms = [], []
    for i in range(n):
        np.random.seed(seed + i)
        wf, mask, _ = gen._generate_single_sample(
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
        zs.append(torch.as_tensor(wf[0, 0]))  # complex (H,W)
        ms.append(torch.as_tensor(mask[0, 0].astype("float32")))
    return zs, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dino_overfit.yaml")
    ap.add_argument("--model-id", default="facebook/dinov2-small",
                    help="HF backbone id; use a DINOv3 id after hf login")
    ap.add_argument("--input-mode", default="realimag", choices=["amplitude", "realimag"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/dino_overfit/result.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model_id}  mode={args.input_mode}")

    zs, ms = gen_patches(args.config, args.n, args.seed)
    occ = float(np.mean([m.mean().item() for m in ms]))
    print(f"generated {len(zs)} patches  size={tuple(zs[0].shape)}  mean occ={occ:.1%}")

    ds = trn.ComplexPatchDataset(zs, ms, input_mode=args.input_mode)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.n, shuffle=True)

    model = seg.DINOSegmenter(model_id=args.model_id, pretrained=True)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable head params: {nt/1e6:.2f}M")

    history = trn.fit(model, loader, device=device, steps=args.steps, lr=args.lr)
    final = history[-1]
    print(f"FINAL  loss {final[1]:.4f}  dice {final[2]:.3f}  iou {final[3]:.3f}")

    # render predictions vs GT for the batch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    x, m = next(iter(loader))
    with torch.no_grad():
        prob = torch.sigmoid(model(x.to(device))).cpu()
    k = min(args.n, 4)
    fig, ax = plt.subplots(3, k, figsize=(4 * k, 9))
    if k == 1:
        ax = ax.reshape(3, 1)
    for j in range(k):
        amp = np.log10(np.abs(zs[j].numpy()) + 1e-9)
        ax[0, j].imshow(amp, aspect="auto", cmap="viridis", origin="lower")
        ax[0, j].set_title(f"log|z| #{j}")
        ax[1, j].imshow(m[j, 0], aspect="auto", cmap="gray", origin="lower")
        ax[1, j].set_title("GT mask")
        ax[2, j].imshow(prob[j, 0], aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=1)
        ax[2, j].set_title("pred prob")
    fig.suptitle(
        f"DINO overfit: {args.model_id} {args.input_mode}  "
        f"dice={final[2]:.3f} iou={final[3]:.3f} ({args.steps} steps)"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
