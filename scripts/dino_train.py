"""Train the DINO frozen-backbone RFI segmenter on the coherent-phase simulator.

Generates full-plane complex visibility samples on-device with the torch-backend
RFISimulator (keeps the GPU hot, no shards/workers), trains only the DPT decoder
head, validates on a fixed-seed set, and saves the best decoder by buried-RFI
recall (the thesis metric). The frozen DINOv3 backbone is never updated.

The primary run is realimag (phase lives in Re/Im); pair it with an amplitude
run for the item-2 A/B. Examples:

    # full run (A100), realimag, real gated DINOv3-S backbone
    python scripts/dino_train.py --device cuda --size 512 --batch-size 8 \
        --input-mode realimag --steps 6000 --ckpt experiments/dino_train/realimag.pt

    # local GPU smoke (one short epoch), ungated dinov2 backbone, no download gate
    python scripts/dino_train.py --smoke --backbone facebook/dinov2-small \
        --ckpt experiments/dino_train/smoke.pt

This script avoids importing the top-level samrfi package; it loads the two
training modules by file path so it stays dependency-light.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src", "samrfi", "training")


def _load_modules():
    """Load the three training modules by file path under a synthetic package,
    so their `from .dino_segmenter import ...` relative imports resolve without
    importing the heavy top-level `samrfi` package (which pulls SAM2)."""
    import types

    pkg = types.ModuleType("samrfi_training")
    pkg.__path__ = [_SRC]
    sys.modules["samrfi_training"] = pkg

    def _load(name, filename):
        spec = importlib.util.spec_from_file_location(
            f"samrfi_training.{name}", os.path.join(_SRC, filename)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"samrfi_training.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    seg = _load("dino_segmenter", "dino_segmenter.py")
    sim_data = _load("sim_data", "sim_data.py")
    trainer = _load("dino_trainer", "dino_trainer.py")
    return seg, sim_data, trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-mode", default="realimag", choices=["realimag", "amplitude"])
    p.add_argument("--detect-floor", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--val-interval", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--backbone",
        default=None,
        help="HF model id for the frozen backbone (default: gated DINOv3-S). "
        "Use facebook/dinov2-small for an ungated smoke run.",
    )
    p.add_argument(
        "--config-only",
        action="store_true",
        help="random-init DINOv3-S config (no download) for pure pipeline smoke",
    )
    p.add_argument("--ckpt", default="experiments/dino_train/best.pt")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast run on local GPU: size 256, 40 steps, batch 2",
    )
    args = p.parse_args()

    if args.smoke:
        args.size, args.steps, args.batch_size = 256, 40, 2
        args.val_interval, args.val_batches, args.patience = 20, 2, 99

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seg, sim_data, trainer = _load_modules()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"CUDA unavailable; falling back to {device}")

    print(
        f"backbone={'config-only' if args.config_only else (args.backbone or 'DINOv3-S')}  "
        f"mode={args.input_mode}  size={args.size}  batch={args.batch_size}  "
        f"steps={args.steps}  device={device}"
    )
    model = seg.DINOSegmenter(
        backbone_size="small",
        model_id=args.backbone,
        config_only=args.config_only,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable (decoder) params: {n_train/1e6:.2f}M")

    stream = sim_data.SimBatchStream(
        size=args.size,
        batch_size=args.batch_size,
        input_mode=args.input_mode,
        device=device,
        detect_floor=args.detect_floor,
    )
    val_set = sim_data.make_val_set(stream, args.val_batches, seed=1234)

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    metadata = {
        "input_mode": args.input_mode,
        "image_size": args.size,
        "detect_floor": args.detect_floor,
        "backbone": args.backbone or ("config-only" if args.config_only else "DINOv3-S"),
        "out_indices": list(model.out_indices),
    }
    history = trainer.train(
        model,
        stream,
        val_set,
        device=device,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        val_interval=args.val_interval,
        patience=args.patience,
        cosine=not args.no_cosine,
        ckpt_path=args.ckpt,
        metadata=metadata,
    )
    print("done. best:", history["best"])


if __name__ == "__main__":
    main()
