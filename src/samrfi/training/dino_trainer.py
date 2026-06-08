"""
Trainer for the DINOv3 frozen-backbone RFI segmenter.

Mirrors the SAM2Trainer scaffold (DiceCE loss, AMP, grad accumulation,
scheduler, early stopping, checkpointing) but for pure dense segmentation:
no prompts, no boxes. Only the DPT decoder head is optimised.

This module deliberately avoids importing the top-level samrfi package so it
stays light; it depends only on torch and the dino_segmenter module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dino_segmenter import complex_to_channels


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Soft-Dice + BCE on logits, matching the intent of monai DiceCELoss.
    logits/target: (B, 1, H, W). target in {0,1}."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(2, 3)) + eps
    den = (prob + target).sum(dim=(2, 3)) + eps
    dice = 1 - (num / den).mean()
    return bce + dice


@torch.no_grad()
def mask_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    iou = (inter / union.clamp(min=1)).item()
    dice = (2 * inter / (pred.sum() + target.sum()).clamp(min=1)).item()
    return dice, iou


class ComplexPatchDataset(Dataset):
    """In-memory complex patches -> (3-channel input, mask) tensors."""

    def __init__(self, complex_patches, masks, input_mode="realimag"):
        self.z = complex_patches  # list/tensor of (H, W) complex
        self.m = masks  # list/tensor of (H, W) {0,1}
        self.input_mode = input_mode

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        z = self.z[i]
        if not torch.is_tensor(z):
            z = torch.as_tensor(z)
        x = complex_to_channels(z.unsqueeze(0), self.input_mode).squeeze(0)  # (3,H,W)
        m = torch.as_tensor(self.m[i], dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        return x, m


def fit(
    model: nn.Module,
    dataloader,
    *,
    device="cuda",
    steps=300,
    lr=1e-3,
    weight_decay=0.0,
    use_amp=False,
    log_interval=25,
    logger=print,
):
    """Train the decoder head. Returns list of (step, loss, dice, iou).

    Iterates the dataloader cyclically for `steps` optimiser steps -- suitable
    both for a one-batch overfit probe and for short real training runs.
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp and device_type == "cuda")

    history = []
    data_iter = _cycle(dataloader)
    model.decoder.train()
    for step in range(1, steps + 1):
        x, m = next(data_iter)
        x, m = x.to(device), m.to(device)
        opt.zero_grad()
        with torch.amp.autocast(device_type, enabled=use_amp and device_type == "cuda"):
            logits = model(x)
            loss = dice_ce_loss(logits, m)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % log_interval == 0 or step == 1 or step == steps:
            dice, iou = mask_metrics(logits.detach().float(), m)
            history.append((step, loss.item(), dice, iou))
            logger(f"step {step:4d}  loss {loss.item():.4f}  dice {dice:.3f}  iou {iou:.3f}")
    return history


def _cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def _noise_floor(amp: torch.Tensor, mask: torch.Tensor, pct: float = 99.0):
    """Per-sample p99 of |z| over clean (non-mask) pixels = the noise floor.

    amp/mask: (B, 1, H, W). Returns (B,) floor in the same units as amp."""
    B = amp.shape[0]
    floors = amp.new_zeros(B)
    a = amp.view(B, -1)
    m = mask.view(B, -1) > 0.5
    for i in range(B):
        clean = a[i][~m[i]]
        floors[i] = torch.quantile(clean, pct / 100.0) if clean.numel() else a[i].max()
    return floors


@torch.no_grad()
def recall_by_band(logits, mask, amp, thr: float = 0.5):
    """Buried/bright RFI recall split at the per-sample noise p99.

    buried = RFI pixels with |z| <= noise p99 (the thesis target).
    bright = RFI pixels with |z| >  noise p99.
    Returns (dice, buried_recall, bright_recall) averaged over the batch where
    each band is present (NaN-safe)."""
    pred = torch.sigmoid(logits) > thr
    m = mask > 0.5
    floor = _noise_floor(amp, mask).view(-1, 1, 1, 1)
    buried = m & (amp <= floor)
    bright = m & (amp > floor)

    inter = (pred & m).sum().float()
    dice = (2 * inter / (pred.sum() + m.sum()).clamp(min=1)).item()

    def rec(sel):
        n = sel.sum()
        return ((pred & sel).sum().float() / n).item() if n > 0 else float("nan")

    return dice, rec(buried), rec(bright)


@torch.no_grad()
def evaluate(model, val_set, device):
    """Aggregate val metrics over a list of (x, m, amp) batches."""
    model.decoder.eval()
    dices, buried, bright = [], [], []
    for x, m, amp in val_set:
        x, m, amp = x.to(device), m.to(device), amp.to(device)
        logits = model(x)
        d, bu, br = recall_by_band(logits, m, amp)
        dices.append(d)
        if bu == bu:  # not NaN
            buried.append(bu)
        if br == br:
            bright.append(br)
    import statistics as _st

    nanmean = lambda xs: _st.mean(xs) if xs else float("nan")  # noqa: E731
    return {
        "dice": nanmean(dices),
        "buried_recall": nanmean(buried),
        "bright_recall": nanmean(bright),
    }


def train(
    model: nn.Module,
    stream,
    val_set,
    *,
    device="cuda",
    steps=4000,
    lr=1e-3,
    weight_decay=1e-4,
    use_amp=True,
    val_interval=200,
    patience=5,
    cosine=True,
    ckpt_path=None,
    metadata=None,
    logger=print,
):
    """Train the decoder head against an on-the-fly stream with validation.

    Selects the best checkpoint by val buried_recall (the thesis metric), with
    dice as the tiebreak; early-stops after `patience` validations without
    improvement. Saves decoder-only state plus metadata. Returns history dict.
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if cosine else None
    )
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_on = use_amp and device_type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=amp_on)

    history = {"train": [], "val": []}
    best = {"buried_recall": -1.0, "dice": -1.0, "step": 0}
    stale = 0

    for step in range(1, steps + 1):
        model.decoder.train()
        x, m, _ = stream.next_batch()
        x, m = x.to(device), m.to(device)
        opt.zero_grad()
        with torch.amp.autocast(device_type, enabled=amp_on):
            logits = model(x)
            loss = dice_ce_loss(logits, m)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()

        if step % val_interval == 0 or step == steps:
            vm = evaluate(model, val_set, device)
            history["train"].append((step, loss.item()))
            history["val"].append((step, vm))
            logger(
                f"step {step:5d}  loss {loss.item():.4f}  "
                f"val dice {vm['dice']:.3f}  buried {vm['buried_recall']:.3f}  "
                f"bright {vm['bright_recall']:.3f}"
            )
            improved = vm["buried_recall"] > best["buried_recall"] + 1e-4 or (
                abs(vm["buried_recall"] - best["buried_recall"]) <= 1e-4
                and vm["dice"] > best["dice"] + 1e-4
            )
            if improved:
                best = {**vm, "step": step}
                stale = 0
                if ckpt_path:
                    _save_ckpt(model, ckpt_path, {**(metadata or {}), "best": best})
                    logger(f"  saved best -> {ckpt_path}")
            else:
                stale += 1
                if stale >= patience:
                    logger(f"early stop at step {step} (best buried {best['buried_recall']:.3f})")
                    break

    history["best"] = best
    return history


def _save_ckpt(model, path, metadata):
    torch.save({"decoder": model.decoder.state_dict(), "metadata": metadata}, path)
