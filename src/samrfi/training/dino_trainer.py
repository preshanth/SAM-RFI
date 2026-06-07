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
