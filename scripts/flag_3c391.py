#!/usr/bin/env python
"""
RFI Flagging for 3C391 using trained SAM-RFI model
Trained on synthetic 1024x1024 patches
"""

from samrfi.inference import RFIPredictor

# Paths
MS_PATH = "/home/pjaganna/Data/3C391/3c391_ctm_mosaic_10s_spw0.ms"
MODEL_PATH = "/home/pjaganna/Data/SAMRFI/training_output/samrfi_data/models/model_sam2-large_stretch-unknown_sigma-unknown_patch-torch_size-unknown_epochs10_20251008_125608.pth"

# Settings (for larger GPU machine)
SAM_CHECKPOINT = "large"
DEVICE = "cuda"
BATCH_SIZE = 8  # Increased for larger GPU (adjust as needed: 4-16)
PATCH_SIZE = 1024  # Trained on 1024x1024 synthetic patches
NUM_ANTENNAS = None  # Process all antennas
NUM_ITERATIONS = 1  # Single pass

print("Loading model...")
predictor = RFIPredictor(
    model_path=MODEL_PATH,
    sam_checkpoint=SAM_CHECKPOINT,
    device=DEVICE,
    batch_size=BATCH_SIZE
)

print("Running prediction...")
flags = predictor.predict_ms(
    ms_path=MS_PATH,
    num_antennas=NUM_ANTENNAS,
    patch_size=PATCH_SIZE,
    stretch="SQRT",
    save_flags=True
)

print(f"Done! Flagged {flags.sum()/flags.size*100:.2f}% of data")
