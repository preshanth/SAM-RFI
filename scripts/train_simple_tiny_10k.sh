#!/bin/bash
# Simple SAM2 training - No LoRA, direct fine-tuning

set -e  # Exit on error

echo "============================================================"
echo "SAM-RFI Simple Training - SAM2-tiny on 1080Ti"
echo "============================================================"
echo ""
echo "Config: configs/training_simple_tiny_10k.yaml"
echo "Model: SAM2-tiny (no LoRA, direct fine-tuning)"
echo "Dataset: 10K training + 1K validation"
echo "Batch size: 4 (fits in 8GB VRAM)"
echo "Native resolution: 1024x1024"
echo ""
echo "Training strategy:"
echo "  - Vision encoder: FROZEN (set freeze_vision_encoder=false to train)"
echo "  - Prompt encoder: FROZEN"
echo "  - Mask decoder: TRAINED"
echo ""
echo "============================================================"
echo ""

# Run training pipeline
python scripts/run_training.py \
  --config configs/training_simple_tiny_10k.yaml \
  --skip-generation \
  2>&1 | tee training_simple_tiny.log

echo ""
echo "============================================================"
echo "Training complete!"
echo "Log saved to: training_simple_tiny.log"
echo "Output directory: ./training_output_simple_tiny_10k"
echo "============================================================"
