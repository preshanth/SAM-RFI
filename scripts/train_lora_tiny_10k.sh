#!/bin/bash
# Train SAM2-tiny with LoRA on 10K synthetic dataset (1080Ti compatible)

set -e  # Exit on error

echo "============================================================"
echo "SAM-RFI LoRA Training - SAM2-tiny on 1080Ti"
echo "============================================================"
echo ""
echo "Config: configs/training_lora_tiny_10k.yaml"
echo "Model: SAM2-tiny with LoRA (rank=16, alpha=32)"
echo "Dataset: 10K training + 1K validation"
echo "Batch size: 4 (fits in 8GB VRAM)"
echo "Native resolution: 1024x1024"
echo ""
echo "============================================================"
echo ""

# Run training pipeline
python scripts/run_training.py --config configs/training_lora_tiny_10k.yaml --skip-generation 2>&1 | tee training_lora_tiny.log

echo ""
echo "============================================================"
echo "Training complete!"
echo "Log saved to: training_lora_tiny.log"
echo "Output directory: ./training_output_lora_tiny_10k"
echo "============================================================"
