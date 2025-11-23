#!/bin/bash
# Run zero-shot SAM3 test on CPU with 32 cores
# Uses smaller 512x512 images and 10 samples for faster testing

echo "=========================================="
echo "SAM3 Zero-Shot Test (CPU Mode)"
echo "=========================================="
echo "Config: 10 samples, 512x512 images"
echo "Cores: 32 (half of 64)"
echo "Device: CPU"
echo "=========================================="
echo ""

python scripts/zeroshot_comparison.py \
    --output results/zeroshot_cpu/ \
    --config configs/zeroshot_cpu_10.yaml \
    --cpu \
    --num-cores 32

echo ""
echo "=========================================="
echo "Test complete! Check results at:"
echo "  results/zeroshot_cpu/comparison_results.json"
echo "  results/zeroshot_cpu/plots/"
echo "=========================================="
