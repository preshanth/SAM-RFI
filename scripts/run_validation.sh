#!/bin/bash
#
# Complete validation pipeline for SAM-RFI
# Generates synthetic datasets and runs GPU validation
#
# Usage: ./run_validation.sh
#

set -e  # Exit on error

# Configuration
TRAIN_SIZE=4000
VAL_SIZE=1000
TRAIN_DIR="./datasets/train_${TRAIN_SIZE}"
VAL_DIR="./datasets/val_${VAL_SIZE}"
OUTPUT_DIR="./validation_output"
MAX_BATCH_SIZE=64

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "SAM-RFI Validation Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Training samples: ${TRAIN_SIZE}"
echo "  Validation samples: ${VAL_SIZE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Max batch size: ${MAX_BATCH_SIZE}"
echo ""

# Check if datasets already exist
if [ -d "${TRAIN_DIR}/exact_masks" ] && [ -d "${VAL_DIR}/exact_masks" ]; then
    echo -e "${YELLOW}Datasets already exist. Skip generation? (y/n)${NC}"
    read -r skip_gen
    if [ "$skip_gen" = "y" ]; then
        echo "Skipping dataset generation..."
        SKIP_GEN=true
    else
        echo "Regenerating datasets..."
        rm -rf "${TRAIN_DIR}" "${VAL_DIR}"
        SKIP_GEN=false
    fi
else
    SKIP_GEN=false
fi

# Step 1: Generate training dataset
if [ "$SKIP_GEN" = false ]; then
    echo ""
    echo "============================================================"
    echo "Step 1: Generating Training Dataset (${TRAIN_SIZE} samples)"
    echo "============================================================"
    echo ""

    samrfi generate-data \
        --source synthetic \
        --config configs/synthetic_train_4k.yaml \
        --output "${TRAIN_DIR}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training dataset generated successfully${NC}"
        echo "  Location: ${TRAIN_DIR}/exact_masks"
    else
        echo -e "${RED}✗ Training dataset generation failed${NC}"
        exit 1
    fi

    # Step 2: Generate validation dataset
    echo ""
    echo "============================================================"
    echo "Step 2: Generating Validation Dataset (${VAL_SIZE} samples)"
    echo "============================================================"
    echo ""

    samrfi generate-data \
        --source synthetic \
        --config configs/synthetic_val_1k.yaml \
        --output "${VAL_DIR}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Validation dataset generated successfully${NC}"
        echo "  Location: ${VAL_DIR}/exact_masks"
    else
        echo -e "${RED}✗ Validation dataset generation failed${NC}"
        exit 1
    fi
fi

# Step 3: Run GPU validation
echo ""
echo "============================================================"
echo "Step 3: Running GPU Validation"
echo "============================================================"
echo ""

# Check if CUDA is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}✗ CUDA not available! GPU validation requires CUDA.${NC}"
    exit 1
fi

# Print GPU info
echo "GPU Information:"
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
echo ""

python validate_gpu.py \
    --dataset "${TRAIN_DIR}/exact_masks" \
    --config configs/a100_validation.yaml \
    --max-batch-size ${MAX_BATCH_SIZE} \
    --num-epochs 1 \
    --output "${OUTPUT_DIR}/validation_report.json"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ GPU validation completed successfully${NC}"
    echo "  Report: ${OUTPUT_DIR}/validation_report.json"
else
    echo ""
    echo -e "${RED}✗ GPU validation failed${NC}"
    exit 1
fi

# Step 4: Summary
echo ""
echo "============================================================"
echo "Validation Complete!"
echo "============================================================"
echo ""
echo "Generated datasets:"
echo "  Training:   ${TRAIN_DIR}/exact_masks (${TRAIN_SIZE} samples)"
echo "  Validation: ${VAL_DIR}/exact_masks (${VAL_SIZE} samples)"
echo ""
echo "Output:"
echo "  Validation report: ${OUTPUT_DIR}/validation_report.json"
echo "  Models: ${OUTPUT_DIR}/models/"
echo "  Plots: ${OUTPUT_DIR}/models/*.png"
echo ""
echo "Next steps:"
echo "  1. Review validation report: cat ${OUTPUT_DIR}/validation_report.json | jq"
echo "  2. Check optimal batch size in report"
echo "  3. Run full training with validation:"
echo ""
echo "     samrfi train \\"
echo "       --config configs/gpu_v100_training.yaml \\"
echo "       --dataset ${TRAIN_DIR}/exact_masks \\"
echo "       --validation-dataset ${VAL_DIR}/exact_masks"
echo ""
echo "Note: Datasets are directories with batch_*.pt files (not .pt files)"
echo ""
