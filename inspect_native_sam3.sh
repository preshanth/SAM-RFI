#!/bin/bash
# Inspect native SAM3 package structure
# Run this to understand the native SAM3 API for comparison

echo "======================================================================"
echo "NATIVE SAM3 PACKAGE INSPECTION"
echo "======================================================================"
echo ""

# Clone SAM3 repo to temporary location
SAM3_DIR="/tmp/sam3_inspection"
echo "[1/4] Cloning SAM3 repository..."
if [ -d "$SAM3_DIR" ]; then
    echo "  Removing existing clone..."
    rm -rf "$SAM3_DIR"
fi

git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
cd "$SAM3_DIR"

echo "  ✓ Cloned to $SAM3_DIR"
echo ""

# Show directory structure
echo "[2/4] Repository structure..."
echo ""
tree -L 2 -I '__pycache__|*.pyc|*.pth|*.pt' || find . -maxdepth 2 -type f -name "*.py" | head -20
echo ""

# Show key files
echo "[3/4] Key files for understanding API..."
echo ""

if [ -f "sam3/model_builder.py" ]; then
    echo "  ✓ sam3/model_builder.py (main entry point)"
fi

if [ -f "sam3/train/train.py" ]; then
    echo "  ✓ sam3/train/train.py (training script)"
fi

if [ -d "sam3/train/configs" ]; then
    echo "  ✓ sam3/train/configs/ (Hydra configs)"
    echo ""
    echo "    Available config files:"
    find sam3/train/configs -name "*.yaml" | head -10 | sed 's/^/      - /'
fi

echo ""

# Extract key functions from model_builder.py
echo "[4/4] Key functions in model_builder.py..."
echo ""
if [ -f "sam3/model_builder.py" ]; then
    grep -n "^def " sam3/model_builder.py | head -20 | sed 's/^/  /'
else
    echo "  ✗ model_builder.py not found"
fi

echo ""
echo "======================================================================"
echo "COMPARISON: Native SAM3 vs Transformers"
echo "======================================================================"
echo ""
echo "NATIVE SAM3 PACKAGE:"
echo "  - Uses: build_sam3_image_model(), Sam3Processor"
echo "  - Training: Hydra configs + sam3/train/train.py"
echo "  - Config: YAML files in sam3/train/configs/"
echo "  - API: Different from transformers (more like original SAM)"
echo ""
echo "TRANSFORMERS SAM3:"
echo "  - Uses: Sam3TrackerModel, Sam3TrackerProcessor"
echo "  - Training: Standard PyTorch loop (like SAM2)"
echo "  - Config: Simple Python dicts/YAML"
echo "  - API: Same as SAM2 (drop-in replacement)"
echo ""
echo "FOR SAM-RFI:"
echo "  - Current code uses transformers (SAM2Model)"
echo "  - If Sam3Tracker works → use transformers (easiest)"
echo "  - If not → rewrite to use native package (more work)"
echo ""
echo "RUN: python validate_sam3.py to test transformers approach"
echo ""
