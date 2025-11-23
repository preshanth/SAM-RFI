#!/bin/bash
# Quick environment check before running validation

echo "======================================================================"
echo "ENVIRONMENT CHECK"
echo "======================================================================"
echo ""

# Check Python version
echo "[1/5] Python version..."
python3 --version
echo ""

# Check pip
echo "[2/5] pip packages..."
echo "  transformers: $(pip show transformers 2>/dev/null | grep Version || echo 'NOT INSTALLED')"
echo "  torch: $(pip show torch 2>/dev/null | grep Version || echo 'NOT INSTALLED')"
echo "  numpy: $(pip show numpy 2>/dev/null | grep Version || echo 'NOT INSTALLED')"
echo "  pillow: $(pip show pillow 2>/dev/null | grep Version || echo 'NOT INSTALLED')"
echo ""

# Check CUDA
echo "[3/5] CUDA availability..."
python3 -c "import torch; print(f'  PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'  Device count: {torch.cuda.device_count()}'); print(f'  Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  ✗ PyTorch not installed"
echo ""

# Check HuggingFace authentication
echo "[4/5] HuggingFace authentication..."
if [ -f ~/.huggingface/token ]; then
    echo "  ✓ HuggingFace token found at ~/.huggingface/token"
else
    echo "  ✗ No HuggingFace token found"
    echo "    Run: huggingface-cli login"
fi
echo ""

# Check git
echo "[5/5] Git status..."
git --version
echo "  Current branch: $(git branch --show-current)"
echo "  Working directory: $(pwd)"
echo ""

echo "======================================================================"
echo "READY FOR VALIDATION?"
echo "======================================================================"
echo ""

# Check if ready
READY=true

if ! python3 -c "import transformers" 2>/dev/null; then
    echo "✗ transformers not installed"
    echo "  Install: pip install transformers"
    READY=false
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "✗ PyTorch not installed"
    echo "  Install: pip install torch torchvision"
    READY=false
fi

if ! [ -f ~/.huggingface/token ]; then
    echo "⚠️  No HuggingFace token (may be needed for facebook/sam3)"
    echo "  Login: huggingface-cli login"
    echo "  Request access: https://huggingface.co/facebook/sam3"
fi

if [ "$READY" = true ]; then
    echo ""
    echo "✓ Environment looks good!"
    echo ""
    echo "NEXT STEPS:"
    echo "1. Ensure you have access to facebook/sam3 on HuggingFace"
    echo "2. Run: python validate_sam3.py"
    echo ""
else
    echo ""
    echo "✗ Environment needs setup"
    echo "  Fix the issues above before running validation"
    echo ""
fi
