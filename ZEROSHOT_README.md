# Zero-Shot SAM3 Test - Quick Start

## What This Does

Tests if **pretrained SAM3** (no training!) can detect RFI using text prompts alone.
Compares against CASA tfcrop/rflag baselines.

**Critical question**: Does SAM3 understand "radio frequency interference" without ever seeing radio astronomy data?

---

## Requirements

### GPU
- **Works on 1080ti** (11GB VRAM) ✅
- Inference only (no training) - low memory usage
- ~8-10GB VRAM for SAM3 inference

### Software
```bash
# 1. Install transformers from GitHub (SAM3 only in dev branch)
pip install git+https://github.com/huggingface/transformers

# 2. Install SAM-RFI
pip install -e .

# 3. Install dependencies
pip install torch torchvision matplotlib tqdm datasets scipy pillow
```

---

## Quick Run

```bash
# Run full zero-shot test (20 synthetic RFI samples)
python scripts/zeroshot_comparison.py --output results/zeroshot/

# Expected runtime on 1080ti: ~15-20 minutes
#   - Data generation: ~5 min
#   - SAM3 inference: ~8-10 min (6 prompts × 20 samples)
#   - CASA comparison: ~2 min
#   - Plotting: ~1 min
```

---

## What It Tests

### SAM3 Zero-Shot (6 text prompts)
- `"radio frequency interference"`
- `"interference pattern"`
- `"corrupted signal region"`
- `"noise contamination"`
- `"anomalous signal"`
- `"RFI"`

### CASA Baselines (algorithmic approximations)
- tfcrop (MAD flagging)
- rflag (SumThreshold)
- Combined (tfcrop + rflag)

---

## Output

```
results/zeroshot/
├── synthetic_data/           # 20 test samples with ground truth
├── comparison_results.json   # Detailed metrics
├── plots/
│   ├── comparison_barchart.png     # SAM3 vs CASA comparison
│   └── sam3_prompts_comparison.png # Best text prompt
```

---

## Expected Results

### Scenario A: Text Prompting Works 🎉
```
SAM3 "radio frequency interference": IoU=0.65, F1=0.75
CASA combined: IoU=0.58, F1=0.70

✅ SAM3 beats CASA with zero-shot text prompts!
→ Next step: Fine-tune to push IoU > 0.80
```

### Scenario B: Text Prompting Fails ❌
```
SAM3 best prompt: IoU=0.15, F1=0.22
CASA combined: IoU=0.58, F1=0.70

❌ Pretrained SAM3 doesn't understand radio data
→ Next step: Train with visual prompts (bounding boxes)
```

### Scenario C: Partial Success ⚠️
```
SAM3 "interference pattern": IoU=0.35, F1=0.48
CASA combined: IoU=0.58, F1=0.70

⚠️ Shows promise but needs fine-tuning
→ Next step: Fine-tune with text+bbox hybrid prompts
```

---

## Memory Optimization (for 1080ti)

If you run out of VRAM, reduce test size:

```yaml
# Edit configs/zeroshot_test_20.yaml

synthetic:
  num_samples: 10  # Reduce from 20 to 10
  num_channels: 512  # Reduce from 1024 to 512
  num_times: 512
```

Or run with smaller batches:
```python
# In scripts/zeroshot_comparison.py, line ~140
# Change batch processing if needed
```

---

## Troubleshooting

### Error: `Sam3Model not found`
```bash
# Install transformers from GitHub (not PyPI)
pip install --upgrade git+https://github.com/huggingface/transformers
```

### Error: `CUDA out of memory`
```bash
# Reduce image size in config
synthetic:
  num_channels: 512  # Instead of 1024
  num_times: 512
```

### Error: `SAM-RFI not available`
```bash
# Install package
pip install -e .
```

---

## Next Steps Based on Results

### If IoU > 0.5 (Success!)
1. Fine-tune SAM3 with text prompts on larger dataset
2. Test on real VLA data
3. Compare trained SAM3 vs CASA on real observations
4. **Write paper!**

### If IoU < 0.2 (Failure)
1. Switch to visual prompting (bounding boxes)
2. Train SAM3 like SAM2 (proven to work)
3. Still compare against CASA
4. Paper focuses on visual prompt approach

### If 0.2 < IoU < 0.5 (Partial)
1. Try hybrid prompting (text + bounding boxes)
2. Fine-tune with best prompt
3. Compare hybrid vs pure visual
4. Paper discusses both approaches

---

## Timeline

- **Today**: Zero-shot test (15-20 min on 1080ti)
- **Tomorrow**: Analyze results, decide approach
- **Day 3-4**: Train SAM3 (if needed)
- **Day 5-7**: Full validation on real data
- **Week 2**: Paper draft

---

## Questions?

Check results in `results/zeroshot/comparison_results.json`

The script prints interpretation automatically:
- ✅ Success: IoU > 0.5
- ⚠️  Partial: IoU 0.2-0.5
- ❌ Failure: IoU < 0.2
