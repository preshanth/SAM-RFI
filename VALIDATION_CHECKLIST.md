# SAM3 Migration Validation Checklist

## Overview

Before starting the SAM3 migration, we need to validate that Sam3Tracker from transformers supports fine-tuning (training the mask decoder weights).

**Current Status:** SAM2 branch uses `transformers` library with simple PyTorch training loop
**Goal:** Migrate to SAM3 with minimal code changes
**Critical Question:** Does Sam3Tracker support fine-tuning via transformers?

---

## Pre-requisites

### 1. HuggingFace Access Token

**Action Required:**
```bash
# Get access to facebook/sam3 on HuggingFace
# Visit: https://huggingface.co/facebook/sam3
# Click "Request Access" button
# Wait for approval email

# Once approved, authenticate:
huggingface-cli login
# Paste your token when prompted
```

**Why:** The SAM3 model checkpoint is gated and requires approval

### 2. Install/Upgrade Dependencies

**Action Required:**
```bash
# Upgrade transformers to latest version (must have SAM3 support)
pip install --upgrade transformers

# Install other dependencies if needed
pip install torch torchvision pillow numpy
```

**Why:** Sam3Tracker was added to transformers recently (Nov 2025)

### 3. GPU Access

**Confirmed:** You have H100 access ✓
**Memory Required:** ~20GB for training (848M params + gradients + batch data)
**H100 Capacity:** 80GB ✓ More than sufficient

---

## Validation Tests

Run these tests **IN ORDER** before making any code changes.

### Test 1: Transformers Approach (RECOMMENDED - Try First)

**Run:**
```bash
cd /home/user/SAM-RFI
python validate_sam3.py
```

**What it tests:**
1. ✓ Can import Sam3TrackerModel and Sam3TrackerProcessor
2. ✓ Can load facebook/sam3 checkpoint
3. ✓ Parameter structure (vision encoder, prompt encoder, mask decoder)
4. ✓ Freezing encoders works (like SAM2 training)
5. ✓ Forward pass with visual prompts (bounding boxes)
6. ✓ Backward pass works (gradient computation)

**Expected Output (SUCCESS):**
```
======================================================================
VALIDATION SUMMARY
======================================================================

✓ Sam3Tracker is available in transformers
✓ Model has 848M total parameters
✓ Mask decoder has ~XX.XM trainable parameters
✓ Encoder freezing works
✓ Forward pass with visual prompts works
✓ Backward pass (gradient computation) works

======================================================================
CONCLUSION: Sam3Tracker SUPPORTS fine-tuning via transformers! ✓
======================================================================
```

**If this succeeds:** Migration is simple (~15 line changes, 2-3 hours)
**If this fails:** Need to use native SAM3 package (more complex, 3-5 days)

---

### Test 2: Native SAM3 Package (Fallback - Only if Test 1 Fails)

**Run:**
```bash
cd /home/user/SAM-RFI
bash inspect_native_sam3.sh
```

**What it does:**
- Clones https://github.com/facebookresearch/sam3
- Shows directory structure
- Lists training configs (Hydra YAML files)
- Compares native API vs transformers API

**When to use this:**
- If Test 1 fails (Sam3Tracker doesn't support training)
- To understand what rewrite would be needed
- To see training examples from Meta

---

## Decision Tree

```
Start
  │
  ├─> Run: python validate_sam3.py
  │
  ├─> SUCCESS? ─────────────────> YES ──> Use Transformers Approach
  │                                │       • Minimal code changes
  │                                │       • Keep existing architecture
  │                                │       • Migration time: 2-3 hours
  │                                │       ✓ RECOMMENDED
  │
  └─> FAILURE? ─────────────────> NO ───> Use Native SAM3 Package
                                          • Rewrite training loop
                                          • Convert to Hydra configs
                                          • Migration time: 3-5 days
                                          • Only if you NEED SAM3 features
                                          ⚠️  OR stay with SAM2
```

---

## What To Check in Validation Results

### Critical Success Indicators

1. **Import Success:**
   ```
   ✓ Sam3TrackerModel found
   ✓ Sam3TrackerProcessor found
   ```
   If ✗ → Need to upgrade transformers

2. **Parameter Structure:**
   ```
   Mask decoder: XX.XM parameters
   ```
   Should be similar to SAM2 (~40-80M params depending on architecture)

3. **Freezing Works:**
   ```
   Trainable params before freezing: XXX.XM
   Trainable params after freezing: XX.XM
   ```
   Should see significant reduction (only mask decoder trainable)

4. **Backward Pass:**
   ```
   ✓ Backward pass successful - mask_decoder has gradients
   ✓ Vision encoder correctly frozen (no gradients)
   ```
   This is THE CRITICAL TEST - proves training will work

### Failure Scenarios

**Scenario 1: Sam3Tracker not found**
```
✗ Sam3Tracker not available in transformers
```
**Fix:** Upgrade transformers: `pip install --upgrade transformers`

**Scenario 2: Can't load model**
```
✗ Failed to load model
Error: 401 Unauthorized
```
**Fix:** Run `huggingface-cli login` and ensure you have access to facebook/sam3

**Scenario 3: Backward pass fails**
```
✗ Backward pass failed
```
**Meaning:** Sam3Tracker doesn't support training via transformers
**Action:** Must use native SAM3 package OR stick with SAM2

---

## After Validation

### If Test 1 SUCCEEDS:

**Next Steps:**
1. Create git worktree for parallel development
2. Update 4 files:
   - `src/samrfi/training/sam2_trainer.py` (~5 lines)
   - `src/samrfi/inference/predictor.py` (~5 lines)
   - `src/samrfi/utils/model_cache.py` (~3 lines)
   - Training configs (~2 lines)
3. Test on small synthetic dataset
4. Compare SAM2 vs SAM3 performance
5. Commit and push

**Estimated Time:** 2-3 hours

---

### If Test 1 FAILS:

**Decision Point:**

**Option A: Use Native SAM3**
- Rewrite training loop to use Hydra
- Adapt data pipeline to native format
- Create SAM3-specific configs
- Time: 3-5 days
- Only worth it if you NEED SAM3 features (text prompts, improved performance)

**Option B: Stay with SAM2**
- Keep current working code
- SAM2 is proven and works well
- Wait for better transformers integration
- Time: 0 hours (no migration)

**Recommendation:**
- If you only need visual prompt training (current use case) → Stay with SAM2
- If you want text prompts ("radio frequency interference") → Invest in native SAM3

---

## Questions to Answer After Validation

1. **Did Test 1 succeed?** (Yes/No)

2. **If No, do we NEED SAM3?**
   - Do we need text prompt capability?
   - Do we need improved performance over SAM2?
   - Is it worth 3-5 days of rewrite?

3. **If Yes to Test 1, proceed with migration?**
   - Risk: Low (drop-in replacement)
   - Effort: Low (2-3 hours)
   - Benefit: Latest model, improved performance

---

## Files Created for Validation

- `validate_sam3.py` - Main validation script (transformers approach)
- `inspect_native_sam3.sh` - Native SAM3 inspection (fallback)
- `VALIDATION_CHECKLIST.md` - This file

**Run validation NOW before any code changes!**
