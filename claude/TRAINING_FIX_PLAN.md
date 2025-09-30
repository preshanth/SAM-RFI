# SAM-RFI Training Fix Plan - Root Cause & Solutions

**Date**: 2025-09-29
**Status**: Ready for Implementation
**Root Cause**: Training entire SAM2 model instead of just mask decoder

## Critical Discovery

**Current Training Approach (WRONG)**:
```python
# Line 357 in src/samrfi/models/training.py
self.optimizer = AdamW(
    self.model.parameters(),  # TRAINS ALL LAYERS
    lr=1e-4,
    weight_decay=1e-2
)
```

**Legacy Training Approach (WORKED)**:
```python
# Lines 86-94 in samrfi/rfitraining.py
# Freeze vision and prompt encoders
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Only train mask decoder
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
```

**Impact**: Training all layers with 10× higher learning rate destroys pretrained features → causes plateau at 0.52-0.54

---

## Fix #1: Freeze Vision & Prompt Encoders (CRITICAL)

### Implementation
**File**: `src/samrfi/models/training.py`
**Location**: In `setup_model()` method, after line 331

**Add**:
```python
# Freeze vision encoder and prompt encoder (only train mask decoder)
# This preserves SAM2's pretrained features from millions of images
for name, param in self.model.named_parameters():
    if any(encoder_name in name for encoder_name in
           ["vision_encoder", "prompt_encoder", "image_encoder"]):
        param.requires_grad = False

# Count trainable parameters
trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in self.model.parameters())
logger.info(f"Froze vision/prompt encoders: {trainable_params:,} / {total_params:,} parameters trainable")
logger.info("Only training mask decoder for efficient fine-tuning")
```

**Rationale**:
- SAM2 vision encoder trained on 11M images - don't destroy
- Mask decoder is task-specific - needs training for RFI
- Reduces trainable parameters by ~90%
- Faster training, less data required
- Matches legacy approach that converged

---

## Fix #2: Reduce Learning Rate & Disable Weight Decay

### Implementation
**File**: `configs/training/v100_config.yaml`

**Change lines 15-16**:
```yaml
# Old (WRONG)
learning_rate: 1e-4
weight_decay: 1e-2

# New (CORRECT)
learning_rate: 1e-5    # 10× reduction, matches legacy
weight_decay: 0        # Disable for fine-tuning mask decoder
```

**Rationale**:
- lr=1e-4 too aggressive for fine-tuning pretrained model
- lr=1e-5 matches legacy SAM1 training (successful)
- weight_decay=0 prevents overfitting with small trainable parameters
- Conservative learning preserves pretrained features

---

## Fix #3: Switch to DiceCELoss

### Implementation
**File**: `src/samrfi/models/training.py`

**Add import** (top of file):
```python
from monai.losses import DiceCELoss
```

**In `_compute_batch_loss()` method, replace line 774**:
```python
# Old (WRONG for imbalanced data)
segmentation_loss = F.binary_cross_entropy_with_logits(union_logits, gt_masks, reduction='mean')

# New (CORRECT for class imbalance)
dice_ce_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
segmentation_loss = dice_ce_loss(
    union_logits.unsqueeze(1),  # Add channel dimension
    gt_masks.unsqueeze(1)
)
```

**Rationale**:
- RFI detection has severe class imbalance (few RFI pixels, many clean pixels)
- BCE treats all pixels equally → model can achieve low loss by predicting "no RFI"
- Dice coefficient rewards overlap → forces model to detect RFI
- Legacy used DiceCELoss successfully
- Standard for medical/scientific segmentation tasks

---

## Advanced Option: Adapter Layers for Vision Encoder

### Question from User
"Is it possible to add RFI segmentation context to the vision encoder without retraining it? Like reweighting?"

### Answer: Yes - Multiple Approaches

#### Option A: LoRA (Low-Rank Adaptation) - RECOMMENDED
**Concept**: Add small trainable matrices to frozen layers
```python
# Add after freezing layers
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,
    target_modules=["vision_encoder.blocks.*.attn.qkv"],  # Attention layers
    lora_dropout=0.1,
    bias="none"
)
self.model = get_peft_model(self.model, lora_config)
```

**Benefits**:
- Only train 0.1-1% of parameters
- Preserve pretrained knowledge
- Add RFI-specific features to vision encoder
- Fast training, minimal overfitting

**Cost**: Requires `peft` library: `pip install peft`

#### Option B: Adapter Layers
**Concept**: Insert small bottleneck layers into frozen network
```python
# After each transformer block in vision encoder
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.up = nn.Linear(adapter_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))  # Residual connection
```

**Benefits**:
- Simple to implement
- Only adapters are trained (~1% of parameters)
- Preserves pretrained features

**Cost**: Need to modify SAM2 architecture

#### Option C: Prompt Tuning
**Concept**: Learn visual prompts (learnable tokens) that condition encoder
```python
# Add learnable prompt tokens to image patches
self.visual_prompt = nn.Parameter(torch.randn(1, num_prompt_tokens, hidden_dim))
```

**Benefits**:
- Minimal parameters (<0.1%)
- No architecture changes
- Conditions encoder for RFI task

**Cost**: May not be as effective as LoRA/adapters

### Recommendation: Start with Fix #1 (Frozen Encoder)

**Reasoning**:
1. **Test frozen encoder first** - May already converge well
2. **If still needs help** - Add LoRA for vision encoder
3. **LoRA is easiest** - One config change with `peft` library
4. **Incremental improvement** - Don't overcomplicate initially

**Implementation Priority**:
1. Apply Fixes #1-3 (freeze, lr, loss)
2. Train and evaluate convergence
3. If convergence good but accuracy low → add LoRA
4. If convergence still poor → check data quality/diversity

---

## Fix #4: Remove Double Processing Bug (Performance)

### Implementation
**File**: `src/samrfi/models/training.py`
**Location**: In `_compute_sam2_loss()` method

**Delete lines 682-691**:
```python
# DELETE THIS BLOCK (full image forward pass that gets discarded)
with torch.set_grad_enabled(True):
    outputs = sam2_model(**inputs)
```

**Keep lines 692-707** (tiling processing)

**Rationale**:
- Wastes 25-30% memory and compute
- Processes image twice, throws away first result
- Not a convergence issue, just inefficiency
- Apply AFTER confirming Fixes #1-3 work

---

## Implementation Steps

### Step 1: Apply Critical Fixes
```bash
# 1. Edit src/samrfi/models/training.py
#    - Add freezing code in setup_model()
#    - Add DiceCELoss import and replace BCE

# 2. Edit configs/training/v100_config.yaml
#    - learning_rate: 1e-5
#    - weight_decay: 0

# 3. Verify changes
git diff
```

### Step 2: Test Training
```bash
# Run training with fixed config
python training/synthetic_training.py \
    --config configs/training/v100_config.yaml \
    --output-dir test_frozen_encoder

# Watch loss - should decrease below 0.3 within 10 epochs
# tail -f test_frozen_encoder/synthetic_training.log
```

### Step 3: Evaluate Results
**Success Criteria**:
- Loss decreases steadily (not plateau)
- Below 0.3 by epoch 5
- Below 0.2 by epoch 10
- Validation loss tracks training loss

**If successful**:
- Apply Fix #4 (remove double processing)
- Create A100 config for larger batches
- Run full 50-epoch training

**If still plateaus**:
- Check data quality/diversity
- Try disabling gaussianity loss temporarily
- Consider adding LoRA to vision encoder

---

## Expected Performance After Fixes

### Current (Broken)
- Loss plateaus at 0.52-0.54 immediately
- No improvement after epoch 2
- 74% pixel accuracy stuck
- Training appears ineffective

### After Fix #1-3 (Expected)
- Loss starts ~0.6, decreases to ~0.2 by epoch 10
- Steady improvement each epoch
- 90%+ pixel accuracy by epoch 10
- Model learns RFI patterns

### After Fix #4 (Performance)
- 20-30% faster training
- 25% less memory usage
- Can increase batch_size on A100
- No impact on convergence (just efficiency)

---

## A100/L40s Configuration (After Fixes)

### Create: `configs/training/a100_config.yaml`
```yaml
model:
  version: "sam2"
  variant: "large"  # Can handle large model with 40GB
  image_size: 1024

training:
  batch_size: 3              # Increase from 1 after bug fix
  gradient_accumulation: 8   # Reduce from 16
  mixed_precision: true      # Re-enable with fixed scaler
  max_epochs: 50
  learning_rate: 1e-5        # Match frozen encoder strategy
  weight_decay: 0            # Disable for mask decoder
  gradient_checkpointing: false  # Disable with 40GB VRAM

dataset:
  loading_strategy: "eager"   # A100 has RAM
  memory_budget_gb: 128       # Adjust for system

hardware:
  target_gpu: "A100"
  memory_limit: "40GB"

optimizer:
  name: "AdamW"
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  name: "cosine"
  warmup_ratio: 0.1

loss:
  gaussianity:
    enabled: false  # Disable initially, re-enable if needed

logging:
  log_every_n_steps: 50
  save_every_n_epochs: 5
```

---

## LoRA Implementation (Optional - After Testing Frozen Encoder)

### If frozen encoder converges but accuracy needs boost

**Install**:
```bash
pip install peft
```

**Add to `setup_model()` in training.py**:
```python
# After freezing encoders, before optimizer setup
if self.config.get("use_lora", False):
    from peft import get_peft_model, LoraConfig

    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=32,
        target_modules=[
            "vision_encoder.blocks.*.attn.qkv",  # Attention Q,K,V
            "vision_encoder.blocks.*.attn.proj", # Attention projection
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="SEMANTIC_SEGMENTATION"
    )

    self.model = get_peft_model(self.model, lora_config)
    logger.info(f"Applied LoRA: {self.model.num_parameters()} total, "
                f"{self.model.num_parameters(trainable_only=True)} trainable")
```

**Config addition**:
```yaml
# In training section
use_lora: true
lora_r: 8
lora_alpha: 32
```

**Benefits**:
- Adds RFI-specific adaptation to vision encoder
- Only trains ~1% extra parameters
- Minimal overfitting risk
- Can improve accuracy 5-10% over frozen encoder

---

## Summary Checklist

**Critical (Do First)**:
- [ ] Add encoder freezing code to `training.py`
- [ ] Switch to DiceCELoss in `training.py`
- [ ] Update `v100_config.yaml` (lr=1e-5, wd=0)
- [ ] Test training for 10 epochs
- [ ] Verify loss decreases below 0.3

**Performance (Do After Convergence)**:
- [ ] Remove double processing bug (lines 682-691)
- [ ] Test performance improvement
- [ ] Create A100 config

**Advanced (Do If Needed)**:
- [ ] Add LoRA if accuracy needs boost
- [ ] Experiment with different LoRA ranks
- [ ] Benchmark frozen vs LoRA accuracy

**Documentation**:
- [ ] Update TRAINING_REALITY.md with results
- [ ] Document final hyperparameters
- [ ] Record convergence curves
- [ ] Compare with legacy training results

---

## Why This Will Work

**Confidence: HIGH**

**Evidence**:
1. Legacy approach (frozen encoders) converged successfully
2. SAM2 is better than SAM1 - should work even better
3. DiceCELoss standard for segmentation with class imbalance
4. lr=1e-5 proven effective for mask decoder fine-tuning
5. RFI detection doesn't need vision encoder retraining

**Theory**:
- Vision encoder: Generic visual features (edges, textures, shapes)
- Mask decoder: Task-specific (RFI vs clean)
- Pretrained features already capture edges/structures in radio data
- Just need to train decoder to interpret them as RFI

**Risk**: LOW
- If doesn't work, can try LoRA
- If still doesn't work, data quality issue
- Worst case: revert to legacy approach