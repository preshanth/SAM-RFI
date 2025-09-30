# LoRA (Low-Rank Adaptation) for SAM2-RFI

**Purpose:** Enable SAM2 vision encoder to learn RFI-specific visual features while preserving pretrained natural image knowledge.

**Created:** 2025-09-29
**Status:** Implementation guide for testing

---

## What is LoRA?

**LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning method that adds small trainable weight matrices to frozen pretrained layers.

### Core Concept

Instead of updating large weight matrices directly:
```
W_new = W_frozen + ΔW
where ΔW = A × B
```

- `W_frozen`: Original SAM2 pretrained weights (remain frozen)
- `A`: Trainable matrix [d × r]
- `B`: Trainable matrix [r × d]
- `r`: Rank (typically 4, 8, or 16) where r << d

### Example Parameter Count

**Original attention layer:** 1024×1024 = 1,048,576 parameters

**LoRA with r=8:** (1024×8) + (8×1024) = 16,384 parameters (~1.5%)

### Benefits

1. **Minimal parameters:** Train 0.1-1% of total model
2. **Preserves pretrained knowledge:** Original weights stay frozen
3. **Task-specific adaptation:** Low-rank updates capture domain features (RFI patterns)
4. **Memory efficient:** Gradient computation only for small matrices
5. **No overfitting risk:** Small parameter space, safe for limited training data

---

## SAM2 Architecture Overview

### 1. Vision/Image Encoder (Hiera)

**Hierarchical MAE with 4 stages:**

- **Stage 1**: Stride 4 features (high resolution, skip connection to decoder)
- **Stage 2**: Stride 8 features (skip connection to decoder)
- **Stage 3**: Stride 16 features (used in memory attention)
- **Stage 4**: Stride 32 features (used in memory attention)

**Each stage contains transformer blocks with:**
- Attention layers:
  - `qkv` projection: Generates Query, Key, Value tensors
  - `proj` (output projection): Projects attention output back
- MLP layers: Feed-forward network for feature transformation

### 2. Prompt Encoder

- **Sparse prompts** (points, boxes): Positional encodings + learned embeddings
- **Dense prompts** (masks): Convolutional encoding

### 3. Mask Decoder

- **Two-way transformer blocks**: Update prompt embeddings ↔ image embeddings
- **Skip connections**: High-resolution features from vision encoder Stages 1 & 2
- **Output**: Segmentation mask predictions + IoU quality scores

### 4. Memory System

- **Memory encoder + Memory bank** for video temporal consistency
- **Not used for static image training** (RFI waterfall images)

---

## Trainable Layers for LoRA

### High Priority (Maximum Impact for RFI)

**Vision Encoder Attention Layers:**
- `vision_encoder.blocks.*.attn.qkv` - Where visual features are extracted
- `vision_encoder.blocks.*.attn.proj` - Output projection of attention

**Rationale:**
- These layers learn visual feature representations
- Adapting them allows learning RFI-specific patterns (frequency drifts, signal morphologies)
- Most effective for domain adaptation (natural images → radio data)

### Medium Priority

**Mask Decoder Attention:**
- `mask_decoder.transformer.layers.*.self_attn.qkv` - Decoder self-attention
- `mask_decoder.transformer.layers.*.cross_attn_token_to_image` - Cross-attention

**Rationale:**
- Less critical since mask decoder is already trainable in frozen encoder approach
- May provide marginal improvement for RFI-specific segmentation refinement

### Lower Priority

**MLP Layers:**
- `vision_encoder.blocks.*.mlp` - Feed-forward networks
- `mask_decoder.*.mlp`

**Rationale:**
- More parameters to train (~3x attention layers)
- Attention is where spatial relationships are learned (more relevant for segmentation)

---

## Minimal Implementation: Vision Encoder LoRA

### 1. Install Dependencies

```bash
pip install peft  # Parameter-Efficient Fine-Tuning library from Hugging Face
```

### 2. Code Integration

**File:** `src/samrfi/models/training.py`

**Add import at top:**
```python
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not available - LoRA disabled")
```

**In `setup_model()` method, after encoder freezing (line 346):**
```python
# Freeze vision encoder and prompt encoder (only train mask decoder)
for name, param in self.model.named_parameters():
    if any(encoder_name in name for encoder_name in
           ["vision_encoder", "prompt_encoder", "image_encoder"]):
        param.requires_grad = False

# Count trainable parameters (mask decoder only)
trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in self.model.parameters())
logger.info(f"Froze vision/prompt encoders: {trainable_params:,} / {total_params:,} parameters trainable")

# Apply LoRA to vision encoder (if enabled)
if self.config.get("training", {}).get("use_lora", False) and PEFT_AVAILABLE:
    lora_config = LoraConfig(
        r=self.config["training"].get("lora_r", 8),              # Low rank
        lora_alpha=self.config["training"].get("lora_alpha", 32), # Scaling factor
        target_modules=[
            "vision_encoder.blocks.*.attn.qkv",   # Attention Q,K,V projection
            "vision_encoder.blocks.*.attn.proj",  # Attention output projection
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="SEMANTIC_SEGMENTATION"
    )

    self.model = get_peft_model(self.model, lora_config)

    # Count trainable parameters after LoRA
    lora_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    logger.info(f"Applied LoRA to vision encoder: {lora_trainable:,} trainable parameters")
    logger.info(f"LoRA overhead: {lora_trainable - trainable_params:,} parameters ({100*(lora_trainable-trainable_params)/total_params:.2f}%)")
else:
    logger.info("Only training mask decoder for efficient fine-tuning")
```

### 3. Configuration

**File:** `configs/training/v100_config.yaml`

**Add to `training` section:**
```yaml
training:
  batch_size: 1
  gradient_accumulation: 16
  learning_rate: 1e-5
  weight_decay: 0

  # LoRA configuration
  use_lora: true        # Enable LoRA for vision encoder
  lora_r: 8            # Low-rank dimension (4, 8, 16)
  lora_alpha: 32       # Scaling factor (typically 2-4× lora_r)
```

**LoRA Rank Selection:**
- `r=4`: Minimal parameters (~0.5%), most constrained
- `r=8`: Balanced (recommended starting point)
- `r=16`: More expressiveness, higher parameter count (~2%)

**LoRA Alpha:**
- Controls scaling of LoRA updates
- Rule of thumb: `lora_alpha = 4 × lora_r`
- Higher alpha = stronger LoRA influence

---

## Expected Impact

### Without LoRA (Frozen Vision Encoder)
- **Trainable:** Mask decoder only (~10% of model)
- **Feature extraction:** Generic natural image features (edges, textures)
- **Assumption:** Generic features sufficient for RFI segmentation
- **Risk:** May miss RFI-specific patterns (frequency drifts, signal morphologies)

### With LoRA (Adapted Vision Encoder)
- **Trainable:** Mask decoder + LoRA adapters (~11-12% of model)
- **Feature extraction:** Natural image features + RFI-specific adaptations
- **Learning:** Attention layers adapt to radio data characteristics
- **Benefit:** Better feature representations for RFI detection

### Performance Estimates (Medium Confidence)

**Frozen encoder baseline:** Expect reasonable convergence if pretrained features generalize

**With LoRA:** Potential 5-15% accuracy improvement from domain-specific features
- Better boundary detection (frequency/time edges)
- Improved handling of RFI morphology variations
- Stronger performance on minority RFI types (transients, narrowband)

**Unknown:** Actual improvement depends on:
1. How well SAM2 natural image features transfer to radio data
2. Training data diversity (RFI morphology coverage)
3. Whether RFI patterns require domain-specific features

---

## Testing Strategy

### Phase 1: Frozen Encoder Baseline
```bash
# Disable LoRA in config
use_lora: false

# Run training
python training/synthetic_training.py --config configs/training/v100_config.yaml

# Evaluate convergence and accuracy
```

**Success Criteria:**
- Loss decreases steadily (not plateau)
- Below 0.3 by epoch 10
- 90%+ pixel accuracy
- Validation loss tracks training

### Phase 2: Add LoRA (If Baseline Insufficient)
```bash
# Enable LoRA in config
use_lora: true
lora_r: 8
lora_alpha: 32

# Run training
python training/synthetic_training.py --config configs/training/v100_config.yaml

# Compare with frozen encoder baseline
```

**Compare:**
- Final loss (training + validation)
- Pixel accuracy
- IoU on different RFI types (broadband, narrowband, transients)
- Convergence speed

### Phase 3: Hyperparameter Tuning (If LoRA Helps)
```bash
# Experiment with rank
lora_r: 4, 8, 16

# Experiment with learning rate
learning_rate: 5e-6, 1e-5, 2e-5
```

---

## Implementation Checklist

**Prerequisites:**
- [ ] `pip install peft` installed
- [ ] Frozen encoder training tested (Phase 1)
- [ ] Baseline performance metrics recorded

**Code Changes:**
- [ ] Add `peft` import to `training.py`
- [ ] Add LoRA application code in `setup_model()`
- [ ] Add parameter counting for LoRA layers
- [ ] Add `use_lora`, `lora_r`, `lora_alpha` to config

**Testing:**
- [ ] Verify LoRA layers applied: check log for parameter counts
- [ ] Confirm only mask decoder + LoRA trainable (not full vision encoder)
- [ ] Compare memory usage vs frozen encoder baseline
- [ ] Monitor training convergence

**Validation:**
- [ ] Accuracy comparison: frozen vs LoRA
- [ ] IoU per RFI type: broadband, narrowband, transients, etc.
- [ ] Inference speed: ensure LoRA doesn't slow prediction
- [ ] Overfitting check: validation loss vs training loss

---

## Alternative Approaches (Future Exploration)

### 1. Prompt Tuning
**Concept:** Learn visual prompt tokens (prepended to image patches)
**Benefits:** Even fewer parameters (<0.1%)
**Implementation:** Add learnable tokens to vision encoder input

### 2. Adapter Layers
**Concept:** Insert small bottleneck layers between transformer blocks
**Benefits:** Simple, ~1% parameters
**Drawback:** Requires modifying SAM2 architecture

### 3. Full Vision Encoder Fine-Tuning
**Concept:** Unfreeze vision encoder, train all parameters
**Benefits:** Maximum flexibility
**Drawbacks:**
- Risk destroying pretrained features
- Requires large training dataset to prevent overfitting
- Much slower training

---

## References

**LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

**SAM2 Architecture:** Meta AI Segment Anything Model 2 (2024)

**SAM LoRA Fine-Tuning:** Multiple implementations for medical imaging (SAMed, Sam_LoRA)

**PEFT Library:** Hugging Face Parameter-Efficient Fine-Tuning toolkit

---

## Summary

**LoRA provides a low-risk way to adapt SAM2's vision encoder to RFI detection:**

1. **Minimal overhead:** ~1% additional parameters
2. **Preserves pretrained knowledge:** Frozen weights + small adaptations
3. **Domain adaptation:** Learn RFI-specific visual features
4. **Easy to test:** Enable/disable via config flag

**Recommendation:**
1. Test frozen encoder first (validate baseline fixes work)
2. Add LoRA if accuracy needs improvement
3. Compare performance quantitatively before deciding

**Key Unknown:** Whether SAM2's natural image features generalize well to radio data. LoRA provides insurance if they don't.