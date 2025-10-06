# Future Directions: Advanced SAM2 Techniques for RFI

## Overview

This document outlines advanced techniques to improve SAM2-RFI beyond the current decoder fine-tuning approach. These methods address key challenges: domain adaptation from synthetic to real data, efficient training, and leveraging traditional flagging algorithms (TFCrop/RFLAG).

---

## Current Approach: Decoder Fine-tuning

**What we do now:**
```python
# Freeze encoder (184M params)
# Train decoder only (~40M params)
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5)
```

**Strengths:**
- ✓ Lightweight (trains 18% of parameters)
- ✓ Fast training
- ✓ Leverages pretrained encoder features

**Limitations:**
- Encoder trained on ImageNet (natural images), not spectrograms
- Fixed encoder may miss RFI-specific patterns (frequency sweeps, narrowband bursts)
- No adaptation mechanism for new RFI types or sites

---

## 1. LoRA (Low-Rank Adaptation)

### Motivation

**Problem:** The frozen encoder extracts ImageNet features, not RFI-specific spectro-temporal patterns.

**Solution:** Add lightweight trainable adapters to the encoder without full fine-tuning.

### How LoRA Works

Instead of updating massive weight matrices, inject low-rank decomposition:

```
Standard layer:     y = W_frozen · x
LoRA-adapted layer: y = W_frozen · x + (B · A) · x

Where:
- W_frozen: Original pretrained weights (frozen)
- A: Low-rank down-projection (d → r), e.g., 1024 → 16
- B: Low-rank up-projection (r → d), e.g., 16 → 1024
- Trainable: Only A and B matrices (~0.5M params vs 40M)
```

### Architecture

```
SAM2-Hiera-Large (224M params)
├─ Image Encoder (frozen)
│   └─ + LoRA adapters in attention layers (trainable)
│       • Blocks [0,1,2,3]: Early feature extraction
│       • Q/K/V projections: rank-16 adapters
│       • Learn RFI-specific edges, sweeps, bursts
│
├─ Prompt Encoder (frozen)
│
└─ Mask Decoder (frozen OR LoRA)
    └─ + LoRA in cross-attention (optional)
```

### Configuration

```yaml
lora:
  enabled: true
  rank: 16                    # Low-rank dimension (8, 16, 32, 64)
  alpha: 32                   # Scaling factor (typically 2×rank)
  dropout: 0.1                # Regularization

  target_modules:
    encoder:
      layers: [0, 1, 2, 3]    # First 4 blocks
      modules:
        - "attn.qkv"          # Q/K/V projections
        - "attn.proj"         # Output projection

    decoder:
      modules:
        - "cross_attn_token_to_image.q_proj"
        - "cross_attn_token_to_image.k_proj"
```

### Expected Benefits

1. **Encoder adaptation:** Learn RFI-specific features while keeping most weights frozen
2. **Fewer parameters:** Train 0.2-1% of model (500K-2M params)
3. **Faster training:** 3-5x speedup vs decoder fine-tuning
4. **Less overfitting:** Low-rank bottleneck acts as regularization
5. **Lower memory:** No optimizer states for frozen layers (~30% reduction)
6. **Modular adapters:** Can train multiple LoRA modules for different RFI types

### Implementation

```python
from peft import LoraConfig, get_peft_model

model = Sam2Model.from_pretrained("facebook/sam2-hiera-large")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "vision_encoder.blocks.0.attn.qkv",
        "vision_encoder.blocks.1.attn.qkv",
        "vision_encoder.blocks.2.attn.qkv",
        "vision_encoder.blocks.3.attn.qkv",
        "mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj",
    ],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 589,824 || all params: 224,589,824 || trainable: 0.26%
```

### Multiple LoRA Adapters

Train specialized adapters for different RFI types:

```
sam2-base.pth (frozen)
├─ lora_narrowband.pth    # GPS, satellites
├─ lora_broadband.pth     # Lightning, transients
├─ lora_sweeps.pth        # Radar chirps
└─ lora_combined.pth      # All types
```

**Runtime adapter swapping:**
```python
model.load_adapter("lora_narrowband.pth")  # Detect satellites
predictions = model(waterfall)

model.load_adapter("lora_sweeps.pth")      # Switch to radar detection
predictions = model(waterfall)
```

### When to Use LoRA

**Use if:**
- Current baseline (decoder-only) shows encoder struggles with RFI patterns
- Need faster training or lower memory
- Want modular RFI-type-specific models

**Skip if:**
- Current baseline works well (encoder features transfer fine)
- Already training fast enough

---

## 2. Support-Set Guided Prompting (SGP)

### Motivation

**Key insight from your work:** Including TFCrop/RFLAG-flagged examples improves performance on real data.

**Why?** They bridge the synthetic → real domain gap:
- Synthetic: Clean mathematical models, perfect noise, idealized bandpass
- Real: Messy artifacts, non-Gaussian noise, calibration errors, hardware issues

**Traditional flaggers encode domain knowledge** about what real RFI looks like.

### What is SGP?

Instead of manual prompts, the model **generates prompts automatically** by learning from a small "support set" of labeled examples.

**Think of it as:** Few-shot learning + Domain adaptation via examples

### How SGP Works

```
Support Set (5-10 examples):
├─ Real observation 1 + TFCrop mask
├─ Real observation 2 + RFLAG mask
├─ Real observation 3 + TFCrop mask
└─ ...
      ↓
  Encode into "RFI prototype"
  (captures TFCrop/RFLAG flagging style)
      ↓
Query: New observation
      ↓
  Compare to prototype
      ↓
  Auto-generate prompts
  (guided by TFCrop/RFLAG patterns)
      ↓
  SAM2 segments using these prompts
      ↓
  Final mask (adapted to real data)
```

### Comparison to Current Approach

| Aspect | Current | SGP |
|--------|---------|-----|
| Training data | 16,000 synthetic samples | Base: 16k synthetic<br>Support: 5-50 real examples |
| Prompts | Derived from ground truth | Auto-generated from support set |
| New RFI type | Retrain model | Show 5 examples, adapts instantly |
| Site adaptation | Fine-tune per site | Swap support set per site |
| Domain gap | Synthetic → Real via fine-tuning | Synthetic → Real via support examples |

### Architecture

```python
class SGPModule(nn.Module):
    """Support-Set Guided Prompting for RFI domain adaptation"""

    def __init__(self, feature_dim=256, prototype_dim=128):
        self.support_encoder = nn.Sequential(
            nn.Linear(feature_dim, prototype_dim),
            nn.ReLU(),
            nn.Linear(prototype_dim, prototype_dim)
        )

    def encode_support_set(self, support_examples):
        """
        Args:
            support_examples: [(image, mask), ...] from TFCrop/RFLAG
        Returns:
            prototype: Averaged feature representation
        """
        features = []
        for img, mask in support_examples:
            # Extract features from masked region
            feat = self.extract_rfi_features(img, mask)
            features.append(feat)

        # Average to create prototype (encodes TFCrop/RFLAG style)
        prototype = torch.stack(features).mean(dim=0)
        return prototype

    def guide_prediction(self, query_features, prototype):
        """
        Compare query to TFCrop/RFLAG prototype
        Generate prompts that match their flagging style
        """
        similarity = F.cosine_similarity(query_features, prototype)
        adjusted_prompts = self.generate_prompts(similarity)
        return adjusted_prompts
```

### Workflow: Two-Stage Transfer Learning

**Stage 1: Synthetic Pre-training (current baseline)**
```python
# Train on 16k synthetic samples
model = SAM2Trainer(synthetic_dataset)
model.train(epochs=20)  # Learn general RFI concept
model.save("synthetic_sam2.pth")
```

**Stage 2: Real-world Adaptation with SGP**
```python
# Load synthetic-trained model
model = load_pretrained("synthetic_sam2.pth")

# Add SGP module
sgp = SupportSetGuidedPrompting(
    encoder=model.vision_encoder,
    prototype_dim=256
)

# Create site-specific support sets from TFCrop/RFLAG
support_sets = {
    "VLA": [
        (vla_obs1, tfcrop_mask1),
        (vla_obs2, tfcrop_mask2),
        (vla_obs3, rflag_mask3),
        # 5-10 examples per site
    ],
    "MeerKAT": [
        (meerkat_obs1, tfcrop_mask1),
        (meerkat_obs2, rflag_mask2),
        # ...
    ],
}

# At inference: adapt to site using its support set
def predict(observation, site="VLA"):
    prototype = sgp.encode_support_set(support_sets[site])
    prompts = sgp.guide_prediction(observation, prototype)
    mask = model(observation, prompts=prompts)
    return mask
```

### Why SGP is Better Than Fine-tuning Alone

**Fine-tuning approach:**
```
Synthetic (16k) → Fine-tune on Real (500) → Single model
```
- ✗ Works for one site/configuration only
- ✗ Needs retraining for new sites
- ✗ Risk of catastrophic forgetting (loses synthetic knowledge)
- ✗ Expensive (requires lots of labeled real data)

**SGP approach:**
```
Synthetic (16k) → Base model (frozen)
                     ↓
                 SGP adapter (learns from support set)
                     ↓
              Swap support sets per site
```
- ✓ One model works for all sites
- ✓ No retraining for new sites (just provide 5-10 examples)
- ✓ Preserves synthetic knowledge (base frozen)
- ✓ Efficient (few examples needed)

### Use Cases for SGP

1. **Site-specific adaptation:**
   - VLA, MeerKAT, ASKAP have different RFI environments
   - Provide 5-10 TFCrop examples per site
   - Model adapts instantly

2. **New RFI type encountered:**
   - Starlink satellites appear (not in training data)
   - Manually flag 5 examples with TFCrop
   - Add to support set → model learns new pattern

3. **Leverage traditional flaggers:**
   - TFCrop/RFLAG have decades of domain knowledge
   - Use their outputs as "teachers" via support sets
   - Bridge synthetic → real gap without massive labeled datasets

4. **Instrument-specific artifacts:**
   - Each telescope has unique systematics
   - Support set captures these per-instrument
   - Single model handles multiple instruments

### When to Use SGP

**Use if:**
- Current model struggles on real data (synthetic → real gap exists)
- You have TFCrop/RFLAG examples from multiple sites
- Need to adapt to new RFI types quickly
- Want to leverage traditional flagger knowledge

**Skip if:**
- Synthetic baseline generalizes well to real data
- Don't have good real examples with traditional flags
- Single-site deployment (fine-tuning is simpler)

---

## 3. Combined Approach: LoRA + SGP

The ultimate system combines both techniques:

### Architecture

```
┌─── Synthetic Pre-training ───┐
│ 16k synthetic samples         │
│ Train decoder only (current)  │
└───────────────────────────────┘
         ↓
    Base SAM2 Model
         ↓
┌─── Add LoRA Adapters ─────────┐
│ Encoder: rank-16 adapters     │
│ Learn RFI-specific features   │
│ Trainable: 0.5M params        │
└───────────────────────────────┘
         ↓
    SAM2 + LoRA
         ↓
┌─── Add SGP Module ────────────┐
│ Support sets per site         │
│ TFCrop/RFLAG examples         │
│ Runtime domain adaptation     │
└───────────────────────────────┘
         ↓
    Final System
```

### Benefits of Combination

1. **LoRA:** Adapts encoder to spectrograms (better RFI-specific features)
2. **SGP:** Adapts to real data per-site (bridges synthetic → real gap)
3. **Together:** Best of both worlds
   - Learn RFI patterns efficiently (LoRA)
   - Adapt to messy real data (SGP)
   - Site flexibility (SGP support sets)
   - Efficient training (LoRA low-rank)

---

## Implementation Roadmap

### Phase 1: Validate Baseline ✓ (Current - In Progress)
```
Goal: Does decoder-only fine-tuning work?
Status: Running 20 epochs on 16k synthetic samples
Next: Evaluate on real observations
```

### Phase 2: Real Data Evaluation
```python
# Test synthetic-trained model on real observations
real_dataset = load_real_observations_with_tfcrop_flags()
metrics = evaluate(model, real_dataset)

# Questions to answer:
# 1. Does it detect real RFI?
# 2. How does it compare to TFCrop/RFLAG?
# 3. Where does it fail? (domain gap analysis)
```

### Phase 3a: Add LoRA (if encoder needs adaptation)
```python
# If Phase 2 shows encoder struggles with RFI patterns:
lora_config = LoraConfig(r=16, target_modules=[...])
model = get_peft_model(base_model, lora_config)

# Train on synthetic data with LoRA
trainer = SAM2Trainer(model, use_lora=True)
trainer.train(epochs=20)

# Compare: LoRA vs baseline on real data
```

### Phase 3b: Add SGP (if domain gap exists)
```python
# If Phase 2 shows synthetic → real gap:

# Collect support sets
support_sets = collect_tfcrop_examples_per_site()

# Implement SGP module
sgp = SGPModule(feature_dim=256)

# Train SGP to match TFCrop/RFLAG patterns
sgp.train(synthetic_model, support_sets)

# Evaluate on real data with SGP
metrics = evaluate_with_sgp(model, sgp, real_dataset)
```

### Phase 4: Combined System (if both help)
```python
# Best of both worlds
base_model = load_synthetic_trained()
lora_model = add_lora_adapters(base_model)
sgp_module = train_sgp(lora_model, support_sets)

# Deploy
def predict(observation, site):
    prototype = sgp_module.encode_support_set(support_sets[site])
    mask = lora_model(observation, sgp_guidance=prototype)
    return mask
```

---

## Decision Tree

```
Start: Evaluate baseline (decoder-only) on real data
         │
         ↓
    Does it work well?
    ├─ YES → Done! Ship it.
    └─ NO → Analyze failure mode
              │
              ↓
         What's the problem?
         ├─ Encoder features poor? → Try LoRA
         ├─ Synthetic ≠ Real? → Try SGP
         └─ Both? → Try LoRA + SGP
```

---

## Key Takeaways

1. **Current approach (decoder fine-tuning) is already lightweight** (18% of params)
   - Good baseline, simple, effective
   - Wait for results before adding complexity

2. **LoRA enables encoder adaptation** without full retraining
   - Use if encoder struggles with RFI-specific patterns
   - 0.2-1% trainable params, 3-5x faster
   - Can create modular RFI-type-specific adapters

3. **SGP leverages TFCrop/RFLAG knowledge** for domain adaptation
   - Bridges synthetic → real gap using few examples
   - Site-specific without retraining
   - Treats traditional flaggers as "teachers"

4. **Don't over-engineer prematurely**
   - Phase 1: Validate baseline on real data first
   - Phase 2: Add LoRA/SGP only if needed
   - Phase 3: Combine if both provide value

5. **SGP is particularly valuable for RFI** because:
   - TFCrop/RFLAG contain decades of domain expertise
   - Few real examples (5-10) can guide synthetic-trained model
   - Handles site/instrument diversity without retraining

---

## References

### LoRA
- **Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **HuggingFace PEFT:** https://github.com/huggingface/peft
- **Application:** Efficient fine-tuning for domain adaptation

### Support-Set Guided Prompting
- **Paper:** "SAM2-SGP: Enhancing SAM2 for Medical Image Segmentation via Support-Set Guided Prompting"
- **Concept:** Few-shot learning via prototype matching
- **RFI Application:** Use TFCrop/RFLAG examples as support sets for domain transfer

### Current SAM2-RFI Implementation
- **Baseline:** Decoder-only fine-tuning on synthetic data
- **Dataset:** 16k synthetic samples (exact ground truth)
- **Status:** In training (20 epochs, A100 GPU)
- **Next:** Evaluate on real observations with TFCrop/RFLAG comparisons

---

**Status:** Waiting for Phase 1 results before proceeding to advanced techniques.
