# SAM3 RFI Prompting Strategy

## Overview

SAM3 introduces **dual-prompting capabilities** that SAM2 lacked:
1. **Visual Prompts** (bounding boxes, points) - inherited from SAM2
2. **Text Prompts** (natural language) - **NEW in SAM3**

This document outlines strategies for leveraging both capabilities for Radio Frequency Interference (RFI) detection in radio astronomy data.

---

## 1. Current Approach: Visual Prompting (SAM2 Legacy)

### How It Works

The existing SAM-RFI pipeline uses **bounding box prompts** extracted from ground truth RFI masks during training:

```python
# From src/samrfi/data/sam_dataset.py
def _get_bounding_box(self, mask):
    """Extract bounding box from mask with random perturbation."""
    y_indices, x_indices = torch.where(mask > 0)
    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    y_min, y_max = y_indices.min().item(), y_indices.max().item()

    # Add random perturbation (20 pixels by default)
    # This makes the model robust to imperfect prompts
    return [x_min, y_min, x_max, y_max]
```

**Training Flow:**
1. Ground truth RFI mask → Extract bounding box
2. Feed image + bounding box → SAM model
3. Model predicts refined mask
4. Compare with ground truth → Compute loss
5. Only train mask decoder (freeze vision + prompt encoders)

**Advantages:**
- ✓ Proven to work (validated on SAM2)
- ✓ Precise spatial guidance
- ✓ No language understanding required
- ✓ Direct migration path from SAM2 → SAM3

**Limitations:**
- ✗ Requires pre-computed masks for training
- ✗ Cannot leverage semantic understanding of "RFI"
- ✗ Bounding boxes don't capture complex RFI morphology well

---

## 2. NEW Capability: Text Prompting

### Potential Text Prompts for RFI

SAM3 can accept natural language descriptions as prompts. Candidate prompts:

#### Generic RFI Prompts
```python
"radio frequency interference"
"RFI"
"interference pattern"
"corrupted signal region"
"anomalous radio emissions"
```

#### Specific RFI Types
```python
# Broadband RFI (affects wide frequency range)
"broadband interference across all frequencies"
"wideband RFI contamination"

# Narrowband RFI (single frequency)
"narrowband interference spike"
"single frequency emission line"

# Intermittent RFI (time-varying)
"intermittent interference bursts"
"transient RFI events"

# Structured RFI (regular patterns)
"periodic interference pattern"
"regular RFI stripes"
```

### Text Prompt Training Strategy

**Option A: Pure Text Prompts**
```python
# Modified SAMDataset to use text instead of boxes
def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    mask = item["label"]

    # Instead of bounding box:
    text_prompt = "radio frequency interference"

    inputs = processor(
        images=image,
        text_prompts=[text_prompt],
        return_tensors="pt"
    )

    return {
        "pixel_values": inputs.pixel_values,
        "input_text": inputs.input_text,
        "ground_truth_mask": mask
    }
```

**Option B: Hybrid Prompts** (Text + Bounding Box)
```python
# Combine text description with spatial guidance
inputs = processor(
    images=image,
    text_prompts=["radio frequency interference"],
    input_boxes=[bbox],  # Coarse spatial hint
    return_tensors="pt"
)
```

**Advantages:**
- ✓ Semantic understanding of RFI concept
- ✓ No need to compute bounding boxes
- ✓ Can generalize across RFI types
- ✓ More flexible prompting at inference

**Limitations:**
- ✗ Untested - SAM3 text prompts may not work well for non-natural images
- ✗ Radio astronomy data looks very different from natural images
- ✗ Requires experimentation to find effective prompts

---

## 3. Hybrid Prompting Strategy

### Rationale

RFI detection is a **spatial + semantic task**:
- **Spatial**: RFI occurs in specific (channel, time) regions
- **Semantic**: RFI has characteristic patterns (broadband, narrowband, transient)

Hybrid prompting leverages both:

```python
# Example: Narrowband RFI detection
text = "narrowband interference spike"
bbox = [0, freq_channel_120, width, freq_channel_130]  # Coarse frequency range

# SAM3 uses text for concept, bbox for spatial refinement
inputs = processor(
    images=waterfall_plot,
    text_prompts=[text],
    input_boxes=[bbox],
    return_tensors="pt"
)
```

### When to Use Each Approach

| RFI Type | Recommended Prompt | Rationale |
|----------|-------------------|-----------|
| **Broadband** | Text only: "broadband RFI" | Affects entire frequency axis, spatial hint not useful |
| **Narrowband** | Hybrid: text + frequency range bbox | Text defines concept, bbox constrains frequency |
| **Intermittent** | Hybrid: text + time range bbox | Text defines transient nature, bbox constrains time |
| **Unknown** | Text only: "radio frequency interference" | Let model learn general RFI concept |

---

## 4. Test Suite Design

### Phase 1: Visual Prompting Validation (Baseline)

**Goal:** Verify SAM3 achieves SAM2-level performance with visual prompts.

**Test Cases:**
1. **Synthetic RFI Patterns**
   - Generate waterfall plots with known RFI
   - Test broadband, narrowband, transient patterns
   - Metrics: IoU, Precision, Recall, F1

2. **Real Measurement Sets**
   - Use existing SAM-RFI test data
   - Compare SAM2 vs SAM3 predictions
   - Expected: SAM3 ≥ SAM2 performance

**Script:**
```bash
# Train SAM3 with visual prompts on synthetic data
python scripts/train_sam3.py --config configs/sam3_visual_baseline.yaml

# Evaluate on test set
python scripts/evaluate_sam3.py --model output/sam3_visual/model_best.pth \
                                --test-data data/synthetic_rfi_test.npz
```

### Phase 2: Text Prompting Exploration

**Goal:** Discover which text prompts work for RFI detection.

**Experiment Design:**
```python
# Test different text prompts on same data
prompts_to_test = [
    "radio frequency interference",
    "RFI",
    "interference pattern",
    "corrupted signal",
    "broadband interference",
    "narrowband spike",
    "anomalous emissions"
]

# For each prompt:
# 1. Fine-tune SAM3 with that prompt
# 2. Evaluate on held-out test set
# 3. Record metrics
# 4. Compare performance across prompts
```

**Expected Outcomes:**
- Some prompts may work better than others
- Domain-specific terminology ("RFI") vs generic ("interference")
- May need multiple prompts for different RFI types

### Phase 3: Hybrid Prompting Optimization

**Goal:** Test if text + bbox outperforms text-only or bbox-only.

**Test Matrix:**

| Prompt Type | Broadband RFI | Narrowband RFI | Transient RFI |
|-------------|---------------|----------------|---------------|
| BBox only   | F1 = ? | F1 = ? | F1 = ? |
| Text only   | F1 = ? | F1 = ? | F1 = ? |
| Text + BBox | F1 = ? | F1 = ? | F1 = ? |

**Hypothesis:**
- Broadband: Text-only performs best (no useful spatial structure)
- Narrowband: Hybrid performs best (frequency range + semantic concept)
- Transient: Hybrid performs best (time range + semantic concept)

### Phase 4: Zero-Shot Generalization Test

**Goal:** Test if SAM3 can detect RFI types it wasn't trained on.

**Approach:**
1. Train on synthetic broadband RFI only
2. Test on real narrowband RFI (never seen during training)
3. Use text prompts: "narrowband interference spike"
4. Measure: Does text prompt enable zero-shot detection?

**This tests SAM3's key advantage:** Semantic understanding via text.

---

## 5. Implementation Roadmap

### Milestone 1: Visual Baseline (1-2 days)
- [x] Migrate SAM2 → SAM3 code
- [ ] Train SAM3 with visual prompts on synthetic data
- [ ] Validate performance matches/exceeds SAM2
- [ ] Document baseline metrics

### Milestone 2: Text Prompt Integration (2-3 days)
- [ ] Modify `sam_dataset.py` to support text prompts
- [ ] Update `sam3_trainer.py` for text-based training
- [ ] Run prompt exploration experiments
- [ ] Identify best-performing prompts

### Milestone 3: Hybrid Prompting (2-3 days)
- [ ] Implement text + bbox combined prompting
- [ ] Run comparison experiments (bbox vs text vs hybrid)
- [ ] Analyze which approach works best per RFI type
- [ ] Document optimal prompting strategy

### Milestone 4: Real Data Validation (3-5 days)
- [ ] Apply best prompting strategy to real measurement sets
- [ ] Compare against traditional RFI flaggers (AOFlagger)
- [ ] Measure deployment performance
- [ ] Write up results

---

## 6. Key Questions to Resolve

### Q1: Do text prompts work on radio astronomy data?
**Why it matters:** SAM3 was trained on natural images (ImageNet, etc.). Radio waterfall plots are fundamentally different.

**Test:** Train SAM3 with text prompt "radio frequency interference" on synthetic data, evaluate on held-out test set.

**Success criteria:** IoU > 0.7 on test set (comparable to visual prompting)

### Q2: What text prompts are most effective?
**Why it matters:** Prompt engineering is crucial for text-based models.

**Test:** Grid search over prompt variations, measure F1 score.

**Success criteria:** Identify 2-3 prompts with F1 > 0.8

### Q3: Does hybrid prompting improve performance?
**Why it matters:** Combining text + spatial may give best of both worlds.

**Test:** A/B test text-only vs text+bbox on same data.

**Success criteria:** Hybrid improves F1 by ≥5% over text-only

### Q4: Can SAM3 generalize to unseen RFI types?
**Why it matters:** Zero-shot detection would be a major advantage over SAM2.

**Test:** Train on Type A RFI, test on Type B with text prompt.

**Success criteria:** F1 > 0.5 on unseen RFI type (vs F1 ≈ 0 for SAM2)

---

## 7. Recommended Starting Point

**Start with Visual Prompting Validation** (Milestone 1):

1. Use existing SAM-RFI pipeline (bounding boxes)
2. Train SAM3 on synthetic RFI data
3. Verify SAM3 matches SAM2 performance
4. This establishes a baseline for comparison

**Why this first?**
- Lowest risk (known to work with SAM2)
- Fastest validation of SAM3 migration
- Establishes performance floor

**Then explore text prompting** (Milestone 2):
- Only after visual baseline is proven
- Can compare text vs visual on same data
- Lower risk of wasting compute

---

## 8. Code Changes Required

### For Text Prompting

**File: `src/samrfi/data/sam_dataset.py`**
```python
class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor=None, prompt_type="visual", text_prompt=None):
        """
        Args:
            prompt_type: "visual", "text", or "hybrid"
            text_prompt: Text prompt if using text/hybrid mode
        """
        self.prompt_type = prompt_type
        self.text_prompt = text_prompt or "radio frequency interference"

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        mask = item["label"]

        if self.prompt_type == "visual":
            # Current approach
            bbox = self._get_bounding_box(mask)
            return {
                "pixel_values": image.permute(2, 0, 1),
                "input_boxes": torch.tensor([bbox], dtype=torch.float32),
                "ground_truth_mask": mask
            }

        elif self.prompt_type == "text":
            # NEW: Text prompting
            return {
                "pixel_values": image.permute(2, 0, 1),
                "input_text": self.text_prompt,
                "ground_truth_mask": mask
            }

        elif self.prompt_type == "hybrid":
            # NEW: Combined prompting
            bbox = self._get_bounding_box(mask)
            return {
                "pixel_values": image.permute(2, 0, 1),
                "input_boxes": torch.tensor([bbox], dtype=torch.float32),
                "input_text": self.text_prompt,
                "ground_truth_mask": mask
            }
```

**File: `src/samrfi/training/sam3_trainer.py`**
```python
# Update forward pass to handle text prompts
outputs = self.model(
    pixel_values=batch["pixel_values"].to(self.device),
    input_boxes=batch.get("input_boxes"),  # Optional
    input_text=batch.get("input_text"),    # NEW: Optional
    multimask_output=False,
)
```

---

## 9. Success Metrics

### Quantitative Metrics
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Critical for astronomy (don't flag real sky signal as RFI!)

### Qualitative Metrics
- Visual inspection of predictions on real data
- Comparison with expert-flagged data
- Robustness to different observatories/instruments

### Performance Targets
- **Visual prompting (baseline)**: F1 > 0.85 (match SAM2)
- **Text prompting**: F1 > 0.80 (within 5% of visual)
- **Hybrid prompting**: F1 > 0.88 (best overall)
- **Zero-shot**: F1 > 0.50 on unseen RFI types

---

## 10. Next Steps

**Immediate Actions:**
1. ✅ SAM3 branch created and migrated
2. ⏳ Push branch to remote (retry after 403 error)
3. ⏳ Create experiment config for visual baseline training
4. ⏳ Run baseline training on synthetic data
5. ⏳ Validate SAM3 performance matches SAM2

**Discussion Points:**
- Which test approach should we prioritize? (Visual baseline vs text exploration)
- What real measurement sets do you have for validation?
- Do you have ground truth RFI masks for real data?
- What compute budget do we have for experiments? (H100 time)

---

## References

- **SAM3 Model Card**: https://huggingface.co/facebook/sam3
- **SAM3 Training Docs**: https://github.com/facebookresearch/sam3
- **SAM-RFI Repository**: Current SAM2 implementation
- **Validation Results**: `validate_sam3.py` confirmed visual prompts work
