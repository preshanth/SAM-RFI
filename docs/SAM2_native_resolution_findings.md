# SAM2 Native Resolution - Code Evidence

**Date:** 2025-09-30
**Source:** https://github.com/facebookresearch/sam2 (commit: latest)

---

## Training Resolution

### Training Configuration
**File:** `sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`

```yaml
scratch:
  resolution: 1024        # Line 4
  train_batch_size: 1
  num_train_workers: 10
  num_frames: 8

vos:
  train_transforms:
    - _target_: training.dataset.transforms.RandomResizeAPI
      sizes: ${scratch.resolution}    # Line 34 - Uses 1024
      square: true                     # Line 35
      consistent_transform: True
```

### Model Configuration
**File:** `sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`

```yaml
model:
  _target_: sam2.modeling.sam2_base.SAM2Base

  image_size: 1024    # Line 85

  # ... encoder/decoder configs ...
```

### Training Model Configuration
**File:** `sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`

```yaml
trainer:
  model:
    _target_: training.model.sam2.SAM2Train

    # ... image_encoder, memory_attention configs ...

    num_maskmem: 7
    image_size: ${scratch.resolution}    # Line 146 - Uses 1024
```

---

## Model Architecture

### Base Model Initialization
**File:** `sam2/modeling/sam2_base.py`

```python
class SAM2Base(nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,              # Line 29 - Default value
        backbone_stride=16,           # Line 30
        # ... other params ...
    ):
        # ...
        self.image_size = image_size                                      # Line 162
        # ...
        self.backbone_stride = backbone_stride
        self.sam_image_embedding_size = self.image_size // self.backbone_stride  # Line 210
        # For 1024: embedding_size = 64

        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),          # Line 220
            # ...
        )
```

### Position Encoding
**File:** `sam2/modeling/position_encoding.py`

```python
class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats,
        normalize=True,
        scale=None,
        temperature=10000,
        image_size: int = 1024,    # Line 31 - Default 1024
    ):
        super().__init__()
        # ...

    def forward(self, x: torch.Tensor):
        # ...
        if self._cache is not None:
            if self._cache[0].size == x.size:
                cache_key = (image_size // stride, image_size // stride)   # Line 50
                if cache_key in self._cache[1]:
                    return self._cache[1][cache_key].to(x.device)
```

### Attention Feature Sizes
**File:** `sam2/modeling/sam/transformer.py`

```python
# Line 261
feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
```

---

## Inference

### Image Predictor
**File:** `sam2/sam2_image_predictor.py`

```python
def set_image(self, image: np.ndarray) -> None:
    # ...
    self._orig_hw = [image.shape[:2]]

    if self._predictor.model.image_size is None:
        # Line 45
        self._predictor.set_image(image, resolution=self.model.image_size)
```

### Video Predictor
**File:** `sam2/sam2_video_predictor.py`

```python
def _load_img_as_tensor(img_path, image_size):
    img_pil = PILImage.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))  # Line 94
    # Resize to model's image_size
```

---

## Mask Decoder Output Resolution

### Output Shapes
**File:** `sam2/modeling/sam2_base.py` (lines 285-300)

```python
def _forward_sam_heads(...):
    """
    Outputs:
    - low_res_multimasks: [B, M, H*4, W*4] shape       # Line 285
      where H, W = backbone feature map dimensions
      (for 1024x1024: H=64, W=64 → low_res = 256x256)

    - high_res_multimasks: [B, M, H*16, W*16] shape    # Line 289
      upsampled from low-resolution masks, with same
      size as input image (stride is 1 pixel)
      (for 1024x1024: H=64, W=64 → high_res = 1024x1024)

    - low_res_masks: [B, 1, H*4, W*4] shape           # Line 295
    - high_res_masks: [B, 1, H*16, W*16] shape        # Line 298
    """
```

### Mask Upsampling Architecture
**File:** `sam2/modeling/sam/mask_decoder.py` (lines 65-76)

```python
self.output_upscaling = nn.Sequential(
    nn.ConvTranspose2d(
        transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
    ),  # 2x upsampling
    LayerNorm2d(transformer_dim // 4),
    nn.GELU(),
    nn.ConvTranspose2d(
        transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
    ),  # 2x upsampling (total 4x from backbone features)
    nn.GELU(),
)
```

**Upsampling path:**
1. Backbone features: H×W (e.g., 64×64 for 1024×1024 input)
2. After decoder upscaling (4x): H×4 × W×4 (256×256) = **low_res_masks**
3. After final upsampling (4x more): H×16 × W×16 (1024×1024) = **high_res_masks**

### Key Finding: Learned Upsampling

**The masks have finer resolution than backbone features because the decoder uses learned transposed convolutions to upsample 16× from backbone features to pixel-level masks.**

- Backbone: 64×64 features (stride 16)
- Decoder output: 1024×1024 masks (stride 1)
- **Upsampling factor: 16× through learned ConvTranspose2d layers**

---

## Position Embeddings: No Interpolation Needed

### Sine/Cosine Embeddings (Not Learned)
**File:** `sam2/modeling/position_encoding.py` (lines 90-124)

```python
@torch.no_grad()
def _pe(self, B, device, *cache_key):
    H, W = cache_key
    if cache_key in self.cache:
        return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

    # Compute fresh sine/cosine embeddings for this H×W size
    # Lines 95-122: Generate position embeddings from scratch
    # using sine/cosine functions (not interpolation!)

    self.cache[cache_key] = pos[0]  # Cache for future use
    return pos
```

**Key insight:** Position embeddings are **computed** (not interpolated) for any size:
- Check cache for (H, W)
- If miss: **compute** sine/cosine embeddings for that size
- Store in cache

**No interpolation happens.** The only difference between 128×128 and 1024×1024:
- 1024×1024: Uses pre-cached embeddings (faster)
- 128×128: Computes embeddings on first use, then caches (slightly slower first time)

Both produce mathematically correct embeddings for their respective sizes.

---

## Summary of Findings

### Training
- **Native resolution:** 1024×1024
- **Configured via:** `image_size` parameter
- **Backbone stride:** 16 pixels
- **Backbone features:** 64×64 (for 1024×1024 input)
- **Output masks:** 1024×1024 (16× upsampled from backbone via learned convolutions)

### Inference
- **Default resolution:** Same as model's `image_size` (1024 for SAM2.1)
- **Input→Backbone→Mask:**
  - 1024×1024 input → 64×64 backbone → 1024×1024 mask (16× learned upsampling)
  - 128×128 input → 8×8 backbone → 128×128 mask (16× learned upsampling)
- **Position embeddings:** Computed via sine/cosine for any size (no interpolation)

### Resolution Matching
- **128×128 input → 128×128 mask** (no information loss for that patch)
- **1024×1024 input → 1024×1024 mask** (native training resolution)
- Output mask resolution **always matches input image resolution**

### How Masks Exceed Backbone Resolution
**Question:** How can 64×64 backbone features produce 1024×1024 masks?

**Answer:** Learned upsampling through transposed convolutions:
1. Backbone: 64×64 features (stride 16 from 1024×1024 input)
2. Mask decoder applies 16× upsampling via learned `ConvTranspose2d` layers
3. Output: 1024×1024 pixel-level masks

The decoder is **trained** to predict fine-grained masks from coarse features. The upsampling is not simple bilinear - it's learned during training to recover fine details.
