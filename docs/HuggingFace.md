# HuggingFace Hub Integration Guide

Complete guide for publishing and downloading SAM-RFI models and datasets using HuggingFace Hub.

## Table of Contents

- [Quick Start](#quick-start)
  - [Publishing a Trained Model](#publishing-a-trained-model)
  - [Downloading Models for Inference](#downloading-models-for-inference)
  - [Publishing Datasets](#publishing-datasets)
- [For Users](#for-users)
  - [Using Published Models](#using-published-models)
  - [Model Versioning](#model-versioning)
  - [Private Repositories](#private-repositories)
- [For Contributors](#for-contributors)
  - [Model Publishing Workflow](#model-publishing-workflow)
  - [Authentication Setup](#authentication-setup)
  - [Troubleshooting](#troubleshooting)

## Quick Start

### Publishing a Trained Model

After training a SAM-RFI model, publish it to HuggingFace Hub for sharing:

```bash
# Upload model with auto-detection of model size
samrfi publish --type model \
  --input ./samrfi_data/sam2_rfi_best.pth \
  --repo-id polarimetic/sam-rfi

# Upload with explicit model size (if auto-detection fails)
samrfi publish --type model \
  --input ./samrfi_data/sam2_rfi_best.pth \
  --repo-id polarimetic/sam-rfi \
  --model-size large

# Upload to private repository
export HF_TOKEN=hf_xxxxx  # Your HuggingFace token
samrfi publish --type model \
  --input ./samrfi_data/sam2_rfi_best.pth \
  --repo-id polarimetic/sam-rfi-private \
  --private
```

The model will be uploaded to:
```
https://huggingface.co/polarimetic/sam-rfi
└── large/model.pth
```

### Downloading Models for Inference

SAM-RFI automatically downloads models from HuggingFace Hub when needed.

#### Command Line Interface

```bash
# Single-pass prediction with HuggingFace model
samrfi predict \
  --model polarimetic/sam-rfi/large \
  --input observation.ms

# Iterative prediction (3 passes)
samrfi predict \
  --model polarimetic/sam-rfi/large \
  --input observation.ms \
  --iterations 3

# Alternative: specify repo and size separately
samrfi predict \
  --model polarimetic/sam-rfi \
  --checkpoint large \
  --input observation.ms
```

#### Python API

```python
from samrfi.inference import RFIPredictor

# Initialize predictor with HuggingFace model
predictor = RFIPredictor(
    model_path="polarimetic/sam-rfi/large",
    device="cuda"
)

# Single-pass prediction
flags = predictor.predict_ms("observation.ms")

# Iterative prediction
flags = predictor.predict_iterative("observation.ms", num_iterations=3)
```

On first use, the model is downloaded to your HuggingFace cache (`~/.cache/huggingface/hub/`). Subsequent runs use the cached version.

### Publishing Datasets

Publish training datasets to HuggingFace Hub for sharing:

```bash
# Publish dataset
samrfi publish --type dataset \
  --input ./datasets/train_4k/exact_masks \
  --repo-id polarimetic/sam-rfi-dataset

# Publish private dataset
export HF_TOKEN=hf_xxxxx
samrfi publish --type dataset \
  --input ./datasets/train_4k/exact_masks \
  --repo-id polarimetic/sam-rfi-dataset-private \
  --private
```

## For Users

### Using Published Models

SAM-RFI supports two model path formats:

1. **Local File Path**
   ```bash
   samrfi predict --model ./models/sam2_rfi_best.pth --input observation.ms
   ```

2. **HuggingFace Repo ID**
   ```bash
   samrfi predict --model polarimetic/sam-rfi/large --input observation.ms
   ```

The system auto-detects which format you're using based on:
- Contains `/` → Potential HuggingFace repo ID
- File exists locally → Local file (takes precedence)
- File doesn't exist + contains `/` → Download from HuggingFace

### Model Versioning

SAM-RFI uses a "latest" versioning approach for simplicity:
- Each model size (tiny, small, base_plus, large) has one current version
- Uploading a new model overwrites the previous version in that size category
- No git-style tags or semantic versioning

**Repository Structure:**
```
polarimetic/sam-rfi/
├── README.md           # Auto-generated model card
├── tiny/model.pth      # Latest tiny model
├── small/model.pth     # Latest small model
├── base_plus/model.pth # Latest base_plus model
└── large/model.pth     # Latest large model
```

### Private Repositories

For private models, set the `HF_TOKEN` environment variable:

**Upload:**
```bash
export HF_TOKEN=hf_xxxxx
samrfi publish --type model \
  --input model.pth \
  --repo-id user/private-models \
  --private
```

**Download/Use:**
```bash
export HF_TOKEN=hf_xxxxx
samrfi predict --model user/private-models/large --input observation.ms
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

## For Contributors

### Model Publishing Workflow

**1. Train Model**
```bash
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --validation-dataset ./datasets/val_1k/exact_masks
```

**2. Verify Checkpoint**
```bash
python -c "import torch; print(torch.load('./samrfi_data/sam2_rfi_best.pth', map_location='cpu').keys())"
```

Expected keys:
- `model_state_dict`
- `config` (includes `sam_checkpoint` for size detection)
- `preprocessing`
- `training_losses`, `validation_losses`

**3. Publish to HuggingFace**
```bash
samrfi publish --type model \
  --input ./samrfi_data/sam2_rfi_best.pth \
  --repo-id polarimetic/sam-rfi
```

**4. Verify Upload**
- Visit: `https://huggingface.co/polarimetic/sam-rfi`
- Check model card (README.md) is generated
- Check model file exists at `{size}/model.pth`

**5. Test Download**
```bash
# Clear cache to force re-download
rm -rf ~/.cache/huggingface/hub/models--preshanth--sam-rfi-models

# Test prediction
samrfi predict \
  --model polarimetic/sam-rfi/large \
  --input test.ms \
  --no-save
```

### Authentication Setup

**HuggingFace Token Management:**

1. **Create Token**
   - Visit: https://huggingface.co/settings/tokens
   - Click "New token"
   - Set permissions: `write` (for publishing)
   - Copy token (starts with `hf_`)

2. **Set Environment Variable**
   ```bash
   # Temporary (current session)
   export HF_TOKEN=hf_xxxxx

   # Permanent (add to ~/.bashrc or ~/.zshrc)
   echo 'export HF_TOKEN=hf_xxxxx' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify Authentication**
   ```bash
   python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
   ```

### Troubleshooting

#### Upload Issues

**Problem:** `Failed to create repository`
- **Cause:** Token doesn't have write permissions or repo name conflicts
- **Solution:**
  ```bash
  # Check token permissions at https://huggingface.co/settings/tokens
  # Try with explicit token:
  samrfi publish --type model \
    --input model.pth \
    --repo-id user/repo \
    --token hf_xxxxx
  ```

**Problem:** `Could not detect model size from checkpoint`
- **Cause:** Checkpoint missing `config.sam_checkpoint` field
- **Solution:** Use `--model-size` flag:
  ```bash
  samrfi publish --type model \
    --input model.pth \
    --repo-id user/repo \
    --model-size large
  ```

**Problem:** `Failed to upload model`
- **Cause:** Network issues or large file size
- **Solution:**
  - Check internet connection
  - Retry upload (HuggingFace supports resume)
  - For very large models, consider `huggingface-cli upload` directly

#### Download Issues

**Problem:** `Failed to download model from {repo_id}`
- **Cause:** Repo doesn't exist, network issues, or private repo without token
- **Solutions:**
  1. Verify repo exists: `https://huggingface.co/{repo_id}`
  2. Check internet connection
  3. For private repos: `export HF_TOKEN=hf_xxxxx`

**Problem:** Path ambiguity (e.g., `./user/repo.pth` treated as HuggingFace ID)
- **Cause:** Local path contains `/` but doesn't exist yet
- **Solution:**
  - Use absolute paths: `/full/path/to/model.pth`
  - Or create the file first before referencing

**Problem:** Model downloaded but prediction fails
- **Cause:** Mismatch between model architecture and CLI `--checkpoint` arg
- **Solution:**
  ```bash
  # Ensure --checkpoint matches model size
  samrfi predict \
    --model user/repo/large \
    --checkpoint large \
    --input obs.ms
  ```

#### Cache Management

**View Cache Location:**
```bash
echo $HF_HOME  # Custom cache location
# Default: ~/.cache/huggingface/hub/
```

**Clear Model Cache:**
```bash
# Clear specific model
rm -rf ~/.cache/huggingface/hub/models--{org}--{repo}

# Clear all HuggingFace cache
rm -rf ~/.cache/huggingface/hub/
```

**Set Custom Cache:**
```bash
export HF_HOME=/path/to/custom/cache
samrfi predict --model user/repo/large --input obs.ms
```

## Advanced Usage

### Batch Publishing Multiple Models

Publish multiple trained models at once:

```bash
#!/bin/bash
# publish_models.sh

REPO_ID="polarimetic/sam-rfi"
MODELS_DIR="./trained_models"

for model_file in $MODELS_DIR/*.pth; do
    echo "Publishing $model_file..."
    samrfi publish --type model \
      --input "$model_file" \
      --repo-id "$REPO_ID"
done
```

### Programmatic Model Upload

Use Python API for custom workflows:

```python
from huggingface_hub import HfApi, create_repo
import torch

# Load checkpoint
checkpoint = torch.load("model.pth", map_location="cpu")
model_size = checkpoint["config"]["sam_checkpoint"]

# Create repo and upload
api = HfApi(token="hf_xxxxx")
create_repo("user/sam-rfi-models", repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj="model.pth",
    path_in_repo=f"{model_size}/model.pth",
    repo_id="user/sam-rfi-models",
    repo_type="model"
)
```

### Dataset Download and Use

Download datasets from HuggingFace (for training):

```python
from datasets import load_dataset

# Download dataset
hf_dataset = load_dataset("polarimetic/sam-rfi-dataset", split="train")

# Convert to SAM-RFI format (if needed)
# Note: Current training uses BatchedDataset format, not HF datasets
```

## Best Practices

1. **Model Naming:** Use organization/project format: `org/sam-rfi-models`
2. **Private Models:** Enable for proprietary training data or work-in-progress
3. **Model Cards:** Auto-generated cards include training metrics - keep checkpoints complete
4. **Cache Management:** Monitor `~/.cache/huggingface/` size, clean periodically
5. **Version Control:** For critical deployments, pin specific git revisions (not yet supported, use local files)
6. **Testing:** Always test downloaded models before production use

## Links

- **HuggingFace Hub:** https://huggingface.co
- **SAM-RFI GitHub:** https://github.com/preshanth/SAM-RFI
- **HuggingFace Hub Documentation:** https://huggingface.co/docs/hub
- **HuggingFace Hub Python Client:** https://huggingface.co/docs/huggingface_hub
