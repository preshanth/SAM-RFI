# SAM-RFI System Architecture and API Design

## Overview

SAM-RFI is a Radio Frequency Interference (RFI) detection system for radio astronomy data using Meta's Segment Anything Model (SAM). The system processes measurement sets from radio telescopes (VLA, ALMA) to identify and flag RFI contamination.

## System Architecture

### Core Components

```
SAM-RFI/
├── src/samrfi/                 # New modular architecture
│   ├── adapters/              # SAM version abstraction
│   ├── models/               # Training pipelines  
│   ├── datasets/             # Data handling
│   ├── core/                # MS loading/flagging
│   └── cli.py               # Command interface
├── samrfi/                   # Legacy implementation
├── training/                 # Training scripts
├── configs/                  # Hardware-specific configs
└── scripts/                  # Utility scripts
```

### Data Flow Architecture

**Production Pipeline (Real MS Processing)**:
```
MS File → MSLoader → TilingProcessor → SAMInferenceEngine → TilingProcessor → MSFlagger → Flagged MS + Stats
    ↓         ↓            ↓                 ↓                    ↓             ↓
Metadata  1024x1024   4×256x256      SAM2 Multi-Masks    1024x1024       Statistics
Extract   Tiles       Patches        + Union Compute     Union Mask      Tracking
```

**Training Pipeline (SAM2 Model Training)**:
```
SimulatedMS → RFI Injection → Ground Truth → SAM2 Training → Trained Model
     ↓            ↓              ↓              ↓
  CASA Data   Realistic RFI   Perfect Masks   Real Forward Pass
```

## API Design

### High-Level Usage

```python
# 1. Basic RFI Detection
from samrfi import RadioRFI, RFIModels

# Load measurement set
rfi_detector = RadioRFI("/path/to/observation.ms")
rfi_detector.load_ms()

# Run SAM2 inference  
model = RFIModels(model_type="sam2", variant="large")
model.load_model()
rfi_detector.run_rfi_detection(model)

# Apply flags
rfi_detector.apply_flags(mode="combine")  # or "replace"
```

### Production Flag Application Pipeline

```python
from samrfi.core import MSFlagApplicator

# Complete pipeline in a single class
applicator = MSFlagApplicator(
    ms_path="/path/to/observation.ms",
    sam_device="cuda",
    sam_variant="large", 
    sam_model_path="/path/to/trained_model.pt",  # or None for untrained
    field_id=0
)

# Process entire measurement set
results = applicator.process_all_baselines()

# Generate comprehensive report
report = applicator.generate_report()
print(report)

# Or process individual baselines
baselines = applicator.loader.get_baselines()
result = applicator.process_baseline(
    baselines[0][0], baselines[0][1], 
    spw_group_id=0, dry_run=False
)
```

### Advanced Component Usage

```python
from samrfi.core import MSLoader, TilingProcessor, SAMInferenceEngine, MSFlagger

# Manual pipeline control for advanced usage
loader = MSLoader("/path/to/observation.ms")
tiling = TilingProcessor(patch_size=256)
inference = SAMInferenceEngine(device="cuda", variant="large", 
                              local_model_path="/path/to/trained.pt")
flagger = MSFlagger("/path/to/observation.ms")

# Process single baseline with full control
baseline_data = loader.load_baseline_data(0, 1, spw_group_id=0)
data_1024 = loader.generate_1024x1024_tiles(baseline_data['data'][:,:,0])

# Create 3-channel input and tile
data_3ch = np.stack([data_1024, data_1024, data_1024], axis=0)  # [3,1024,1024]
tiling_result = tiling.create_patches(data_3ch)
patches = tiling_result['patches'].transpose(1, 0, 2, 3)  # [4,3,256,256]

# Run SAM2 inference
patch_masks = inference.predict_patches(patches)
union_mask = inference.compute_union_mask(patch_masks)

# Reconstruct and apply flags
mask_patches = np.stack(patch_masks, axis=0)[np.newaxis]  # [1,4,256,256]
full_mask = tiling.reconstruct_from_patches(mask_patches, tiling_result['metadata'])
flags_binary = full_mask.squeeze() > 0.5

flagger.write_baseline_flags(0, 1, flags_binary, baseline_data['metadata'])
```

### Training Pipeline

```python
# 1. Generate Synthetic Training Data
from samrfi.datasets import SimulatedMS, ObservationConfig, RFIConfig

obs_config = ObservationConfig(
    num_antennas=27,
    num_spw=2,
    channels_per_spw=512,
    start_frequency=1.4e9,
    total_duration=1024.0,
    thermal_noise_sigma=1e-3
)

rfi_config = RFIConfig(
    broadband_probability=0.25,
    narrowband_lines=15,
    transient_events=8,
    periodic_signals=3,
    satellite_passes=2
)

simulator = SimulatedMS(obs_config)
simulator.create_ms_with_rfi("training_data.ms", rfi_config)

# 2. Train SAM2 Model
python training/synthetic_training.py --config configs/training/v100_config.yaml
```

### Command Line Interface

```bash
# Basic RFI detection
samrfi process /path/to/observation.ms --antenna 1 --model sam2

# Get system information
samrfi info

# Training with specific hardware
samrfi train --config configs/training/v100_config.yaml --output results/
```

## Core Classes and Interfaces

### MSLoader (Memory-Efficient Data Loading)

```python
class MSLoader:
    def __init__(self, ms_path: str, field_id: int = 0)
    def get_field_ids() -> List[int]
    def get_baselines() -> List[Tuple[int, int]]
    def get_spw_groups() -> List[int]
    def load_baseline_data(ant1: int, ant2: int, spw_group_id: int) -> Dict
    def apply_existing_flags_as_zeros(data: np.ndarray, flags: np.ndarray) -> np.ndarray
    def generate_1024x1024_tiles(data: np.ndarray) -> np.ndarray

class MSMetadataExtractor:
    def __init__(self, ms_path: str)
    def get_antenna_info() -> Dict
    def get_field_info() -> Dict
    def get_spw_info() -> Dict
    def get_scan_spw_mapping() -> Dict
```

### MSFlagger (Flag Writing and Statistics)

```python
class MSFlagger:
    def __init__(self, ms_path: str)
    def write_baseline_flags(ant1: int, ant2: int, flags: np.ndarray,
                           metadata: Dict, mode: str = "combine")
    def get_flagging_statistics() -> Dict
    def generate_flagging_report() -> str

class StatisticsTracker:
    def add_baseline_stats(baseline_stats: Dict)
    def get_summary_statistics() -> Dict
    def get_per_baseline_stats() -> List[Dict]
```

### TilingProcessor (Generic Patch Operations)

```python
class TilingProcessor:
    def __init__(self, patch_size: int = 256)
    def create_patches(data: np.ndarray, overlap: int = 0) -> Dict[str, Any]
    def reconstruct_from_patches(patches: np.ndarray, metadata: Dict) -> np.ndarray
    def get_patch_coordinates(patch_idx: int, patches_per_side: int) -> Tuple[int, int]
    def get_pixel_coordinates(patch_idx: int, patches_per_side: int, 
                             local_y: int, local_x: int) -> Tuple[int, int]
```

### SAMInferenceEngine (Model Operations)

```python
class SAMInferenceEngine:
    def __init__(self, device: str = "cuda", variant: str = "large", 
                 local_model_path: Optional[str] = None)
    def predict_patches(patches: np.ndarray, generate_prompts: bool = True) -> List[np.ndarray]
    def compute_union_mask(mask_list: List[np.ndarray]) -> np.ndarray
    def predict_and_union(patches: np.ndarray) -> np.ndarray
    def get_model_info() -> Dict[str, Any]
    def generate_simple_prompts(image_shape: Tuple[int, int], 
                               n_points: int = 10) -> Dict[str, Any]
```

### MSFlagApplicator (Complete Pipeline)

```python
class MSFlagApplicator:
    def __init__(self, ms_path: str, sam_device: str = "cuda",
                 sam_variant: str = "large", sam_model_path: Optional[str] = None,
                 field_id: int = 0)
    def process_all_baselines(dry_run: bool = False) -> Dict[str, Any]
    def process_baseline(ant1: int, ant2: int, spw_group_id: int, 
                        dry_run: bool = False) -> Dict[str, Any]
    def get_ms_info() -> Dict[str, Any]
    def generate_report() -> str

### SAM2Adapter (Model Abstraction)

```python
class SAM2Adapter:
    def __init__(self, device: str = "cuda", variant: str = "large")
    def load_model(local_model_path: str = None)
    def predict_single(image: np.ndarray) -> np.ndarray
    def predict_batch(images: List[np.ndarray]) -> List[np.ndarray]
    def set_device(device: str)
```

### GPUOptimizedTrainer (Hardware-Aware Training)

```python
class GPUOptimizedTrainer:
    def __init__(self, config: Dict)
    def setup_model(sam_adapter: SAM2Adapter, dataset_size: int)
    def train_epoch(dataloader: DataLoader, epoch: int) -> Dict
    def validate(dataloader: DataLoader) -> Dict
    def save_checkpoint(path: str, epoch: int, metrics: Dict)
```

## Configuration System

### Hardware-Specific Configurations

```yaml
# configs/training/v100_config.yaml
model:
  version: "sam2"
  variant: "large"  # Options: tiny, small, base_plus, large
  local_model_path: ""  # Path to local model (empty = HuggingFace download)

training:
  batch_size: 2
  gradient_accumulation: 8
  mixed_precision: false
  max_epochs: 200
  learning_rate: 1e-4

dataset:
  loading_strategy: "eager"  # Options: eager, lazy, hybrid
  cache_observations: 2
  memory_budget_gb: 64

hardware:
  target_gpu: "V100"
  memory_limit: "32GB"
  time_limit: "flexible"
```

### Dataset Loading Strategies

1. **Eager Loading**: Load all training data into RAM at startup
   - Best performance (~2x speed improvement)
   - Requires sufficient RAM (auto-switches if dataset > memory_budget_gb)

2. **Lazy Loading**: Load samples on-demand during training
   - Minimal RAM usage
   - Uses memory mapping for efficiency

3. **Hybrid Loading**: Cache recently accessed observations
   - LRU cache with configurable size
   - Balances memory usage and performance

## Data Pipeline Architecture

### Measurement Set Processing

```
MS File → MSMetadataExtractor → Field/Baseline/SPW Discovery
    ↓
MSLoader → Scan-based SPW Grouping → Baseline Data Loading
    ↓
1024x1024 Tile Generation → Zero-pad flagged data → SAM Input
    ↓
SAM2Adapter → RFI Mask Prediction → Post-processing
    ↓
MSFlagger → Flag Combination → Statistics Tracking → Updated MS
```

### SAM2 Training Architecture

```
Training Batch → Prompt Generation → SAM2 Forward Pass → Loss Computation → Backpropagation
     ↓                  ↓                   ↓                   ↓
[3,1024,1024]     Point/Box Prompts    pred_masks        BCE + IoU Loss
RFI Images        from Ground Truth    iou_scores        Scalar Tensor
```

**Training Components**:
- **GPUOptimizedTrainer**: Hardware-aware training with memory management
- **SAM2Adapter**: Transformers integration with fallback to official SAM2
- **RFISyntheticDataset**: Converts MS data to SAM2-compatible format
- **Prompt Generation**: 128 random points + bounding boxes from ground truth

**Loss Computation**:
```python
def _compute_sam2_loss(images, masks):
    for sample in batch:
        # 1. Generate prompts from ground truth RFI regions
        points, boxes = generate_prompts_from_mask(gt_mask)
        
        # 2. Process through SAM2 
        inputs = processor(image=sample, input_points=points, input_boxes=boxes)
        outputs = sam2_model(**inputs)
        
        # 3. Extract best prediction
        pred_mask = outputs.pred_masks[0, best_idx]
        pred_score = outputs.iou_scores[0, best_idx]
        
        # 4. Compute losses
        seg_loss = BCE(pred_mask, gt_mask)
        score_loss = abs(pred_score - actual_iou)
        
    return total_loss.squeeze()  # Ensure scalar for backward()
```

### Synthetic Data Generation

```
CASA Simulator → Realistic Baselines → Thermal Noise Addition
    ↓
RFI Injection (5 types) → Ground Truth Masks → Training Dataset
    ↓
RFISyntheticDataset → Channel Processing → SAM2 Format
```

## Training System Architecture

### Loss Function Implementation

The training uses a two-component loss function based on the working SAM2 implementation:

```python
# Segmentation Loss: Binary Cross-Entropy
seg_loss = (-gt_mask * log(pred_mask + eps) - 
           (1 - gt_mask) * log(1 - pred_mask + eps)).mean()

# Score Loss: IoU Prediction
intersection = (gt_mask * (pred_mask > 0.5)).sum()
iou = intersection / (union + eps)
score_loss = abs(pred_scores - iou).mean()

# Combined Loss
total_loss = seg_loss + 0.05 * score_loss
```

### SAM2 Training Pipeline

1. **Image Processing**: Convert complex visibilities to RGB channels
2. **Prompt Generation**: Extract points and bounding boxes from ground truth
3. **SAM2 Forward Pass**: Image encoding → Prompt encoding → Mask decoding
4. **Loss Computation**: Segmentation + Score loss with real predictions
5. **Gradient Updates**: Hardware-optimized backpropagation

### Dataset Architecture

```python
# Training data flow
RFISyntheticDataset(loading_strategy="eager")
    ↓
Complex Visibilities [baselines, time, freq, pols]
    ↓
Channel Extraction: [real², imag², log_amp, phase]
    ↓
RGB Image Generation [3, 1024, 1024]
    ↓
PyTorch DataLoader → Batch Processing → SAM2 Training
```

## Model Download and Caching

### Local Model Management

```bash
# Download models to local directory
python scripts/download_models.py --variant large --update-config

# Available variants and sizes
- tiny: ~38MB
- small: ~184MB  
- base_plus: ~615MB
- large: ~2.4GB
```

### Model Loading Priority

1. **Local Path**: `local_model_path` in config (highest priority)
2. **HuggingFace**: Download from `facebook/sam2-hiera-{variant}`
3. **Official SAM2**: Fallback to official repository implementation

## Error Handling and Robustness

### Graceful Degradation

1. **Connection Issues**: Local model fallback when HuggingFace unavailable
2. **Memory Constraints**: Auto-switch from eager to lazy loading
3. **Model Loading**: Fallback chain from local → HuggingFace → official SAM2
4. **Training Errors**: Skip corrupted samples, continue training

### Validation and Testing

```python
# End-to-end validation pipeline
python validate_end_to_end.py --ms /path/to/test.ms --model sam2_large

# Unit tests
pytest tests/  # New architecture
pytest testing/  # Legacy compatibility
```

## Performance Optimization

### Hardware-Specific Optimizations

**V100 (32GB VRAM)**:
- SAM2-hiera-large model
- Batch size 2 with gradient accumulation (effective batch 16)
- Eager dataset loading (leverages 256GB system RAM)
- Gradient checkpointing enabled

**H200 (80GB VRAM)**:
- Larger batch sizes (4+)
- Mixed precision (FP16)
- Model compilation via torch.compile
- Time-optimized for 2-hour training windows

### Memory Management

1. **Scan-based SPW Grouping**: Handles mixed time dimensions intelligently
2. **Baseline-by-baseline Processing**: Constant ~1.2GB memory usage for large MS
3. **Immediate Flag Writing**: No accumulation of large flag arrays
4. **Memory Mapping**: Efficient access to large .npy training files

## Integration Points

### Radio Astronomy Software

- **CASA Integration**: Uses casatools for MS reading/writing
- **AOFlagger Compatibility**: Can be used alongside or instead of AOFlagger  
- **Pipeline Integration**: Designed for integration into observatory pipelines

### Machine Learning Frameworks

- **PyTorch**: Native training pipeline
- **HuggingFace Transformers**: Model loading and caching
- **MONAI**: Medical imaging loss functions (DiceCE)

## Testing and Validation

### ML Pipeline Testing

```python
# tests/test_ml_pipeline.py - Real ML component validation
pytest tests/test_ml_pipeline.py -v -s

# Step-by-step pipeline testing:
# 1. SimulatedMS generates realistic RFI baseline (30% contamination)
# 2. Extract complex visibility waterfall using CASA tools  
# 3. Apply R=Gradient, G=Amplitude, B=Phase channel mapping
# 4. Load SAM2-tiny and run actual inference
# 5. Compute union of SAM2 multi-mask outputs
# 6. Calculate training loss (BCE, IoU, accuracy) with ground truth
```

### Validation Test Suite

```bash
# Run pipeline validation tests
pytest tests/validation/ -v

# Available tests:
# - test_new_loader_flagger.py: MS loading and flagging infrastructure
# - test_flag_pipeline.py: Complete SAM2 flag application pipeline
# - test_simulated_ms.py: CASA-based synthetic data generation
# - test_single_ms.py: Single MS generation workflow
# - test_end_to_end_validation.py: End-to-end pipeline validation
```

### Performance Testing

```python
# Test with different model variants
pytest tests/test_ml_pipeline.py::TestMLPipeline::test_4_sam2_inference -v -s

# Expected baseline performance for untrained SAM2-tiny:
# - Segmentation loss: 0.5-0.7
# - IoU: 0.0-0.3  
# - Pixel accuracy: 40-70%
# - Processing time: ~1-2 seconds per 1024x1024 tile
```

## Usage Examples

### Simple RFI Detection

```python
from samrfi import RadioRFI, RFIModels

# Load and process measurement set
detector = RadioRFI("/path/to/observation.ms") 
detector.load_ms()

# Apply SAM2 RFI detection
model = RFIModels(model_type="sam2")
detector.run_rfi_detection(model)
detector.apply_flags()
```

### Custom Training

```python
from samrfi.datasets import SimulatedMS
from samrfi.models import GPUOptimizedTrainer
from samrfi.adapters import SAM2Adapter

# Generate training data
simulator = SimulatedMS(obs_config)
simulator.create_ms_with_rfi("training.ms", rfi_config)

# Train model
trainer = GPUOptimizedTrainer(config)
sam_adapter = SAM2Adapter(variant="large")
trainer.setup_model(sam_adapter, dataset_size)

for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.validate(val_loader)
```

### Production Pipeline

```python
from samrfi.core import MSLoader, MSFlagger
from samrfi.adapters import SAM2Adapter

# Production-scale processing
loader = MSLoader("/data/large_observation.ms")
flagger = MSFlagger("/data/large_observation.ms") 
sam = SAM2Adapter(variant="large", local_model_path="/models/sam2-large")

# Process systematically
total_baselines = len(loader.get_all_baselines())
for i, (ant1, ant2, spw_group, field_id) in enumerate(loader.get_all_baselines()):
    print(f"Processing baseline {i+1}/{total_baselines}")
    
    data = loader.load_baseline_data(ant1, ant2, spw_group, field_id)
    tiles = loader.generate_1024x1024_tiles(data['visibilities'])
    flags = sam.predict_batch(tiles)
    flagger.write_baseline_flags(ant1, ant2, flags, spw_group, field_id)

print("RFI detection complete")
stats = flagger.get_flagging_statistics()
report = flagger.generate_flagging_report()
```

This architecture provides a robust, scalable system for RFI detection in radio astronomy data with proper abstraction layers, hardware optimization, and production-ready interfaces.