# SAM-RFI End-to-End Validation

This document describes the comprehensive end-to-end validation pipeline that tests the complete SAM-RFI workflow from synthetic data generation through model training to inference and evaluation.

## Overview

The validation pipeline tests:
- Synthetic measurement set generation with realistic RFI patterns
- Memory-efficient data loading and processing
- SAM2 model training (simulated for validation)
- Inference and performance evaluation  
- Memory management for GTX 1080Ti constraints
- Visualization and result analysis

## Quick Start

### Run Complete Validation
```bash
# Full validation pipeline
python validate_end_to_end.py

# Quick test (minimal data, faster)
python validate_end_to_end.py --quick-test

# Custom output directory
python validate_end_to_end.py --output-dir my_validation_results
```

### Run with Pytest
```bash
# Quick pytest validation
pytest tests/test_end_to_end_validation.py -v -k quick

# Full pytest suite (slower)
pytest tests/test_end_to_end_validation.py -v

# With output visible
pytest tests/test_end_to_end_validation.py -v -s

# GPU tests (requires CUDA)
pytest tests/test_end_to_end_validation.py -v -m gpu
```

## Validation Steps

### Step 1: Synthetic Data Generation
- Creates realistic measurement set with controllable RFI
- Generates clean, corrupted, and ground truth versions
- Configurable observation parameters (antennas, frequencies, duration)
- Multiple RFI pattern types: broadband, narrowband, transient, periodic, satellite

### Step 2: Memory-Efficient Data Loading  
- Tests chunked loading with adaptive batch sizing
- Validates memory monitoring and safety checks
- Extracts training patches from waterfall data
- Ensures compatibility with GTX 1080Ti memory constraints

### Step 3: Model Training Simulation
- Simulates SAM2 training pipeline
- Tests GPU memory management
- Tracks training metrics (loss, accuracy, memory usage)
- Uses GTX 1080Ti-optimized configuration

### Step 4: Inference and Evaluation
- Runs inference on validation patches
- Calculates performance metrics (accuracy, precision, recall, F1)
- Tests prediction quality against ground truth
- Validates inference memory usage

### Step 5: Visualization
- Creates comprehensive validation plots
- Shows training curves and performance metrics
- Visualizes sample predictions vs ground truth
- Saves results for analysis

## Configuration

### GTX 1080Ti Optimization
The validation uses `configs/training/gtx1080ti_config.yaml`:

```yaml
training:
  batch_size: 1                  # Small batch for 11GB VRAM
  gradient_accumulation: 8       # Effective batch size of 8
  mixed_precision: true          # FP16 memory saving
  patch_size: 512               # Smaller patches for memory
  max_patches_per_epoch: 500    # Limit patches per epoch
  
hardware:
  target_gpu: "GTX1080Ti"
  memory_limit: "11GB"
```

### Quick Test Mode
For faster validation during development:
- Reduced antenna count (4 instead of 6)
- Smaller observation duration (10 vs 30 minutes)
- Fewer training patches (20 vs 500)
- Reduced epochs (2 vs 10)

## Expected Results

### Performance Targets
- **Memory Usage**: < 10GB peak on GTX 1080Ti
- **Training Time**: < 10 minutes for quick test, < 1 hour for full
- **Accuracy**: > 70% on synthetic data (baseline threshold)
- **Memory Efficiency**: Adaptive batching prevents OOM errors

### Output Files
```
validation_results/
├── synthetic_data/           # Generated measurement sets
│   ├── measurement_sets/     # CASA-compatible .ms files
│   └── ground_truth/         # NumPy arrays with data and masks
├── models/                   # Trained model checkpoints
├── plots/                    # Validation visualizations
├── logs/                     # Training and validation logs
└── validation_summary.json   # Complete results summary
```

## Interpreting Results

### Training Plots
- **Loss Curve**: Should show decreasing trend
- **Accuracy Curve**: Should show increasing trend  
- **Memory Usage**: Should stay within GPU limits

### Performance Metrics
- **Accuracy**: Overall correctness of RFI detection
- **Precision**: Fraction of detected RFI that is actually RFI
- **Recall**: Fraction of actual RFI that was detected
- **F1-Score**: Harmonic mean of precision and recall

### Sample Predictions
Visual comparison of:
- **Red Channel**: Original waterfall data intensity
- **Green Overlay**: Model predictions (detected RFI)
- **Blue Overlay**: Ground truth RFI locations

## Troubleshooting

### Memory Issues
```bash
# If OOM errors occur, try quick test mode
python validate_end_to_end.py --quick-test

# Or check memory usage
nvidia-smi
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run CPU-only validation
CUDA_VISIBLE_DEVICES="" python validate_end_to_end.py --quick-test
```

### Dependencies
```bash
# Install all required packages
pip install -e .[dev,training,notebooks]

# Check for missing dependencies
python -c "import matplotlib, torch, numpy, casacore"
```

## Integration with CI/CD

### Automated Testing
```bash
# In CI pipeline, run quick validation
pytest tests/test_end_to_end_validation.py -v -k "quick and not gpu"

# For GPU CI runners
pytest tests/test_end_to_end_validation.py -v -m gpu
```

### Performance Regression Detection
The validation can detect:
- Memory usage regressions
- Training convergence issues  
- Inference accuracy degradation
- Loading performance problems

## Extending the Validation

### Adding New RFI Patterns
Modify `RFIConfig` in the validation script:
```python
rfi_config = RFIConfig(
    broadband_probability=0.03,
    narrowband_lines=5,
    transient_events=8,
    # Add new pattern parameters
    custom_pattern_strength=0.1
)
```

### Custom Hardware Configurations
Create new config files in `configs/training/` and specify:
```bash
python validate_end_to_end.py --config configs/training/my_gpu_config.yaml
```

### Additional Metrics
Extend the `_calculate_metrics` method to include:
- IoU (Intersection over Union)
- Matthews Correlation Coefficient
- ROC AUC scores
- Per-RFI-type performance

## Benchmarking Against Other Flaggers

The validation creates measurement sets compatible with:
```bash
# AOFlagger
aoflagger validation_results/synthetic_data/measurement_sets/validation_obs_corrupted.ms

# CASA tfcrop
casa -c "flagdata(vis='validation_obs_corrupted.ms', mode='tfcrop')"

# Compare results against ground truth
python compare_flaggers.py validation_results/
```

This allows direct performance comparison between SAM-RFI and traditional flagging algorithms on identical synthetic data with known ground truth.