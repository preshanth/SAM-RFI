# SAM-RFI Training Guide

Complete guide for training SAM2 models on synthetic RFI data using the unified pipeline.

## Overview

The training pipeline creates realistic measurement sets using CASA simulator, injects RFI with proper amplitudes, and trains SAM2 for RFI detection. The system is optimized for GTX 1080Ti hardware constraints but supports other GPUs via configuration files.

## Training Pipeline Architecture

### Data Flow
```python
# 1. Generate MS with realistic CASA data + RFI + flags
simulator.create_ms_with_rfi(ms_path, rfi_config, save_training_data=True)

# 2. Extract training arrays from MS automatically:
# - corrupted_visibilities.npy: [baselines, times, channels, pols]  
# - rfi_mask.npy: [baselines, times, channels, pols]

# 3. RFISyntheticDataset loads .npy files → converts to SAM2 format:
# - waterfall: amplitude [time, freq] → RGB image [3, H, W]
# - mask: RFI flags [time, freq] → ground truth [H, W]
```

### Dataset Configuration
- **Fixed Dataset**: 2 training MS (702 baselines) + 1 validation MS (351 baselines)
- **Natural Tiles**: 1024×1024 time×frequency (no artificial resizing)
- **Heavy RFI**: 25%+ contamination with 5 realistic RFI morphologies:
  - Broadband RFI with polynomial frequency variation
  - Narrowband persistent lines (15 lines)
  - Transient pulses and repeated bursts
  - Periodic signals with 10-30% duty cycles
  - Satellite passes with linear frequency drift
- **Training Samples**: 2,808 training + 1,404 validation (baselines × 4 polarizations)
- **Thermal Noise**: Configurable `thermal_noise_sigma=1e-3` in observation config

## Hardware Configurations

### GTX 1080Ti (11GB VRAM)
```yaml
# configs/training/gtx1080ti_config.yaml
training:
  batch_size: 1              # Small batch for 11GB VRAM
  gradient_accumulation: 8   # Accumulate to effective batch size of 8
  mixed_precision: true      # FP16 to save memory
  max_epochs: 10            # Configurable epochs
  learning_rate: 1e-4       # Conservative learning rate
  gradient_checkpointing: true   # Save memory at cost of speed
  patch_size: 512           # Smaller patches for memory
```

### V100 (16GB VRAM)
```yaml
# configs/training/v100_config.yaml - Memory-constrained, time-flexible
training:
  batch_size: 2
  gradient_accumulation: 4
  mixed_precision: true
  max_epochs: 15
```

### H200 (80GB VRAM)
```yaml
# configs/training/h200_config.yaml - Time-constrained, memory-rich
training:
  batch_size: 4
  gradient_accumulation: 2
  mixed_precision: true
  compile_model: true        # Use torch.compile for speed
  max_epochs: 20
```

## Launch Instructions

### Full Training Pipeline
```bash
cd /home/pjaganna/Software/SAM-RFI

# Complete pipeline: dataset generation → visualization → training
python training/synthetic_training.py

# With custom parameters:
python training/synthetic_training.py \
  --output-dir ./my_training_run \
  --config configs/training/gtx1080ti_config.yaml
```

### Skip Dataset Generation
```bash
# If dataset already exists, skip to training
python training/synthetic_training.py --skip-dataset --skip-viz

# Dataset generation only (for debugging)
python training/synthetic_training.py --skip-training
```

### Command Line Arguments
```bash
--output-dir DIR          # Output directory (default: ./sam_rfi_synthetic_training)
--config CONFIG.yaml      # Training configuration (default: gtx1080ti_config.yaml)
--skip-dataset            # Skip dataset generation (use existing)
--skip-viz               # Skip PNG visualization generation
--skip-training          # Skip training (dataset generation only)
--viz-samples N          # Number of visualization samples (default: 10)
```

## Training Process

### Phase 1: Dataset Generation (~15 minutes)
1. **MS Creation**: Creates 3 measurement sets using CASA simulator
2. **RFI Injection**: Adds realistic RFI with proper thermal noise
3. **Data Extraction**: Saves .npy files in training format
4. **Visualization**: Generates PNG plots for first 5 baselines per MS

### Phase 2: Model Loading
1. **SAM2 Download**: Auto-downloads `facebook/sam2-hiera-base-plus` (~1.6GB)
2. **Model Initialization**: Sets up SAM2 with GTX 1080Ti optimizations
3. **Dataset Loading**: Creates PyTorch datasets from .npy files

### Phase 3: Training Loop (~2-3 hours for 10 epochs)
1. **Training**: 2,808 samples with gradient accumulation
2. **Validation**: 1,404 samples every epoch
3. **Checkpointing**: Model saved every 2 epochs
4. **Monitoring**: Progress logged every 10 steps

## Expected Outputs

### Generated Files
```
sam_rfi_synthetic_training/
├── synthetic_ms/
│   ├── measurement_sets/
│   │   ├── sam_rfi_train_obs_000.ms
│   │   ├── sam_rfi_train_obs_001.ms
│   │   └── sam_rfi_val_obs_000.ms
│   └── ground_truth/
│       ├── sam_rfi_train_obs_000/
│       │   ├── corrupted_visibilities.npy
│       │   └── rfi_mask.npy
│       └── ...
├── visualizations/
│   ├── train_sam_rfi_train_obs_000_viz.png
│   └── ...
├── dataset_summary.json
├── model_checkpoints/
├── training_logs/
└── synthetic_training.log
```

### Visualization Outputs
- **4-panel PNG plots**: Clean/Corrupted/Mask/RFI-only for visual inspection
- **SAM tiles**: 1024×1024 waterfall plots during MS generation
- **Training metrics**: Loss curves, validation accuracy, memory usage

## Performance Expectations

### GTX 1080Ti Timings
- **Dataset Generation**: ~15 minutes (3 MS files)
- **Model Download**: ~5 minutes (first run only)
- **Training**: ~2-3 hours for 10 epochs
- **Memory Usage**: ~9GB VRAM peak with gradient checkpointing

### Training Metrics
- **Effective Batch Size**: 8 (1 × 8 gradient accumulation)
- **Steps per Epoch**: ~351 (2,808 samples / 8 effective batch)
- **Total Steps**: 3,510 (10 epochs × 351 steps)
- **Validation Steps**: ~176 (1,404 samples / 8 effective batch)

## Monitoring and Debugging

### Log Files
- **`synthetic_training.log`**: Complete training log with timestamps
- **Console output**: Real-time progress and memory usage
- **TensorBoard logs**: Training curves and metrics (if enabled)

### Common Issues
1. **CUDA OOM**: Reduce `batch_size` or enable `gradient_checkpointing`
2. **Dataset not found**: Run without `--skip-dataset` first
3. **SAM2 download fails**: Check internet connection, model cached in `~/.cache/huggingface/`
4. **Training hangs**: Check GPU memory usage with `nvidia-smi`

### Success Indicators
- Dataset generation completes without errors
- PNG visualizations show realistic RFI patterns
- Training loss decreases over epochs
- Validation metrics improve
- Model checkpoints saved successfully

## Customization

### Adjusting Training Parameters
Edit `configs/training/gtx1080ti_config.yaml`:
```yaml
training:
  max_epochs: 20           # Increase for longer training
  learning_rate: 5e-5      # Lower for stability
  batch_size: 2            # If you have more VRAM
  patch_size: 1024         # Full resolution (if memory allows)
```

### RFI Configuration
Modify RFI scenarios in `training/synthetic_training.py`:
```python
# Heavy RFI scenario (~25% total)
RFIConfig(
    broadband_probability=0.15,  # Increase broadband
    narrowband_lines=20,         # More persistent lines
    transient_events=12,         # More transient events
    periodic_signals=5,          # More periodic patterns
    satellite_passes=3           # More satellite passes
)
```

### Observation Parameters
Adjust data characteristics in `ObservationConfig`:
```python
ObservationConfig(
    num_antennas=27,             # VLA configuration
    thermal_noise_sigma=2e-3,    # Higher noise level
    total_duration=2048.0,       # Longer observation
    start_frequency=3.0e9,       # S-band instead of L-band
)
```

## Next Steps After Training

1. **Evaluate Model**: Run inference on test data
2. **Compare Performance**: Benchmark against AOFlagger
3. **Real Data Testing**: Apply to actual VLA observations
4. **Model Export**: Convert to ONNX for deployment
5. **Integration**: Incorporate into SAM-RFI inference pipeline

## Restarting Training

If you already have generated datasets and want to restart training:

### Skip Dataset Generation
```bash
# Skip dataset, go straight to training (most common restart)
python training/synthetic_training.py \
  --skip-dataset \
  --skip-viz \
  --output-dir my_training_run \
  --config configs/training/gtx1080ti_config.yaml
```

### Skip Only Visualization Issues
```bash
# If dataset generation worked but visualization failed
python training/synthetic_training.py \
  --skip-dataset \
  --skip-viz \
  --output-dir my_training_run
```

### Resume from Checkpoint
```bash
# TODO: Add checkpoint resumption once implemented
# python training/synthetic_training.py --resume-from-checkpoint path/to/checkpoint
```

## Troubleshooting

### Common Issues and Solutions

**Issue**: `AttributeError: 'GPUOptimizedTrainer' object has no attribute 'use_mixed_precision'`
**Solution**: Fixed in latest code - GTX1080Ti optimization settings now properly configured

**Issue**: `FileNotFoundError: clean_visibilities.npy`  
**Solution**: Use `--skip-viz` to bypass broken visualization step

**Issue**: CUDA Out of Memory
**Solutions**:
- Reduce `batch_size` to 1 in config
- Enable `gradient_checkpointing: true`
- Reduce `patch_size` to 256 or 512

**Issue**: Training hangs or very slow
**Solutions**:
- Check `nvidia-smi` for GPU utilization
- Reduce `dataloader_num_workers` to 1
- Set `pin_memory: false`

### General Debugging Steps
1. Check `synthetic_training.log` for detailed error messages
2. Verify GPU memory with `nvidia-smi`
3. Test dataset generation alone with `--skip-training`
4. Try smaller configuration (reduce batch_size, patch_size)
5. Ensure CASA tools are properly installed and accessible

### Expected Training Behavior
- **Loss should decrease** over epochs (SAM probability loss)
- **GPU memory usage**: ~9GB peak for GTX 1080Ti
- **Training time**: ~2-3 hours for 10 epochs
- **Log frequency**: Progress every 10 steps, validation every epoch

For issues with dataset generation, refer to the `SimulatedMS` class in `src/samrfi/datasets/simulated_ms.py` which handles CASA integration and RFI injection.