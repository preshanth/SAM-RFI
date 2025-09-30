# SAM-RFI Implementation Plan

## Current Status: Phase 6 Complete ✅ - Production Pipeline and Testing Infrastructure

### Production Pipeline Infrastructure - OPERATIONAL ✅
**Complete Flag Application System** (`src/samrfi/core/`):
- **MSFlagApplicator**: End-to-end pipeline orchestration for measurement set processing
- **TilingProcessor**: Generic 1024×1024 → 4×256×256 patch operations 
- **SAMInferenceEngine**: SAM2 model loading, inference, and union computation
- **Core Infrastructure**: Cleaned MSLoader/MSFlagger (removed _v2 suffix, updated imports)
- **Status**: Production-ready pipeline for applying trained models to real measurement sets

### SAM2 Training Pipeline - REQUIRES DEBUGGING ⚠️
**Training Status** (`training/synthetic_training.py`, `src/samrfi/models/training.py`):
- **Issue**: Training loss plateaus at 0.52-0.54, not converging despite optimizations
- **Union-based Loss**: Fixed conflicting gradients by computing union of SAM2 masks for training
- **New Channel Mapping**: R=Gradient, G=Amplitude, B=Phase (improved RFI edge detection)
- **Loss Normalization**: Fixed gaussianity loss explosion with bounded statistical functions
- **Next Steps**: Further debugging required to achieve training convergence

### Training Data Generation - OPERATIONAL ✅  
**SimulatedMS Integration** (`src/samrfi/datasets/simulated_ms.py`):
- **CASA Integration**: Realistic baseline visibilities with proper thermal noise
- **Heavy RFI Contamination**: 25%+ occupancy with 5 morphologically diverse RFI types
- **Fixed Dataset**: 2,808 training + 1,404 validation samples from 3 MS files
- **Ground Truth Quality**: Pixel-perfect RFI masks for supervised learning
- **Natural Dimensions**: 1024×1024 tiles matching SAM2 input requirements

### Real MS Processing Pipeline - OPERATIONAL ✅
**MSLoader/MSFlagger** (`src/samrfi/core/loader_v2.py`, `flagger_v2.py`):
- **Production Ready**: Memory-efficient MS loading and flagging system
- **SPW-Aware Processing**: Scan-based grouping with temporal structure awareness
- **Statistics Tracking**: Comprehensive metrics (TP/TN/FP/FN, precision, recall, F1)
- **Status**: Ready for real MS processing with trained SAM2 models

## Phase 6 Achievements

### Production Pipeline Implementation - COMPLETE ✅
**Core Infrastructure Cleanup**:
- **Removed _v2 suffix**: Renamed `loader_v2.py`/`flagger_v2.py` to `loader.py`/`flagger.py`
- **Updated imports**: All components now use clean `from samrfi.core import MSLoader, MSFlagger`
- **Deleted legacy code**: Removed old implementations that weren't using current architecture
- **Clean exports**: Updated `__init__.py` to export production-ready classes

**New Pipeline Components**:
- **TilingProcessor**: Generic tiling operations (any size → 256×256 patches + reconstruction)
- **SAMInferenceEngine**: SAM2 model loading, batch inference, union computation
- **MSFlagApplicator**: Complete pipeline orchestrator (MS → tiling → inference → flagging)

**Pipeline Architecture**:
```python
# Production flag application:
applicator = MSFlagApplicator(ms_path, sam_variant="large", sam_model_path="/path/to/trained.pt")
results = applicator.process_all_baselines()  # Process entire MS
report = applicator.generate_report()         # Generate statistics
```

### Validation and Testing Infrastructure - COMPLETE ✅
**Real ML Pipeline Testing** (`tests/test_ml_pipeline.py`):
- **Step 1**: SimulatedMS generates realistic baseline with heavy RFI (30% contamination)
- **Step 2**: Extract complex visibility waterfall using CASA tools
- **Step 3**: Apply new R=Gradient, G=Amplitude, B=Phase channel mapping
- **Step 4**: Load SAM2-tiny model and run actual inference
- **Step 5**: Compute union of SAM2 multi-mask outputs
- **Step 6**: Calculate training loss (BCE, IoU, accuracy) with ground truth

**Test Organization** (`tests/validation/`):
- **Moved pipeline tests**: Only tests using current architecture components
- **Removed useless tests**: Deleted tests that don't exercise real ML functionality
- **Documentation**: `VALIDATION_TESTS.md` explains what each test validates and how to run

## Phase 4 Achievements

### SimulatedMS Pipeline - OPERATIONAL ✅
- **CASA Integration**: Uses `casatools.simulator` for realistic baseline visibilities
- **Proper Thermal Noise**: `sigma_simple=1e-3` for realistic noise levels  
- **Heavy RFI Contamination**: 25%+ occupancy with 5 realistic RFI types
- **Memory Efficient**: Baseline-by-baseline processing (~1.2GB constant memory)
- **SAM-Ready Visualization**: 1024×1024 tile plots with ground truth masks

### Realistic RFI Types Implemented ✅
1. **Broadband RFI**: Polynomial frequency variation (parabolic/cubic envelopes)
2. **Narrowband Lines**: 15+ persistent frequency lines across all time
3. **Transient Pulses**: Short bursts and repeated burst patterns  
4. **Periodic Signals**: Time-varying pulse trains with 10-30% duty cycles
5. **Satellite Passes**: Linear frequency drift over time

### Working Implementation
```python
# Memory-efficient processing: 351 baselines at constant ~1.2GB memory
simulator = SimulatedMS(obs_config)
simulator.create_ms_with_rfi("output.ms", rfi_config, include_rfi_flags=True)

# Generates:
# - CASA MS with realistic noise + 25%+ RFI contamination
# - SAM 1024×1024 tile visualizations showing clean/corrupted/mask/RFI-only
# - Perfect ground truth masks for supervised learning
```

## Phase 7: Training Pipeline Debugging (Current Focus)

### Training Issues to Resolve
1. **Loss Plateau Investigation**:
   - Training loss stuck at 0.52-0.54 despite union-based approach and channel mapping improvements
   - Need to investigate learning rate, weight decay, and SAM2 parameter freezing
   - Consider different prompt generation strategies or model initialization

2. **Training Pipeline Validation**:
   - Use `tests/test_ml_pipeline.py` to baseline untrained SAM2 performance
   - Compare training loss computation with test loss computation for consistency
   - Validate new channel mapping produces meaningful gradients for learning

3. **Alternative Training Approaches**:
   - Test different SAM2 variants (tiny/small vs large) for training convergence
   - Investigate fine-tuning vs training from scratch approaches
   - Consider reducing prompt complexity from 128 to 16 points for faster iteration

### Code Changes Required

**Update `training/synthetic_training.py`**:
```python
# Replace current approach:
# generator = SyntheticDatasetGenerator(...)

# With new SimulatedMS approach:
from samrfi.datasets import SimulatedMS, ObservationConfig, RFIConfig

obs_config = ObservationConfig(
    num_antennas=27,
    num_spw=2, 
    channels_per_spw=512,  # 1024 total channels
    start_frequency=1.4e9,
    total_duration=1024.0,  # 1024 time steps
    integration_time=1.0
)

rfi_config = RFIConfig(
    broadband_probability=0.25,  # 25% occupancy
    narrowband_lines=15,
    transient_events=8,
    periodic_signals=3,
    satellite_passes=2
)

# Generate training MS files
for i in range(num_training_ms):
    simulator = SimulatedMS(obs_config)
    ms_path = f"training_ms_{i:03d}.ms"
    simulator.create_ms_with_rfi(ms_path, rfi_config, include_rfi_flags=True)
```

**Training Loss Evaluation**:
```python
# Residual evaluation should target noise floor of ~1e-3
target_noise_rms = 1e-3
training_loss = mse_loss(sam_output_residuals, target_noise_rms)

# Success metric: SAM residuals approach thermal noise levels
success_threshold = 2.0 * target_noise_rms  # 2× noise floor
```

### Expected Training Improvements
- **Better Convergence**: Realistic noise structure provides proper loss targets
- **Robust Performance**: 25%+ RFI loading trains SAM for challenging scenarios  
- **Real-World Applicable**: CASA-generated data matches actual observation characteristics
- **Scalable**: Memory-efficient processing allows larger training datasets

## Phase 6: Advanced Features (Future)

### Integration with Real MS Data
- **Real MS Validation**: Test SimulatedMS patterns against actual VLA/ALMA data
- **Cross-Validation**: Compare synthetic RFI patterns with real RFI observations
- **Adaptive RFI Generation**: Tune RFI parameters based on real data statistics

### Performance Optimizations  
- **GPU Acceleration**: Move RFI generation to GPU for faster training data creation
- **Parallel Processing**: Generate multiple MS files simultaneously 
- **Caching**: Cache generated MS files for repeated training runs

### Advanced RFI Models
- **Site-Specific RFI**: Model RFI patterns specific to VLA, ALMA, etc.
- **Temporal Correlations**: RFI patterns that evolve over long time scales
- **Realistic Satellite Catalogs**: Use actual satellite orbital data for realistic passes

## File Status

### Current Working Files ✅
- `src/samrfi/datasets/simulated_ms.py` - Complete CASA-based RFI simulation
- `src/samrfi/datasets/__init__.py` - Updated exports  
- `test_single_ms.py` - Working test script
- `CLAUDE.md` - Updated documentation

### Files Needing Updates
- `training/synthetic_training.py` - Replace with SimulatedMS approach
- Training configs in `configs/training/` - Update for new pipeline
- Test scripts - Update to use new SimulatedMS patterns

## Success Criteria for Phase 5

- [ ] **Training Script Updated**: `training/synthetic_training.py` uses SimulatedMS
- [ ] **Memory Efficient**: Training dataset generation stays within memory limits
- [ ] **Quality Metrics**: SAM training residuals approach 1e-3 noise floor
- [ ] **Scalable**: Can generate 100+ training MS files efficiently  
- [ ] **Validation**: Generated training data improves SAM performance on real data

## Context for Next Session

**Where We Are**: Realistic RFI simulation is complete and working. CASA generates proper baseline visibilities with 1e-3 noise, and we inject 25%+ heavy RFI contamination with 5 realistic signal types.

**What's Next**: Update the training pipeline to use this new SimulatedMS approach instead of the old synthetic visibility generation. This should provide much better training data quality.

**Key Implementation**: The training script needs to create multiple MS files using `SimulatedMS`, then extract 1024×1024 tiles for SAM training, using the perfect ground truth masks we generate.

**Expected Outcome**: SAM training should converge faster and perform better on real data, since the training data now matches realistic observation characteristics.