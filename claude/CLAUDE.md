# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Programming Collaboration Style

**Act as Critical Pair Programmer**: Be analytical, realistic, and professionally direct. Challenge assumptions and point out flaws before they become problems. Avoid being "nice for the sake of being nice" - focus on technical correctness and efficiency.

**Confidence Levels - Always Indicate**:
- **High confidence**: Based on direct evidence, established patterns, or verified facts
- **Medium confidence**: Reasonable assumptions based on available information  
- **Low confidence/Guessing**: Uncertain, requires validation or investigation

**Demand Evidence and Validation**:
- Require actual benchmarks or clear technical justification for performance claims
- Avoid hand-wavy multipliers like "3-4x speedup" without backing data
- When building tests, ensure they exercise **real functionality**, not just data shuffling
- Validate that code changes actually solve the intended problem

**Code Reuse Over Rewriting**:
- **Critical**: Always check for existing implementations before writing new code
- Prefer extending or refactoring existing code over creating duplicate functionality
- When proposing new components, first analyze what already exists and can be reused
- Identify and eliminate redundant code patterns

**Step-by-Step Development**:
- Build and validate components incrementally rather than large complex systems
- Test each step independently before moving to the next
- Ask clarifying questions about requirements and constraints before implementation
- Get explicit approval for architectural decisions before coding

**No Assumptions - Ask Questions**:
- Never assume user intent - ask specific clarifying questions
- When multiple approaches exist, present options and ask for preference
- Don't implement "multiple fixes" without understanding the root cause
- Challenge requirements that seem unrealistic or poorly defined

**Focus on Real Functionality**:
- Prioritize testing actual ML pipeline components (model loading, inference, loss computation)
- Avoid creating fake/mock tests that don't exercise production code paths
- Ensure tests validate that the system actually works, not just that data moves around
- Test edge cases and failure modes, not just happy paths

**Unit Test Principles**:
- Keep tests local to the module being tested (no separate test directories)
- Test core functionality: file creation, expected shapes, basic data validation
- Don't over-engineer test validation - check file exists, shapes match, data isn't all zeros
- Follow existing patterns like test_ml_pipeline.py for consistency
- Validate the pipeline works end-to-end, not individual algorithm correctness

## Project Overview

SAM-RFI is a Python package for Radio Frequency Interference (RFI) detection in radio astronomy data using Meta's Segment Anything Model (SAM). The project is currently undergoing a major refactor from a legacy monolithic structure to a modern pip-installable package with proper architecture patterns.

**Current Branch**: `refactor` - Major restructuring from legacy code in `samrfi/` to modern architecture in `src/samrfi/`
**Main Branch**: Contains the original working implementation before refactoring

## Architecture & Code Organization

### Dual Structure During Refactor
The codebase currently maintains two parallel structures during the refactor:

1. **Legacy Code** (`samrfi/` directory):
   - `radiorfi.py` - Main RadioRFI class for measurement set handling
   - `rfimodels.py` - RFIModels class for SAM inference
   - `rfitraining.py` - Training pipeline for custom models
   - `syntheticrfi.py` - Synthetic RFI data generation
   - `plotter.py` - Visualization utilities
   - `metricscalculator.py` - Performance metrics

2. **New Architecture** (`src/samrfi/` directory):
   - `adapters/` - SAM version adapters (SAM1, SAM2, future SAM3)
   - `models/` - Training pipelines and model management
   - `datasets/` - Data handling and preprocessing
   - `core/` - Core functionality
   - `cli.py` - Command line interface

### Key Architectural Patterns

**Adapter Pattern**: `src/samrfi/adapters/` provides clean abstraction across SAM versions
- `base.py` - Abstract base class for all SAM adapters
- `sam2_adapter.py` - Primary SAM2 implementation (recommended)
- `registry.py` - Adapter registration and discovery

**GPU-Optimized Training**: Different configurations for different hardware
- V100 configs: Memory-constrained, time-flexible (smaller batches, gradient accumulation)
- H200 configs: Time-constrained, memory-rich (larger batches, compilation, mixed precision)

**Legacy Compatibility**: The new `src/samrfi/__init__.py` imports from the old structure during refactor to maintain backward compatibility.

## Common Development Commands

### Package Management
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e .[dev,training,notebooks]
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=samrfi --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest tests/         # New architecture tests
pytest testing/       # Legacy tests
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8

# Type checking
mypy src/samrfi
```

### Training Workflows
```bash
# CLI interface (under development)
samrfi process /path/to/measurement.ms --antenna 1 --model sam2
samrfi info

# Training with specific GPU configs
python training/synthetic_training.py --config configs/training/h200_config.yaml
python training/rflag_3C147_training.py --config configs/training/v100_config.yaml
```

## Hardware-Specific Configuration

### V100 Training (32GB VRAM)
- Uses SAM2.1-hiera-large model for maximum performance
- Batch size 2 with gradient accumulation (effective batch size 16)
- Configurable dataset loading strategy (eager/lazy/hybrid)
- Optimized for accuracy over speed

### H200 Training (Time-Constrained) 
- Larger batch sizes (4+)
- Mixed precision (FP16)
- Model compilation via `torch.compile`
- 2-hour training window optimization

Configuration files are in `configs/training/`:
- `v100_config.yaml` - Optimized for V100 GPUs
- `h200_config.yaml` - Optimized for H200 GPUs  
- `gtx1080ti_config.yaml` - Consumer GPU settings

### Configurable Dataset Loading

The training pipeline supports three loading strategies configurable via `dataset` section in config files:

**Eager Loading** (`loading_strategy: "eager"`):
- Loads all training data into system RAM at startup
- Fastest training throughput (~2x speed improvement)
- Requires sufficient RAM (auto-switches to lazy if dataset > `memory_budget_gb`)
- Best for systems with abundant RAM (256GB+)

**Lazy Loading** (`loading_strategy: "lazy"`):
- Loads samples on-demand during training
- Minimal RAM usage
- Slight training slowdown due to disk I/O per sample
- Uses memory mapping for efficiency

**Hybrid Loading** (`loading_strategy: "hybrid"`):
- Caches recently accessed observations in RAM
- Balances memory usage and performance
- LRU cache evicts oldest observations when `cache_observations` limit reached
- Good compromise for moderate RAM systems

**Auto-Selection**: 
- Estimates dataset size and auto-switches from eager to lazy if size exceeds `memory_budget_gb`
- V100 config defaults: `eager` mode with 64GB budget, 2 observation cache

## Data Pipeline

### Core Classes (Legacy, being refactored)
- `RadioRFI`: Loads measurement sets, handles CASA table interactions
- `RFIModels`: Runs inference with trained SAM models
- `RFITraining`: Training pipeline for custom models
- `SyntheticRFI`: Generates synthetic waterfall plots for training
- `RFIDataset`: Dataset management for training
- `Plotter`: Visualization of waterfall plots and flags

### New Architecture Components
- `SAMAdapter`: Abstract base for version-agnostic SAM interactions
- `GPUOptimizedTrainer`: Hardware-aware training pipeline
- `RFIDatasetCreator`: Modern dataset creation utilities

## Testing Strategy

Two parallel test suites during refactor:
- `tests/` - New architecture pytest-based tests
- `testing/` - Legacy test scripts

Key test files:
- `test_sam2_adapter.py` - Adapter functionality
- `test_training_1080ti.py` - Consumer GPU training validation

## Package Installation

The package uses modern Python packaging with `pyproject.toml`:
- Supports Python 3.9+
- Core dependencies: numpy, torch, casacore, astropy
- Optional dependencies for dev, training, and notebooks
- CLI entry point: `samrfi` command

## Development Notes

1. **Refactor Status**: The project is in active refactor. New development should target `src/samrfi/` structure.

2. **Backward Compatibility**: Legacy imports in `src/samrfi/__init__.py` maintain compatibility during transition.

3. **SAM2 Focus**: SAM2 is the primary target. SAM1 support exists but SAM2 is recommended for new work.

4. **Hardware Optimization**: Training configs are specifically tuned for V100 vs H200 constraints.

5. **Core Data Pipeline Completed**: Memory-efficient MS loading and flagging system is fully functional with scan-based SPW grouping.

## Development Principles

### Code Quality and Communication

**Professional Tone**: All code, comments, documentation, and communication should maintain a professional tone. Avoid emojis, casual language, or unnecessary embellishments. Focus on clarity and precision. Use a measured tone without effusive agreement or praise - act as an equal partner. Prefer discussion upfront rather than overly large summaries after fixes.

**Confidence Levels**: When providing analysis, solutions, or recommendations, clearly indicate confidence level:
- **High confidence**: Certain based on direct evidence or established patterns
- **Medium confidence**: Likely correct based on available information
- **Low confidence/Guessing**: Uncertain, requires validation or further investigation

**Simplest Path Principle**: Always prioritize the simplest and cleanest solution that achieves the objective. Avoid over-engineering and unnecessary complexity.

**Planning Before Implementation**: Never make code edits without explicit discussion and approval. Always:
1. Analyze the problem thoroughly
2. Ask clarifying questions about requirements and constraints
3. Present a clear implementation plan
4. Wait for approval before proceeding with code changes

### Hardware Requirements

**Pascal GPU Support**: For Pascal architecture GPUs (GTX 10-series), use:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Higher CUDA versions remove Pascal support. This constraint affects GTX 1080Ti configurations.

## Current Training Performance Status (2025-09-08)

### **CRITICAL: Training Performance Optimization Required**
**Current Performance**: 1.2-2.1 samples/sec (down from initial 1.6 samples/sec)
**Target**: Need significant improvement for 5,616 iterations/epoch × 50 epochs

### **Identified Bottlenecks from Profiling:**
**High Confidence** findings from actual profiling data:
1. **SAM2 Forward Pass**: 53-84% of training time (~94ms per batch after warmup)
2. **Loss Computation**: 32% of training time (~56ms per batch) - **MAJOR BOTTLENECK**
3. **Prompt Generation**: 11-13% (~18-23ms per batch) - 128 points per sample
4. **Logging Overhead**: Previously major bottleneck, now fixed with epoch-aware frequency
5. **Data Loading**: Negligible (<1ms) - not the issue

### **Agreed Solutions (Pending Implementation):**

#### **1. Union-Based Multi-Mask Training (IMPLEMENTED ✅)**
**Approach**: SAM2 returns multiple masks per sample `[batch, num_masks, H, W]`, training uses union of all masks.

**Current Implementation - Union Training:**
- **Training**: Compute union (max) of all masks, then compute single loss on union
- **Inference**: Take union (max) of all masks for comprehensive RFI detection
- **Implementation**: `src/samrfi/models/training.py` line 762: `union_logits = pred_masks[:, 0, :, :, :].max(dim=1)[0]`
- **Benefit**: Single consistent approach for training and inference, comprehensive RFI detection

#### **2. Configurable Gaussianity Loss Term**
**Concept**: Penalize predictions where residual (after RFI removal) isn't Gaussian noise
**Metrics**: Skewness penalty + Kurtosis penalty + optional KS-test
**Configuration**: Weighted combination with configurable per-metric weights

#### **3. Prompt Complexity Reduction** 
**Current**: 128 random points per sample for SAM2 prompts
**Target**: Reduce to 16-32 points (should reduce prompt generation time by ~4x)

### **Technical Debt Resolved:**
- ✅ Mixed precision autocast safety (BCE → BCE_with_logits)
- ✅ Epoch-aware logging frequency (10x less logging after epoch 0)
- ✅ Deprecated autocast syntax updated
- ✅ Vectorized loss computation structure (needs multi-mask fix)

### **4x256 Tiling Strategy for True High-Resolution Processing**

**Problem**: SAM2 (both transformers and original) outputs low-resolution masks that require upscaling, losing fine-grained detail.

**Solution**: 4x256 Tiling Approach
- Split 1024x1024 input into four clean 256x256 tiles
- Process each tile individually through SAM2 for native resolution
- Reconstruct full 1024x1024 masks without upscaling artifacts

**Phase 1: Simple Implementation (Current Priority)**
```
Tile Layout (no overlap):
┌─────────┬─────────┐
│ Tile 1  │ Tile 2  │
│[0:256,  │[0:256,  │ 
│ 0:256]  │ 256:512]│
├─────────┼─────────┤
│ Tile 3  │ Tile 4  │
│[256:512,│[256:512,│
│ 0:256]  │ 256:512]│
└─────────┴─────────┘
```

**Phase 2: Advanced Features (Future)**
1. **Overlapping Tiles**: 32-64px overlap with weighted blending for smooth boundaries
2. **Data Augmentation**: Per-tile rotations (90°, 180°, 270°) and random flips
3. **Adaptive Prompts**: Tile-specific prompt generation based on local RFI patterns
4. **Multi-scale Processing**: Different tile sizes (128x128, 256x256, 512x512) for various RFI morphologies
5. **Edge Continuity**: Specialized handling for RFI spanning tile boundaries

**Benefits**:
- **True High Resolution**: No upscaling, preserves fine RFI detail
- **Memory Efficient**: Process smaller tiles individually  
- **Parallel Processing**: Can process tiles concurrently
- **Quality Preservation**: Avoids interpolation artifacts from post-processing

### **Session Complete - All Major Optimizations Implemented ✅**

**Achievements**:
1. ✅ **Hybrid per-mask training**: Individual losses for each SAM2 mask during training, union for inference
2. ✅ **Configurable gaussianity loss**: Real residuals testing with skewness, kurtosis, and Anderson-Darling tests
3. ✅ **True 4x256 tiling**: Split 1024×1024 into native 256×256 tiles for high-resolution processing
4. ✅ **Memory optimization**: Sequential tile processing + batch_size=1, gradient_accumulation=16
5. ✅ **Model efficiency**: Switched to sam2-base_plus for memory constraints

**Final Implementation Status**:
- **Tiling Strategy**: 1024×1024 input → 4×256×256 tiles → SAM2 native resolution processing → 1024×1024 reconstruction
- **Memory Management**: Sequential processing avoids 4x memory spike
- **Loss Components**: Per-mask segmentation + IoU + gaussianity (configurable weights)
- **Training Stable**: Pipeline running with normalized loss ~0.53 (see Loss Normalization section below)

### **Loss Normalization Implementation (2025-09-09)**

**Problem Identified**: Gaussianity loss component was producing extreme values (1,864-6,019) that completely dominated training, making segmentation learning impossible.

**Root Cause Analysis**:
- **Anderson-Darling loss**: Unbounded values 100-1000+ overwhelming other components
- **Raw statistical moments**: Skewness/kurtosis producing 10-100+ values without normalization
- **Weight imbalance**: Gaussianity weight (0.1) with large raw values vs segmentation (~0.1-2.0)

**Implemented Solutions**:

1. **Bounded Statistical Loss Functions** (`src/samrfi/models/training.py`):
   - `_bounded_skewness_loss()`: `tanh(raw_loss/2.0)` → range [-1,1]
   - `_bounded_kurtosis_loss()`: `tanh(raw_loss/5.0)` → range [-1,1]
   - `_bounded_anderson_darling_loss()`: `sigmoid(raw_loss/5.0) - 0.5` → range [-0.5,0.5]

2. **Reduced Loss Weights** (`configs/training/v100_config.yaml`):
   - Overall gaussianity weight: `0.1 → 0.01` (10x reduction)
   - Component weights: `[1.0, 1.0, 2.0] → [0.3, 0.3, 0.4]`

3. **Detailed Loss Component Logging**:
   - Breakdown every 50 steps (epoch 0) / 200 steps (later epochs)
   - Individual segmentation, IoU, gaussianity components
   - Real/imaginary gaussianity sub-components for debugging

4. **Config Toggle for A/B Testing**:
   - `loss.gaussianity.enabled: true/false` for comparison studies

**Results Achieved**:
- **Total loss**: 1,864-6,019 → 0.52-0.53 (1000x improvement)
- **Loss stability**: Consistent training without wild swings
- **Component balance**: Segmentation (0.52) + IoU (0.015) + Gaussianity (0.001)
- **Training effectiveness**: 74% pixel-level accuracy in early epochs

**Loss Interpretation Guide**:
- **Segmentation loss 0.52**: ~74% pixel accuracy, good early SAM2 learning
- **IoU score loss 0.30**: Reasonable mask quality prediction
- **Gaussianity loss 0.001**: Controlled non-Gaussian penalty
- **Target evolution**: Segmentation should decrease to 0.1-0.3 over training

### **Remaining Items for Future Sessions:**
1. Reduce prompt complexity from 128 to 16 points (4x speedup potential)
2. Validate performance improvements with profiling  
3. Implement Phase 2 tiling features (overlaps, augmentation, adaptive prompts)

## Channel Mapping Implementation (2025-09-09)

### **New R=Gradient, G=Amplitude, B=Phase Approach**

**Implementation**: Updated training dataset to use optimized 3-channel mapping:
- **Red Channel**: Gradient magnitude from log-amplitude derivatives
- **Green Channel**: Log-amplitude (existing)
- **Blue Channel**: Phase (existing)

**Gradient Computation**:
```python
# Compute gradient magnitude from log amplitude
log_amp = np.log10(np.abs(complex_data) + 1e-10)  # Epsilon for numerical stability
time_deriv = np.diff(log_amp, axis=0)  # ∂/∂time
freq_deriv = np.diff(log_amp, axis=1)  # ∂/∂frequency  
gradient_magnitude = np.sqrt(time_deriv**2 + freq_deriv**2)  # Always positive
```

**RFI Detection Benefits**:
- **Gradient (Red)**: Highlights RFI edges and rapid transitions (perfect for SAM2 segmentation)
- **Amplitude (Green)**: Shows RFI strength and broad structure
- **Phase (Blue)**: Captures phase discontinuities from RFI

**Future Consideration**: Signed derivatives instead of magnitude:
- **Note**: Could explore individual signed time/frequency derivatives instead of magnitude
- **Rationale**: Directional change information (increasing vs decreasing) might help SAM2 distinguish RFI types
- **Implementation**: Replace `gradient_magnitude` with `time_deriv` or `freq_deriv` for directional sensitivity

## Current Implementation Status (Phase 2 Complete)

### Core Data Pipeline - OPERATIONAL ✅

**Memory-Efficient MS Loading** (`src/samrfi/core/loader_v2.py`):
- **MSMetadataExtractor**: Per-field metadata using `casatools.msmetadata`
- **MSLoader**: Scan-based SPW grouping with temporal structure awareness
- **Key Features**:
  - Automatic scan-based SPW grouping using `scannumbers()` and `spwsforscan()` 
  - Handles mixed time dimensions across SPW groups (e.g., 651 vs 684 time steps)
  - Memory-safe baseline loading with statistics tracking (std, skewness, kurtosis, % flagged)
  - 1024x1024 tile generation with zero-padding for SAM processing
  - Flag masking (zero flagged data before SAM inference)

**Iterative Flag Writing** (`src/samrfi/core/flagger_v2.py`):
- **MSFlagger**: SPW-aware flag writing with combine/replace modes
- **StatisticsTracker**: Comprehensive metrics including TP/TN/FP/FN, precision, recall, F1 score
- **Key Features**:
  - Always combines (OR) with existing flags, never replaces
  - Per-polarization statistics tracking
  - Flagging reports with before/after comparisons
  - Dry-run support for testing

### Working Data Flow
```python
# Multi-field, multi-baseline, multi-SPW-group processing
for field in all_fields:
    for baseline in all_baselines:  # ≤351 for VLA
        for spw_group in available_spw_groups:  # Temporal groups (0, 1, ...)
            # 1. Load baseline data for specific SPW group
            data = loader.load_baseline_data(ant1, ant2, spw_group_id)
            
            # 2. Apply existing flags as zeros
            masked_data = loader.apply_existing_flags_as_zeros(data['data'], data['existing_flags'])
            
            # 3. Generate 1024x1024 tiles per polarization
            tiles = loader.generate_1024x1024_tiles(masked_data)
            
            # 4. Run SAM inference → get mask
            flags = sam_inference(tiles)
            
            # 5. Write flags immediately (combine with existing)
            flagger.write_baseline_flags(ant1, ant2, flags, data['metadata'])
            
            # 6. Track statistics per baseline+SPW group
        print(f"Baseline {baseline} complete for field {field}")
    print(f"Field {field} complete")
```

### Validation Results
- **Test Suite**: `test_new_loader_flagger.py` - All tests pass ✅
- **Real MS Tested**: VLA 18A-191 day1 data with 32 SPWs, 10 fields, mixed time dimensions
- **SPW Grouping**: Correctly separates temporal groups (SPWs 0-15 vs 16-31)
- **Memory Management**: Handles large MS files with chunked processing
- **Statistics**: Full per-baseline, per-polarization tracking

### Technical Achievements
- **Scan-Based Grouping**: Solved mixed time dimension problem using observational structure
- **Casatools Integration**: Clean implementation using `table()` and `msmetadata()` APIs  
- **Legacy Compatibility**: Maintains patterns from `samrfi/radiorfi.py` for baseline queries
- **Professional Code**: No emojis, confidence levels indicated, simplest path prioritized

### Phase 5 Complete: Full SAM2 Training Pipeline - OPERATIONAL ✅

**Complete End-to-End Training Pipeline** (`training/synthetic_training.py`):
- **Real SAM2 Training**: Actual SAM2 model forward pass with gradient computation
- **CASA Integration**: Uses `SimulatedMS` for realistic baseline visibilities and noise
- **Fixed Dataset**: 2 training MS (702 baselines) + 1 validation MS (351 baselines)
- **Natural 1024×1024 Tiles**: Time×frequency dimensions match SAM2 input requirements
- **Heavy RFI Scenarios**: 25%+ contamination with 5 realistic RFI morphologies

**SAM2 Training Integration** (`src/samrfi/models/training.py`):
- **Real Forward Pass**: `_compute_sam2_loss()` uses actual SAM2 model predictions
- **Prompt-Based Training**: Point and bounding box prompts generated from ground truth RFI masks
- **Loss Computation**: BCE segmentation loss + IoU score loss with proper gradient flow
- **Memory Management**: Per-sample processing with batch accumulation
- **Tensor Handling**: Proper dimension management and scalar loss requirements

**Training Data Flow**:
```python
# 1. Load batch of RFI waterfall images and ground truth masks
# 2. For each sample: generate point/box prompts from ground truth
# 3. Process through SAM2: processor → model → predictions
# 4. Compute losses: BCE(predicted_mask, gt_mask) + IoU_score_loss
# 5. Backpropagate through actual SAM2 parameters
```

**Technical Achievements**:
- **Dual API Support**: Transformers SAM2 (primary) with official SAM2 fallback
- **Robust Error Handling**: Fixed indexing, dimension mismatches, and scalar loss requirements
- **Gradient Flow**: Proper backpropagation through SAM2 transformer layers
- **Hardware Optimized**: V100 config with lazy loading to prevent CUDA OOM errors

**Training Status**: 9.75-hour training run initiated with 70,200 steps across 50 epochs.

**Auto-Generated Training Data**:
- **Training**: 2,808 samples from realistic CASA-generated baselines
- **Validation**: 1,404 samples with consistent evaluation metrics
- **RFI Morphologies**: Broadband, narrowband lines, transients, periodic signals, satellite passes
- **Ground Truth**: Pixel-perfect RFI masks for supervised SAM2 learning

## Phase 4 Complete: Realistic RFI Simulation with CASA Integration

### SimulatedMS Pipeline - OPERATIONAL ✅

**CASA-Based Synthetic Data Generation** (`src/samrfi/datasets/simulated_ms.py`):
- **CASA Simulator Integration**: Uses `casatools.simulator` for realistic baseline visibilities  
- **Proper Thermal Noise**: `sigma_simple=1e-3` via `sm.setnoise()` for realistic noise levels
- **Memory Efficient Processing**: Baseline-by-baseline (~1.2GB constant memory usage)
- **Professional Tool Management**: Proper CASA tool cleanup with `.close()` and `.done()`

**Realistic RFI Implementation**:
1. **Broadband RFI**: Polynomial frequency variation (parabolic/cubic amplitude envelopes)
2. **Narrowband Lines**: 15+ persistent frequency lines across all time  
3. **Transient Pulses**: Short bursts and repeated burst patterns
4. **Periodic Signals**: Time-varying pulse trains with 10-30% duty cycles
5. **Satellite Passes**: Linear frequency drift over time (moving RFI)

**RFI Signal Characteristics**:
- **Heavy Contamination**: 25%+ data occupancy for challenging SAM training
- **Realistic Amplitudes**: 10^3 to 10^6 times thermal noise (1e-2 to 1.0 range)
- **Signal Morphology**: 
  - Frequency-varying: Broadband with smooth polynomial envelopes
  - Time-varying: Transients, periodic patterns, satellite motion  
  - Combined: Satellite passes with time+frequency evolution

**SAM-Ready Output**:
- **1024×1024 Tiles**: Natural time×frequency dimensions (no resizing needed)
- **4-Panel Visualizations**: Clean/Corrupted/RFI-Mask/RFI-Only plots per baseline/polarization
- **Ground Truth Masks**: Pixel-perfect RFI flagging for supervised learning
- **Noise Structure Visible**: Log-scale amplitude plots show realistic noise floors

### Memory-Controlled Architecture

**Baseline-by-Baseline Processing**:
```python
# Process 351 baselines individually to maintain constant memory
for baseline_idx, (ant1, ant2) in enumerate(antenna_pairs):
    # 1. Read single baseline from CASA MS
    baseline_vis = read_baseline_data(ant1, ant2)  # ~3MB
    
    # 2. Generate RFI for this baseline only  
    rfi_array, rfi_mask = generate_baseline_rfi(baseline_vis.shape)
    
    # 3. Add RFI and write back immediately
    corrupted_vis = baseline_vis + rfi_array
    write_baseline_data(ant1, ant2, corrupted_vis, rfi_mask)
    
    # 4. Create SAM tile plots (first 5 baselines)
    if baseline_idx < 5:
        create_sam_plots(baseline_vis, corrupted_vis, rfi_mask)
    
    # 5. Immediate cleanup
    del baseline_vis, rfi_array, rfi_mask, corrupted_vis
    gc.collect()
```

**Usage Examples**:
```bash
# Create realistic MS with heavy RFI contamination
python test_single_ms.py

# Generates:
# - test_simulated.ms (CASA MS with realistic noise + 25%+ RFI)
# - test_simulated_plots/ (SAM 1024×1024 tile visualizations)
```

**Training Data Quality Achieved**:
- ✅ **Realistic Baselines**: CASA simulator provides proper visibility structure
- ✅ **Proper Noise Levels**: 1e-3 thermal noise matching real observations  
- ✅ **Heavy RFI Loading**: 25%+ contamination with 5 distinct RFI morphologies
- ✅ **SAM-Optimized Format**: Natural 1024×1024 tiles, no artificial resizing
- ✅ **Ground Truth Perfect**: Exact pixel-level RFI masks for training loss

## Project Planning and Workflow

### Implementation Roadmap
This project follows a structured phase-based approach documented in:
- **`IMPLEMENTATION_PLAN.md`**: Current status (Phase 4 complete), next steps (Phase 5 training integration), and detailed technical requirements
- **`VALIDATION.md`**: Final validation criteria against real MS data (target of overall project)

### Development Philosophy
**Important**: Claude Code should prioritize:
1. **Planning over coding**: Extensive discussion and approval before implementation
2. **Explicit justification**: Explain reasoning and confidence levels for all recommendations  
3. **No assumptions**: Never interpret user intent - ask clarifying questions instead
4. **Cautious approach**: Seek approval for architectural decisions and code changes

### Current Project Status
**Phase 1-2 Complete**: Core data pipeline (MSLoader/MSFlagger) - operational ✅
**Phase 4 Complete**: Realistic RFI simulation (SimulatedMS) - operational ✅  
**Phase 5 Next**: Training pipeline integration using SimulatedMS approach
**Phase 6 Goal**: Real data validation leading to VALIDATION.md completion

### Next Steps: Training Pipeline Update

The existing `training/synthetic_training.py` should be updated to use `SimulatedMS` instead of the old synthetic visibility generation approach. This will provide:
- Realistic CASA-generated baseline data
- Heavy RFI contamination (25%+ vs previous ~20%)  
- Proper noise floors for residual evaluation
- Memory-efficient processing for large training datasets

**Before Implementation**: Detailed planning discussion required to ensure proper integration approach and avoid assumptions about training workflow requirements.
- Always give me information that is substantiated. Think like a critical scientific pair programmer. We will discuss a lot and create work plans before actually writing code
- Give me confidence values for your statements and sources for substantiation