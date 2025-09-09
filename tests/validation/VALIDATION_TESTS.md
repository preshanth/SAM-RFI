# SAM-RFI Pipeline Validation Tests

This directory contains validation tests for the SAM-RFI pipeline components. All tests use the current production architecture and are designed to validate end-to-end functionality.

## Overview

These tests validate the complete SAM-RFI pipeline from data generation through flag application:

- **Data Generation**: CASA-based synthetic MS creation with realistic RFI
- **Core Pipeline**: MS loading, tiling, SAM2 inference, and flag application  
- **Model Integration**: SAM2 adapter functionality and inference engine
- **End-to-End**: Complete workflow validation

## Test Files

### 1. Core Pipeline Tests

#### `test_new_loader_flagger.py`
**Purpose**: Validates MS loading and flagging infrastructure  
**Components Tested**:
- `MSLoader` - Memory-efficient MS data loading
- `MSFlagger` - Flag writing and statistics tracking  
- `MSMetadataExtractor` - MS metadata extraction
- `StatisticsTracker` - Performance metrics

**Usage**:
```bash
python tests/validation/test_new_loader_flagger.py /path/to/test.ms
```

**What it validates**:
- Scan-based SPW grouping handles mixed time dimensions
- Baseline data loading works across all antenna pairs
- Flag writing combines properly with existing flags
- Statistics tracking provides accurate TP/TN/FP/FN metrics
- Memory usage remains constant during processing

---

#### `test_flag_pipeline.py`
**Purpose**: Tests complete SAM2 flag application pipeline  
**Components Tested**:
- `MSFlagApplicator` - Complete pipeline orchestration
- `TilingProcessor` - 1024×1024 → 4×256×256 tiling  
- `SAMInferenceEngine` - SAM2 model integration
- Union-based mask computation

**Usage**:
```bash
python tests/validation/test_flag_pipeline.py /path/to/test.ms --n_baselines 2
```

**What it validates**:
- 1024×1024 tiles split correctly into 256×256 patches
- SAM2 inference runs on patches (untrained model)
- Union computation combines multi-mask SAM2 outputs
- Reconstructed masks match original dimensions
- Flags are applied correctly to measurement set
- Pipeline handles multiple polarizations properly

---

### 2. Data Generation Tests

#### `test_simulated_ms.py`  
**Purpose**: Validates CASA-based synthetic data generation  
**Components Tested**:
- `SimulatedMS` - CASA simulator integration
- `ObservationConfig` - Observation parameter management
- `RFIConfig` - RFI injection configuration
- Realistic RFI morphologies (broadband, narrowband, transients, etc.)

**Usage**:
```bash
python tests/validation/test_simulated_ms.py
```

**What it validates**:
- CASA simulator creates realistic baseline visibilities
- Thermal noise levels match specifications (1e-3 sigma)
- RFI injection produces 25%+ contamination
- Generated MS files are valid CASA format
- Memory usage stays constant during baseline-by-baseline processing
- Ground truth masks are pixel-perfect for training

---

#### `test_single_ms.py`
**Purpose**: Tests single MS generation workflow  
**Components Tested**:
- `SimulatedMS` end-to-end workflow
- Visualization generation for SAM training data
- Memory-efficient single MS processing

**Usage**:
```bash  
python tests/validation/test_single_ms.py
```

**What it validates**:
- Single MS creation completes without memory issues
- 1024×1024 visualization tiles are generated correctly
- RFI patterns are diverse and realistic
- Generated data suitable for SAM2 training

---

### 3. End-to-End Validation

#### `test_end_to_end_validation.py`
**Purpose**: Complete pipeline validation from MS generation through flagging  
**Components Tested**: 
- Full pipeline integration
- Real MS data compatibility
- Performance benchmarking

**Usage**:
```bash
python tests/validation/test_end_to_end_validation.py --ms /path/to/real.ms
```

**What it validates**:
- Pipeline processes real measurement sets
- Performance metrics meet expectations  
- Flag quality assessment against known RFI
- Memory and compute resource usage
- Integration with existing radio astronomy workflows

---

## Running All Tests

### Prerequisites
```bash
# Install package in development mode
pip install -e .

# Ensure CASA tools are available
python -c "from casatools import table, msmetadata; print('CASA OK')"

# Ensure SAM2 model access
python -c "from samrfi.adapters import SAM2Adapter; print('SAM2 OK')"
```

### Test Execution

**Individual Tests**:
```bash
# Core pipeline (requires existing MS)
python tests/validation/test_new_loader_flagger.py /path/to/test.ms
python tests/validation/test_flag_pipeline.py /path/to/test.ms

# Data generation (creates test MS)  
python tests/validation/test_simulated_ms.py
python tests/validation/test_single_ms.py

# End-to-end (comprehensive)
python tests/validation/test_end_to_end_validation.py --ms /path/to/test.ms
```

**Batch Testing**:
```bash
# Run all validation tests with synthetic data
cd tests/validation
./run_all_tests.sh
```

### Expected Outcomes

**Successful Test Run**:
- All components load without import errors
- MS files are read/written correctly using casatools
- SAM2 model loads and runs inference (even untrained)
- Tiling operations preserve data dimensions
- Flag application modifies MS files appropriately
- Memory usage remains bounded during processing
- Statistics and reports are generated correctly

**Common Failures**:
- **CASA not available**: Install CASA or use conda environment
- **SAM2 model loading**: Check GPU availability and model download
- **MS file permissions**: Ensure read/write access to test data
- **Memory issues**: Reduce batch sizes or use smaller test MS

## Test Data Requirements

**For Core Pipeline Tests**:
- Real measurement set with multiple baselines, SPWs, and polarizations
- Preferably VLA or ALMA data with some existing flags
- Size: 100MB-1GB for reasonable test runtime

**For Data Generation Tests**:  
- No external data required
- Tests create synthetic MS files automatically
- Generated files: ~50-500MB depending on configuration

**For End-to-End Tests**:
- Real MS with known RFI contamination
- Ground truth flagging for validation comparison
- Size: 500MB-5GB for comprehensive testing

## Integration with CI/CD

These tests are designed for:
- **Development validation**: Run locally during feature development  
- **Pull request checks**: Automated testing of pipeline changes
- **Release validation**: Comprehensive testing before releases
- **Performance benchmarking**: Track pipeline performance over time

**Automation Ready**:
All tests support command-line execution and return proper exit codes for automated testing systems.