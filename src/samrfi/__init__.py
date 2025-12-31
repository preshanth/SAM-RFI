"""
SAM-RFI: Radio Frequency Interference Detection with SAM2
==========================================================

A Python package for detecting and flagging Radio Frequency Interference (RFI)
in radio astronomy data using Meta's Segment Anything Model 2 (SAM2).

Key Features:
- SAM2-based segmentation with HuggingFace transformers
- Physically realistic synthetic data generation
- Complete training pipeline with validation tracking
- Iterative flagging for deep RFI cleaning
- GPU-accelerated training and inference

Modules:
--------
data : Data loading and preprocessing
    - MSLoader: Load CASA measurement sets
    - Preprocessor: Patchify, normalize, and preprocess data
    - SAMDataset: PyTorch Dataset wrapper
    - TorchDataset: Efficient torch-backed datasets with shared memory

data_generation : Dataset generators
    - SyntheticDataGenerator: Generate physically realistic synthetic RFI
    - MSDataGenerator: Convert MS files to training datasets

training : Model training
    - SAM2Trainer: Train SAM2 models with HuggingFace transformers

inference : Apply trained models
    - RFIPredictor: Single-pass and iterative RFI prediction

config : Configuration management
    - ConfigLoader: Load and validate YAML configs

Usage:
------
>>> # Core data operations
>>> from rfi_toolbox.preprocessing import Preprocessor
>>> from rfi_toolbox.datasets import TorchDataset
>>> from samrfi.data_generation import SyntheticDataGenerator
>>>
>>> # Optional: CASA-dependent operations
>>> from rfi_toolbox.io import MSLoader  # Requires pip install samrfi[casa]
>>>
>>> # Optional: GPU/transformers-dependent operations
>>> from samrfi.training import SAM2Trainer  # Requires pip install samrfi[gpu]
>>> from samrfi.inference import RFIPredictor  # Requires pip install samrfi[gpu]
>>>
>>> # Full workflow example (requires [gpu,casa])
>>> loader = MSLoader('observation.ms')
>>> loader.load(num_antennas=5)
>>> preprocessor = Preprocessor(loader.data)
>>> dataset = preprocessor.create_dataset(patch_size=128)
>>> trainer = SAM2Trainer(dataset, device='cuda')
>>> trainer.train(num_epochs=10, batch_size=4)
>>> predictor = RFIPredictor('model.pth', device='cuda')
>>> flags = predictor.predict_ms('observation.ms')
"""

__version__ = "2.0.0"
__author__ = "Derod Deal, Preshanth Jagannathan"

# Config module
from .config import ConfigLoader

# SAM2-specific data modules
from .data import BatchedDataset, HFDatasetWrapper, SAMDataset

# Data generation module
from .data_generation import SyntheticDataGenerator

# Note: Shared utilities (MSLoader, Preprocessor, BatchWriter, TorchDataset)
# are in rfi_toolbox. Import directly when needed:
#   from rfi_toolbox.io import MSLoader
#   from rfi_toolbox.preprocessing import Preprocessor
#   from rfi_toolbox.datasets import BatchWriter, TorchDataset
# Note: MSDataGenerator requires CASA and is not imported by default
# Use: from samrfi.data_generation.ms_generator import MSDataGenerator

# Note: ModelCache, RFIPredictor, and SAM2Trainer require transformers and are not imported by default
# Use: from samrfi.utils.model_cache import ModelCache
# Use: from samrfi.inference import RFIPredictor
# Use: from samrfi.training import SAM2Trainer


__all__ = [
    # Data
    "Preprocessor",
    "SAMDataset",
    "BatchedDataset",
    "TorchDataset",
    "BatchWriter",
    "HFDatasetWrapper",
    # Data generation
    "SyntheticDataGenerator",
    # Config
    "ConfigLoader",
]
