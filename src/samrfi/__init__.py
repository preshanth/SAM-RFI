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

Shared primitives (MSLoader, Preprocessor, TorchDataset, BatchWriter,
SyntheticDataGenerator, segmentation/statistics metrics) live in the external
``rfi_toolbox`` package and are imported directly from there. The ``samrfi.*``
modules below provide only the SAM2-specific pieces layered on top.

Modules:
--------
data : SAM2-specific dataset wrappers
    - SAMDataset: SAM2 Dataset wrapper (bbox prompts + processor)
    - BatchedDataset: Streaming reader for generated batch_*.pt directories
    - HFDatasetWrapper: Convert datasets to/from HuggingFace format
    - AdaptivePatcher, RAMCachedDataset, GPU transforms

data_generation : Dataset generators
    - MSDataGenerator: Convert MS files to training datasets
    (SyntheticDataGenerator lives in rfi_toolbox.data_generation)

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
>>> from rfi_toolbox.data_generation import SyntheticDataGenerator
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

# Note: Shared utilities from rfi_toolbox - import directly when needed:
#   from rfi_toolbox.io import MSLoader
#   from rfi_toolbox.preprocessing import Preprocessor
#   from rfi_toolbox.datasets import BatchWriter, TorchDataset
#   from rfi_toolbox.data_generation import SyntheticDataGenerator
# Note: MSDataGenerator requires CASA and is not imported by default
# Use: from samrfi.data_generation import MSDataGenerator

# Note: ModelCache, RFIPredictor, and SAM2Trainer require transformers and are not imported by default
# Use: from samrfi.utils.model_cache import ModelCache
# Use: from samrfi.inference import RFIPredictor
# Use: from samrfi.training import SAM2Trainer


__all__ = [
    # SAM2-specific data modules (shared primitives live in rfi_toolbox)
    "SAMDataset",
    "BatchedDataset",
    "HFDatasetWrapper",
    # Config
    "ConfigLoader",
]
