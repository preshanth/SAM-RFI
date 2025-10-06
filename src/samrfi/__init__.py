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
    - NumpyDataset: Efficient numpy-backed datasets

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
>>> from samrfi.data import MSLoader, Preprocessor
>>> from samrfi.training import SAM2Trainer
>>> from samrfi.inference import RFIPredictor
>>>
>>> # Load and preprocess data
>>> loader = MSLoader('observation.ms')
>>> loader.load(num_antennas=5)
>>> preprocessor = Preprocessor(loader.data)
>>> dataset = preprocessor.create_dataset(patch_size=128)
>>>
>>> # Train model
>>> trainer = SAM2Trainer(dataset, device='cuda')
>>> trainer.train(num_epochs=10, batch_size=4)
>>>
>>> # Predict RFI
>>> predictor = RFIPredictor('model.pth', device='cuda')
>>> flags = predictor.predict_ms('observation.ms')
"""

__version__ = "2.0.0"
__author__ = "Derod Deal, Preshanth Jagannathan"

# Data module
from .data import (
    MSLoader,
    Preprocessor,
    SAMDataset,
    BatchedDataset,
    NumpyDataset,
    BatchWriter,
    HFDatasetWrapper,
)

# Data generation module
from .data_generation import (
    SyntheticDataGenerator,
    MSDataGenerator,
)

# Training module
from .training import SAM2Trainer

# Inference module
from .inference import RFIPredictor

# Config module
from .config import ConfigLoader

# Utilities
from .utils import ModelCache

__all__ = [
    # Data
    "MSLoader",
    "Preprocessor",
    "SAMDataset",
    "BatchedDataset",
    "NumpyDataset",
    "BatchWriter",
    "HFDatasetWrapper",
    # Data generation
    "SyntheticDataGenerator",
    "MSDataGenerator",
    # Training
    "SAM2Trainer",
    # Inference
    "RFIPredictor",
    # Config
    "ConfigLoader",
    # Utils
    "ModelCache",
]
