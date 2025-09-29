"""
Training data generators for SAM-RFI

Provides base classes and concrete implementations for generating training data
from various sources (CASA simulator, pure Python, etc.)
"""

from .rfi_generator import RFIGenerator
from .base import TrainingDataGenerator
from .casa_generator import CASATrainingGenerator
from .python_generator import PurePythonTrainingGenerator

__all__ = [
    'RFIGenerator', 
    'TrainingDataGenerator',
    'CASATrainingGenerator', 
    'PurePythonTrainingGenerator'
]