"""Data generation modules for SAM-RFI"""

from .synthetic_generator import SyntheticDataGenerator

__all__ = ["SyntheticDataGenerator"]

# Note: MSDataGenerator requires CASA and is not imported by default
# Use: from samrfi.data_generation.ms_generator import MSDataGenerator
