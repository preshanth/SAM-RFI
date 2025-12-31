"""Data generation modules for SAM-RFI

NOTE: SyntheticDataGenerator is in rfi_toolbox. Import directly:
    from rfi_toolbox.data_generation import SyntheticDataGenerator
"""

# SAM2-specific data generation
from .ms_generator import MSDataGenerator

__all__ = ["MSDataGenerator"]
