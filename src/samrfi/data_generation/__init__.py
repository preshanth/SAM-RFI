"""Data generation modules for SAM-RFI

NOTE: SyntheticDataGenerator has been moved to rfi_toolbox for sharing across ML methods.
For backward compatibility, we provide both:
- rfi_toolbox.data_generation.SyntheticDataGenerator (recommended)
- samrfi.data_generation.synthetic_generator.SyntheticDataGenerator (deprecated, will be removed)
"""

# Forward import from rfi_toolbox (recommended)
from rfi_toolbox.data_generation import SyntheticDataGenerator

# SAM2-specific data generation
from .ms_generator import MSDataGenerator

__all__ = ["SyntheticDataGenerator", "MSDataGenerator"]
