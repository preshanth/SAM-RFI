"""
SAM-RFI: SAM-based Radio Frequency Interference Detection for Radio Astronomy

A modern, pip-installable package for RFI detection using Meta's Segment Anything Model.
"""

__version__ = "1.0.0"

# Package metadata
__author__ = "Derod Deal, Preshanth Jagannathan"
__email__ = "dealderod@gmail.com, pjaganna@nrao.edu"
__license__ = "MIT"
__description__ = "SAM-based Radio Frequency Interference Detection for Radio Astronomy"

# Import core components (will be implemented in later stages)
# For now, we'll import what exists to avoid import errors

try:
    # These will be implemented in later stages
    from .core import *
    from .adapters import *
    from .models import *
except ImportError as e:
    # During refactor stages, some modules may not exist yet
    import warnings
    warnings.warn(
        f"Some SAM-RFI modules not yet available during refactor: {e}. "
        "This is expected during package restructuring.",
        ImportWarning
    )

# Package information
def get_version() -> str:
    """Get the current package version."""
    return __version__

def get_info() -> dict:
    """Get package information."""
    return {
        "name": "samrfi",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "status": "Refactor Stage 1 - Package Infrastructure"
    }

# For backwards compatibility during refactor
# These will be removed once full refactor is complete
try:
    import sys
    import os
    
    # Add the old samrfi directory to path temporarily
    old_samrfi_path = os.path.join(os.path.dirname(__file__), '..', '..', 'samrfi')
    if os.path.exists(old_samrfi_path):
        sys.path.insert(0, os.path.dirname(old_samrfi_path))
        
        # Import key classes from old structure for compatibility
        from samrfi.radiorfi import RadioRFI
        from samrfi.rfimodels import RFIModels
        from samrfi.syntheticrfi import SyntheticRFI
        from samrfi.rfitraining import RFITraining
        from samrfi.rfidataset import RFIDataset
        from samrfi.metricscalculator import RadioRFIMetricsCalculator, SyntheticRFIMetricsCalculator
        from samrfi.plotter import Plotter
        
        # Clean up path
        sys.path.remove(os.path.dirname(old_samrfi_path))
        
except ImportError:
    # If old imports fail, that's ok - we're in transition
    pass

__all__ = [
    "get_version",
    "get_info",
    # Legacy exports (will be updated in later stages)
    "RadioRFI",
    "RFIModels", 
    "SyntheticRFI",
    "RFITraining",
    "RFIDataset",
    "RadioRFIMetricsCalculator",
    "SyntheticRFIMetricsCalculator",
    "Plotter",
]