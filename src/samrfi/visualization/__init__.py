"""
Interactive and static visualization tools for SAM-RFI.

This module provides visualization capabilities for radio astronomy data:
- Interactive MS waterfall exploration (HoloViz + Datashader)
- Static publication-quality plots (Matplotlib)
- Multi-flagger comparison plots

Installation:
    pip install samrfi[viz]  # For interactive visualization
"""

__all__ = []

# Try to import interactive visualization (requires viz extras)
try:
    from .ms_explorer import MSWaterfallExplorer, create_explorer_from_ms

    __all__.extend(["MSWaterfallExplorer", "create_explorer_from_ms"])
    HAS_HOLOVIZ = True
except ImportError:
    HAS_HOLOVIZ = False

# Static visualization (matplotlib - always available)
# from .static import plot_waterfall_static, plot_4panel_comparison
# __all__.extend(["plot_waterfall_static", "plot_4panel_comparison"])
