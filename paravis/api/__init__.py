"""
paravis.api — Public API for headless/script usage.

Convenience wrappers around paravis.core that provide sensible defaults
and return NumPy arrays or xarray-compatible results.
"""

from .indices import compute_indices, list_available_indices
from .raoq import compute_rao_q
from .visualization import plot_raster, plot_comparison

__all__ = [
    "compute_indices",
    "list_available_indices",
    "compute_rao_q",
    "plot_raster",
    "plot_comparison",
]
