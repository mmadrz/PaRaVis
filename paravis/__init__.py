"""
PaRaVis — Parallel Rao's Q Visualization and Analysis
======================================================

A library for spectral index computation, Rao's Q diversity analysis,
and raster data visualization.

Main components:
    paravis.api         — Public API for headless/script usage
    paravis.core        — Core computation (no Qt dependency)
    paravis.gui         — PySide6 desktop application
    paravis.workers     — Background processing threads
    paravis.utils       — Shared utilities (settings, system profiler)
"""

from .__version__ import __version__, __version_info__  # noqa: F401


def __getattr__(name):
    """Lazy import of public API to avoid circular imports at package level."""
    if name in ("compute_indices", "compute_rao_q", "plot_raster", "plot_comparison"):
        import paravis.api as _api
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

