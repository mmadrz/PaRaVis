"""
Shared data models used across core sub-packages.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RasterProfile:
    """Metadata for a loaded raster file."""
    path: str
    shape: Tuple[int, int]
    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    dtype: Optional[str] = None
    nodata: Optional[float] = None
    size_mb: float = 0.0


@dataclass
class WindowConfig:
    """Configuration for moving-window operations (Rao's Q)."""
    window_size: int = 15
    step_size: int = 1
    na_tolerance: float = 0.3
    max_pixels: int = 500_000
