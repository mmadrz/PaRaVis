"""Spectral index computation engine."""

from .engine import (
    compute_index,
    compute_indices,
    compute_indices_dask,
    is_index_computable,
    get_available_indices,
    get_default_band_mapping,
)
from .registry import register_index, list_custom_indices
from .constants import get_default_constants, merge_constants
from .models import SpectralIndex, BandMapping

__all__ = [
    "compute_index",
    "compute_indices",
    "compute_indices_dask",
    "is_index_computable",
    "get_available_indices",
    "get_default_band_mapping",
    "register_index",
    "list_custom_indices",
    "get_default_constants",
    "merge_constants",
    "SpectralIndex",
    "BandMapping",
]
