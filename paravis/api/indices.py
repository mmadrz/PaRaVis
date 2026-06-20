"""
Public API for spectral index computation.

Convenience wrappers around paravis.core.indices with sensible defaults.
"""
from typing import Dict, List, Optional, Union

import numpy as np

from paravis.core.raster import read_raster
from paravis.core.indices import (
    compute_index as _compute_index,
    compute_indices as _compute_indices,
    get_available_indices,
    is_index_computable,
    get_default_band_mapping,
)


def list_available_indices() -> List[str]:
    """List all available spectral index names.

    Returns
    -------
    List[str]
    """
    return [idx.name for idx in get_available_indices()]


def compute_indices(
    raster_path: str,
    indices: Optional[List[str]] = None,
    band_mapping: Optional[Dict[int, str]] = None,
    constants: Optional[Dict[str, float]] = None,
    max_pixels: int = 1_000_000,
) -> Dict[str, np.ndarray]:
    """Compute spectral indices from a raster file.

    Parameters
    ----------
    raster_path : str
        Path to a multi-band GeoTIFF.
    indices : List[str], optional
        Names of indices to compute. If None, computes all computable ones.
    band_mapping : Dict[int, str], optional
        Mapping from band number to spectral code.
        Defaults to Landsat 8/9 mapping.
    constants : Dict[str, float], optional
        Override constants for index computation.
    max_pixels : int
        Maximum pixels to read (auto-downsamples beyond this).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from index name to 2D result array.
    """
    if band_mapping is None:
        band_mapping = get_default_band_mapping()

    # Read raster
    data_3d, transform, crs = read_raster(raster_path, max_pixels=max_pixels)

    # Ensure 3D
    if data_3d.ndim == 2:
        data_3d = data_3d[np.newaxis, :, :]

    # Determine which indices to compute
    if indices is None:
        all_indices = get_available_indices()
        idx_constants = constants or {}
        indices = [
            idx.name for idx in all_indices
            if is_index_computable(idx.name, idx_constants, band_mapping)
        ]

    # Compute
    results = _compute_indices(data_3d, band_mapping, indices, constants)
    return results


def compute_index(
    raster_path: str,
    index_name: str,
    band_mapping: Optional[Dict[int, str]] = None,
    constants: Optional[Dict[str, float]] = None,
    max_pixels: int = 1_000_000,
) -> np.ndarray:
    """Compute a single spectral index from a raster file.

    Parameters
    ----------
    raster_path : str
        Path to a multi-band GeoTIFF.
    index_name : str
        Name of the index (e.g. 'NDVI').
    band_mapping : Dict[int, str], optional
        Band mapping. Defaults to Landsat 8/9.
    constants : Dict[str, float], optional
        Override constants.
    max_pixels : int
        Maximum pixels to read.

    Returns
    -------
    np.ndarray
        2D array of the computed index.
    """
    if band_mapping is None:
        band_mapping = get_default_band_mapping()

    data_3d, _, _ = read_raster(raster_path, max_pixels=max_pixels)
    if data_3d.ndim == 2:
        data_3d = data_3d[np.newaxis, :, :]

    return _compute_index(data_3d, band_mapping, index_name, constants)
