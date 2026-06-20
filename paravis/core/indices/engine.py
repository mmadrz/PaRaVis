"""
Core spectral index computation engine.

Uses spyndex for formula parsing and evaluation. Pure Python — no Qt dependency.
"""
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import spyndex

from .models import SpectralIndex, BandMapping


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_available_indices() -> List[SpectralIndex]:
    """Return the list of all spectral indices known to spyndex.

    Returns
    -------
    List[SpectralIndex]
    """
    results: List[SpectralIndex] = []
    for name, idx in spyndex.indices.items():
        bands = list(getattr(idx, "bands", []))
        constants = list(getattr(idx, "constants", []))
        formula = getattr(idx, "formula", "")
        reference = getattr(idx, "reference", "")
        long_name = getattr(idx, "long_name", name)
        results.append(
            SpectralIndex(
                name=name,
                formula=formula,
                bands=bands,
                constants=constants,
                reference=reference,
                long_name=long_name,
            )
        )
    return results


def get_default_band_mapping() -> Dict[int, str]:
    """Return the default Landsat 8/9 band mapping."""
    return {1: "A", 2: "B", 3: "G", 4: "R", 5: "N", 6: "S1", 7: "S2", 8: "T"}


def is_index_computable(
    idx_name: str,
    constants_override: Dict[str, float],
    band_mapping: Dict[int, str],
) -> bool:
    """Check whether a given index can be computed with the provided mapping.

    Parameters
    ----------
    idx_name : str
        Name of the spectral index.
    constants_override : Dict[str, float]
        User-supplied constant overrides.
    band_mapping : Dict[int, str]
        Mapping from band numbers to spectral codes.

    Returns
    -------
    bool
    """
    if idx_name not in spyndex.indices:
        return False

    idx = spyndex.indices[idx_name]
    required_bands = getattr(idx, "bands", [])
    required_constants = getattr(idx, "constants", [])
    available_bands = set(band_mapping.values())

    # All required bands must be mapped
    for band in required_bands:
        if band not in available_bands:
            return False

    # All required constants must be provided (either default or overridden).
    # A constant in spyndex.constants with default=None is NOT sufficient —
    # it would cause a TypeError in spyndex's eval().
    for const in required_constants:
        if const in constants_override:
            if constants_override[const] is not None:
                continue
        # Check spyndex default (must be non-None)
        spyndex_const = spyndex.constants.get(const)
        if spyndex_const is not None and spyndex_const.default is not None:
            continue
        return False

    return True


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_index(
    raster_data: np.ndarray,
    band_mapping: Dict[int, str],
    index_name: str,
    constants: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute a single spectral index from raster data.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    band_mapping : Dict[int, str]
        Mapping from band number (1-based) to spectral code.
    index_name : str
        Name of the index (e.g. 'NDVI').
    constants : Dict[str, float], optional
        Override constants for the index.

    Returns
    -------
    np.ndarray
        2D array of the computed index.
    """
    from spyndex import computeIndex

    # Build the parameter dict for spyndex: code -> band array
    params: Dict[str, np.ndarray] = {}

    for band_num, code in band_mapping.items():
        if 1 <= band_num <= raster_data.shape[0]:
            params[code] = raster_data[band_num - 1]

    # Fill in default constants from spyndex (skip None-valued ones —
    # they are wavelength references that cause TypeError in eval()).
    default_consts = {
        k: v.default for k, v in spyndex.constants.items()
        if v.default is not None
    }
    params.update(default_consts)
    if constants:
        # Also drop any None-valued overrides that might sneak in
        clean_consts = {k: v for k, v in constants.items() if v is not None}
        params.update(clean_consts)

    try:
        result = computeIndex(index_name, **params)
        return np.asarray(result, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute index '{index_name}': {exc}")


def compute_indices(
    raster_data: np.ndarray,
    band_mapping: Dict[int, str],
    index_names: List[str],
    constants: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Compute multiple spectral indices from raster data.

    Passes **all** index names to ``spyndex.computeIndex`` in a single
    call — more efficient than calling ``compute_index`` in a loop since
    the parameter dict is built once.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    band_mapping : Dict[int, str]
        Mapping from band number to spectral code.
    index_names : List[str]
        Names of indices to compute.
    constants : Dict[str, float], optional
        Override constants.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from index name to 2D result array.
    """
    from spyndex import computeIndex

    # Filter to valid index names
    valid_names = [n for n in index_names if n in spyndex.indices]
    if not valid_names:
        return {}

    # Build parameter dict (same as compute_index)
    params: Dict[str, np.ndarray] = {}
    for band_num, code in band_mapping.items():
        if 1 <= band_num <= raster_data.shape[0]:
            params[code] = raster_data[band_num - 1]

    default_consts = {
        k: v.default
        for k, v in spyndex.constants.items()
        if v.default is not None
    }
    params.update(default_consts)
    if constants:
        clean_consts = {k: v for k, v in constants.items() if v is not None}
        params.update(clean_consts)

    try:
        result = computeIndex(valid_names, **params)
    except Exception as exc:
        # Fall back to one-by-one if the batch call fails
        results: Dict[str, np.ndarray] = {}
        for name in valid_names:
            try:
                results[name] = compute_index(
                    raster_data, band_mapping, name, constants
                )
            except Exception as exc2:
                print(f"  ⚠ Skipping '{name}': {exc2}")
        return results

    result_np = np.asarray(result, dtype=np.float32)
    if result_np.ndim == 2:
        return {valid_names[0]: result_np}
    return {
        name: result_np[i] for i, name in enumerate(valid_names)
    }


# ---------------------------------------------------------------------------
# Dask-accelerated batch computation
# ---------------------------------------------------------------------------

def compute_indices_dask(
    raster_data: np.ndarray,
    band_mapping: Dict[int, str],
    index_names: List[str],
    constants: Optional[Dict[str, float]] = None,
    tile_size: int = 512,
    num_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute multiple spectral indices in parallel using dask + spyndex.

    Converts the numpy array to a chunked dask array and calls
    ``spyndex.computeIndex`` with **all** index names at once.  The
    returned dask array is then computed using dask's threaded scheduler,
    which processes chunks in parallel.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    band_mapping : Dict[int, str]
        Mapping from band number to spectral code.
    index_names : List[str]
        Names of indices to compute.
    constants : Dict[str, float], optional
        Override constants.
    tile_size : int
        Chunk size for dask (height and width).  Default 512.
    num_workers : int, optional
        Number of threads for dask's threaded scheduler.  ``None`` means
        dask chooses (usually ``os.cpu_count()``).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from index name to 2D result array.
    """
    import dask
    import dask.array as da
    from spyndex import computeIndex

    # Validate that all requested indices exist
    valid_names: List[str] = []
    for name in index_names:
        if name in spyndex.indices:
            valid_names.append(name)

    if not valid_names:
        return {}

    # ------------------------------------------------------------------
    # 1. Build params dict with dask arrays
    # ------------------------------------------------------------------
    # Chunk each band independently: band dimension is size-1 so each
    # band can be processed on its own.
    chunks = (1, tile_size, tile_size)
    data_dask = da.from_array(raster_data, chunks=chunks)

    params: Dict[str, object] = {}
    for band_num, code in band_mapping.items():
        if 1 <= band_num <= data_dask.shape[0]:
            params[code] = data_dask[band_num - 1]

    # Default spyndex constants (skip None-valued wavelength references)
    default_consts = {
        k: v.default
        for k, v in spyndex.constants.items()
        if v.default is not None
    }
    params.update(default_consts)

    if constants:
        clean_consts = {k: v for k, v in constants.items() if v is not None}
        params.update(clean_consts)

    # ------------------------------------------------------------------
    # 2. Compute all indices via spyndex (returns a lazy dask array)
    # ------------------------------------------------------------------
    try:
        result_dask = computeIndex(valid_names, **params)  # type: ignore[arg-type]
    except Exception as exc:
        raise RuntimeError(f"spyndex.computeIndex failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Trigger parallel computation
    # ------------------------------------------------------------------
    if num_workers is not None:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            result_np = result_dask.compute()
    else:
        result_np = result_dask.compute()

    # ------------------------------------------------------------------
    # 4. Split into per-index 2D arrays
    # ------------------------------------------------------------------
    result_np = np.asarray(result_np, dtype=np.float32)

    if result_np.ndim == 2:
        # Only one index was requested
        return {valid_names[0]: result_np}

    return {
        name: result_np[i] for i, name in enumerate(valid_names)
    }
