"""
Custom index registry — allows users to register new spectral indices.

Usage:
    from paravis.core.indices import register_index

    @register_index(name="CUSTOM_VI", bands=["N", "R"])
    def custom_vi(nir, red):
        return (nir - red) / (nir + red + 0.5)
"""
from typing import Callable, Dict, List, Optional

import numpy as np

# Storage for custom index functions
_custom_indices: Dict[str, Callable] = {}


def register_index(
    name: str,
    bands: List[str],
    description: str = "",
):
    """Decorator to register a custom spectral index.

    Parameters
    ----------
    name : str
        Unique index name (uppercase convention, e.g. 'CUSTOM_VI').
    bands : List[str]
        Spectral codes required by this index.
    description : str
        Human-readable description.
    """
    def decorator(func: Callable) -> Callable:
        _custom_indices[name] = {
            "func": func,
            "bands": bands,
            "description": description,
        }
        return func
    return decorator


def list_custom_indices() -> Dict[str, dict]:
    """Return all registered custom indices."""
    return dict(_custom_indices)


def compute_custom_index(
    name: str,
    band_arrays: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute a custom index by name.

    Parameters
    ----------
    name : str
        Registered index name.
    band_arrays : Dict[str, np.ndarray]
        Mapping from spectral code to band data.

    Returns
    -------
    np.ndarray
        Computed index.
    """
    if name not in _custom_indices:
        raise KeyError(f"Custom index '{name}' not found. Register it with @register_index.")

    entry = _custom_indices[name]
    kwargs = {code: band_arrays[code] for code in entry["bands"]}
    return entry["func"](**kwargs)
