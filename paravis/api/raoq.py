"""
Public API for Rao's Q diversity computation.

Implements the ecological Rao's quadratic entropy formula:

    Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

where each unique spectral profile in a window is treated as a
"species" (i), dᵢⱼ is the spectral distance between species i and j,
and pᵢ, pⱼ are their relative abundances (proportion of pixels
belonging to each unique profile).
"""
from typing import Optional

import numpy as np

from paravis.core.raster import read_raster
from paravis.core.raoq import (
    compute_rao_q as _compute_rao_q_cpu,
    RaoQConfig,
    RaoQResult,
)
from paravis.core.raoq.gpu import compute_rao_q_gpu, is_gpu_available


def compute_rao_q(
    raster_path: str,
    window_size: int = 15,
    step_size: int = 1,
    na_tolerance: float = 0.3,
    backend: str = "auto",
    n_workers: int = 4,
    max_pixels: int = 500_000,
    simplify: int = 2,
    distance_metric: str = "euclidean",
    p_minkowski: int = 2,
) -> np.ndarray:
    """Compute Rao's Q diversity from a raster file.

    For each moving window, unique spectral profiles are identified
    as "species", and Rao's Q is computed as:

        Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

    where pᵢ is the relative abundance (frequency) of each unique
    spectral profile in the window.

    Parameters
    ----------
    raster_path : str
        Path to a multi-band GeoTIFF.
    window_size : int
        Size of the moving window (odd number).
    step_size : int
        Step between windows (default 1).
    na_tolerance : float
        Maximum allowed fraction of NA pixels (0.0 to 1.0).
    backend : str
        'auto', 'cpu', 'gpu', or 'parallel'. If 'auto', tries GPU first.
    n_workers : int
        Number of CPU workers for 'parallel' backend.
    max_pixels : int
        Maximum pixels to read (auto-downsamples beyond this).
    simplify : int
        Number of decimal places to round the output to (0 = integers, default 2).
    distance_metric : str
        Distance metric: 'euclidean', 'manhattan', 'chebyshev', or 'minkowski'.
    p_minkowski : int
        Exponent for Minkowski distance (ignored for other metrics).

    Returns
    -------
    np.ndarray
        2D array of Rao's Q diversity values.
    """
    config = RaoQConfig(
        window_size=window_size,
        step_size=step_size,
        na_tolerance=na_tolerance,
        n_workers=n_workers,
        simplify=simplify,
        distance_metric=distance_metric,
        p_minkowski=p_minkowski,
    )

    # Read raster
    data_3d, _, _ = read_raster(raster_path, max_pixels=max_pixels)
    if data_3d.ndim == 2:
        data_3d = data_3d[np.newaxis, :, :]

    # Choose backend
    if backend == "auto":
        if is_gpu_available():
            result = compute_rao_q_gpu(data_3d, config)
        else:
            result = _compute_rao_q_cpu(data_3d, config)
    elif backend == "gpu":
        result = compute_rao_q_gpu(data_3d, config)
    elif backend == "parallel":
        from paravis.core.raoq import compute_rao_q_parallel
        result = compute_rao_q_parallel(data_3d, config)
    else:
        result = _compute_rao_q_cpu(data_3d, config)

    return result
