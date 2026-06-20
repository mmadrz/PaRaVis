"""
Rao's Q diversity computation — CPU implementation.

Implements the ecological Rao's quadratic entropy formula:

    Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

where each unique spectral profile in a window is treated as a
"species" (i), dᵢⱼ is the spectral distance between species i and j,
and pᵢ, pⱼ are their relative abundances (proportion of pixels
belonging to each unique profile).

Supports multiple distance metrics:
  - ``"euclidean"``     — L2 norm (default)
  - ``"manhattan"``     — L1 norm
  - ``"chebyshev"``     — L∞ norm (max absolute difference)
  - ``"minkowski"``     — Lp norm (generalised, p via ``p_minkowski``)
  - ``"canberra"``      — weighted L1, sensitive near zero
  - ``"braycurtis"``    — ecological dissimilarity, bounded [0, 1]
Pure NumPy — no Qt dependency. Supports both single-threaded and
multi-process parallel execution.
"""
import time
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .models import RaoQConfig, RaoQResult


# ---------------------------------------------------------------------------
# Shared memory for parallel workers (fork-based)
# ---------------------------------------------------------------------------
# When using ``ProcessPoolExecutor`` with ``fork``, each worker inherits the
# parent's address space but then *unpickles* task arguments, creating a fresh
# copy of every numpy array in the argument tuple.  To avoid duplicating the
# full raster in every worker we use ``initializer`` — the reference is set
# once after fork (no pickle), and ``raster_data`` is excluded from task args.
_shared_raster_data = None


def _init_worker(raster_data_ref):
    """Worker-process initialiser — stores a reference to the raster data.

    Called once in each child immediately after ``fork()``, before any
    tasks are received.  ``raster_data_ref`` is the already-existing array
    inherited from the parent (CoW), so this is just a pointer assignment
    — no memory is copied.
    """
    global _shared_raster_data
    _shared_raster_data = raster_data_ref


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _compute_pairwise_distances(
    profiles: np.ndarray,
    metric: str = "euclidean",
    p: int = 2,
) -> np.ndarray:
    """Compute pairwise distances between an array of spectral profiles.

    Parameters
    ----------
    profiles : np.ndarray
        2D array of shape (n_species, n_bands).
    metric : str
        One of ``"euclidean"``, ``"manhattan"``, ``"chebyshev"``,
        ``"minkowski"``, ``"canberra"``,
        ``"braycurtis"``.
    p : int
        Exponent for Minkowski distance (ignored for other metrics).

    Returns
    -------
    np.ndarray
        1D array of upper-triangular pairwise distances, ordered
        as (0,1), (0,2), …, (1,2), (1,3), … (matching the iteration
        order of the Rao's Q loop).
    """
    n = profiles.shape[0]
    # Collect all pairwise differences at once
    # i_idx, j_idx are the upper-triangular indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = profiles[i_idx] - profiles[j_idx]  # (n_pairs, n_bands)

    if metric == "euclidean":
        return np.sqrt(np.sum(diffs ** 2, axis=1))
    elif metric == "manhattan":
        return np.sum(np.abs(diffs), axis=1)
    elif metric == "chebyshev":
        return np.max(np.abs(diffs), axis=1)
    elif metric == "minkowski":
        return np.sum(np.abs(diffs) ** p, axis=1) ** (1.0 / p)
    elif metric == "canberra":
        # Σ |a_k - b_k| / (|a_k| + |b_k|) — weighted L1
        a = profiles[i_idx]
        b = profiles[j_idx]
        denom = np.abs(a) + np.abs(b)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denom > 1e-15, np.abs(diffs) / denom, 0.0)
        return np.sum(ratio, axis=1)
    elif metric == "braycurtis":
        # Σ |a_k - b_k| / Σ (a_k + b_k) — ecological dissimilarity
        a = profiles[i_idx]
        b = profiles[j_idx]
        num = np.sum(np.abs(diffs), axis=1)
        denom = np.sum(np.abs(a) + np.abs(b), axis=1)
        return np.where(denom > 1e-15, num / denom, 0.0)
    else:
        raise ValueError(f"Unknown distance metric: {metric!r}")


# ---------------------------------------------------------------------------
# Padded-strip builder (batched processing helper)
# ---------------------------------------------------------------------------

def _build_padded_strip(
    raster_data: np.ndarray,
    batch_start: int,
    batch_end: int,
    half: int,
    simplify: int = 0,
) -> np.ndarray:
    """Build a padded strip from the original raster for a batch of rows.

    The result is equivalent to taking the slice
    ``padded_full[:, batch_start:batch_end + 2*half, :]`` from the
    full NaN-padded raster, but without constructing the full padded
    array — only the rows needed for this batch plus context are
    extracted and padded.  This keeps peak memory low for large rasters.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape ``(n_bands, height, width)`` — original raster.
    batch_start, batch_end : int
        Row range for this batch (``0 ≤ batch_start < batch_end ≤ height``).
    half : int
        Half the window size (``window_size // 2``).
    simplify : int
        Truncation precision (see ``RaoQConfig.simplify``).  ``0`` skips
        truncation.

    Returns
    -------
    np.ndarray
        Padded strip of shape ``(n_bands, n_rows + 2*half, width + 2*half)``,
        where ``n_rows = batch_end - batch_start``.
    """
    n_bands, height, width = raster_data.shape
    n_rows = batch_end - batch_start
    ws = 2 * half + 1

    # ---- Data rows to extract (including context) -----------------------
    top_context = min(half, batch_start)
    bottom_context = min(half, height - batch_end)

    strip_start = batch_start - top_context
    strip_end = batch_end + bottom_context

    strip = raster_data[:, strip_start:strip_end, :]
    if strip.dtype != np.float32:
        strip = strip.astype(np.float32)

    # ---- NaN padding where context is unavailable (image edges) ---------
    pad_top = half - top_context
    pad_bottom = half - bottom_context

    padded = np.pad(
        strip,
        ((0, 0), (pad_top, pad_bottom), (half, half)),
        mode="constant",
        constant_values=np.nan,
    )

    # ---- Truncate precision (no rounding) --------------------------------
    if simplify:
        fac = 10 ** simplify
        padded = np.trunc(padded * fac) / fac

    return padded


# ---------------------------------------------------------------------------
# Core window computation
# ---------------------------------------------------------------------------

def _compute_rao_q_window(
    window_data: np.ndarray,
    valid_mask: np.ndarray,
    na_tolerance: float,
    distance_metric: str = "euclidean",
    p_minkowski: int = 2,
) -> float:
    """Compute Rao's Q for a single window using the species-abundance approach.

    For each window:
      1. Find unique spectral profiles (species) among valid pixels.
      2. Compute relative abundance pᵢ = countᵢ / total for each species.
      3. Compute pairwise distances dᵢⱼ between species profiles
         using the selected metric.
      4. Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

    Parameters
    ----------
    window_data : np.ndarray
        2D array of shape (n_pixels, n_bands).
    valid_mask : np.ndarray
        Boolean mask of valid (non-NA) pixels.
    na_tolerance : float
        Maximum allowed fraction of NA pixels.
    distance_metric : str
        Distance metric to use (default ``"euclidean"``).
    p_minkowski : int
        Exponent for Minkowski distance (ignored for other metrics).

    Returns
    -------
    float
        Rao's Q value, or NaN if too many NAs.
    """
    n_pixels_total = window_data.shape[0]
    valid_count = np.sum(valid_mask)
    valid_ratio = valid_count / n_pixels_total

    if valid_ratio < (1.0 - na_tolerance) or valid_count < 2:
        return np.nan

    # Filter to valid pixels only
    valid_data = window_data[valid_mask]

    # Find unique spectral profiles (species) and their frequencies
    unique_profiles, counts = np.unique(valid_data, axis=0, return_counts=True)
    n_species = len(unique_profiles)

    # Only one species → no diversity
    if n_species < 2:
        return 0.0

    # Relative abundances (float32 to match GPU precision)
    p = counts.astype(np.float32) / valid_count

    # Pairwise distances between unique profiles
    distances = _compute_pairwise_distances(
        unique_profiles, metric=distance_metric, p=p_minkowski
    )

    # Reconstruct full distance matrix from upper-triangular values
    # (float32 to match GPU kernel precision)
    dist_matrix = np.zeros((n_species, n_species), dtype=np.float32)
    idx = 0
    for i in range(n_species):
        for j in range(i + 1, n_species):
            dist_matrix[i, j] = distances[idx]
            dist_matrix[j, i] = distances[idx]
            idx += 1

    # Rao's Q: Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ
    total = 0.0
    for i in range(n_species):
        for j in range(n_species):
            total += dist_matrix[i, j] * p[i] * p[j]

    return total


def compute_rao_q(
    raster_data: np.ndarray,
    config: Optional[RaoQConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute Rao's Q diversity for a raster using a moving window.

    For each window, unique spectral profiles are identified as
    "species", and Rao's Q is computed as:

        Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

    where pᵢ is the relative abundance (frequency) of each unique
    spectral profile in the window.

    **Memory efficiency**: processes the raster in row-strips (batches)
    so that only a few rows plus the window context are padded at any
    time, rather than the full raster.  This avoids OOM on large rasters.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    config : RaoQConfig, optional
        Configuration parameters.
    progress_callback : Callable[[int, int], None], optional
        Called with (current, total) after each row is processed.

    Returns
    -------
    np.ndarray
        2D array of Rao's Q values (same spatial extent as input).
    """
    if config is None:
        config = RaoQConfig()

    n_bands, height, width = raster_data.shape
    ws = config.window_size
    half = ws // 2
    n_pixels = ws * ws

    result = np.full((height, width), np.nan, dtype=np.float32)
    total_windows = height * width

    # ---- Batched (strip) processing -- like the GPU batched approach ----
    # Process a few rows per iteration to keep peak memory low.
    batch_rows = max(1, config.cpu_batch_size // width)
    batch_rows = min(batch_rows, height)

    for batch_start in range(0, height, batch_rows):
        batch_end = min(batch_start + batch_rows, height)
        n_rows = batch_end - batch_start

        # Build the padded strip for this batch only
        padded = _build_padded_strip(
            raster_data, batch_start, batch_end, half,
            simplify=config.simplify,
        )

        # Process row by row — window-by-window extraction
        for local_row in range(n_rows):
            row = batch_start + local_row

            # View of all windows in this row (no contiguous copy)
            row_windows = sliding_window_view(
                padded[:, local_row:local_row + ws, :], ws, axis=-1
            )

            # Extract each window individually to avoid a huge
            # (width, ws*ws, n_bands) contiguous allocation
            for col in range(width):
                window = row_windows[:, :, col, :]  # (n_bands, ws, ws) — view
                flat = np.ascontiguousarray(
                    window.transpose(1, 2, 0).reshape(n_pixels, n_bands)
                )  # (n_pixels, n_bands) — small contiguous copy
                valid = ~np.any(np.isnan(flat), axis=1)
                value = _compute_rao_q_window(
                    flat, valid, config.na_tolerance,
                    distance_metric=config.distance_metric,
                    p_minkowski=config.p_minkowski,
                )
                result[row, col] = value

        if progress_callback is not None:
            progress_callback(batch_end * width, total_windows)

    return result


def _process_chunk(args):
    """Helper for parallel processing — processes a set of rows.

    Reads ``raster_data`` from the module-level ``_shared_raster_data``
    (set once per worker by :func:`_init_worker`) to avoid pickling the
    full array with every task.  Sub-batches within the chunk using
    ``cpu_batch_size`` so each worker only pads a few rows at a time,
    keeping peak memory low even with many workers.
    """
    config, row_range = args
    raster_data = _shared_raster_data
    n_bands, height, width = raster_data.shape
    ws = config.window_size
    half = ws // 2
    n_pixels = ws * ws

    chunk_rows = len(row_range)
    chunk_result = np.full((chunk_rows, width), np.nan, dtype=np.float32)

    # Sub-batch within this chunk — same logic as compute_rao_q
    batch_rows = max(1, config.cpu_batch_size // width)
    batch_rows = min(batch_rows, chunk_rows)

    for batch_offset in range(0, chunk_rows, batch_rows):
        batch_local_start = batch_offset
        batch_local_end = min(batch_offset + batch_rows, chunk_rows)
        global_start = row_range[batch_local_start]
        global_end = row_range[batch_local_end - 1] + 1

        # Build the padded strip for this sub-batch only
        padded = _build_padded_strip(
            raster_data, global_start, global_end, half,
            simplify=config.simplify,
        )

        # Process each row in the sub-batch
        for local_idx in range(batch_local_start, batch_local_end):
            row_in_padded = local_idx - batch_local_start

            # View of all windows for this row (no contiguous copy)
            row_windows = sliding_window_view(
                padded[:, row_in_padded:row_in_padded + ws, :], ws, axis=-1
            )

            # Extract each window individually to avoid a huge
            # (width, ws*ws, n_bands) contiguous allocation
            for col in range(width):
                window = row_windows[:, :, col, :]  # (n_bands, ws, ws) — view
                flat = np.ascontiguousarray(
                    window.transpose(1, 2, 0).reshape(n_pixels, n_bands)
                )  # (n_pixels, n_bands) — small contiguous copy
                valid = ~np.any(np.isnan(flat), axis=1)
                value = _compute_rao_q_window(
                    flat, valid, config.na_tolerance,
                    distance_metric=config.distance_metric,
                    p_minkowski=config.p_minkowski,
                )
                chunk_result[local_idx, col] = value

    return row_range, chunk_result


def compute_rao_q_parallel(
    raster_data: np.ndarray,
    config: Optional[RaoQConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute Rao's Q using multiple CPU processes.

    For each window, unique spectral profiles are identified as
    "species", and Rao's Q is computed as:

        Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

    where pᵢ is the relative abundance (frequency) of each unique
    spectral profile in the window.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    config : RaoQConfig, optional
        Configuration parameters.
    progress_callback : Callable[[int, int], None], optional
        Called with (windows_done, total_windows) after each chunk completes.

    Returns
    -------
    np.ndarray
        2D array of Rao's Q values.
    """
    if config is None:
        config = RaoQConfig()

    height, width = raster_data.shape[1], raster_data.shape[2]
    total_windows = height * width
    rows_per_chunk = max(1, height // config.n_workers)
    row_ranges = [
        list(range(i, min(i + rows_per_chunk, height)))
        for i in range(0, height, rows_per_chunk)
    ]

    result = np.full((height, width), np.nan, dtype=np.float32)

    # Build task args WITHOUT raster_data — the workers read it from
    # _shared_raster_data (set by _init_worker after fork).  This avoids
    # pickling the full array with every task.
    task_args = [(config, rows) for rows in row_ranges]

    windows_done = 0
    with ProcessPoolExecutor(
        max_workers=config.n_workers,
        initializer=_init_worker,
        initargs=(raster_data,),
    ) as executor:
        futures = {executor.submit(_process_chunk, arg): arg for arg in task_args}
        for future in as_completed(futures):
            rows, chunk = future.result()
            for local_idx, row in enumerate(rows):
                result[row, :] = chunk[local_idx, :]
            windows_done += len(rows) * width
            if progress_callback is not None:
                progress_callback(windows_done, total_windows)

    return result
