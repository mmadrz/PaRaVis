"""
Memory-efficient raster reading and data normalization.

These functions have zero Qt dependency — usable in any Python context.
"""
from typing import Optional, Tuple

import numpy as np
import rasterio


def read_raster(
    file_path: str,
    max_pixels: int = 1_000_000,
    band: Optional[int] = None,
) -> Tuple[np.ndarray, object, object]:
    """Read a raster file with automatic downsampling for large files.

    Parameters
    ----------
    file_path : str
        Path to the raster file.
    max_pixels : int
        Maximum number of pixels to read; will downsample beyond this.
    band : int, optional
        Band index to read (1-based). If None, reads ALL bands.

    Returns
    -------
    data : np.ndarray
        2D array (single band) or 3D array (all bands) of raster data
        (may be downsampled). Nodata values are converted to NaN.
    transform : affine.Affine or similar
        Geo-transform of the (possibly downsampled) raster.
    crs : object
        Coordinate reference system.
    """
    with rasterio.open(file_path) as src:
        nodata = src.nodata
        h, w = src.height, src.width
        total_pixels = h * w

        if total_pixels > max_pixels:
            factor = int(np.sqrt(total_pixels / max_pixels))
            if factor > 1:
                out_shape = (h // factor, w // factor)
                data = src.read(band, out_shape=out_shape)
            else:
                data = src.read(band)
        else:
            data = src.read(band)

        # Convert nodata to NaN for proper handling
        if nodata is not None:
            data = data.astype(np.float64, copy=False)
            data[data == nodata] = np.nan
        else:
            # Ensure float type for consistent handling
            data = data.astype(np.float64, copy=False)

        transform = src.transform
        crs = src.crs

    return data, transform, crs


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range — memory efficient.

    Parameters
    ----------
    data : np.ndarray
        Input array (may contain NaN).

    Returns
    -------
    np.ndarray
        Normalized array in [0, 1] range.
    """
    if data.size == 0:
        return data

    # Ensure float type for consistent handling
    data = data.astype(np.float64, copy=False)

    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
        return np.zeros_like(data, dtype=np.float64)

    return (data - min_val) / (max_val - min_val)


def downsample_data(data: np.ndarray, max_pixels: int = 1_000_000) -> np.ndarray:
    """Downsample large arrays to reduce memory usage.

    Parameters
    ----------
    data : np.ndarray
        2D input array.
    max_pixels : int
        Maximum allowed pixels.

    Returns
    -------
    np.ndarray
        Downsampled array.
    """
    h, w = data.shape
    total_pixels = h * w

    if total_pixels <= max_pixels:
        return data

    factor = int(np.sqrt(total_pixels / max_pixels))
    if factor < 1:
        factor = 1

    downsampled = data[::factor, ::factor]
    print(
        f"Downsampled from {h}x{w} ({total_pixels / 1e6:.1f}M pixels) "
        f"to {downsampled.shape[0]}x{downsampled.shape[1]} "
        f"({downsampled.size / 1e6:.1f}M pixels)"
    )
    return downsampled
