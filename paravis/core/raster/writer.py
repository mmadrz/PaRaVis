"""
Raster writing utilities — save NumPy arrays as GeoTIFF files.
"""
from typing import Optional

import numpy as np
import rasterio


def write_geotiff(
    data: np.ndarray,
    path: str,
    transform: Optional[object] = None,
    crs: Optional[object] = None,
    nodata: Optional[float] = None,
    compress: bool = True,
    dtype: Optional[str] = None,
) -> str:
    """Write a 2D array to a GeoTIFF file.

    Parameters
    ----------
    data : np.ndarray
        2D array to write.
    path : str
        Output file path.
    transform : object, optional
        Affine transform.
    crs : object, optional
        Coordinate reference system.
    nodata : float, optional
        No-data value.
    compress : bool
        Enable LZW compression.
    dtype : str, optional
        Output data type (e.g. 'float32'). Inferred from data if None.

    Returns
    -------
    str
        Path to the written file.
    """
    if dtype is None:
        dtype = str(data.dtype)

    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": dtype,
        "height": data.shape[0],
        "width": data.shape[1],
        "crs": crs,
        "transform": transform,
        "compress": "LZW" if compress else None,
        "nodata": nodata,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)

    return path
