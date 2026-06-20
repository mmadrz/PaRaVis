"""
Raster utility functions — coordinate conversion, colormap helpers.
"""
from typing import Optional, Tuple


def decimal_degrees_to_dms(dd: float) -> Tuple[int, int, float]:
    """Convert decimal degrees to degrees-minutes-seconds.

    Parameters
    ----------
    dd : float
        Decimal degrees value.

    Returns
    -------
    Tuple[int, int, float]
        (degrees, minutes, seconds)
    """
    sign = -1 if dd < 0 else 1
    abs_dd = abs(dd)
    degrees = int(abs_dd)
    minutes = int((abs_dd - degrees) * 60)
    seconds = ((abs_dd - degrees) * 60 - minutes) * 60
    return sign * degrees, minutes, seconds


def get_cmap_with_nan(
    cmap_name: str,
    bad_color: str = "white",
    bad_alpha: float = 1.0,
):
    """Get a matplotlib colormap with NaN values rendered as a specific color.

    Parameters
    ----------
    cmap_name : str
        Name of the matplotlib colormap.
    bad_color : str
        Color name for NaN pixels.
    bad_alpha : float
        Alpha for NaN pixels.

    Returns
    -------
    Colormap or None
        Modified colormap, or None if matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt

        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad(color=bad_color, alpha=bad_alpha)
        return cmap
    except Exception:
        return None
