"""
Public API for raster visualization.

Provides quick-plot functions for use in scripts and notebooks.
"""
from typing import Optional, List, Tuple
import numpy as np

from paravis.core.raster import read_raster, normalize_data


def plot_raster(
    raster_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    max_pixels: int = 1_000_000,
    show_colorbar: bool = True,
):
    """Quick-plot a raster file.

    Parameters
    ----------
    raster_path : str
        Path to a raster file.
    title : str, optional
        Plot title.
    cmap : str
        Matplotlib colormap name.
    figsize : Tuple[int, int]
        Figure size in inches.
    max_pixels : int
        Maximum pixels to read.
    show_colorbar : bool
        Whether to show a colorbar.
    """
    import matplotlib.pyplot as plt

    data, transform, _ = read_raster(raster_path, max_pixels=max_pixels)
    if data.ndim == 3:
        data = data[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(title or raster_path.split("/")[-1])
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    return fig, ax


def plot_comparison(
    file_paths: List[str],
    labels: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None,
    max_pixels: int = 500_000,
):
    """Plot multiple rasters side by side for comparison.

    Parameters
    ----------
    file_paths : List[str]
        Paths to raster files.
    labels : List[str], optional
        Labels for each subplot.
    cmap : str
        Colormap name.
    figsize : Tuple[int, int], optional
        Figure size.
    max_pixels : int
        Maximum pixels to read.
    """
    import matplotlib.pyplot as plt

    n = len(file_paths)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    if labels is None:
        labels = [p.split("/")[-1] for p in file_paths]

    for i, (path, label) in enumerate(zip(file_paths, labels)):
        data, _, _ = read_raster(path, max_pixels=max_pixels)
        if data.ndim == 3:
            data = data[0]
        im = axes[0, i].imshow(data, cmap=cmap, interpolation="nearest")
        axes[0, i].set_title(label)
        plt.colorbar(im, ax=axes[0, i], shrink=0.75)

    plt.tight_layout()
    return fig, axes
