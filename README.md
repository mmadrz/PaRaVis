# PaRaVis

> **Pa**rallel **Ra**o's Q **Vis**ualization

<p align="center">
  <img src="paravis/gui/theme/logo.png" alt="PaRaVis Logo" width="1200">
</p>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/paravis?style=flat-square&logo=pypi)](https://pypi.org/project/paravis/)
[![Python versions](https://img.shields.io/pypi/pyversions/paravis?style=flat-square&logo=python)](https://pypi.org/project/paravis/)
[![License](https://img.shields.io/pypi/l/paravis?style=flat-square)](https://github.com/mmadrz/paravis/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-369%20passed-success?style=flat-square)](https://github.com/mmadrz/paravis/actions)
[![Coverage](https://codecov.io/gh/mmadrz/PaRaVis/branch/main/graph/badge.svg)](https://codecov.io/gh/mmadrz/PaRaVis)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macOS-lightgrey?style=flat-square)](https://github.com/mmadrz/paravis)

</div>

A cross-platform Python library and desktop GUI for **spectral index computation**, **Rao's Q diversity analysis**, and **raster data visualization**. PaRaVis combines a professional PySide6 interface with a lightweight headless API — equally at home in interactive exploration and automated HPC batch pipelines.

---

## Features

- **200+ Spectral Indices** — NDVI, EVI, SAVI, NDWI, NBR, BAI, and more via [Spyndex](https://github.com/awesome-spectral-indices/spyndex). Register custom indices with the `@register_index` decorator.
- **Rao's Q Diversity** — Moving-window quadratic entropy with three backends: CPU ([NumPy](https://github.com/numpy/numpy)), multi-core parallel (`ProcessPoolExecutor`), and GPU-accelerated ([CuPy](https://github.com/cupy/cupy) + custom CUDA `RawKernel`).
- **6 Distance Metrics** — Euclidean, Manhattan, Chebyshev, Minkowski, Canberra, and Bray-Curtis — all supported on every backend.
- **Raster I/O** — Downsampling for large files, LZW-compressed GeoTIFF output, multi-band support.
- **Desktop GUI** — Three-panel layout, light/dark themes, splash screen, and full-screen mode for a better experience.
- **Headless API** — Zero-Qt core modules usable in scripts, notebooks, and HPC clusters.

---

## Quick Start

```bash
pip install paravis[gui]         # install with GUI
paravis                          # launch the desktop app
```

```python
from paravis.api import compute_indices, compute_rao_q

# Compute vegetation indices
results = compute_indices("scene.tif", indices=["NDVI", "EVI"])
print(results["NDVI"].mean())

# Rao's Q diversity (auto-selects GPU if available)
diversity = compute_rao_q("scene.tif", window_size=15, backend="auto")
```

---

## Installation

```bash
pip install paravis               # headless (scripts, notebooks, HPC)
pip install paravis[gui]          # + desktop GUI
pip install paravis[gpu]          # + GPU acceleration (see note below)
pip install paravis[gui,gpu,dev]  # everything

> **GPU prerequisite:** `paravis[gpu]` installs [CuPy](https://github.com/cupy/cupy), which requires the **NVIDIA CUDA Toolkit** on your system. Check your CUDA version with `nvcc --version`, then install the matching variant manually if `pip install paravis[gpu]` fails:
> ```bash
> pip install cupy-cuda12x   # for CUDA 12.x
> pip install cupy-cuda11x   # for CUDA 11.x
> ```
> See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for details.
```

---

### Dependencies


| Core | Optional |
|:-----|:---------|
| <a href="https://github.com/numpy/numpy"><img src="https://img.shields.io/badge/numpy-%3E%3D1.21-013243?style=flat-square&logo=numpy&logoColor=white" alt="numpy"></a> <a href="https://github.com/rasterio/rasterio"><img src="https://img.shields.io/badge/rasterio-%3E%3D1.3-2C8EBB?style=flat-square" alt="rasterio"></a> <a href="https://github.com/awesome-spectral-indices/spyndex"><img src="https://img.shields.io/badge/spyndex-%3E%3D0.4-FF6F00?style=flat-square" alt="spyndex"></a> <a href="https://github.com/matplotlib/matplotlib"><img src="https://img.shields.io/badge/matplotlib-%3E%3D3.5-11557C?style=flat-square" alt="matplotlib"></a> <a href="https://github.com/mwaskom/seaborn"><img src="https://img.shields.io/badge/seaborn-%3E%3D0.12-7DB0D6?style=flat-square" alt="seaborn"></a> <a href="https://github.com/giampaolo/psutil"><img src="https://img.shields.io/badge/psutil-%3E%3D5.0-003D7A?style=flat-square" alt="psutil"></a> | <a href="https://github.com/qtproject/pyside-pyside6"><img src="https://img.shields.io/badge/PySide6-%3E%3D6.5-41CD52?style=flat-square&logo=qt&logoColor=white" alt="PySide6"></a> <a href="https://github.com/cupy/cupy"><img src="https://img.shields.io/badge/cupy-%3E%3D12.0-E90000?style=flat-square" alt="cupy"></a> |

> **Note:** `rasterio` requires the GDAL system library. On Linux: `apt install libgdal-dev`. On Windows/macOS the pip wheel includes it.

---

## How Rao's Q Works

Rao's Quadratic Entropy measures spectral diversity within a local neighbourhood using a **moving-window approach**: for each window, the spectral distance between every pair of pixels is weighted by their relative abundances:

$$Q = \sum_{i=1}^{N} \sum_{j=1}^{N} d_{ij} \cdot p_i \cdot p_j$$

Where $N$ = total number of pixels within a window, $p_i$ and $p_j$ = relative abundances for pixel $i$ and $j$, and $d_{ij}$ = pairwise spectral distance between pixel $i$ and $j$.

The `simplify` parameter controls **input truncation** precision before species identification — higher precision means more distinct species detected, yielding a more detailed diversity map but at greater computational cost.

Six distance metrics are available: Euclidean, Manhattan, Chebyshev, Minkowski (tunable $p$), Canberra, and Bray-Curtis. All work on every backend.

> **→ Full theory, formulas, and parameter details in [docs/guide.md](docs/guide.md)**

---

## Backends

| Backend | Command | Use case |
|:--------|:--------|:---------|
| **CPU** ([NumPy](https://github.com/numpy/numpy)) | `backend="cpu"` | Small rasters (< 1000×1000 px), any hardware, limited RAM |
| **Parallel CPU** | `backend="parallel"` | Multi-core workstations, HPC nodes, medium-to-large rasters |
| **GPU** (CUDA kernel) | `backend="gpu"` or `"auto"` | Large rasters (> 1000×1000 px), NVIDIA GPUs, batch processing |

All three implement the **identical species-abundance formula**, producing bitwise-comparable results. The GPU uses three automatic paths: a fast shared-memory CUDA kernel, a per-thread fallback kernel, and a pure [CuPy](https://github.com/cupy/cupy) fallback if compilation fails.

> **→ Full backend comparison and GPU architecture in [docs/guide.md](docs/guide.md)**

---

## API Overview

```python
from paravis.api import (
    compute_indices,      # → dict of 2D arrays
    compute_index,        # → single 2D array
    compute_rao_q,        # → 2D Rao's Q map
    plot_raster,          # → (figure, axes)
    plot_comparison,      # → (figure, axes)
    list_available_indices, # → list of index names
)
```

```python
# Spectral indices with custom band mapping
compute_indices("sentinel2.tif", indices=["NDVI", "NDWI"],
                band_mapping={4: "R", 8: "N"})

# Rao's Q with full control
compute_rao_q("scene.tif", window_size=11, na_tolerance=0.3,
              backend="parallel", n_workers=8,
              simplify=2, distance_metric="braycurtis")

# Read / write rasters
from paravis.core.raster import read_raster, write_geotiff
data, transform, crs = read_raster("input.tif")
write_geotiff(result, "output.tif", transform=transform, crs=crs)
```

> **→ Full API reference with all parameters in [docs/guide.md](docs/guide.md)**

---

## Desktop GUI

```bash
paravis                    # launch
python -m paravis.gui.app  # or via module
```

| Left | Centre | Right |
|:-----|:-------|:------|
| Browse 200+ spectral indices from Spyndex, configure band-to-raster mapping, set output options and parallel workers, then run batch index computation with progress tracking. | Compute Rao's Quadratic Entropy diversity maps in single or batch mode. Choose GPU/CPU backend, configure distance metric, window size, simplify precision, and NA tolerance with real-time speed feedback. | Load computed rasters and generate individual plots, statistics (box/histogram/KDE/violin), difference heatmaps, side-by-side split views, or time-series GIF animations. Customise colormaps, DPI, and output format. |

Light/dark themes, keyboard shortcuts, persistent settings.

> **→ Full GUI walkthrough in [docs/guide.md](docs/guide.md)**

---

## System Requirements

- **Python** ≥ 3.9
- **RAM**: 4 GB minimum, 8 GB recommended for GUI
- **GPU** (optional): NVIDIA with Compute Capability 6.0+, driver ≥ 450.0, [CuPy](https://github.com/cupy/cupy) ≥ 12.0

---

## Citation

If PaRaVis contributes to your research, please cite:

```bibtex
@article{FATHI2024102739,
  title = {PaRaVis: An automatic Python graphical package for ensemble analysis of plant beta diversity using remote sensing proxies},
  journal = {Ecological Informatics},
  volume = {82},
  pages = {102739},
  year = {2024},
  issn = {1574-9541},
  doi = {10.1016/j.ecoinf.2024.102739},
  url = {https://www.sciencedirect.com/science/article/pii/S1574954124002814},
  author = {Mohammad Reza Fathi and Hooman Latifi and Hamed Gholizadeh and Siddhartha Khare},
}
```
