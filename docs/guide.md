# PaRaVis v2.0 — User Guide 

> **Pa**rallel **Ra**o's Q **Vis**ualization and Analysis

A cross-platform desktop application and Python library for spectral index computation, Rao's Q diversity analysis, and raster data visualisation. Built with PySide6, NumPy, Rasterio, Spyndex, and CuPy.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Launching the GUI](#3-launching-the-gui)
4. [Interface Walkthrough](#4-interface-walkthrough)
   - 4.1 [Menu Bar](#41-menu-bar)
   - 4.2 [Status Bar](#42-status-bar)
   - 4.3 [Left Panel — Indices](#43-left-panel--indices)
   - 4.4 [Centre Panel — Rao's Q](#44-centre-panel--raos-q)
   - 4.5 [Right Panel — Visualisation](#45-right-panel--visualisation)
5. [Workflows](#5-workflows)
   - 5.1 [Computing Spectral Indices](#51-computing-spectral-indices)
   - 5.2 [Computing Rao's Q Diversity](#52-computing-raos-q-diversity)
   - 5.3 [Batch Processing Rao's Q](#53-batch-processing-raos-q)
   - 5.4 [Visualising Rasters](#54-visualising-rasters)
6. [Dialogs](#6-dialogs)
   - 6.1 [Band Mapping](#61-band-mapping)
   - 6.2 [Constants Editor](#62-constants-editor)
   - 6.3 [Index Table (Expanded)](#63-index-table-expanded)
   - 6.4 [Settings](#64-settings)
   - 6.5 [About](#65-about)
7. [Headless & Script Usage](#7-headless--script-usage)
8. [GPU Acceleration](#8-gpu-acceleration)
9. [Keyboard Shortcuts](#9-keyboard-shortcuts)
10. [Troubleshooting](#10-troubleshooting)

---
<details>
<summary><h2>1. Overview</h2></summary>

PaRaVis provides a complete environment for remote-sensing raster analysis:

- **Spectral Indices** — Compute 100+ published vegetation, water, burn, and soil indices via the Spyndex library. Register your own with the `@register_index` decorator. Built-in band-mapping presets for Landsat 8/9, Sentinel-2, and Sentinel-1, plus fully custom configurations.
- **Rao's Q Diversity** — Rao's quadratic entropy (Q) over a moving window using a **species-abundance approach**. Three backends: single-threaded CPU (NumPy), multi-core parallel (`ProcessPoolExecutor`), and GPU-accelerated (CuPy with a custom CUDA `RawKernel`). Six distance metrics available on every backend.
- **Raster I/O** — Read GeoTIFF, ERDAS IMAGINE, ENVI, and other GDAL-supported formats. Automatic downsampling for large files. Write LZW-compressed GeoTIFFs.
- **Visualisation** — Five plot modes: individual, statistics (histogram, box, density), difference + heatmap, split-view comparison, and time series with GIF animation export.
- **Desktop GUI** — Resizable three-panel layout, drag-and-drop file loading, light/dark themes, splash screen, and persistent settings.
- **Headless API** — All core modules are Qt-free and can be used in Jupyter notebooks, Python scripts, or HPC batch jobs.

---
</details>

<details>
<summary><h2>1.1 Rao's Q Theory</h2></summary>

### What Is Rao's Q?

Rao's Quadratic Entropy (also called Rao's Q) is a measure of **spectral variation within a local neighbourhood**. Unlike simple spectral indices (e.g., NDVI) which measure a single value at each pixel, Rao's Q quantifies how much pixel values vary inside a moving window — capturing spectral heterogeneity across the image.

This makes it valuable for:
- **Remote sensing**: Detecting sub-pixel mixing, urban complexity, and vegetation structural variation.
- **Change detection**: Identifying areas where spectral heterogeneity changes over time.
- **Land cover analysis**: Distinguishing homogeneous areas (e.g., water, bare soil) from heterogeneous ones (e.g., urban edges, ecotones).

### How Rao's Q Works

The algorithm slides a window across the image and, for each position, measures how spectrally varied the pixels inside are. To do this efficiently, **pixels with identical spectral values are grouped together** — a window with 225 pixels might only have 10 distinct value combinations, so only 45 pairwise comparisons are needed instead of 25,200. The result is a measure of the average spectral difference between any two pixels in that window: higher values mean more spectral variation, lower values mean more uniformity.

### The `simplify` Parameter

The `simplify` parameter controls how precisely pixel values are compared before grouping them. Values are **truncated** (not rounded) to a set number of decimal places — keeping more decimals means finer distinctions are preserved.

**GUI range:** 0–6, **default:** 2.

| Value | Meaning | Effect on distinct value groups | Computation cost |
|-------|---------|---------------------------------|-----------------|
| **0** | Off — no truncation | Nearly every pixel is unique | Highest (full precision) |
| **1** | Keep 1 decimal | Most aggressive grouping | Lowest (fewest groups) |
| **2** (default) | Keep 2 decimals | Good balance | Low |
| **3** | Keep 3 decimals | Subtle grouping | Low–moderate |
| **4** | Keep 4 decimals | Minimal grouping | Moderate |
| **5** | Keep 5 decimals | Very fine distinctions | High |
| **6** | Keep 6 decimals | Near full precision | Highest (near full precision) |

**How it works in practice:** a pixel value of `0.1234567` becomes `0.12` at `simplify=2` (truncated, not rounded). Two pixels with values `0.1234567` and `0.1235567` would be grouped as having the same value at level 2 (both become `0.12`), but would be treated as distinct at level 4 (becoming `0.1234` vs `0.1235`).

**Key point:** `simplify=0` skips truncation entirely and keeps full float32 precision. Higher values mean more decimal places are kept, so more spectral variation is detected, producing a more detailed heterogeneity map at the cost of slower computation.

**When to adjust:**
- **0** — no truncation; use only when you need absolute precision (slowest). Often indistinguishable from 6 in practice.
- **1** — maximum speed; ideal for noisy data, large rasters, or exploratory runs.
- **2 (default)** — recommended starting point for most analyses.
- **3–6** — progressively finer spectral discrimination; use with clean, well-calibrated data or final production maps.

### The `na_tolerance` Parameter

Controls NaN tolerance per window:

- `na_tolerance=0.0`: All pixels must be valid. Any NaN → result is NaN.
- `na_tolerance=0.3` (default): Up to 30% NaN allowed.
- `na_tolerance=1.0`: Any number of NaN allowed (as long as ≥2 valid pixels exist).

---
</details>

<details>
<summary><h2>1.2 Distance Metrics</h2></summary>

All six metrics are supported on **every backend** (CPU, parallel CPU, GPU CUDA kernel, and pure CuPy fallback). They are defined for two spectral profiles $a = (a_1, a_2, ..., a_B)$ and $b = (b_1, b_2, ..., b_B)$ where $B$ is the number of bands.

### Euclidean (L2 Norm)

$$d_{ij} = \sqrt{\sum_{k=1}^{B} (a_k - b_k)^2}$$

The default metric. Standard straight-line distance in spectral space. Sensitive to overall magnitude differences.

### Manhattan (L1 Norm)

$$d_{ij} = \sum_{k=1}^{B} |a_k - b_k|$$

Sum of absolute differences. Less sensitive to extreme values than Euclidean.

### Chebyshev (L∞ Norm)

$$d_{ij} = \max_{k} |a_k - b_k|$$

Maximum single-band difference. Emphasises the band with the largest change.

### Minkowski (Lp Norm)

$$d_{ij} = \left( \sum_{k=1}^{B} |a_k - b_k|^p \right)^{1/p}$$

Generalised norm. $p=1$ = Manhattan, $p=2$ = Euclidean. Tunable via the `p_minkowski` parameter.

### Canberra

$$d_{ij} = \sum_{k=1}^{B} \frac{|a_k - b_k|}{|a_k| + |b_k|}$$

Weighted by band magnitude. Each band contributes at most 1 to the total. **Sensitive near zero** — small differences in small values can contribute significantly. Useful when bands have different dynamic ranges.

### Bray-Curtis

$$d_{ij} = \frac{\sum_{k=1}^{B} |a_k - b_k|}{\sum_{k=1}^{B} (a_k + b_k)}$$

Ecological dissimilarity index. Bounded in $[0, 1]$. **0** = identical profiles, **1** = completely disjoint. Widely used in community ecology and vegetation analysis.

### CPU Implementation

On CPU, distances are computed vectorised using NumPy:

```python
i_idx, j_idx = np.triu_indices(n, k=1)       # upper-triangular pairs
diffs = profiles[i_idx] - profiles[j_idx]

if metric == "euclidean":
    return np.sqrt(np.sum(diffs ** 2, axis=1))
elif metric == "manhattan":
    return np.sum(np.abs(diffs), axis=1)
# ... etc
```

### GPU Implementation (CUDA Kernel)

The same math is implemented in CUDA C within the `RawKernel`. Each distance metric is a branch inside a loop over species pairs:

```cuda
for (int i = start_i; i < end_i; i++) {
    for (int j = i + 1; j < n_species; j++) {
        if (metric_id == 0) {        // euclidean
            for (int b = 0; b < n_bands; b++) {
                float diff = A - B;
                dist += diff * diff;
            }
            dist = sqrtf(dist);
        } else if (metric_id == 1) { // manhattan
            for (int b = 0; b < n_bands; b++)
                dist += fabsf(A - B);
        }
        // ... etc
        partial += dist * ci * cj;
    }
}
```

---
</details>

<details>
<summary><h2>2. Installation</h2></summary>

```bash
# Core library only (headless — for scripts and notebooks)
pip install paravis

# With the desktop GUI
pip install paravis[gui]

# With GPU acceleration (requires NVIDIA GPU + CUDA)
pip install paravis[gpu]

# Full installation (GUI + GPU + dev tools)
pip install paravis[gui,gpu,dev]

# From source
git clone https://github.com/mmadrz/paravis.git
cd paravis
pip install -e .[gui,gpu,dev]
```

### Dependencies

| Package      | Minimum | Required | Purpose                    | Notes |
|--------------|---------|----------|----------------------------|-------|
| **numpy**    | ≥1.21   | Yes      | Core numerical arrays      | All array operations, sliding windows, unique species detection |
| **rasterio** | ≥1.3    | Yes      | GeoTIFF I/O                | Reads/writes all GDAL-supported formats. Requires GDAL system library. On Linux: `apt install libgdal-dev`. Windows/macOS pip wheels are self-contained. |
| **spyndex**  | ≥0.4    | Yes      | Spectral index formulas    | Provides 190+ published indices (NDVI, EVI, SAVI, etc.) and the `@register_index` decorator |
| **matplotlib**| ≥3.5   | Yes      | Plotting & colormaps       | Used for all visualisation (individual, stats, heatmap, comparison). The `plt.colormaps[name]` API (≥3.5) replaces the removed `get_cmap()`. |
| **seaborn**  | ≥0.12   | Yes      | Statistical visualisations | Histograms, box plots, density plots, and summary statistics tables |
| **psutil**   | ≥5.0    | Yes      | Hardware detection         | CPU core count and RAM detection for the system profiler. Required for accurate hardware info on all platforms. |
| **PySide6**  | ≥6.5    | GUI      | Desktop application        | Provides the Qt GUI framework. `QAction` moved to `QtGui` in 6.5. Only needed with `[gui]` extra. |
| **cupy**     | ≥12.0   | GPU      | GPU-accelerated Rao's Q    | NVIDIA CUDA toolkit. Enables custom CUDA `RawKernel` compilation at runtime. Only needed with `[gpu]` extra. Requires compatible NVIDIA driver (≥450.0). |

---
</details>

<details>
<summary><h2>3. Launching the GUI</h2></summary>

```bash
# Console script (after pip install paravis[gui])
paravis

# Python module
python -m paravis.gui.app
```

When the GUI starts, a splash screen displays the PaRaVis logo, version number, and a loading progress bar. Once initialisation completes, the main window opens with three panels and a status bar.

---
</details>

<details>
<summary><h2>4. Interface Walkthrough</h2></summary>

The main window is divided into three resizable columns separated by a splitter. The layout is fully customisable — drag the splitter handles, toggle panels via the View menu, and reset everything from View → Reset Layout.

```
+----------------------------------------------------------------------+
|  [View] | [Full Screen]  [Info]                               [Exit] |
+--------------+--------------------+----------------------------------+
|  INDICES     |     RAO'S Q        |       VISUALIZATION              |
|  +----------+|  +--------------+  |  +---------+  +---------------+  |
|  | Input    ||  | Single/Batch |  |  |  Files  |  | Plot Options  |  |
|  +----------+|  | tabs         |  |  +---------+  +---------------+  |
|  | Index    ||  |              |  |  |         |  |  Canvas       |  |
|  | Table    ||  | Parameters   |  |  |         |  |               |  |
|  +----------+|  +--------------+  |  |         |  |               |  |
|  | Output   ||  | Progress/Log |  |  |         |  |               |  |
|  +----------+|  +--------------+  |  +---------+  +---------------+  |
+--------------+--------------------+----------------------------------+
|                                            Ready  |  [GPU] [CPU/RAM] |
+----------------------------------------------------------------------+
```

### 4.1 Menu Bar

The menu bar is intentionally minimal — a single **View** dropdown plus three direct-action buttons.

**View dropdown** provides:
- Toggle visibility of each panel (Indices, Rao's Q, Visualisation)
- Theme switcher — Light or Dark (applied immediately via QSS)
- Full Screen (`F11`) — enters and exits full-screen mode
- Reset Layout — restores the default splitter proportions and shows all panels

**Direct buttons** sit directly on the menu bar:
- **Full Screen** (`F11`) — same as the View menu item, provided as a convenient one-click target
- **Info** — opens the About dialog showing version, authors, and citation information
- **Exit** (`Ctrl+Q`) — a red-tinted `QPushButton` anchored to the top-right corner of the menu bar. Styled with `background-color: #c0392b; color: white; font-weight: bold`.

### 4.2 Status Bar

The status bar at the bottom of the window shows live system information:

- **GPU** — model name, total VRAM, free VRAM, CUDA kernel compilation status (updates when a Rao's Q job runs)
- **CPU / RAM** — logical core count and total system memory
- **Status messages** — brief text such as "Ready", "Files loaded", "Computing indices…", or "Stopped"

Sections are separated by vertical bars.

### 4.3 Left Panel — Indices

The Indices panel is the main interface for spectral index computation.

**Input group**
- **Configure** button — opens the Band Mapping dialog, where you assign each raster band to a spectral channel (Coastal, Blue, Green, Red, RedEdge, NIR, SWIR1, SWIR2, Thermal, PAN, SAR_VV, SAR_VH, SAR_HV, SAR_HH)
- **Add TIFFs...** — standard file dialog to select one or more raster files
- **Clear** — removes all files from the list
- A label shows how many files are currently loaded

**Index Selection group**
- **Search box** — filters the index table by name in real time as you type
- **Select All / Select None** — toggles every checkbox in the table
- **Show only computable** — when checked, hides indices whose required bands aren't available in the current band mapping
- **Expand** button — opens the Index Table Expanded dialog, a larger scrollable version of the table
- **Index table** — each row shows:
  - A checkbox to select the index
  - The index name (e.g. "NDVI")
  - A green checkmark or red cross indicating computability
  - The required bands (e.g. "N, R")
  - The formula (e.g. "(N - R) / (N + R)")
- Below the table, a summary line shows the number of selected indices, available bands, and computable count

**Output Settings group**
- **Output folder** — choose where result GeoTIFFs will be saved
- **Date pattern** — optional regular expression to extract a date from filenames for time-series naming (e.g. `\d{8}`)
- **Constants** — opens the Constants Editor to override default Spyndex constants (e.g. `L=0.5` for SAVI)
- **Workers** — spinner for the number of parallel workers (1–16)
- **Tile size** — processing tile dimension in pixels (128–4096)
- **Scale denominator** — raw pixel values are divided by this number to obtain reflectance (e.g. 10000 for Sentinel-2 Level-2A)

**Progress group**
- A progress bar shows computation progress
- **Run** starts processing, **Stop** cancels, **Clear Log** clears the log output
- The log display shows each index as it is computed, timing information, and any errors

### 4.4 Centre Panel — Rao's Q

Two tabs provide single-job and batch processing modes.

#### Single Tab

**Input Files group**
- **Select Input Files** — choose one or more raster files (only the first is used in single mode)
- **Output Folder** — choose the output directory
- **Name** — output filename (defaults to "Rao_Q")

**Processing Mode group**
- **GPU / CPU** toggle — switches between GPU-accelerated and CPU processing
- When GPU is selected, a badge shows the CuPy backend status, VRAM usage, and CUDA kernel state

**Parameters group** (grid layout)

| Control       | Description                                      |
|---------------|--------------------------------------------------|
| Window Size   | Moving window side length in pixels (odd number) |
| Distance      | Metric: `euclidean`, `manhattan`, `chebyshev`, `minkowski`, `canberra`, `braycurtis` |
| Minkowski p   | Exponent for Minkowski distance (1–10, ignored for other metrics) |
| NA Tolerance  | Maximum fraction of NaN pixels allowed in a window (0.0–1.0) |
| Block Size    | Processing block dimension in pixels              |
| Workers       | CPU worker count (1–32, only used in CPU/Parallel mode) |
| Simplify      | Decimal places to truncate input values (0 = no truncation — full float32 precision, 2 = default, max 6). Higher = more decimal places kept → more species, slower; lower = fewer species, faster. Note: 0 is the most precise (slowest) |

**Progress group**
- Bar shows `current_window / total_windows`
- Log displays real-time system profile and per-window timing
- Run / Stop / Clear Log buttons

#### Batch Tab

Process multiple jobs sequentially:
- **Job list table** — columns for input file path and output name
- **Add Job** — configure input files and output name for a new job
- **Remove Job** — remove the selected job
- **Clear All** — empty the job list
- **Processing Mode** — GPU/CPU toggle applied to the entire batch
- **Output Folder** — common output directory for all jobs
- **Progress** — shows the current job name, per-job progress, and a consolidated log
- **Start Batch / Stop** controls

### 4.5 Right Panel — Visualisation

Three control tabs switch between file management, plot type, and output options.

**Files tab**
- **Load Files** — add raster files to the visualiser
- **Clear** — remove all files
- **File combo** — selects the active raster for individual or stats plots
- **Stats combo** — selects one or two files for comparison plots (diff, split)

**Plot Type tab** (nested sub-tabs)
- **Individual** — single raster displayed with a chosen colormap, an optional title, and a colour bar
- **Stats** — histogram, box plot, or density plot of the raster's pixel values, accompanied by a summary statistics table (min, max, mean, std, percentiles)
- **Diff + Heatmap** — pixel-wise difference map and a 2D heatmap between two rasters or two time points within a single raster's filename pattern

  **Heatmap Features:**
  - **Colormap** — select from any matplotlib colormap (viridis, plasma, RdYlGn, jet, etc.)
  - **Simplify** — controls input data truncation precision before species identification, not output display (0 = no truncation — full float32 precision, 2 = two decimal places, etc.). Higher precision means more unique spectral profiles detected, producing a finer-grained diversity map
  - **Auto grid sizing** — `_auto_adjust_grid()` calculates optimal grid dimensions using a square-root heuristic with +10 offset for better spacing
  - **Adaptive font size** — font size scales with `min(cell_width, cell_height)`, with a 4 pt minimum. Text is automatically hidden when cells are smaller than 0.35 inches to prevent overlap
  - **NaN rendering** — NaN values are rendered transparently in both the difference map and heatmap
- **Split** — side-by-side comparison with an adjustable vertical or horizontal split line that you can drag
- **Time Series** — multi-temporal overlay plot with GIF animation export (via Matplotlib's `PillowWriter`)

**Options tab**
- **Max Pixels** — downsampling threshold (`0.25M`, `0.5M`, `1M`, `2M`, `Unlimited`)
- **Quality** — render quality: `Fast (Draft)`, `Normal`, or `High Quality`
- **Cache** — enable or disable raster data caching between plot types
- **Output** — DPI (72–600), image format (`PNG`, `JPG`, `PDF`, `SVG`), transparent background toggle, and GIF frames-per-second

**Actions**
- **Generate Plot** — renders the current selection
- **Save Plot** (`Ctrl+S`) — exports the current plot to file

When no plot is active, the panel cycles between the PaRaVis logo and a "Select options and click Generate Plot" prompt in a cross-fade animation.

---
</details>

<details>
<summary><h2>5. Workflows</h2></summary>

### 5.1 Computing Spectral Indices

1. **Configure band mapping** — Click Configure in the Input group. Assign each raster band to a spectral channel using the spinner controls. For standard sensors, click Auto-detect.
2. **Add input files** — Click Add TIFFs... and select one or more rasters.
3. **Select indices** — Use the search box or scroll through the table. Check the indices you want. Enable "Show only computable" to hide incompatible indices.
4. **Set output options** — Choose an output folder, tile size, worker count, and scale denominator. Adjust constants if needed.
5. **Click Run** — The progress bar advances, and the log shows each index as it completes.

Output: one GeoTIFF per index per input file, written to the output folder.

### 5.2 Computing Rao's Q Diversity

1. **Select input files** — Click Select Input Files and choose your raster(s).
2. **Choose output folder** — Click Output Folder.
3. **Set processing mode** — Toggle GPU if available, or stay with CPU/Parallel.
4. **Configure parameters** — Window size (odd), distance metric, NA tolerance, block size, worker count, and simplify level.
5. **Click Run** — The progress bar reports `current_window / total_windows`. The log prints the system profile and per-window distance computation speed.

Output: a single GeoTIFF with per-pixel Rao's Q values.

### 5.3 Batch Processing Rao's Q

1. Switch to the Batch tab.
2. Click **Add Job** for each raster you want to process, configuring input files and output name.
3. Set the common output folder and processing mode (GPU/CPU).
4. Click **Start Batch** — jobs run sequentially. Each job's log is appended to the consolidated log. Stop cancels the remaining queue.

### 5.4 Visualising Rasters

1. Load one or more raster files via the Files tab.
2. Switch to the Plot Type tab and choose a mode:
   - **Individual** — select a file, choose a colormap (e.g. `viridis`, `RdYlGn`, `jet`), optionally enter a title, and click Generate Plot.
   - **Stats** — select a file, choose histogram / box plot / density, click Generate Plot.
   - **Diff + Heatmap** — select two files (or two time points parsed from filenames), click Generate Plot.
   - **Split** — select two files, drag the split line to the desired position, click Generate Plot.
   - **Time Series** — select multiple files in chronological order, choose a colormap, click Generate Plot for a static view or set GIF FPS and click Export GIF.
3. Adjust options (DPI, quality, format) and click **Save Plot** (`Ctrl+S`) to export.

---
</details>

<details>
<summary><h2>6. Dialogs</h2></summary>

### 6.1 Band Mapping

`BandMappingDialog` — assign raster band numbers to spectral channels.

- The left column lists available bands detected from the loaded raster.
- The right column lists spectral channels (Coastal, Blue, Green, Red, RedEdge, NIR, SWIR1, SWIR2, Thermal, PAN, SAR_VV, SAR_VH, SAR_HV, SAR_HH).
- Use each channel's spinner to select which band index is mapped to it.
- **Auto-detect** attempts to match common sensor configurations based on band count.
- Changes are reflected immediately in the index table's computability column.

### 6.2 Constants Editor

`ConstantsEditorDialog` — view and override Spyndex constant values.

- Each row shows a constant name and its current value.
- Double-click a value cell to edit.
- **Reset** restores all constants to their Spyndex defaults.
- Useful for tuning indices like SAVI (`L`), ARVI (`gamma`), or EVI (`C1`, `C2`, `L`).

### 6.3 Index Table (Expanded)

`IndexTableExpandDialog` — a larger, scrollable version of the index selection table.

- Shows all indices in a full-width table.
- Search and filter work identically to the panel version.
- Checkbox selections are synchronised with the main panel in real time.

### 6.4 Settings

`SettingsDialog` — application preferences persisted with `QSettings`.

- **Theme** — Light or Dark (applied immediately, no restart needed).
- **Auto-save layout** — when enabled, window geometry, splitter positions, and panel visibility are saved on exit and restored on next launch.
- **Confirm exit** — when enabled, a confirmation dialog appears before quitting.

### 6.5 About

`AboutDialog` — application identification.

- Name and version (v2.0.1).
- Short description.
- Author and contributor credits.
- DOI / citation information.
- MIT License notice.

---
</details>

<details>
<summary><h2>7. Headless & Script Usage</h2></summary>

All core computation is accessible without the GUI. The `paravis.api` module provides convenience wrappers (with automatic raster loading), and `paravis.core` gives direct access to the underlying engines for advanced usage.

### 7.1 Spectral Indices API

```python
from paravis.api import list_available_indices, compute_indices, compute_index

# List all available indices (190+ via Spyndex)
all_indices = list_available_indices()
print(f"{len(all_indices)} indices available")

# Compute all computable indices from a raster file
# (auto-detects which indices can be computed from available bands)
results = compute_indices("landsat_scene.tif")
for name, data in results.items():
    print(f"{name}: shape={data.shape}, mean={data.mean():.4f}")

# Compute specific indices only
results = compute_indices(
    "landsat_scene.tif",
    indices=["NDVI", "EVI", "SAVI", "NDWI"],
)

# Compute a single index
ndvi = compute_index("landsat_scene.tif", "NDVI")

# Custom band mapping (e.g., for Sentinel-2)
results = compute_indices(
    "sentinel2_scene.tif",
    indices=["NDVI", "NDWI"],
    band_mapping={4: "R", 8: "N"},   # Sentinel-2: B4=Red, B8=NIR
)

# Override constants (e.g., soil adjustment factor for SAVI)
results = compute_indices(
    "scene.tif",
    indices=["SAVI"],
    constants={"L": 0.5},
)
```

### 7.2 Rao's Q API

```python
from paravis.api import compute_rao_q

# Minimal call — 15×15 window, Euclidean, auto backend
result = compute_rao_q("scene.tif")

# Full parameter control
result = compute_rao_q(
    raster_path="scene.tif",
    window_size=11,          # Window size (odd number)
    step_size=1,             # Step size (1 = every pixel)
    na_tolerance=0.3,        # Max 30% NaN pixels allowed per window
    backend="auto",          # "auto" | "cpu" | "gpu" | "parallel"
    n_workers=4,             # Number of parallel CPU workers
    max_pixels=500_000,      # Auto-downsample if raster exceeds this
    simplify=2,              # Truncate input to N decimal places
    distance_metric="manhattan",
    p_minkowski=2,           # Exponent for Minkowski distance
)

# Force GPU (will raise if no GPU available)
result = compute_rao_q("scene.tif", backend="gpu")

# Force parallel CPU
result = compute_rao_q("scene.tif", backend="parallel", n_workers=8)

# Bray-Curtis with large window
result = compute_rao_q(
    "scene.tif",
    window_size=21,
    distance_metric="braycurtis",
    backend="parallel",
)

# Save result as GeoTIFF
from paravis.core.raster import write_geotiff, read_raster

write_geotiff(result, "rao_q_output.tif", nodata=-9999)
```

### 7.3 Low-Level Core API

For maximum control, use the core modules directly:

```python
import numpy as np
from paravis.core.raoq import compute_rao_q, compute_rao_q_parallel
from paravis.core.raoq.models import RaoQConfig, RaoQResult
from paravis.core.raster import read_raster, write_geotiff

# Read raster manually
data_3d, transform, crs = read_raster("scene.tif", max_pixels=1_000_000)

# Create a configuration
config = RaoQConfig(
    window_size=15,
    step_size=1,
    na_tolerance=0.3,
    n_workers=4,
    distance_metric="euclidean",
    p_minkowski=2,
    simplify=2,
    gpu_batch_size=50000,   # windows per GPU batch (default)
    cpu_batch_size=10000,   # windows per CPU batch (default)
)

# Single-threaded CPU
result = compute_rao_q(data_3d, config)

# Multi-process parallel
result_parallel = compute_rao_q_parallel(data_3d, config)

# With progress callback
def on_progress(current, total):
    print(f"  {current / total:.0%}")

result = compute_rao_q(data_3d, config, progress_callback=on_progress)

# Save to GeoTIFF
write_geotiff(result, "output.tif", transform=transform, crs=crs)
```

### 7.4 GPU Detection and Control

```python
from paravis.core.raoq.gpu import is_gpu_available, get_gpu_info, compute_rao_q_gpu

# Check GPU status
print("GPU available:", is_gpu_available())

if is_gpu_available():
    info = get_gpu_info()
    print(f"Device:  {info['name']}")
    print(f"VRAM:    {info['total_gb']:.1f} GB total, {info['free_gb']:.1f} GB free")
    print(f"Compute: {info['compute_capability']}")

    # Compute directly on GPU
    from paravis.core.raoq.gpu import compute_rao_q_gpu
    data_3d, _, _ = read_raster("scene.tif")
    result = compute_rao_q_gpu(data_3d, config)
```

### 7.5 Visualization API

```python
from paravis.api import plot_raster, plot_comparison

# Single band
fig, ax = plot_raster(
    "scene.tif",
    band=1,
    cmap="viridis",
    title="Band 4 — Near Infrared",
    colorbar=True,
)

# Comparison of two results
fig, axes = plot_comparison(
    "species_richness.tif",
    "rao_q_diversity.tif",
    labels=["Richness", "Rao's Q"],
    cmap="plasma",
)
```

### 7.6 Custom Index Registration

```python
from paravis.core.indices import register_index, SpectralIndex

@register_index
class NDWI_MOD(SpectralIndex):
    name = "NDWI_MOD"
    formula = "(G - N) / (G + N)"
    bands = ["G", "N"]      # Green and NIR
    reference = "McFeeters, 1996"
    description = "Modified Normalized Difference Water Index"

# Now usable anywhere in PaRaVis
from paravis.api import list_available_indices
print("NDWI_MOD" in list_available_indices())  # True

# Compute with raster
from paravis.api import compute_index
ndwi_mod = compute_index(
    "scene.tif",
    "NDWI_MOD",
    band_mapping={2: "G", 4: "N"},
)
```

```python
from paravis.api import compute_indices, compute_rao_q, plot_comparison

# ── Spectral Indices ──
# Band mapping: {band_number: spectral_code}, e.g. Landsat 8/9
results = compute_indices(
    "scene.tif",
    indices=["NDVI", "EVI", "SAVI"],
    band_mapping={4: "N", 3: "R", 2: "G"},  # band 4→NIR, band 3→Red, band 2→Green
)
for name, arr in results.items():
    print(f"{name}: {arr.shape}  mean={arr.mean():.4f}")

# ── Rao's Q Diversity ──
diversity = compute_rao_q(
    "scene.tif",
    window_size=15,
    distance_metric="euclidean",
    na_tolerance=0.2,
    backend="gpu",           # or "auto", "cpu", "parallel"
    simplify=2,
)

# ── Visual Comparison ──
fig, axes = plot_comparison(
    ["scene_2000.tif", "scene_2020.tif"],
    labels=["2000", "2020"],
    cmap="RdYlGn",
)
fig.savefig("comparison.png", dpi=150)
```

### Custom Indices

Register your own spectral index with the `@register_index` decorator:

```python
from paravis.core.indices.registry import register_index, compute_custom_index

@register_index(name="NDWI_MOD", bands=["G", "N"],
                description="Modified Normalised Difference Water Index")
def ndwi_modified(green, nir):
    return (green - nir) / (green + nir + 1e-10)

# Compute with real or simulated data
import numpy as np
band_arrays = {"G": np.random.rand(100, 100).astype(np.float32),
               "N": np.random.rand(100, 100).astype(np.float32)}
result = compute_custom_index("NDWI_MOD", band_arrays)
```

### Batch Script

The `examples/batch_processing.py` file demonstrates processing an entire directory of GeoTIFFs with Rao's Q:

```bash
python examples/batch_processing.py path/to/input_dir path/to/output_dir
```

---
</details>

<details>
<summary><h2>8. GPU Acceleration</h2></summary>

When CuPy is installed and an NVIDIA GPU is available, PaRaVis compiles custom CUDA `RawKernel`s at runtime for maximum-performance Rao's Q computation.

```python
from paravis.core.raoq.gpu import is_gpu_available, get_gpu_info

print("GPU available:", is_gpu_available())
info = get_gpu_info()
print(f"Device:  {info['name']}")
print(f"VRAM:    {info['total_gb']:.1f} GB total, {info['free_gb']:.1f} GB free")
print(f"Compute: {info['compute_capability']}")
```

### Requirements

- **NVIDIA GPU** with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
- **CuPy** ≥ 12.0 (`pip install cupy` or `pip install paravis[gpu]`)
- **NVIDIA driver** ≥ 450.0
- The custom CUDA kernel falls back to **pure CuPy** computation if compilation fails

### Three GPU Computation Paths

PaRaVis implements three GPU paths, automatically selected based on the window size and band count of your data:

#### Path A: Shared-Memory CUDA Kernel (Fastest)

**Triggered when** the total shared memory required fits within 99 KB:

$$4 \times (2 + 3 \times n_{pixels}) + 4 \times n_{pixels} \times n_{bands} \leq 99{,}376 \text{ bytes}$$

**How it works (one CUDA block per window):**

1. **Thread 0** counts valid (non-NaN) pixels and writes their indices to shared memory (`s_int[0] = valid_count`).
2. **All threads** cooperatively load pixel data from global memory into shared memory.
3. **Thread 0** identifies unique spectral profiles (species) by comparing each pixel's band vector against a growing list of species representatives. The count per species is tallied in `s_species_count[]`.
4. **All threads** participate in computing pairwise distances between species. Each thread handles a subset of species pairs, accumulating partial sums.
5. **Warp-shuffle reduction** (`__shfl_xor_sync`) combines partial sums across threads in each warp.
6. **Final reduction** sums across warps, and thread 0 writes the result.

The kernel uses the upper-triangular formula for efficiency:

$$Q = \frac{2}{n^2} \times \sum_{i < j} d_{ij} \times \text{count}_i \times \text{count}_j$$

**Shared memory layout** (dynamic, sized at kernel launch):
```
s_int[0]                          = valid_count
s_int[1]                          = n_species
s_int[2 .. n_pixels+1]            = valid_indices[]
s_int[n_pixels+2 .. 2n_pixels+1]  = species_rep[]
s_int[2n_pixels+2 .. 3n_pixels+1] = species_count[]
s_float[3n_pixels+2 ..]           = pixel_data[valid_count × n_bands]
```

#### Path B: Per-Thread Fallback CUDA Kernel

**Triggered when** shared memory is insufficient for Path A (many bands or large window).

Each CUDA **thread** processes one entire window independently, reading pixel data directly from global memory. The algorithm is identical to Path A (species identification → pairwise distances → weighted sum), but without cooperative shared-memory loading.

```cuda
// One thread per window
int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
// ... species-abundance logic using global memory reads
```

This path still benefits from GPU parallelism (thousands of threads) but lacks the shared-memory speed advantage of Path A.

#### Path C: Pure CuPy Fallback

**Triggered when** the custom CUDA kernel fails to compile (incompatible driver, older GPU, missing `nvcc`).

Uses CuPy's native array operations instead of custom CUDA C. Still runs on the GPU, but with higher overhead due to per-window Python-level loops and temporary array allocations.

```python
# Pure CuPy approach (vectorised per window)
for i in range(n_total):
    sub = d_w3d[i][valid_idx]
    unique_profiles, counts = cp.unique(sub, axis=0, return_counts=True)
    p = counts / nv
    a = unique_profiles[:, None, :]    # (n_species, 1, n_bands)
    b = unique_profiles[None, :, :]    # (1, n_species, n_bands)
    diffs = a - b
    dists = cp.sqrt(cp.sum(diffs**2, axis=-1))  # Euclidean
    Q = cp.sum(dists * p[:, None] * p[None, :])
```

### Batch Processing on GPU

Rows are processed in **batches** (configurable via `gpu_batch_size` in `RaoQConfig`, default 50,000 windows). For each batch:

1. Sliding windows are extracted from the GPU-resident padded raster using CuPy's `sliding_window_view`.
2. NaN masks are computed in parallel.
3. One CUDA kernel launch processes the entire batch.
4. Results are copied back to host.

This minimises kernel launch overhead and maximises GPU utilisation.

### Batch Processing on CPU

The CPU implementation uses the **same strip-based batching strategy** (configurable via `cpu_batch_size` in `RaoQConfig`, default 10,000 windows per batch). Instead of padding the entire raster upfront (which could double peak memory), each batch:

1. Extracts only the rows needed for the current batch *plus* `window_size // 2` context rows above and below from the **original raster** — no full-raster padded copy is ever created.
2. Pads that strip with NaN only at the image edges (top of first batch, bottom of last batch, and left/right borders).
3. Extracts sliding windows using NumPy's `sliding_window_view` and processes row-by-row.
4. Discards the strip before moving to the next batch, keeping peak memory at `O(batch_rows × width)` instead of `O(height × width)`.

This is especially beneficial for large rasters where the full padded array would exceed available RAM, and in parallel mode where each worker independently builds its own compact strip.

**Both CPU and GPU batching are mathematically identical** — the same species-abundance formula applied to the same sliding windows — guaranteeing identical results to within float32 rounding error.

### Species-Abundance Equivalence

**All three GPU paths implement the exact same species-abundance formula as the CPU.** This guarantees that CPU and GPU results are identical to within float32 rounding error (~1e-5). The 28+ test cases in `tests/test_cpu_gpu_equivalence.py` verify this across all metrics, window sizes, and NaN fractions.

### Fallback Behaviour

- `backend="auto"` — tries GPU first; falls back to single-threaded CPU if no GPU is detected
- If the CUDA kernel fails to compile → uses **pure CuPy** (Path C) — still on GPU, just slower
- If CuPy is missing entirely → `is_gpu_available()` returns `False`, CPU backends are used
- If GPU runs out of memory → reduce window size or increase `gpu_batch_size`

---
</details>

<details>
<summary><h2>9. Keyboard Shortcuts</h2></summary>

| Shortcut      | Action                               |
|---------------|--------------------------------------|
| `Ctrl+O`      | Open raster files (file dialog)      |
| `Ctrl+Q`      | Quit the application                 |
| `Ctrl+S`      | Save current plot                    |
| `Ctrl+,`      | Open Settings dialog                 |
| `F11`         | Toggle full-screen mode              |
| `Escape`      | Stop current computation             |

---
</details>

<details>
<summary><h2>10. Troubleshooting</h2></summary>

### GUI won't start

```bash
# Check Python version
python --version               # must be ≥ 3.9

# Verify installation
python -c "import paravis; print(paravis.__version__)"   # should print "2.0.1"

# Check Qt platform
python -c "from PySide6.QtWidgets import QApplication; app = QApplication([]); print('Qt OK')"
```

On Linux, the application forces the XCB Qt platform plugin. If you encounter display problems:

```bash
QT_QPA_PLATFORM=wayland python -m paravis.gui.app
```

### GPU not detected

1. Run `nvidia-smi` to verify the NVIDIA driver is working.
2. Check CuPy: `python -c "import cupy; print(cupy.__version__)"`.
3. If CuPy is missing: `pip install cupy`.
4. If the custom CUDA kernel fails to compile, the engine falls back to pure-CuPy (slower but functional).

### "Cannot import name 'QAction' from 'PySide6.QtWidgets'"

PySide6 ≥ 6.5 is required. Upgrade:

```bash
pip install --upgrade PySide6
```

### Main window appears blank

- Make sure `QT_QPA_PLATFORM` is not set to `offscreen`.
- If running over SSH, use X forwarding (`ssh -X`) or a VNC session.
- Try switching between Light and Dark themes.

### Tests fail

GUI tests must run in isolated subprocesses because Qt C++ objects cannot be shared across test modules. Always use the provided runner:

```bash
python scripts/run_tests_isolated.py
```

For a quick smoke test of non-GUI modules only:

```bash
pytest tests/ -v -p no:cupyx
```

GPU tests are automatically skipped when no CUDA device is available.

---
</details>

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

---

*© 2026 PaRaVis Contributors. MIT License.*
