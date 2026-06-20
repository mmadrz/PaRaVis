"""
Widget panel for raster visualization, comparison, and GIF generation.

Faithful port of v1 VisualizationWidget with:
- Individual plot, Stats, Diff+Heatmap, Split, Time Series tabs
- File management (load, clear, file dict, metadata)
- Colormap selection with NaN handling
- Quality/performance settings (Fast/Normal/High, DPI, format, alignment)
- Cache management, status bar with file count
- Worker threads for background plotting
- Blinking logo on empty state
- Navigation toolbar on generated plots
- GIF generation (split animation + time series)
"""
import os
from typing import Dict, List, Optional
from functools import lru_cache

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame,
    QFileDialog, QMessageBox, QGroupBox, QTabWidget, QComboBox,
    QProgressBar, QLineEdit, QSpinBox, QCheckBox, QSlider, QDialog,
    QDialogButtonBox, QListWidget, QListWidgetItem,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QSequentialAnimationGroup, QPauseAnimation
from PySide6.QtGui import QPixmap, QFont

from paravis.core.raster import read_raster
from paravis.gui.components.mpl_canvas import MplCanvas
from paravis.gui.components.workers import PlotWorker, GifWorker

# Try importing matplotlib and seaborn
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    import seaborn as sns
    import pandas as pd
    from matplotlib.animation import FuncAnimation, PillowWriter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def normalize_data(data):
    """Normalize data to 0-1 range, handling NaN and all data types."""
    if data.size == 0:
        return data
    # Convert to float to handle all data types (int, uint, etc.)
    data = data.astype(np.float64, copy=False)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
        return np.zeros_like(data, dtype=np.float64)
    return (data - min_val) / (max_val - min_val)


def read_raster_memory_efficient(file_path, max_pixels=500000):
    """Read raster with automatic downsampling for large files.
    Handles nodata values by converting them to NaN.
    """
    import rasterio
    with rasterio.open(file_path) as src:
        # Read nodata value from the file
        nodata = src.nodata
        
        data = src.read(1)
        h, w = data.shape
        total_pixels = h * w
        if total_pixels > max_pixels:
            factor = int(np.sqrt(total_pixels / max_pixels))
            if factor > 1:
                out_shape = (h // factor, w // factor)
                data = src.read(1, out_shape=out_shape)
        
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


def get_cmap_with_nan(cmap_name, bad_color='white', bad_alpha=1.0):
    """Get a colormap with NaN values rendered as a specific color."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    try:
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad(color=bad_color, alpha=bad_alpha)
        return cmap
    except Exception:
        try:
            cmap = plt.cm.get_cmap(cmap_name).copy()
            cmap.set_bad(color=bad_color, alpha=bad_alpha)
            return cmap
        except Exception:
            return cmap_name


def decimal_degrees_to_dms(dd):
    """Convert decimal degrees to DMS."""
    degrees = int(dd)
    minutes = int((dd - degrees) * 60)
    seconds = ((dd - degrees) * 60 - minutes) * 60
    return degrees, minutes, seconds


# ------------------------------------------------------------------
# GIF Worker (Split Animation)
# ------------------------------------------------------------------
class SplitGIFWorker(PlotWorker):
    """Worker for generating split-view animation GIFs."""

    def __init__(self, left_data, right_data, file1_name, file2_name,
                 colormap, output_dir, output_name, dpi, fps=10, parent=None):
        super().__init__(lambda: None, parent)
        self.left_data = left_data
        self.right_data = right_data
        self.file1_name = file1_name
        self.file2_name = file2_name
        self.colormap = colormap
        self.output_dir = output_dir
        self.output_name = output_name
        self.dpi = dpi
        self.fps = fps

    def run(self):
        try:
            max_size = 500
            if self.left_data.shape[0] > max_size or self.left_data.shape[1] > max_size:
                factor = max(1, max(self.left_data.shape[0], self.left_data.shape[1]) // max_size)
                self.left_data = self.left_data[::factor, ::factor]
                self.right_data = self.right_data[::factor, ::factor]

            fig, ax = plt.subplots(figsize=(8, 6), dpi=72)
            combined = np.copy(self.left_data)
            split_pos = int(0.5 * self.left_data.shape[1])
            combined[:, split_pos:] = self.right_data[:, split_pos:]
            im = ax.imshow(combined, cmap=get_cmap_with_nan(self.colormap), origin="upper")
            cbar = fig.colorbar(im, ax=ax, label="Rao's Q Value", pad=0.01)
            split_line = ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            fig.tight_layout()

            def animate(frame):
                split_frac = frame / 100.0
                split_pixel = int(split_frac * self.left_data.shape[1])
                combined = np.copy(self.left_data)
                combined[:, split_pixel:] = self.right_data[:, split_pixel:]
                im.set_array(combined)
                split_line.set_xdata([split_pixel, split_pixel])
                return [im, split_line]

            frames = np.arange(0, 101, 5)
            anim = FuncAnimation(fig, animate, frames=frames,
                                 interval=1000 / self.fps, blit=True)
            output_path = os.path.join(
                self.output_dir,
                f"SplitPlotAnimation_{self.file1_name}_{self.file2_name}_{self.output_name}.gif"
            )
            anim.save(output_path, writer=PillowWriter(fps=self.fps), dpi=self.dpi)
            plt.close(fig)

            if self.is_running:
                self.finished.emit(output_path)
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# File Selection Dialog
# ------------------------------------------------------------------
class FileSelectionDialog(QDialog):
    """Dialog for selecting files in chronological order."""

    def __init__(self, file_dict, parent=None):
        super().__init__(parent)
        self.file_dict = file_dict
        self.selected_files = []
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Select Files for Time Series")
        self.setModal(True)
        self.resize(400, 500)

        layout = QVBoxLayout()
        label = QLabel("Select files in chronological order:")
        label.setStyleSheet("font-weight: bold; margin: 10px;")
        layout.addWidget(label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for file_name in sorted(self.file_dict.keys()):
            self.list_widget.addItem(QListWidgetItem(file_name))
        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def _on_accept(self):
        self.selected_files = [item.text() for item in self.list_widget.selectedItems()]
        if len(self.selected_files) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 files for time series")
            return
        super().accept()


# ======================================================================
# MAIN VISUALIZATION WIDGET
# ======================================================================
class VisualizationWidget(QWidget):
    """Panel for raster data visualization (v1 faithful port)."""

    def __init__(self):
        super().__init__()
        self.file_dict: Dict[str, str] = {}  # name -> path
        self.file_metadata: Dict[str, dict] = {}
        self.subtraction_data = None
        self.current_canvas: Optional[MplCanvas] = None
        self.current_figure = None
        self.plot_worker: Optional[PlotWorker] = None
        self.gif_worker = None
        self.timeseries_worker = None
        self.timeseries_selected_files: List[str] = []
        self._gif_output_dir: str = ""
        self.max_pixels = 500_000
        self._plot_generation = 0  # Counter to track plot generations

        # Logo for empty state (alpha-blinking)
        logo_path = os.path.join(os.path.dirname(__file__), "..", "theme", "logo.png")
        self.logo_pixmap = QPixmap(logo_path) if os.path.exists(logo_path) else None
        self._logo_blink_anim = None
        self._logo_effect = None

        self._setup_ui()
        self._start_logo_blink()

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # Header
        header = QLabel("Visualization")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet("background-color: #c0c0c0; margin: 4px 0;")
        line.setFixedHeight(1)
        main_layout.addWidget(line)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Control panel with tabs
        top_panel = self._create_control_panel()
        main_layout.addWidget(top_panel)

        # Plot panel
        bottom_panel = self._create_plot_panel()
        main_layout.addWidget(bottom_panel, 1)

        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(8, 2, 8, 2)
        self.generate_btn = QPushButton("Generate Plot")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setProperty("class", "primary")
        action_layout.addWidget(self.generate_btn, 1)
        self.save_btn = QPushButton("Save Plot")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        self.save_btn.setProperty("class", "secondary")
        action_layout.addWidget(self.save_btn, 1)
        action_group.setLayout(action_layout)
        main_layout.addWidget(action_group)

        # Status bar
        status_bar = QFrame()
        status_bar.setObjectName("vizStatusBar")
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(4, 2, 4, 2)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        self.file_count_label = QLabel("No files loaded")
        self.file_count_label.setStyleSheet("color: #888888; font-size: 11px;")
        status_layout.addWidget(self.file_count_label)
        status_bar.setLayout(status_layout)
        main_layout.addWidget(status_bar)

        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # Control panel
    # ------------------------------------------------------------------

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        control_tabs = QTabWidget()
        control_tabs.setMinimumHeight(240)

        # ── Tab 1: Files ──
        files_tab = QWidget()
        files_layout = QVBoxLayout()
        files_layout.setContentsMargins(10, 6, 10, 6)
        files_layout.setSpacing(6)

        file_btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Files")
        self.load_btn.clicked.connect(self._load_files)
        file_btn_layout.addWidget(self.load_btn, 1)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_files)
        file_btn_layout.addWidget(self.clear_btn, 1)
        files_layout.addLayout(file_btn_layout)

        file_sel_layout = QHBoxLayout()
        file_sel_layout.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        file_sel_layout.addWidget(self.file_combo, 1)
        file_sel_layout.addWidget(QLabel("Stats:"))
        self.files_multiple = QComboBox()
        self.files_multiple.setEditable(False)
        file_sel_layout.addWidget(self.files_multiple, 1)
        files_layout.addLayout(file_sel_layout)
        files_tab.setLayout(files_layout)
        control_tabs.addTab(files_tab, "Files")

        # ── Tab 2: Plot Type ──
        plot_type_tab = QWidget()
        plot_type_layout = QVBoxLayout()
        plot_type_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_tabs = QTabWidget()

        ind_tab = self._create_individual_tab()
        self.plot_tabs.addTab(ind_tab, "Individual")

        stats_tab = self._create_stats_tab()
        self.plot_tabs.addTab(stats_tab, "Stats")

        diff_heat_tab = self._create_diff_heat_tab()
        self.plot_tabs.addTab(diff_heat_tab, "Diff + Heatmap")

        split_tab = self._create_split_tab()
        self.plot_tabs.addTab(split_tab, "Split")

        timeseries_tab = self._create_timeseries_tab()
        self.plot_tabs.addTab(timeseries_tab, "Time Series")

        plot_type_layout.addWidget(self.plot_tabs)
        plot_type_tab.setLayout(plot_type_layout)
        control_tabs.addTab(plot_type_tab, "Plot Type")

        # ── Tab 3: Options ──
        perf_out_tab = QWidget()
        perf_out_layout = QVBoxLayout()
        perf_out_layout.setContentsMargins(10, 8, 10, 8)
        perf_out_layout.setSpacing(6)

        perf_header = QLabel("Performance")
        perf_header.setStyleSheet("font-weight: 600; font-size: 11px; color: #009688;")
        perf_out_layout.addWidget(perf_header)
        perf_row = QHBoxLayout()
        perf_row.addWidget(QLabel("Max Pixels:"))
        self.max_pixels_combo = QComboBox()
        self.max_pixels_combo.addItems(['0.25M', '0.5M', '1M', '2M', '5M', 'Unlimited'])
        self.max_pixels_combo.setCurrentText('0.5M')
        self.max_pixels_combo.currentTextChanged.connect(self._change_max_pixels)
        perf_row.addWidget(self.max_pixels_combo, 1)
        perf_row.addWidget(QLabel("Quality:"))
        self.plot_quality = QComboBox()
        self.plot_quality.addItems(['Fast (Draft)', 'Normal', 'High Quality'])
        self.plot_quality.setCurrentText('Normal')
        perf_row.addWidget(self.plot_quality, 1)
        self.cache_checkbox = QCheckBox("Cache")
        self.cache_checkbox.setChecked(True)
        perf_row.addWidget(self.cache_checkbox)
        perf_out_layout.addLayout(perf_row)

        out_header = QLabel("Output Settings")
        out_header.setStyleSheet("font-weight: 600; font-size: 11px; color: #009688;")
        perf_out_layout.addWidget(out_header)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("DPI:"))
        self.save_dpi = QComboBox()
        self.save_dpi.addItems(['100', '200', '300', '500', '600', '800', '1000'])
        self.save_dpi.setCurrentText('300')
        out_row.addWidget(self.save_dpi, 1)
        out_row.addWidget(QLabel("Format:"))
        self.save_format = QComboBox()
        self.save_format.addItems(['png', 'jpg', 'svg', 'tiff', 'pdf'])
        self.save_format.setCurrentText('png')
        out_row.addWidget(self.save_format, 1)
        out_row.addWidget(QLabel("H-Align:"))
        self.ha_combo = QComboBox()
        self.ha_combo.addItems(['center', 'right', 'left'])
        self.ha_combo.setCurrentText('center')
        out_row.addWidget(self.ha_combo, 1)
        out_row.addWidget(QLabel("V-Align:"))
        self.va_combo = QComboBox()
        self.va_combo.addItems(['center', 'top', 'bottom', 'baseline'])
        self.va_combo.setCurrentText('center')
        out_row.addWidget(self.va_combo, 1)
        self.transparent_bg = QCheckBox("Transparent")
        self.transparent_bg.setChecked(False)
        out_row.addWidget(self.transparent_bg)
        perf_out_layout.addLayout(out_row)

        perf_out_tab.setLayout(perf_out_layout)
        control_tabs.addTab(perf_out_tab, "Options")

        layout.addWidget(control_tabs)
        panel.setLayout(layout)
        return panel

    def _create_individual_tab(self):
        tab = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        layout.addWidget(QLabel("CMAP:"))
        self.ind_colormap = QComboBox()
        if MATPLOTLIB_AVAILABLE:
            cmaps = sorted(plt.colormaps())
            self.ind_colormap.addItems(cmaps)
            self.ind_colormap.setCurrentText('nipy_spectral')
        self.ind_colormap.setMinimumWidth(180)
        layout.addWidget(self.ind_colormap, 1)
        tab.setLayout(layout)
        return tab

    def _create_stats_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        controls_layout = QHBoxLayout()
        self.stats_plot_type = QComboBox()
        self.stats_plot_type.addItems(['Box Plot', 'Histogram', 'KDE Plot', 'Violin Plot'])
        controls_layout.addWidget(self.stats_plot_type, 1)
        self.stats_palette = QComboBox()
        if MATPLOTLIB_AVAILABLE:
            self.stats_palette.addItems(sorted(plt.colormaps()))
        self.stats_palette.setCurrentText('Set2')
        controls_layout.addWidget(self.stats_palette, 1)
        layout.addLayout(controls_layout)

        sample_layout = QHBoxLayout()
        self.stats_sample = QCheckBox("Use sampled data")
        self.stats_sample.setChecked(True)
        sample_layout.addWidget(self.stats_sample)
        sample_layout.addStretch()
        sample_layout.addWidget(QLabel("Sample size:"))
        self.stats_sample_size = QSpinBox()
        self.stats_sample_size.setRange(1000, 100000)
        self.stats_sample_size.setValue(10000)
        sample_layout.addWidget(self.stats_sample_size)
        layout.addLayout(sample_layout)

        tab.setLayout(layout)
        return tab

    def _create_diff_heat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File A:"))
        self.diff_file1 = QComboBox()
        file_layout.addWidget(self.diff_file1, 1)
        file_layout.addWidget(QLabel("File B:"))
        self.diff_file2 = QComboBox()
        file_layout.addWidget(self.diff_file2, 1)
        layout.addLayout(file_layout)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("CMAP:"))
        self.diff_colormap = QComboBox()
        if MATPLOTLIB_AVAILABLE:
            cmaps = sorted(plt.colormaps())
            self.diff_colormap.addItems(cmaps)
            self.diff_colormap.setCurrentText('RdBu')
        controls_layout.addWidget(self.diff_colormap, 1)
        controls_layout.addWidget(QLabel("Grid:"))
        self.heatmap_window = QSlider(Qt.Orientation.Horizontal)
        self.heatmap_window.setRange(10, 150)
        self.heatmap_window.setValue(30)
        self.heatmap_window.setEnabled(False)
        controls_layout.addWidget(self.heatmap_window, 1)
        self.heatmap_label = QLabel("30")
        self.heatmap_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        controls_layout.addWidget(self.heatmap_label)
        self.adaptive_grid_btn = QPushButton("Auto Grid")
        self.adaptive_grid_btn.clicked.connect(self._auto_adjust_grid)
        self.adaptive_grid_btn.setEnabled(False)
        controls_layout.addWidget(self.adaptive_grid_btn)
        layout.addLayout(controls_layout)

        self.show_heatmap_checkbox = QCheckBox("Show Grid Values")
        self.show_heatmap_checkbox.setChecked(False)
        self.show_heatmap_checkbox.toggled.connect(self._toggle_heatmap_controls)
        layout.addWidget(self.show_heatmap_checkbox)

        self.heatmap_window.valueChanged.connect(lambda v: self.heatmap_label.setText(str(v)))
        tab.setLayout(layout)
        return tab

    def _create_split_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Left:"))
        self.split_file1 = QComboBox()
        file_layout.addWidget(self.split_file1, 1)
        file_layout.addWidget(QLabel("Right:"))
        self.split_file2 = QComboBox()
        file_layout.addWidget(self.split_file2, 1)
        layout.addLayout(file_layout)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Split:"))
        self.split_size = QSlider(Qt.Orientation.Horizontal)
        self.split_size.setRange(0, 100)
        self.split_size.setValue(50)
        self.split_size.valueChanged.connect(lambda v: self.split_label.setText(f"{v}%"))
        self.split_size.valueChanged.connect(self._on_split_changed)
        controls_layout.addWidget(self.split_size, 1)
        self.split_label = QLabel("50%")
        self.split_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        controls_layout.addWidget(self.split_label)
        controls_layout.addWidget(QLabel("CMAP:"))
        self.split_colormap = QComboBox()
        if MATPLOTLIB_AVAILABLE:
            cmaps = sorted(plt.colormaps())
            self.split_colormap.addItems(cmaps)
            self.split_colormap.setCurrentText('viridis')
        self.split_colormap.currentTextChanged.connect(self._on_split_changed)
        controls_layout.addWidget(self.split_colormap, 1)
        layout.addLayout(controls_layout)

        gif_layout = QHBoxLayout()
        gif_layout.addWidget(QLabel("FPS:"))
        self.gif_fps = QSpinBox()
        self.gif_fps.setRange(5, 30)
        self.gif_fps.setValue(10)
        self.gif_fps.setMaximumWidth(60)
        gif_layout.addWidget(self.gif_fps)
        gif_layout.addStretch()
        self.gif_btn = QPushButton("Generate Split GIF")
        self.gif_btn.clicked.connect(self._generate_gif)
        gif_layout.addWidget(self.gif_btn)
        layout.addLayout(gif_layout)

        tab.setLayout(layout)
        return tab

    def _create_timeseries_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        row1 = QHBoxLayout()
        self.select_timeseries_btn = QPushButton("Select Files")
        self.select_timeseries_btn.setProperty("class", "primary")
        self.select_timeseries_btn.clicked.connect(self._select_timeseries_files)
        row1.addWidget(self.select_timeseries_btn)
        self.timeseries_files_label = QLabel("No files selected")
        self.timeseries_files_label.setWordWrap(True)
        self.timeseries_files_label.setStyleSheet("color: #888888;")
        row1.addWidget(self.timeseries_files_label, 1)
        row1.addWidget(QLabel("FPS:"))
        self.timeseries_fps = QSpinBox()
        self.timeseries_fps.setRange(5, 30)
        self.timeseries_fps.setValue(8)
        row1.addWidget(self.timeseries_fps)
        row1.addWidget(QLabel("CMAP:"))
        self.timeseries_colormap = QComboBox()
        if MATPLOTLIB_AVAILABLE:
            cmaps = sorted(plt.colormaps())
            self.timeseries_colormap.addItems(cmaps)
            self.timeseries_colormap.setCurrentText('viridis')
        row1.addWidget(self.timeseries_colormap, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("DPI:"))
        self.timeseries_dpi = QComboBox()
        self.timeseries_dpi.addItems(['100', '150', '200', '300'])
        self.timeseries_dpi.setCurrentText('150')
        row2.addWidget(self.timeseries_dpi)
        row2.addWidget(QLabel("FPS:"))
        self.normalize_checkbox = QCheckBox("Normalization")
        self.normalize_checkbox.setChecked(True)
        row2.addWidget(self.normalize_checkbox)
        row2.addStretch()
        self.generate_timeseries_btn = QPushButton("Generate GIF")
        self.generate_timeseries_btn.setProperty("class", "primary")
        self.generate_timeseries_btn.clicked.connect(self._generate_timeseries_gif)
        self.generate_timeseries_btn.setEnabled(False)
        row2.addWidget(self.generate_timeseries_btn, 1)
        layout.addLayout(row2)

        tab.setLayout(layout)
        return tab

    # ------------------------------------------------------------------
    # Plot panel
    # ------------------------------------------------------------------

    def _create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot_frame = QFrame()
        plot_frame.setObjectName("plotFrame")
        plot_frame.setStyleSheet("border: none;")
        plot_layout = QGridLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Logo — centred, blinking with alpha
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.logo_pixmap and not self.logo_pixmap.isNull():
            scaled = self.logo_pixmap.scaled(
                450, 450, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.logo_label.setPixmap(scaled)
        self.logo_label.setVisible(True)
        self._logo_effect = QGraphicsOpacityEffect(self.logo_label)
        self._logo_effect.setOpacity(1.0)
        self.logo_label.setGraphicsEffect(self._logo_effect)
        plot_layout.addWidget(self.logo_label, 0, 0, Qt.AlignmentFlag.AlignCenter)

        plot_frame.setLayout(plot_layout)
        layout.addWidget(plot_frame)

        self.plot_frame = plot_frame
        self.plot_layout = plot_layout

        panel.setLayout(layout)
        return panel

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _change_max_pixels(self, value):
        if value == 'Unlimited':
            self.max_pixels = float('inf')
        elif value.endswith('M'):
            self.max_pixels = int(float(value[:-1]) * 1000000)
        else:
            self.max_pixels = 500000
        self._clear_data_cache()
        self.status_label.setText(f"● Max pixels set to {value}")

    def _clear_data_cache(self):
        self.file_metadata.clear()
        self._get_file_data.cache_clear()

    @lru_cache(maxsize=10)
    def _get_file_data(self, file_path):
        try:
            data, transform, crs = read_raster_memory_efficient(file_path, self.max_pixels)
            return data, transform, crs
        except Exception:
            return None, None, None

    def _load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Rao's Q Output Files", "",
            "GeoTIFF Files (*.tif *.tiff);;All Files (*.*)"
        )
        if files:
            self._load_files_direct(files)

    def load_files_direct(self, files):
        """Load files programmatically (used by drag-drop and main menu)."""
        self._load_files_direct(files)

    def _load_files_direct(self, files):
        if not files:
            return
        for file_path in files:
            file_name = os.path.basename(file_path).split('.')[0]
            self.file_dict[file_name] = file_path
            try:
                import rasterio
                with rasterio.open(file_path) as src:
                    self.file_metadata[file_name] = {
                        'shape': src.shape,
                        'crs': src.crs,
                        'transform': src.transform,
                        'size_mb': src.shape[0] * src.shape[1] * 8 / (1024 * 1024),
                    }
            except Exception:
                pass

        self._update_selectors()
        total_size = sum(m.get('size_mb', 0) for m in self.file_metadata.values())
        self.file_count_label.setText(
            f"{len(self.file_dict)} file(s) loaded ({total_size:.1f} MB)"
        )
        self.status_label.setText("● Files loaded successfully")
        self.status_label.setVisible(True)

    def _clear_files(self):
        self.file_dict.clear()
        self.file_metadata.clear()
        self._clear_data_cache()
        self.subtraction_data = None
        self.timeseries_selected_files = []
        self.timeseries_files_label.setText("No files selected")
        self.generate_timeseries_btn.setEnabled(False)
        self._update_selectors()
        self.file_count_label.setText("No files loaded")
        self.status_label.setText("● Files cleared")
        self._clear_plot()

    def _update_selectors(self):
        file_list = list(self.file_dict.keys())
        self.file_combo.clear()
        self.files_multiple.clear()
        self.diff_file1.clear()
        self.diff_file2.clear()
        self.split_file1.clear()
        self.split_file2.clear()

        self.file_combo.addItems(file_list)
        self.files_multiple.addItems(file_list)
        self.diff_file1.addItems(file_list)
        self.diff_file2.addItems(file_list)
        self.split_file1.addItems(file_list)
        self.split_file2.addItems(file_list)

        if len(file_list) >= 2:
            self.diff_file2.setCurrentIndex(1)
            self.split_file2.setCurrentIndex(1)

    def _on_split_changed(self):
        """Auto-regenerate split plot when slider or colormap changes (debounced)."""
        if self.plot_tabs.currentIndex() == 3:  # Split tab
            # Debounce: wait 300ms after last change before regenerating
            if not hasattr(self, '_split_debounce_timer'):
                from PySide6.QtCore import QTimer
                self._split_debounce_timer = QTimer()
                self._split_debounce_timer.setSingleShot(True)
                self._split_debounce_timer.timeout.connect(self._on_generate)
            self._split_debounce_timer.start(300)

    def _select_timeseries_files(self):
        if not self.file_dict:
            QMessageBox.warning(self, "Warning", "Please load files first")
            return
        dialog = FileSelectionDialog(self.file_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.timeseries_selected_files = dialog.selected_files
            self.timeseries_files_label.setText(
                f"Selected: {', '.join(self.timeseries_selected_files)}"
            )
            self.generate_timeseries_btn.setEnabled(True)
            self.status_label.setText(
                f"● {len(self.timeseries_selected_files)} files selected for time series"
            )

    def _auto_adjust_grid(self):
        file1_name = self.diff_file1.currentText()
        if not file1_name:
            return
        metadata = self.file_metadata.get(file1_name)
        if metadata:
            nrows, ncols = metadata['shape']
            recommended = max(15, min(30, (ncols + nrows) // 150)) + 10
            recommended = max(10, min(recommended, 40))
            self.heatmap_window.setValue(recommended)
            self.heatmap_label.setText(str(recommended))
            QMessageBox.information(
                self, "Grid Adjusted",
                f"Grid size set to {recommended}×{recommended}\n"
                f"(Based on image dimensions: {ncols}×{nrows} pixels)"
            )

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _on_generate(self):
        current_tab = self.plot_tabs.currentIndex()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("● Generating plot...")
        self.status_label.setVisible(True)

        if self.plot_worker:
            self.plot_worker.stop()
            # Don't wait() — the worker may be in a long matplotlib call
            # that doesn't check is_running. The generation counter will
            # ignore its result when it eventually finishes.

        # Increment generation counter to invalidate any stale workers
        self._plot_generation += 1
        current_generation = self._plot_generation

        if current_tab == 0:
            self._plot_individual(current_generation)
        elif current_tab == 1:
            self._plot_statistics(current_generation)
        elif current_tab == 2:
            self._plot_diff_heat(current_generation)
        elif current_tab == 3:
            self._plot_split(current_generation)

    def _on_save(self):
        if self.current_figure is None:
            QMessageBox.warning(self, "Warning", "No plot to save. Please generate a plot first.")
            return

        dpi = int(self.save_dpi.currentText())
        fmt = self.save_format.currentText()

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"plot.{fmt}", f"Image Files (*.{fmt})"
        )
        if file_path:
            try:
                self.current_figure.savefig(
                    file_path, dpi=dpi, bbox_inches='tight',
                    transparent=self.transparent_bg.isChecked()
                )
                self.status_label.setText(f"● Plot saved to {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", f"Plot saved to\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot: {str(e)}")

    def _start_logo_blink(self):
        """Simple logo alpha-blinking: fade in → fade out, loop."""
        if not hasattr(self, 'logo_label') or self.logo_label is None:
            return
        if self._logo_effect is None:
            self._logo_effect = QGraphicsOpacityEffect(self.logo_label)
            self.logo_label.setGraphicsEffect(self._logo_effect)

        self.logo_label.setVisible(True)
        self._logo_effect.setOpacity(1.0)

        if self._logo_blink_anim is None:
            group = QSequentialAnimationGroup()
            group.setLoopCount(-1)

            # 1. Logo fades out (0→2s)
            fade_out = QPropertyAnimation(self._logo_effect, b"opacity")
            fade_out.setDuration(2000)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            group.addAnimation(fade_out)

            # 2. Pause (2→3s) — logo invisible
            group.addPause(1000)

            # 3. Logo fades in (3→5s)
            fade_in = QPropertyAnimation(self._logo_effect, b"opacity")
            fade_in.setDuration(2000)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            group.addAnimation(fade_in)

            # 4. Pause (5→6s) — logo fully visible
            group.addPause(1000)

            self._logo_blink_anim = group

        if self._logo_blink_anim.state() != QPropertyAnimation.State.Running:
            self._logo_blink_anim.start()

    def _stop_logo_blink(self):
        """Stop the blinking animation and hide the logo."""
        if self._logo_blink_anim:
            self._logo_blink_anim.stop()
        try:
            if hasattr(self, 'logo_label') and self.logo_label is not None:
                self.logo_label.setVisible(False)
        except RuntimeError:
            # C++ object already deleted
            pass

    def _clear_plot(self):
        self._stop_logo_blink()

        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Recreate logo label only
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.logo_pixmap and not self.logo_pixmap.isNull():
            scaled = self.logo_pixmap.scaled(
                450, 450, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.logo_label.setPixmap(scaled)
        # Re-attach alpha-blinking effect
        self._logo_effect = QGraphicsOpacityEffect(self.logo_label)
        self._logo_effect.setOpacity(1.0)
        self.logo_label.setGraphicsEffect(self._logo_effect)
        self.plot_layout.addWidget(self.logo_label, 0, 0, Qt.AlignmentFlag.AlignCenter)
        self._start_logo_blink()

        if self.current_figure:
            plt.close(self.current_figure)
        self.current_canvas = None
        self.current_figure = None
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def closeEvent(self, event):
        """Clean up workers and timers on widget close."""
        if self.plot_worker:
            self.plot_worker.stop()
            self.plot_worker.wait(1000)  # Wait up to 1 second
        if hasattr(self, '_split_debounce_timer'):
            self._split_debounce_timer.stop()
        if hasattr(self, '_logo_blink_anim') and self._logo_blink_anim:
            self._logo_blink_anim.stop()
        super().closeEvent(event)

    def _display_plot(self, fig, generation=None):
        # Ignore stale results from previous generations
        if generation is not None and generation != self._plot_generation:
            plt.close(fig)
            return

        self.current_figure = fig

        if self.transparent_bg.isChecked():
            fig.patch.set_alpha(0.0)
            for ax in fig.get_axes():
                ax.patch.set_alpha(0.0)

        self._stop_logo_blink()

        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)

        self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(canvas)

        self.current_canvas = canvas
        self.save_btn.setEnabled(True)
        self.status_label.setText("● Plot generated successfully")
        self.progress_bar.setVisible(False)

    def _display_gif(self, gif_path):
        """Display a GIF animation in the plot section."""
        self._stop_logo_blink()

        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        from PySide6.QtWidgets import QLabel, QSizePolicy
        from PySide6.QtGui import QMovie

        gif_label = QLabel()
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gif_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        movie = QMovie(gif_path)
        movie.setCacheMode(QMovie.CacheMode.CacheAll)
        movie.setSpeed(100)
        gif_label.setMovie(movie)
        movie.start()

        self.plot_layout.addWidget(gif_label)
        self.current_canvas = None
        self.current_figure = None
        self.save_btn.setEnabled(False)
        self._gif_movie = movie  # Keep reference to prevent garbage collection
        self.status_label.setText("● Time series GIF displayed")
        self.progress_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Individual Plot
    # ------------------------------------------------------------------

    def _plot_individual(self, generation=None):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available")
            self.progress_bar.setVisible(False)
            return

        file_name = self.file_combo.currentText()
        if not file_name:
            QMessageBox.warning(self, "Warning", "Please select a file")
            self.progress_bar.setVisible(False)
            return

        file_path = self.file_dict.get(file_name)
        if not file_path:
            QMessageBox.warning(self, "Warning", "File not found")
            self.progress_bar.setVisible(False)
            return

        quality = self.plot_quality.currentText()
        use_cache = self.cache_checkbox.isChecked()

        def create_plot():
            data, transform, _ = (
                self._get_file_data(file_path)
                if use_cache
                else read_raster_memory_efficient(file_path, self.max_pixels)
            )
            if data is None:
                raise Exception("Failed to read file")

            data = normalize_data(data)
            dpi_val = 72 if quality == 'Fast (Draft)' else 100 if quality == 'Normal' else 150
            fig = plt.figure(figsize=(8, 6), dpi=dpi_val)
            ax = fig.add_subplot(111)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=1))

            num_rows, num_cols = data.shape
            x = np.arange(0, num_cols) * transform.a + transform.c
            y = np.arange(0, num_rows) * transform.e + transform.f

            im = ax.imshow(
                np.ma.masked_invalid(data),
                cmap=get_cmap_with_nan(self.ind_colormap.currentText()),
                extent=[x.min(), x.max(), y.min(), y.max()],
                interpolation='bilinear' if quality != 'Fast (Draft)' else 'nearest',
            )
            cbar = fig.colorbar(im, label="Value", pad=0.01)
            cbar.ax.get_yaxis().label.set_fontsize(12)

            x_ticks = np.linspace(x.min(), x.max(), num=min(3, len(ax.get_xticks())))
            y_ticks = np.linspace(y.min(), y.max(), num=min(3, len(ax.get_yticks())))
            ax.set_xticklabels([
                "%.0f° %.0f' %.0f\" %s" % (
                    *decimal_degrees_to_dms(val), "W" if val < 0 else "E"
                ) for val in x_ticks
            ])
            ax.set_yticklabels([
                "%.0f° %.0f' %.0f\" %s" % (
                    *decimal_degrees_to_dms(val), "S" if val < 0 else "N"
                ) for val in y_ticks
            ])

            plt.xticks(rotation=0, fontsize=10, ha=self.ha_combo.currentText())
            plt.yticks(rotation=90, fontsize=10, va=self.va_combo.currentText())
            plt.xlabel("Longitude", fontsize=12)
            plt.ylabel("Latitude", fontsize=12)
            plt.grid(False)
            fig.tight_layout()
            return fig

        self.plot_worker = PlotWorker(create_plot)
        self.plot_worker.finished.connect(lambda fig: self._display_plot(fig, generation))
        self.plot_worker.error.connect(lambda e: self._show_error(str(e)))
        self.plot_worker.start()

    # ------------------------------------------------------------------
    # Statistics Plot
    # ------------------------------------------------------------------

    def _plot_statistics(self, generation=None):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available")
            self.progress_bar.setVisible(False)
            return

        selected_files = [
            self.files_multiple.itemText(i) for i in range(self.files_multiple.count())
        ]
        if not selected_files:
            QMessageBox.warning(self, "Warning", "Please load files first")
            self.progress_bar.setVisible(False)
            return

        plot_type = self.stats_plot_type.currentText()
        color_palette = self.stats_palette.currentText()
        use_sampling = self.stats_sample.isChecked()
        sample_size = self.stats_sample_size.value()

        def create_plot():
            data_dict = {"Files": [], "Pixel Values": []}
            for file_name in selected_files:
                file_path = self.file_dict.get(file_name)
                if file_path:
                    data, _, _ = read_raster_memory_efficient(file_path, self.max_pixels)
                    data = data.flatten()
                    data = data[~np.isnan(data)]
                    data = normalize_data(data)

                    if use_sampling and len(data) > sample_size:
                        indices = np.random.choice(len(data), sample_size, replace=False)
                        data = data[indices]

                    Q1 = np.percentile(data, 25)
                    Q3 = np.percentile(data, 75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    filtered = data[(data >= lower) & (data <= upper)]

                    for value in filtered:
                        data_dict["Files"].append(file_name)
                        data_dict["Pixel Values"].append(value)

            data_df = pd.DataFrame(data_dict)
            if data_df.empty:
                raise Exception("No valid data after filtering")

            quality = self.plot_quality.currentText()
            dpi_val = 72 if quality == 'Fast (Draft)' else 100
            fig = plt.figure(figsize=(8, 6), dpi=dpi_val)
            colors = sns.color_palette(color_palette, len(selected_files))

            if plot_type == "Box Plot":
                sns.boxplot(data=data_df, x="Files", y="Pixel Values", width=0.5, palette=colors)
                if quality != 'Fast (Draft)':
                    sns.stripplot(
                        data=data_df, x="Files", y="Pixel Values",
                        palette=colors, alpha=0.3, jitter=0.2, size=1,
                    )
            elif plot_type == "Violin Plot":
                sns.violinplot(
                    data=data_df, x="Files", y="Pixel Values",
                    inner="quartile", width=0.8, palette=colors,
                )
                if quality != 'Fast (Draft)':
                    sns.stripplot(
                        data=data_df, x="Files", y="Pixel Values",
                        palette=colors, alpha=0.3, jitter=0.2, size=1,
                    )
            elif plot_type == "Histogram":
                sns.histplot(
                    data=data_df, x="Pixel Values", stat="density",
                    hue="Files", common_norm=False, palette=colors, alpha=0.5,
                )
            elif plot_type == "KDE Plot":
                sns.kdeplot(
                    data=data_df, x="Pixel Values", hue="Files",
                    palette=colors, lw=1.5,
                )

            plt.xlabel(
                "Rao's Value" if plot_type in ["Histogram", "KDE Plot"] else "Files",
                fontsize=12,
            )
            plt.ylabel(
                "Rao's Value" if plot_type in ["Box Plot", "Violin Plot"] else "Density",
                fontsize=12,
            )
            plt.grid(True, alpha=0.3)
            sns.despine(trim=True, offset=5)
            plt.xticks(rotation=45, fontsize=10)
            fig.tight_layout()
            return fig

        self.plot_worker = PlotWorker(create_plot)
        self.plot_worker.finished.connect(lambda fig: self._display_plot(fig, generation))
        self.plot_worker.error.connect(lambda e: self._show_error(str(e)))
        self.plot_worker.start()

    # ------------------------------------------------------------------
    # Diff + Heatmap
    # ------------------------------------------------------------------

    def _plot_diff_heat(self, generation=None):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available")
            self.progress_bar.setVisible(False)
            return

        file1_name = self.diff_file1.currentText()
        file2_name = self.diff_file2.currentText()
        if not file1_name or not file2_name:
            QMessageBox.warning(self, "Warning", "Please select two files")
            self.progress_bar.setVisible(False)
            return
        if file1_name == file2_name:
            QMessageBox.warning(self, "Warning", "Please select two different files")
            self.progress_bar.setVisible(False)
            return

        show_heatmap = self.show_heatmap_checkbox.isChecked()
        target_grid_size = self.heatmap_window.value()
        quality = self.plot_quality.currentText()
        use_cache = self.cache_checkbox.isChecked()

        def create_plot():
            file1_path = self.file_dict.get(file1_name)
            file2_path = self.file_dict.get(file2_name)

            if use_cache:
                data1, transform1, _ = self._get_file_data(file1_path)
                data2, transform2, _ = self._get_file_data(file2_path)
            else:
                data1, transform1, _ = read_raster_memory_efficient(file1_path, self.max_pixels)
                data2, transform2, _ = read_raster_memory_efficient(file2_path, self.max_pixels)

            if data1 is None or data2 is None:
                raise Exception("Failed to read files")

            if data1.shape != data2.shape:
                min_rows = min(data1.shape[0], data2.shape[0])
                min_cols = min(data1.shape[1], data2.shape[1])
                data1 = data1[:min_rows, :min_cols]
                data2 = data2[:min_rows, :min_cols]

            data1_norm = normalize_data(data1)
            data2_norm = normalize_data(data2)
            diff_data = np.abs(data1_norm - data2_norm)
            self.subtraction_data = diff_data

            nrows, ncols = diff_data.shape
            transform = transform2 if transform2 is not None else transform1
            x_coords = np.arange(0, ncols) * transform.a + transform.c
            y_coords = np.arange(0, nrows) * transform.e + transform.f

            num_cells_x = max(2, min(25, ncols // target_grid_size))
            num_cells_y = max(2, min(25, nrows // target_grid_size))
            cell_width_pixels = ncols / num_cells_x
            cell_height_pixels = nrows / num_cells_y

            fig_width_inches = 10
            fig_height_inches = 8
            cell_width_inches = fig_width_inches / num_cells_x
            cell_height_inches = fig_height_inches / num_cells_y
            cell_size_inches = min(cell_width_inches, cell_height_inches)
            optimal_font_size = max(4, min(14, (cell_size_inches * 0.85) / 0.14))

            dpi_val = 100 if quality == 'Fast (Draft)' else 150
            fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi_val)
            ax = fig.add_subplot(111)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))

            vmin = np.nanmin(diff_data)
            vmax = np.nanmax(diff_data)
            interpolation = 'bilinear' if quality != 'Fast (Draft)' else 'nearest'
            im = ax.imshow(
                np.ma.masked_invalid(diff_data),
                cmap=get_cmap_with_nan(self.diff_colormap.currentText()),
                extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                vmin=vmin, vmax=vmax, interpolation=interpolation,
            )

            # Always add colorbar for the diff plot
            cbar = fig.colorbar(im, label="Difference Value", pad=0.01)
            cbar.ax.get_yaxis().label.set_fontsize(12)

            if show_heatmap:
                grid_means = np.zeros((num_cells_y, num_cells_x))
                for i in range(num_cells_y):
                    for j in range(num_cells_x):
                        r_start = int(i * cell_height_pixels)
                        r_end = int(min((i + 1) * cell_height_pixels, nrows))
                        c_start = int(j * cell_width_pixels)
                        c_end = int(min((j + 1) * cell_width_pixels, ncols))
                        grid = diff_data[r_start:r_end, c_start:c_end]
                        valid = np.isfinite(grid)
                        grid_means[i, j] = np.nanmean(grid[valid]) if np.any(valid) else np.nan

                for j in range(1, num_cells_x):
                    x_idx = int(j * cell_width_pixels)
                    if x_idx < len(x_coords):
                        ax.axvline(x=x_coords[x_idx], color='black', linewidth=0.5, alpha=0.3)
                for i in range(1, num_cells_y):
                    y_idx = int(i * cell_height_pixels)
                    if y_idx < len(y_coords):
                        ax.axhline(y=y_coords[y_idx], color='black', linewidth=0.5, alpha=0.3)

                cmap = plt.colormaps[self.diff_colormap.currentText()]
                max_val = np.nanmax(grid_means)
                min_val = np.nanmin(grid_means)
                val_range = max_val - min_val
                if val_range < 0.1:
                    fmt_str = "{:.3f}"
                elif val_range < 1:
                    fmt_str = "{:.2f}"
                else:
                    fmt_str = "{:.2f}"

                skip_text = cell_size_inches < 0.35

                for i in range(num_cells_y):
                    for j in range(num_cells_x):
                        if not np.isnan(grid_means[i, j]) and not skip_text:
                            # Cell boundaries in pixel coordinates
                            c_start = int(j * cell_width_pixels)
                            c_end = int(min((j + 1) * cell_width_pixels, ncols))
                            r_start = int(i * cell_height_pixels)
                            r_end = int(min((i + 1) * cell_height_pixels, nrows))

                            # Cell center in data coordinates
                            cx = int((c_start + c_end) / 2)
                            cy = int((r_start + r_end) / 2)
                            x_center = x_coords[cx] if cx < len(x_coords) else x_coords[-1]
                            y_center = y_coords[cy] if cy < len(y_coords) else y_coords[-1]

                            # Cell boundaries in data coordinates (for clipping)
                            x_left = x_coords[c_start] if c_start < len(x_coords) else x_coords[0]
                            x_right = x_coords[c_end - 1] if c_end - 1 < len(x_coords) else x_coords[-1]
                            y_bottom = y_coords[r_end - 1] if r_end - 1 < len(y_coords) else y_coords[-1]
                            y_top = y_coords[r_start] if r_start < len(y_coords) else y_coords[0]

                            value_str = fmt_str.format(grid_means[i, j])
                            norm_val = (grid_means[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                            bg_color = cmap(norm_val)
                            brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                            text_color = 'black' if brightness > 0.5 else 'white'

                            # Create a clipping rectangle for this cell
                            from matplotlib.patches import Rectangle
                            clip_rect = Rectangle(
                                (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
                                transform=ax.transData, fill=False, edgecolor='none'
                            )

                            ax.text(
                                x_center, y_center, value_str,
                                color=text_color, ha='center', va='center',
                                fontsize=optimal_font_size, fontweight='bold',
                                clip_on=True, clip_path=clip_rect,
                            )

            x_ticks = np.linspace(x_coords.min(), x_coords.max(), num=min(3, len(ax.get_xticks())))
            y_ticks = np.linspace(y_coords.min(), y_coords.max(), num=min(3, len(ax.get_yticks())))
            ax.set_xticklabels([
                "%.0f° %.0f' %.0f\" %s" % (
                    *decimal_degrees_to_dms(val), "W" if val < 0 else "E"
                ) for val in x_ticks
            ])
            ax.set_yticklabels([
                "%.0f° %.0f' %.0f\" %s" % (
                    *decimal_degrees_to_dms(val), "S" if val < 0 else "N"
                ) for val in y_ticks
            ])
            plt.xticks(rotation=0, fontsize=10, ha=self.ha_combo.currentText())
            plt.yticks(rotation=90, fontsize=10, va=self.va_combo.currentText())
            plt.xlabel("Longitude", fontsize=12)
            plt.ylabel("Latitude", fontsize=12)
            plt.grid(False)
            fig.tight_layout()
            return fig

        self.plot_worker = PlotWorker(create_plot)
        self.plot_worker.finished.connect(lambda fig: self._display_plot(fig, generation))
        self.plot_worker.error.connect(lambda e: self._show_error(str(e)))
        self.plot_worker.start()

    # ------------------------------------------------------------------
    # Split Plot
    # ------------------------------------------------------------------

    def _plot_split(self, generation=None):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available")
            self.progress_bar.setVisible(False)
            return

        file1_name = self.split_file1.currentText()
        file2_name = self.split_file2.currentText()
        if not file1_name or not file2_name:
            QMessageBox.warning(self, "Warning", "Please select two files")
            self.progress_bar.setVisible(False)
            return

        quality = self.plot_quality.currentText()

        def create_plot():
            file1_path = self.file_dict.get(file1_name)
            file2_path = self.file_dict.get(file2_name)

            data1, transform1, _ = read_raster_memory_efficient(file1_path, self.max_pixels)
            data2, transform2, _ = read_raster_memory_efficient(file2_path, self.max_pixels)

            left_data = normalize_data(data1)
            right_data = normalize_data(data2)

            if left_data.shape != right_data.shape:
                min_rows = min(left_data.shape[0], right_data.shape[0])
                min_cols = min(left_data.shape[1], right_data.shape[1])
                left_data = left_data[:min_rows, :min_cols]
                right_data = right_data[:min_rows, :min_cols]

            split_size = self.split_size.value() / 100.0
            split_position = int(split_size * left_data.shape[1])
            combined_data = np.copy(left_data)
            combined_data[:, split_position:] = right_data[:, split_position:]

            num_rows, num_cols = combined_data.shape
            transform = transform2 if transform2 is not None else transform1
            x = np.arange(0, num_cols) * transform.a + transform.c
            y = np.arange(0, num_rows) * transform.e + transform.f

            dpi_val = 72 if quality == 'Fast (Draft)' else 100
            fig = plt.figure(figsize=(8, 6), dpi=dpi_val)
            ax = fig.add_subplot(111)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))

            interpolation = 'bilinear' if quality != 'Fast (Draft)' else 'nearest'
            im = ax.imshow(
                np.ma.masked_invalid(combined_data),
                cmap=get_cmap_with_nan(self.split_colormap.currentText()),
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="upper", interpolation=interpolation,
            )
            cbar = fig.colorbar(im, label="Value", pad=0.01)
            cbar.ax.get_yaxis().label.set_fontsize(12)

            split_x = x[split_position] if split_position < len(x) else x[-1]
            ax.axvline(x=split_x, color="red", linestyle="--", linewidth=2)

            ax.text(
                x[0] + (split_x - x[0]) / 2, y[-1] + (y[0] - y[-1]) * 0.05,
                file1_name, color='white', fontsize=10, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            )
            ax.text(
                split_x + (x[-1] - split_x) / 2, y[-1] + (y[0] - y[-1]) * 0.05,
                file2_name, color='white', fontsize=10, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            )

            # Format lat/long as DMS (same as other plots)
            x_ticks = np.linspace(x.min(), x.max(), num=min(3, len(ax.get_xticks())))
            y_ticks = np.linspace(y.min(), y.max(), num=min(3, len(ax.get_yticks())))
            ax.set_xticklabels([
                "%.0f° %.0f' %.0f\" %s" % (
                    *decimal_degrees_to_dms(val), "W" if val < 0 else "E"
                ) for val in x_ticks
            ])
            ax.set_yticklabels([
                "%.0f° %.0f' %.0f\" %s" % (
                    *decimal_degrees_to_dms(val), "S" if val < 0 else "N"
                ) for val in y_ticks
            ])
            plt.xticks(rotation=0, fontsize=10, ha=self.ha_combo.currentText())
            plt.yticks(rotation=90, fontsize=10, va=self.va_combo.currentText())

            plt.xlabel("Longitude", fontsize=12)
            plt.ylabel("Latitude", fontsize=12)
            plt.grid(False)
            fig.tight_layout()
            return fig

        self.plot_worker = PlotWorker(create_plot)
        self.plot_worker.finished.connect(lambda fig: self._display_plot(fig, generation))
        self.plot_worker.error.connect(lambda e: self._show_error(str(e)))
        self.plot_worker.start()

    # ------------------------------------------------------------------
    # GIF Generation
    # ------------------------------------------------------------------

    def _generate_gif(self):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available for GIF generation")
            return

        file1_name = self.split_file1.currentText()
        file2_name = self.split_file2.currentText()
        if not file1_name or not file2_name:
            QMessageBox.warning(self, "Warning", "Please select two files for GIF animation")
            return
        if file1_name == file2_name:
            QMessageBox.warning(self, "Warning", "Please select two different files")
            return

        file1_path = self.file_dict.get(file1_name)
        file2_path = self.file_dict.get(file2_name)
        if not file1_path or not file2_path:
            QMessageBox.warning(self, "Warning", "Files not found")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for GIF")
        if not output_dir:
            return

        output_name = "split_animation"
        dpi = int(self.save_dpi.currentText())
        fps = self.gif_fps.value()
        colormap = self.split_colormap.currentText()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("● Generating split GIF animation...")

        data1, _, _ = read_raster_memory_efficient(file1_path, max_pixels=250000)
        data2, _, _ = read_raster_memory_efficient(file2_path, max_pixels=250000)
        left_data = normalize_data(data1)
        right_data = normalize_data(data2)

        self.gif_worker = SplitGIFWorker(
            left_data, right_data, file1_name, file2_name,
            colormap, output_dir, output_name, dpi, fps,
        )
        self.gif_worker.finished.connect(self._on_gif_complete)
        self.gif_worker.error.connect(self._on_gif_error)
        self.gif_worker.start()

    def _generate_timeseries_gif(self):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available for GIF generation")
            return

        if not self.timeseries_selected_files:
            QMessageBox.warning(self, "Warning", "Please select files for time series first")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Time Series GIF"
        )
        if not output_dir:
            return

        output_name = "timeseries"
        dpi = int(self.timeseries_dpi.currentText())
        fps = self.timeseries_fps.value()
        colormap = self.timeseries_colormap.currentText()
        normalize_all = self.normalize_checkbox.isChecked()

        file_paths = [self.file_dict[name] for name in self.timeseries_selected_files]

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText("● Generating time series GIF...")

        self.timeseries_worker = GifWorker(
            file_paths, self.timeseries_selected_files, colormap,
            output_dir, output_name, dpi, fps, normalize_all,
        )
        self.timeseries_worker.finished.connect(self._on_timeseries_complete)
        self.timeseries_worker.error.connect(self._on_timeseries_error)
        self.timeseries_worker.progress.connect(self._on_worker_progress)
        self.timeseries_worker.start()

    def _on_worker_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_timeseries_complete(self, output_path):
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            f"● Time series GIF saved to {os.path.basename(output_path)}"
        )
        QMessageBox.information(self, "Success", f"Time series GIF saved to:\n{output_path}")

        # Display the GIF in the plot section
        self._display_gif(output_path)

    def _on_timeseries_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.status_label.setText("● Time series GIF generation failed")
        QMessageBox.critical(
            self, "Error", f"Failed to generate time series GIF:\n{error_msg}"
        )

    def _toggle_heatmap_controls(self, checked):
        self.heatmap_window.setEnabled(checked)
        self.adaptive_grid_btn.setEnabled(checked)

    def _on_gif_complete(self, output_path):
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            f"● Split GIF saved to {os.path.basename(output_path)}"
        )
        QMessageBox.information(self, "Success", f"Split GIF saved to:\n{output_path}")

    def _on_gif_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.status_label.setText("● Split GIF generation failed")
        QMessageBox.critical(
            self, "Error", f"Failed to generate split GIF:\n{error_msg}"
        )

    def _show_error(self, msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", str(msg))


__all__ = ["VisualizationWidget"]
