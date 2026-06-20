"""
Widget panel for Rao's Q diversity computation — v1 faithful port.

Exact match of v1 RaoQWidget: method names, signals, parameter layout
(QGridLayout), progress reporting (current/total windows), and
BatchProcessingManager interface.
"""
import os
import gc
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QMessageBox, QGroupBox, QTabWidget, QSpinBox,
    QRadioButton, QProgressBar, QTextEdit, QSlider, QLineEdit,
    QComboBox, QListWidget, QInputDialog, QDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from paravis.core.raoq import RaoQConfig
from paravis.core.raoq.gpu import is_gpu_available, GPU_AVAILABLE, GPU_BACKEND, CUSTOM_KERNEL_AVAILABLE
from paravis.utils.system import SystemProfiler
from paravis.workers.raoq_worker import RaoQWorker


class BatchProcessingManager(QThread):
    """Manages batch processing of multiple Rao's Q jobs (v1 parity)."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int)  # current, total (v1 style)
    finished_signal = Signal(bool, str)
    current_job_signal = Signal(str, int)

    def __init__(self, job_list, parameters, parent=None):
        super().__init__(parent)
        self.job_list = job_list
        self.parameters = parameters
        self.current_worker = None
        self.is_running = True

    def run(self):
        try:
            total_jobs = len(self.job_list)
            self.log_signal.emit(f"\n{'='*70}")
            self.log_signal.emit(f"📋 BATCH PROCESSING STARTED — {total_jobs} jobs")
            self.log_signal.emit(f"{'='*70}\n")

            for job_idx, (input_files, output_name) in enumerate(self.job_list, 1):
                if not self.is_running:
                    self.log_signal.emit("\n⚠️ Batch processing stopped by user")
                    break

                self.current_job_signal.emit(f"Job {job_idx}/{total_jobs}: {output_name}", job_idx)

                self.log_signal.emit(f"\n{'─'*70}")
                self.log_signal.emit(f"▶️  STARTING JOB {job_idx}/{total_jobs}")
                self.log_signal.emit(f"📝 Output: {output_name}")
                self.log_signal.emit(f"{'─'*70}")

                output_path = os.path.join(
                    self.parameters['output_folder'], f"{output_name}.tif"
                )

                self.current_worker = RaoQWorker(
                    raster_paths=input_files,
                    output_path=output_path,
                    distance_m=self.parameters['distance_m'],
                    window=self.parameters['window'],
                    na_tolerance=self.parameters['na_tolerance'],
                    block_size=self.parameters.get('block_size', 1024),
                    num_workers=self.parameters['num_workers'],
                    p_minkowski=self.parameters.get('p_minkowski', 2),
                    use_gpu=self.parameters['use_gpu'],
                    simplify=self.parameters.get('simplify', 2),
                )
                self.current_worker.log_signal.connect(self.log_signal)
                self.current_worker.progress_signal.connect(lambda c, t: None)
                self.current_worker.finished_signal.connect(
                    lambda success, msg, name=output_name: self.log_signal.emit(
                        f"✅ Job '{name}': {msg}" if success else f"❌ Job '{name}': {msg}"
                    )
                )
                self.current_worker.start()
                self.current_worker.wait()

                if not self.is_running:
                    break

            if self.is_running:
                self.log_signal.emit(f"\n{'='*70}")
                self.log_signal.emit(f"✅ BATCH PROCESSING COMPLETED — {total_jobs} jobs")
                self.log_signal.emit(f"{'='*70}")
                self.finished_signal.emit(True, f"Batch processing completed! Processed {total_jobs} jobs.")
            else:
                self.log_signal.emit(f"\n⚠️ Batch stopped after {job_idx-1} jobs")
                self.finished_signal.emit(False, "Batch processing stopped by user")

        except Exception as e:
            self.log_signal.emit(f"\n❌ Batch error: {e}")
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self.is_running = False
        if self.current_worker:
            self.current_worker.stop()


class RaoQWidget(QWidget):
    """Panel for computing Rao's Q diversity (v1 faithful port)."""

    def __init__(self):
        super().__init__()
        self.input_files = []
        self.output_folder = ""
        self.worker = None
        self.batch_manager = None
        self.batch_jobs = []

        # Auto-detect system
        self.sys_profile = SystemProfiler.get_auto_config(n_bands_guess=3, window_guess=5)
        self.sys_info_str = SystemProfiler.summary_string(n_bands_guess=3, window_guess=5)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Title
        title = QLabel("Rao's Q")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet("background-color: #c0c0c0; margin: 4px 0;")
        line.setFixedHeight(1)
        layout.addWidget(line)

        # Tab widget
        self.tab_widget = QTabWidget()

        self.single_tab = QWidget()
        self.setup_single_tab()
        self.tab_widget.addTab(self.single_tab, "Single")

        self.batch_tab = QWidget()
        self.setup_batch_tab()
        self.tab_widget.addTab(self.batch_tab, "Batch")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

        # Connect mode toggles
        self.gpu_radio.toggled.connect(self.on_mode_changed)
        self.cpu_radio.toggled.connect(self.on_mode_changed)
        self.on_mode_changed()

        self.batch_gpu_radio.toggled.connect(self.on_batch_mode_changed)
        self.batch_cpu_radio.toggled.connect(self.on_batch_mode_changed)
        self.on_batch_mode_changed()

        # Log system profile
        self.log_system_profile()

    def log_system_profile(self):
        profile = SystemProfiler.summary_string(n_bands_guess=3, window_guess=5)
        for line_text in profile.split('\n'):
            self.log_display.append(line_text)
        self.log_display.append("")

    # ------------------------------------------------------------------
    # Single tab
    # ------------------------------------------------------------------

    def setup_single_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # ── Input Files group ──
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(4)

        row1 = QHBoxLayout()
        self.input_btn = QPushButton("Select Input Files")
        self.input_btn.clicked.connect(self.select_input_files)
        row1.addWidget(self.input_btn, 1)

        self.output_btn = QPushButton("Output Folder")
        self.output_btn.clicked.connect(self.select_output_folder)
        row1.addWidget(self.output_btn, 1)

        row1.addWidget(QLabel("Name:"))
        self.output_name = QLineEdit("Rao_Q")
        self.output_name.setMaximumWidth(120)
        row1.addWidget(self.output_name, 1)
        input_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.input_display = QTextEdit()
        self.input_display.setPlaceholderText("No files selected")
        self.input_display.setMaximumHeight(50)
        self.input_display.setReadOnly(True)
        row2.addWidget(self.input_display, 2)

        out_col = QVBoxLayout()
        out_col.setSpacing(2)
        out_col.addWidget(QLabel("Output path:"))
        self.output_path_display = QLineEdit()
        self.output_path_display.setReadOnly(True)
        out_col.addWidget(self.output_path_display)
        row2.addLayout(out_col, 1)
        input_layout.addLayout(row2)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # ── Processing Mode group ──
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)

        self.gpu_radio = QRadioButton("GPU")
        self.cpu_radio = QRadioButton("CPU")

        if GPU_AVAILABLE:
            self.gpu_radio.setEnabled(True)
            self.gpu_radio.setChecked(True)
            mode_layout.addWidget(self.gpu_radio)

            gpu_mem = self.sys_profile.get('gpu', {}).get('total_gb', 0)
            mem_str = f" | {gpu_mem:.0f}GB" if gpu_mem > 0 else ""
            if GPU_BACKEND == "CuPy" and CUSTOM_KERNEL_AVAILABLE:
                gpu_info_text = QLabel(f"{GPU_BACKEND}{mem_str} — CUDA")
                gpu_info_text.setStyleSheet("color: #009688; font-weight: 600; font-size: 10px;")
            else:
                gpu_info_text = QLabel(f"{GPU_BACKEND}{mem_str}")
                gpu_info_text.setStyleSheet("color: #009688; font-size: 10px;")
            mode_layout.addWidget(gpu_info_text)
        else:
            self.gpu_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)
            mode_layout.addWidget(self.gpu_radio)
            no_gpu_info = QLabel("GPU unavailable")
            no_gpu_info.setStyleSheet("color: #e67e22; font-size: 10px;")
            mode_layout.addWidget(no_gpu_info)

        mode_layout.addWidget(self.cpu_radio)
        mode_layout.addStretch()

        cpu_cores_available = self.sys_profile.get('cpu', {}).get('logical_cores', 8)
        mode_layout.addWidget(QLabel("Cores:"))
        self.cpu_cores_spin = QSpinBox()
        self.cpu_cores_spin.setRange(1, cpu_cores_available)
        if self.sys_profile.get('recommended', {}).get('use_gpu', False):
            auto_cores = self.sys_profile.get('recommended', {}).get('gpu_workers', 2)
        else:
            auto_cores = self.sys_profile.get('recommended', {}).get('cpu_workers', 4)
        self.cpu_cores_spin.setValue(auto_cores)
        self.cpu_cores_spin.setMaximumWidth(60)
        self.cpu_cores_spin.setToolTip(
            f"Available CPU cores: {cpu_cores_available} | Auto-detected: {auto_cores}"
        )
        mode_layout.addWidget(self.cpu_cores_spin)
        mode_layout.addWidget(QLabel(f"(Avail: {cpu_cores_available})"))

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # ── Parameters group ──
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        grid_layout = QGridLayout()

        grid_layout.addWidget(QLabel("Distance Metric:"), 0, 0)
        self.distance_combo = QComboBox()
        self.distance_combo.addItems(["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis"])
        self.distance_combo.currentTextChanged.connect(self.on_distance_changed)
        grid_layout.addWidget(self.distance_combo, 0, 1)

        grid_layout.addWidget(QLabel("Minkowski p:"), 0, 2)
        self.p_spin = QSpinBox()
        self.p_spin.setRange(2, 50)
        self.p_spin.setValue(2)
        self.p_spin.setVisible(False)
        grid_layout.addWidget(self.p_spin, 0, 3)

        grid_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 101)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(3)
        grid_layout.addWidget(self.window_spin, 1, 1)

        grid_layout.addWidget(QLabel("NA Tolerance:"), 1, 2)
        na_container = QWidget()
        na_container_layout = QHBoxLayout()
        na_container_layout.setContentsMargins(0, 0, 0, 0)
        self.na_slider = QSlider(Qt.Orientation.Horizontal)
        self.na_slider.setRange(0, 100)
        self.na_slider.setValue(50)
        self.na_slider.valueChanged.connect(self.update_na_label)
        self.na_label = QLabel("50%")
        na_container_layout.addWidget(self.na_slider, 1)
        na_container_layout.addWidget(self.na_label)
        na_container.setLayout(na_container_layout)
        grid_layout.addWidget(na_container, 1, 3)

        grid_layout.addWidget(QLabel("Simplify (decimals):"), 2, 0)
        self.simplify_spin = QSpinBox()
        self.simplify_spin.setRange(0, 6)
        self.simplify_spin.setValue(2)
        self.simplify_spin.setToolTip("Number of decimal places to keep (0 = integers only)")
        grid_layout.addWidget(self.simplify_spin, 2, 1)

        grid_layout.addWidget(QLabel("Block Size:"), 2, 2)
        self.block_spin = QSpinBox()
        self.block_spin.setRange(128, 10000)
        self.block_spin.setSingleStep(256)
        auto_block = self.sys_profile.get('recommended', {}).get('block_size', 1024)
        self.block_spin.setValue(auto_block)
        self.block_spin.setToolTip(
            f"Auto-detected: {auto_block}. Larger = more windows per batch = higher memory usage."
        )
        grid_layout.addWidget(self.block_spin, 2, 3)

        params_layout.addLayout(grid_layout)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # ── Progress group ──
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        progress_layout.addWidget(self.status_label)

        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setProperty("class", "primary")
        self.run_btn.clicked.connect(self.run_processing)
        run_layout.addWidget(self.run_btn, 1)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("class", "danger")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        run_layout.addWidget(self.stop_btn, 1)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        run_layout.addWidget(self.clear_log_btn, 1)

        self.expand_log_btn = QPushButton("Log Window")
        self.expand_log_btn.setCheckable(True)
        self.expand_log_btn.toggled.connect(self._toggle_log_expand)
        run_layout.addWidget(self.expand_log_btn, 1)

        progress_layout.addLayout(run_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # ── Log group ──
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        self.log_display.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        self.log_group = log_group
        log_group.setMaximumHeight(180)
        layout.addWidget(log_group)
        self._log_window = None
        self._log_window_text = None
        self._progress_start_time = None

        self.single_tab.setLayout(layout)

    # ------------------------------------------------------------------
    # Batch tab
    # ------------------------------------------------------------------

    def setup_batch_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # ── Job List group ──
        job_group = QGroupBox("Job List")
        job_layout = QVBoxLayout()

        self.job_list_widget = QListWidget()
        self.job_list_widget.setMaximumHeight(200)
        job_layout.addWidget(self.job_list_widget)

        job_buttons_layout = QHBoxLayout()
        self.add_job_btn = QPushButton("Add Job")
        self.add_job_btn.clicked.connect(self.add_batch_job)
        job_buttons_layout.addWidget(self.add_job_btn, 1)

        self.remove_job_btn = QPushButton("Remove Selected")
        self.remove_job_btn.clicked.connect(self.remove_batch_job)
        job_buttons_layout.addWidget(self.remove_job_btn, 1)

        self.clear_jobs_btn = QPushButton("Clear All")
        self.clear_jobs_btn.clicked.connect(self.clear_batch_jobs)
        job_buttons_layout.addWidget(self.clear_jobs_btn, 1)

        job_layout.addLayout(job_buttons_layout)
        job_group.setLayout(job_layout)
        layout.addWidget(job_group)

        # ── Output Settings group ──
        batch_output_group = QGroupBox("Output Settings")
        batch_output_layout = QHBoxLayout()
        self.batch_output_btn = QPushButton("Select Output Folder")
        self.batch_output_btn.clicked.connect(self.select_batch_output_folder)
        batch_output_layout.addWidget(self.batch_output_btn)

        self.batch_output_display = QLineEdit()
        self.batch_output_display.setReadOnly(True)
        self.batch_output_display.setPlaceholderText("No output folder selected")
        batch_output_layout.addWidget(self.batch_output_display, 1)

        batch_output_group.setLayout(batch_output_layout)
        layout.addWidget(batch_output_group)

        # ── Processing Mode group ──
        batch_mode_group = QGroupBox("Processing Mode")
        batch_mode_layout = QHBoxLayout()
        batch_mode_layout.setSpacing(6)

        self.batch_gpu_radio = QRadioButton("GPU")
        self.batch_cpu_radio = QRadioButton("CPU")

        if GPU_AVAILABLE:
            self.batch_gpu_radio.setEnabled(True)
            self.batch_gpu_radio.setChecked(True)
            batch_mode_layout.addWidget(self.batch_gpu_radio)

            gpu_mem = self.sys_profile.get('gpu', {}).get('total_gb', 0)
            mem_str = f" | {gpu_mem:.0f}GB" if gpu_mem > 0 else ""
            if GPU_BACKEND == "CuPy" and CUSTOM_KERNEL_AVAILABLE:
                gpu_info = QLabel(f"{GPU_BACKEND}{mem_str} — CUDA")
                gpu_info.setStyleSheet("color: #009688; font-weight: 600; font-size: 10px;")
            else:
                gpu_info = QLabel(f"{GPU_BACKEND}{mem_str}")
                gpu_info.setStyleSheet("color: #009688; font-size: 10px;")
            batch_mode_layout.addWidget(gpu_info)
        else:
            self.batch_gpu_radio.setEnabled(False)
            self.batch_cpu_radio.setChecked(True)
            batch_mode_layout.addWidget(self.batch_gpu_radio)
            no_gpu_info = QLabel("GPU unavailable")
            no_gpu_info.setStyleSheet("color: #e67e22; font-size: 10px;")
            batch_mode_layout.addWidget(no_gpu_info)

        batch_mode_layout.addWidget(self.batch_cpu_radio)
        batch_mode_layout.addStretch()

        cpu_cores_available = self.sys_profile.get('cpu', {}).get('logical_cores', 8)
        batch_mode_layout.addWidget(QLabel("Cores:"))
        self.batch_cpu_cores_spin = QSpinBox()
        self.batch_cpu_cores_spin.setRange(1, cpu_cores_available)
        if self.sys_profile.get('recommended', {}).get('use_gpu', False):
            auto_cores = self.sys_profile.get('recommended', {}).get('gpu_workers', 2)
        else:
            auto_cores = self.sys_profile.get('recommended', {}).get('cpu_workers', 4)
        self.batch_cpu_cores_spin.setValue(auto_cores)
        self.batch_cpu_cores_spin.setMaximumWidth(60)
        batch_mode_layout.addWidget(self.batch_cpu_cores_spin)
        batch_mode_layout.addWidget(QLabel(f"(Avail: {cpu_cores_available})"))

        batch_mode_group.setLayout(batch_mode_layout)
        layout.addWidget(batch_mode_group)

        # ── Parameters group ──
        batch_params_group = QGroupBox("Parameters")
        batch_params_layout = QVBoxLayout()

        batch_grid_layout = QGridLayout()

        batch_grid_layout.addWidget(QLabel("Distance Metric:"), 0, 0)
        self.batch_distance_combo = QComboBox()
        self.batch_distance_combo.addItems(["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis"])
        self.batch_distance_combo.currentTextChanged.connect(self.on_batch_distance_changed)
        batch_grid_layout.addWidget(self.batch_distance_combo, 0, 1)

        batch_grid_layout.addWidget(QLabel("Minkowski p:"), 0, 2)
        self.batch_p_spin = QSpinBox()
        self.batch_p_spin.setRange(2, 50)
        self.batch_p_spin.setValue(2)
        self.batch_p_spin.setVisible(False)
        batch_grid_layout.addWidget(self.batch_p_spin, 0, 3)

        batch_grid_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.batch_window_spin = QSpinBox()
        self.batch_window_spin.setRange(3, 101)
        self.batch_window_spin.setSingleStep(2)
        self.batch_window_spin.setValue(3)
        batch_grid_layout.addWidget(self.batch_window_spin, 1, 1)

        batch_grid_layout.addWidget(QLabel("NA Tolerance:"), 1, 2)
        batch_na_container = QWidget()
        batch_na_container_layout = QHBoxLayout()
        batch_na_container_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_na_slider = QSlider(Qt.Orientation.Horizontal)
        self.batch_na_slider.setRange(0, 100)
        self.batch_na_slider.setValue(50)
        self.batch_na_slider.valueChanged.connect(self.update_batch_na_label)
        self.batch_na_label = QLabel("50%")
        batch_na_container_layout.addWidget(self.batch_na_slider, 1)
        batch_na_container_layout.addWidget(self.batch_na_label)
        batch_na_container.setLayout(batch_na_container_layout)
        batch_grid_layout.addWidget(batch_na_container, 1, 3)

        batch_grid_layout.addWidget(QLabel("Simplify (decimals):"), 2, 0)
        self.batch_simplify_spin = QSpinBox()
        self.batch_simplify_spin.setRange(0, 6)
        self.batch_simplify_spin.setValue(2)
        self.batch_simplify_spin.setToolTip("Number of decimal places to keep (0 = integers only)")
        batch_grid_layout.addWidget(self.batch_simplify_spin, 2, 1)

        batch_grid_layout.addWidget(QLabel("Block Size:"), 2, 2)
        self.batch_block_spin = QSpinBox()
        self.batch_block_spin.setRange(128, 10000)
        self.batch_block_spin.setSingleStep(256)
        auto_block = self.sys_profile.get('recommended', {}).get('block_size', 1024)
        self.batch_block_spin.setValue(auto_block)
        self.batch_block_spin.setToolTip(f"Auto-detected: {auto_block} (based on {self.sys_profile.get('ram', {}).get('available_gb', 16):.1f} GB RAM)")
        batch_grid_layout.addWidget(self.batch_block_spin, 2, 3)

        batch_params_layout.addLayout(batch_grid_layout)
        batch_params_group.setLayout(batch_params_layout)
        layout.addWidget(batch_params_group)

        # ── Batch Progress group ──
        batch_progress_group = QGroupBox("Batch Progress")
        batch_progress_layout = QVBoxLayout()

        self.batch_progress_bar = QProgressBar()
        batch_progress_layout.addWidget(self.batch_progress_bar)

        self.batch_current_job_label = QLabel("No job running")
        batch_progress_layout.addWidget(self.batch_current_job_label)

        self.batch_status_label = QLabel()
        self.batch_status_label.setVisible(False)
        batch_progress_layout.addWidget(self.batch_status_label)

        batch_run_layout = QHBoxLayout()
        self.batch_run_btn = QPushButton("Run Batch")
        self.batch_run_btn.setProperty("class", "primary")
        self.batch_run_btn.clicked.connect(self.run_batch_processing)
        batch_run_layout.addWidget(self.batch_run_btn, 1)

        self.batch_stop_btn = QPushButton("Stop Batch")
        self.batch_stop_btn.setProperty("class", "danger")
        self.batch_stop_btn.clicked.connect(self.stop_batch_processing)
        self.batch_stop_btn.setEnabled(False)
        batch_run_layout.addWidget(self.batch_stop_btn, 1)

        self.batch_clear_log_btn = QPushButton("Clear Log")
        self.batch_clear_log_btn.clicked.connect(self.clear_batch_log)
        batch_run_layout.addWidget(self.batch_clear_log_btn, 1)

        batch_progress_layout.addLayout(batch_run_layout)
        batch_progress_group.setLayout(batch_progress_layout)
        layout.addWidget(batch_progress_group)

        # ── Batch Log group ──
        batch_log_group = QGroupBox("Batch Log")
        batch_log_layout = QVBoxLayout()
        self.batch_log_display = QTextEdit()
        self.batch_log_display.setReadOnly(True)
        self.batch_log_display.setMaximumHeight(150)
        self.batch_log_display.setFont(QFont("Courier", 9))
        batch_log_layout.addWidget(self.batch_log_display)
        batch_log_group.setLayout(batch_log_layout)
        layout.addWidget(batch_log_group)

        self.batch_tab.setLayout(layout)

    # ------------------------------------------------------------------
    # Mode change handlers
    # ------------------------------------------------------------------

    def on_mode_changed(self):
        is_cpu = self.cpu_radio.isChecked()
        self.cpu_cores_spin.setEnabled(is_cpu)
        if is_cpu:
            auto_val = self.sys_profile.get('recommended', {}).get('cpu_workers', 4)
            tip_text = f"CPU mode: using all available cores ({auto_val})"
        else:
            auto_val = self.sys_profile.get('recommended', {}).get('gpu_workers', 2)
            tip_text = f"GPU mode: leaving cores free for data transfer ({auto_val})"
        self.cpu_cores_spin.setValue(auto_val)
        self.cpu_cores_spin.setToolTip(tip_text)

    def on_batch_mode_changed(self):
        is_cpu = self.batch_cpu_radio.isChecked()
        self.batch_cpu_cores_spin.setEnabled(is_cpu)
        if is_cpu:
            auto_val = self.sys_profile.get('recommended', {}).get('cpu_workers', 4)
        else:
            auto_val = self.sys_profile.get('recommended', {}).get('gpu_workers', 2)
        self.batch_cpu_cores_spin.setValue(auto_val)

    def update_na_label(self, value):
        self.na_label.setText(f"{value}%")

    def update_batch_na_label(self, value):
        self.batch_na_label.setText(f"{value}%")

    def on_distance_changed(self, text):
        self.p_spin.setVisible(text == "minkowski")

    def on_batch_distance_changed(self, text):
        self.batch_p_spin.setVisible(text == "minkowski")

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def select_input_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Files", "", "TIFF Files (*.tif *.tiff)"
        )
        if files:
            self.input_files = files
            self.input_display.setText("\n".join(files))
            self.log(f"✅ Selected {len(files)} file(s)")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_path_display.setText(folder)
            self.log(f"✅ Output folder: {folder}")

    def select_batch_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Batch")
        if folder:
            self.batch_output_display.setText(folder)
            self.batch_log(f"✅ Batch output folder: {folder}")

    # ------------------------------------------------------------------
    # Batch job management
    # ------------------------------------------------------------------

    def add_batch_job(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Files for Job", "", "TIFF Files (*.tif *.tiff)"
        )
        if files:
            output_name, ok = QInputDialog.getText(
                self, "Job Output Name",
                f"Enter output name for job (without extension):\nFiles: {len(files)} bands"
            )
            if ok and output_name:
                self.batch_jobs.append((files, output_name))
                self.update_job_list()
                self.batch_log(f"✅ Added job: {output_name} ({len(files)} files)")

    def remove_batch_job(self):
        current_row = self.job_list_widget.currentRow()
        if current_row >= 0:
            removed = self.batch_jobs.pop(current_row)
            self.update_job_list()
            self.batch_log(f"🗑️ Removed job: {removed[1]}")

    def clear_batch_jobs(self):
        self.batch_jobs.clear()
        self.update_job_list()
        self.batch_log("🗑️ Cleared all jobs")

    def update_job_list(self):
        self.job_list_widget.clear()
        for idx, (files, output_name) in enumerate(self.batch_jobs, 1):
            self.job_list_widget.addItem(f"{idx}. {output_name} ({len(files)} bands)")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def run_processing(self):
        if not self.input_files:
            QMessageBox.warning(self, "Warning", "Please select input files!")
            return
        if not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select output folder!")
            return

        output_path = os.path.join(self.output_folder, f"{self.output_name.text()}.tif")
        use_gpu = self.gpu_radio.isChecked() and GPU_AVAILABLE
        distance_m = self.distance_combo.currentText()

        if use_gpu and not GPU_AVAILABLE:
            self.log("\u26a0\ufe0f GPU not available. Switching to CPU mode.")
            use_gpu = False

        cpu_cores = self.cpu_cores_spin.value()
        simplify_val = self.simplify_spin.value()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.log_display.clear()
        self._progress_start_time = None

        self.worker = RaoQWorker(
            raster_paths=self.input_files,
            output_path=output_path,
            distance_m=distance_m,
            window=self.window_spin.value(),
            na_tolerance=self.na_slider.value() / 100.0,
            block_size=self.block_spin.value(),
            num_workers=cpu_cores,
            p_minkowski=self.p_spin.value(),
            use_gpu=use_gpu,
            simplify=simplify_val,
        )

        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_processing_finished)
        self.worker.start()

        if use_gpu:
            if GPU_BACKEND == "CuPy" and CUSTOM_KERNEL_AVAILABLE:
                self.status_label.setText(
                    f"GPU ULTRA-PARALLEL - Custom CUDA kernel | Simplify: {simplify_val}"
                )
            else:
                self.status_label.setText(
                    f"GPU processing - {GPU_BACKEND} | Simplify: {simplify_val}"
                )
        else:
            self.status_label.setText(
                f"CPU parallel processing with {cpu_cores} cores | Simplify: {simplify_val}"
            )
        self.status_label.setVisible(True)

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Stop", "Stop processing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.log("\n⚠️ Stopping...")

    def update_progress(self, current, total):
        if total > 0:
            from time import time as now
            if self._progress_start_time is None:
                self._progress_start_time = now()

            pct = (current / total) * 100
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat("%v / %m windows  (%p%)")

            elapsed = now() - self._progress_start_time
            if elapsed > 0 and current > 0:
                speed = current / elapsed
                self.status_label.setText(
                    f"Progress: {pct:.1f}%  |  {current:,} of {total:,} windows"
                    f"  |  {speed:.0f} win/s"
                )
            else:
                self.status_label.setText(
                    f"Progress: {pct:.1f}%  |  {current:,} of {total:,} windows"
                )

    def on_processing_finished(self, success, message):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if success:
            self.status_label.setText("Completed!")
            self.log(f"\n✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Failed!")
            self.log(f"\n❌ {message}")
            QMessageBox.critical(self, "Error", message)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_batch_processing(self):
        if not self.batch_jobs:
            QMessageBox.warning(self, "Warning", "Please add at least one job to the batch list!")
            return

        batch_output_folder = self.batch_output_display.text()
        if not batch_output_folder:
            QMessageBox.warning(self, "Warning", "Please select output folder for batch processing!")
            return

        use_gpu = self.batch_gpu_radio.isChecked() and GPU_AVAILABLE
        distance_m = self.batch_distance_combo.currentText()

        if use_gpu and not GPU_AVAILABLE:
            self.batch_log("\u26a0\ufe0f GPU not available. Switching to CPU mode.")
            use_gpu = False

        parameters = {
            'output_folder': batch_output_folder,
            'distance_m': distance_m,
            'window': self.batch_window_spin.value(),
            'na_tolerance': self.batch_na_slider.value() / 100.0,
            'block_size': self.batch_block_spin.value(),
            'num_workers': self.batch_cpu_cores_spin.value(),
            'p_minkowski': self.batch_p_spin.value(),
            'use_gpu': use_gpu,
            'simplify': self.batch_simplify_spin.value(),
        }

        self.batch_run_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.add_job_btn.setEnabled(False)
        self.remove_job_btn.setEnabled(False)
        self.clear_jobs_btn.setEnabled(False)

        self.batch_log_display.clear()
        self.batch_progress_bar.setValue(0)
        self.batch_status_label.setText("Processing batch...")

        self.batch_manager = BatchProcessingManager(self.batch_jobs, parameters)
        self.batch_manager.log_signal.connect(self.batch_log)
        self.batch_manager.progress_signal.connect(self.update_batch_progress)
        self.batch_manager.current_job_signal.connect(self.update_current_job)
        self.batch_manager.finished_signal.connect(self.on_batch_finished)
        self.batch_manager.start()

    def stop_batch_processing(self):
        if self.batch_manager and self.batch_manager.isRunning():
            reply = QMessageBox.question(
                self, "Stop", "Stop batch processing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.batch_manager.stop()
                self.batch_log("\n⚠️ Stopping batch...")

    def update_batch_progress(self, current, total):
        if total > 0:
            pct = (current / total) * 100
            self.batch_progress_bar.setRange(0, total)
            self.batch_progress_bar.setValue(current)
            self.batch_progress_bar.setFormat("%v / %m jobs  (%p%)")
            self.batch_status_label.setText(f"Batch progress: {pct:.1f}%  |  {current:,} of {total:,} jobs")

    def update_current_job(self, job_description, job_number):
        self.batch_current_job_label.setText(job_description)

    def on_batch_finished(self, success, message):
        self.batch_run_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.add_job_btn.setEnabled(True)
        self.remove_job_btn.setEnabled(True)
        self.clear_jobs_btn.setEnabled(True)
        if success:
            self.batch_status_label.setText("Batch completed!")
            self.batch_log(f"\n✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.batch_status_label.setText("Batch failed!")
            self.batch_log(f"\n❌ {message}")
            QMessageBox.critical(self, "Error", message)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _toggle_log_expand(self, expanded: bool):
        """Open or close a separate log window."""
        if expanded:
            self._log_window = QDialog(self)
            self._log_window.setWindowTitle("Processing Log")
            self._log_window.resize(800, 600)
            layout = QVBoxLayout(self._log_window)
            self._log_window_text = QTextEdit()
            self._log_window_text.setReadOnly(True)
            self._log_window_text.setFont(QFont("Courier", 9))
            # Copy current log content
            self._log_window_text.setPlainText(self.log_display.toPlainText())
            self._log_window_text.moveCursor(
                self._log_window_text.textCursor().MoveOperation.End
            )
            layout.addWidget(self._log_window_text)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self._close_log_window)
            layout.addWidget(close_btn)
            self._log_window.finished.connect(
                lambda: self.expand_log_btn.setChecked(False)
            )
            self._log_window.show()
            self.expand_log_btn.setText("Log Window")
        else:
            self._close_log_window()

    def _close_log_window(self):
        """Close the external log window if open."""
        if self._log_window is not None:
            self._log_window.close()
            self._log_window.deleteLater()
            self._log_window = None
            self._log_window_text = None
        self.expand_log_btn.setText("Log Window")

    def log(self, message):
        self.log_display.append(message)
        self.log_display.ensureCursorVisible()
        if self._log_window_text is not None:
            self._log_window_text.append(message)
            self._log_window_text.ensureCursorVisible()

    def batch_log(self, message):
        self.batch_log_display.append(message)
        self.batch_log_display.ensureCursorVisible()

    def clear_log(self):
        self.log_display.clear()

    def clear_batch_log(self):
        self.batch_log_display.clear()


__all__ = ["RaoQWidget"]
