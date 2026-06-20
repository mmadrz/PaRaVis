"""
About dialog — application information and credits.
"""
from multiprocessing import cpu_count

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Qt

from paravis.__version__ import __version__
from paravis.utils.system import SystemProfiler
from paravis.core.raoq.gpu import GPU_AVAILABLE, GPU_BACKEND, CUSTOM_KERNEL_AVAILABLE


class AboutDialog(QDialog):
    """About dialog with application information and system status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About PaRaVis")
        self.setFixedSize(480, 360)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 20)

        header = QLabel("🌍 PaRaVis")
        header.setStyleSheet("font-size: 22px; font-weight: 700; color: #00695c;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        version = QLabel(f"v{__version__}")
        version.setStyleSheet("font-size: 13px; color: #009688; font-weight: 500;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #c0c0c0; max-height: 1px;")
        layout.addWidget(divider)

        desc = QLabel(
            "A suite for spectral index computation, Rao's Q diversity\n"
            "analysis, and raster data visualization.\n\n"
            "Built with PySide6, NumPy, Rasterio, and CuPy."
        )
        desc.setStyleSheet("font-size: 12px; color: #555555; line-height: 1.6;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setStyleSheet("background-color: #c0c0c0; max-height: 1px;")
        layout.addWidget(divider2)

        profiler = SystemProfiler()
        cpu_info = profiler.get_cpu_info()
        gpu_info = profiler.get_gpu_info()

        # Build GPU info string like v1 (directly uses GPU_AVAILABLE/GPU_BACKEND/CUSTOM_KERNEL_AVAILABLE)
        if GPU_AVAILABLE:
            gpu_str = f"GPU: {gpu_info.get('name', 'Available')} ({GPU_BACKEND})"
            if CUSTOM_KERNEL_AVAILABLE:
                gpu_str += " — CUDA"
        else:
            gpu_str = "GPU: Not Available"

        sys_info = QLabel(
            f"CPU Cores: {cpu_info['logical_cores']}  |  {gpu_str}"
        )
        sys_info.setStyleSheet("font-size: 11px; color: #888888;")
        sys_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sys_info)

        if GPU_AVAILABLE and GPU_BACKEND == "CuPy" and CUSTOM_KERNEL_AVAILABLE:
            kernel_info = QLabel("✓ Custom CUDA Kernel Active — True GPU Parallelism")
            kernel_info.setStyleSheet("font-size: 11px; color: #009688; font-weight: 600;")
            kernel_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(kernel_info)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setProperty("class", "primary")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)


__all__ = ["AboutDialog"]
