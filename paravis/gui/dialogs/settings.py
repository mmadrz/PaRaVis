"""
Settings dialog — application preferences.
"""
from multiprocessing import cpu_count

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
    QWidget, QFormLayout, QSpinBox, QCheckBox,
)

from paravis.utils.settings import AppSettings


class SettingsDialog(QDialog):
    """Application settings dialog."""

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setFixedSize(500, 400)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 20)

        title = QLabel("⚙️ Application Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #00695c;")
        layout.addWidget(title)

        tabs = QTabWidget()

        # --- General tab ---
        general = QWidget()
        general_layout = QVBoxLayout()
        general_layout.setSpacing(12)

        recent_layout = QFormLayout()
        self.max_recent = QSpinBox()
        self.max_recent.setRange(5, 30)
        self.max_recent.setValue(int(self.settings.get("max_recent_files", 15)))
        recent_layout.addRow("Max recent files:", self.max_recent)
        general_layout.addLayout(recent_layout)

        tile_layout = QFormLayout()
        self.default_tile = QSpinBox()
        self.default_tile.setRange(128, 4096)
        self.default_tile.setSingleStep(128)
        self.default_tile.setValue(int(self.settings.get("default_tile_size", 1024)))
        tile_layout.addRow("Default tile size:", self.default_tile)
        general_layout.addLayout(tile_layout)

        workers_layout = QFormLayout()
        self.default_workers = QSpinBox()
        self.default_workers.setRange(1, cpu_count())
        self.default_workers.setValue(int(self.settings.get("default_workers", min(cpu_count(), 4))))
        workers_layout.addRow("Default CPU workers:", self.default_workers)
        general_layout.addLayout(workers_layout)

        self.auto_save = QCheckBox("Automatically save window layout on exit")
        self.auto_save.setChecked(self.settings.get("auto_save_layout", "true") == "true")
        general_layout.addWidget(self.auto_save)

        self.confirm_exit = QCheckBox("Confirm before closing")
        self.confirm_exit.setChecked(self.settings.get("confirm_exit", "true") == "true")
        general_layout.addWidget(self.confirm_exit)

        general_layout.addStretch()
        general.setLayout(general_layout)
        tabs.addTab(general, "General")

        # --- Performance tab ---
        perf = QWidget()
        perf_layout = QVBoxLayout()
        perf_layout.setSpacing(12)

        self.gpu_accel = QCheckBox("Enable GPU acceleration when available")
        self.gpu_accel.setChecked(self.settings.get("gpu_acceleration", "true") == "true")
        perf_layout.addWidget(self.gpu_accel)

        self.auto_batch = QCheckBox("Auto-detect optimal batch size")
        self.auto_batch.setChecked(self.settings.get("auto_batch_size", "true") == "true")
        perf_layout.addWidget(self.auto_batch)

        self.compression = QCheckBox("Use compression for output GeoTIFFs")
        self.compression.setChecked(self.settings.get("tiff_compression", "true") == "true")
        perf_layout.addWidget(self.compression)

        perf_layout.addStretch()
        perf.setLayout(perf_layout)
        tabs.addTab(perf, "Performance")

        layout.addWidget(tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 24px;
                background-color: #009688;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #00897b; }
        """)
        save_btn.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def save_and_accept(self):
        self.settings.set("max_recent_files", self.max_recent.value())
        self.settings.set("default_tile_size", self.default_tile.value())
        self.settings.set("default_workers", self.default_workers.value())
        self.settings.set("auto_save_layout", "true" if self.auto_save.isChecked() else "false")
        self.settings.set("confirm_exit", "true" if self.confirm_exit.isChecked() else "false")
        self.settings.set("gpu_acceleration", "true" if self.gpu_accel.isChecked() else "false")
        self.settings.set("auto_batch_size", "true" if self.auto_batch.isChecked() else "false")
        self.settings.set("tiff_compression", "true" if self.compression.isChecked() else "false")
        self.accept()


__all__ = ["SettingsDialog"]
