"""
Main application window — 3-column layout with menu bar and status bar.
"""
import os
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QScrollArea,
    QLabel, QStatusBar, QFrame, QSizePolicy,
    QMessageBox, QStyle, QPushButton,
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QFont, QKeySequence, QDesktopServices

from paravis.utils.settings import AppSettings
from paravis.utils.system import SystemProfiler

from .widgets.indices_panel import IndicesWidget
from .widgets.raoq_panel import RaoQWidget
from .widgets.viz_panel import VisualizationWidget

from paravis.gui.dialogs.about import AboutDialog

# v2 implementations — no v1 dependency
from paravis.core.raoq.gpu import GPU_AVAILABLE, GPU_BACKEND, CUSTOM_KERNEL_AVAILABLE


class MainWindow(QMainWindow):
    """Main application window with modern professional interface."""

    def __init__(self, splash=None):
        super().__init__()
        self.splash = splash
        self.settings = AppSettings()
        self.current_theme = self.settings.get_theme()
        self.init_ui()
        self.restore_state()
        self.apply_theme(self.current_theme)

    def init_ui(self):
        """Initialize the complete user interface."""
        self._update_splash("Building interface...")

        self.setWindowTitle("PaRaVis — Indices · Rao's Q · Visualization")
        self.setMinimumSize(1200, 720)

        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()

        # Drag & drop support
        self.setAcceptDrops(True)

        self._update_splash("Ready!")

    def _update_splash(self, message):
        if self.splash and self.splash.isVisible():
            self.splash.update_status(message)

    def _create_menu_bar(self):
        """Build the application menu bar — minimal design with View menu only."""
        menubar = self.menuBar()

        # ---- View Menu ----
        view_menu = menubar.addMenu("&View")

        self.toggle_indices = QAction("Indices Panel", self)
        self.toggle_indices.setCheckable(True)
        self.toggle_indices.setChecked(True)
        self.toggle_indices.triggered.connect(lambda: self._toggle_panel(0))
        view_menu.addAction(self.toggle_indices)

        self.toggle_raoq = QAction("Rao's Q Panel", self)
        self.toggle_raoq.setCheckable(True)
        self.toggle_raoq.setChecked(True)
        self.toggle_raoq.triggered.connect(lambda: self._toggle_panel(1))
        view_menu.addAction(self.toggle_raoq)

        self.toggle_viz = QAction("Visualization Panel", self)
        self.toggle_viz.setCheckable(True)
        self.toggle_viz.setChecked(True)
        self.toggle_viz.triggered.connect(lambda: self._toggle_panel(2))
        view_menu.addAction(self.toggle_viz)

        view_menu.addSeparator()

        theme_menu = view_menu.addMenu("Theme")
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)

        self.light_theme_action = QAction("Light", self, checkable=True)
        self.light_theme_action.setChecked(self.current_theme == "light")
        self.light_theme_action.triggered.connect(lambda: self._switch_theme("light"))
        self.theme_group.addAction(self.light_theme_action)
        theme_menu.addAction(self.light_theme_action)

        self.dark_theme_action = QAction("Dark", self, checkable=True)
        self.dark_theme_action.setChecked(self.current_theme == "dark")
        self.dark_theme_action.triggered.connect(lambda: self._switch_theme("dark"))
        self.theme_group.addAction(self.dark_theme_action)
        theme_menu.addAction(self.dark_theme_action)

        view_menu.addSeparator()

        fullscreen_action = QAction("Toggle Full Screen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        view_menu.addSeparator()

        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout_action)

        # ---- Actions directly on menu bar ----
        menubar.addSeparator()

        full_btn = QAction("Full Screen", self)
        full_btn.setShortcut(QKeySequence("F11"))
        full_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        full_btn.triggered.connect(self._toggle_fullscreen)
        menubar.addAction(full_btn)

        # App info button
        info_action = QAction("Info", self)
        info_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        info_action.triggered.connect(self._show_about)
        menubar.addAction(info_action)

        # Exit button — placed at top-right corner with red styling
        exit_btn = QPushButton("Exit", self)
        exit_btn.setShortcut(QKeySequence("Ctrl+Q"))
        exit_btn.clicked.connect(self.close)
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border: none;
                padding: 4px 12px;
                margin: 2px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        menubar.setCornerWidget(exit_btn, Qt.Corner.TopRightCorner)

    def _create_central_widget(self):
        """Build the 3-column layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setChildrenCollapsible(True)
        splitter.setObjectName("mainSplitter")

        # LEFT — Indices
        self._update_splash("Loading Indices module...")
        left_sa = QScrollArea()
        left_sa.setWidgetResizable(True)
        left_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_sa.setFrameShape(QFrame.NoFrame)
        left_sa.viewport().setAutoFillBackground(False)
        self.indices_widget = IndicesWidget()
        left_sa.setWidget(self.indices_widget)
        splitter.addWidget(left_sa)

        # MIDDLE — Rao's Q
        self._update_splash("Loading Rao's Q module...")
        mid_sa = QScrollArea()
        mid_sa.setWidgetResizable(True)
        mid_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        mid_sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        mid_sa.setFrameShape(QFrame.NoFrame)
        mid_sa.viewport().setAutoFillBackground(False)
        self.raoq_widget = RaoQWidget()
        mid_sa.setWidget(self.raoq_widget)
        splitter.addWidget(mid_sa)

        # RIGHT — Visualization
        self._update_splash("Loading Visualization module...")
        right_sa = QScrollArea()
        right_sa.setWidgetResizable(True)
        right_sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_sa.setFrameShape(QFrame.NoFrame)
        right_sa.viewport().setAutoFillBackground(False)
        self.viz_widget = VisualizationWidget()
        right_sa.setWidget(self.viz_widget)
        splitter.addWidget(right_sa)

        # Equal split
        splitter.setSizes([467, 467, 467])
        main_layout.addWidget(splitter, 1)

    def _create_status_bar(self):
        """Build the status bar with system info (v1 faithful port)."""
        status = QStatusBar()
        status.setObjectName("appStatusBar")
        self.setStatusBar(status)

        # Ready indicator (v1 style)
        self.status_indicator = QLabel()
        self.status_indicator.setVisible(False)
        status.addPermanentWidget(self.status_indicator)

        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #d0d0d0; font-size: 11px;")
        status.addPermanentWidget(sep1)

        # ── System Info (live detected) ──
        sys_prof = SystemProfiler()
        gpu_info = sys_prof.get_gpu_info()
        ram_info = sys_prof.get_ram_info()
        cpu_info = sys_prof.get_cpu_info()

        # GPU status
        if GPU_AVAILABLE:
            gpu_mem_str = f" {gpu_info['total_gb']:.0f}GB" if gpu_info['total_gb'] > 0 else ""
            kernel_str = " [CUDA]" if CUSTOM_KERNEL_AVAILABLE else ""
            gpu_text = f"GPU:{gpu_mem_str}{kernel_str}"
            gpu_label = QLabel(f"🚀 {gpu_text}")
            gpu_label.setStyleSheet("color: #009688; font-size: 11px; padding: 2px 8px;")
        else:
            gpu_label = QLabel("💻 CPU Mode")
            gpu_label.setStyleSheet("color: #888888; font-size: 11px; padding: 2px 8px;")
        status.addPermanentWidget(gpu_label)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #d0d0d0; font-size: 11px;")
        status.addPermanentWidget(sep2)

        # CPU + RAM combined
        ram_str = f" {ram_info['total_gb']:.0f}GB" if ram_info['total_gb'] > 0 else ""
        cpu_label = QLabel(f"🧠 {cpu_info['logical_cores']}C{ram_str} RAM")
        cpu_label.setStyleSheet("color: #888888; font-size: 11px; padding: 2px 8px;")
        status.addPermanentWidget(cpu_label)

        # Left side message
        self.status_message = QLabel(" ")
        self.status_message.setStyleSheet("color: #666666; font-size: 11px; padding: 2px 8px;")
        status.addWidget(self.status_message, 1)

    # ------------------------------------------------------------------
    # Actions (v1 faithful port)
    # ------------------------------------------------------------------

    def _on_open_files(self):
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Raster Files", self.settings.get_last_dir(),
            "Raster Files (*.tif *.tiff *.img *.dat);;All Files (*.*)"
        )
        if files:
            self.settings.set_last_dir(os.path.dirname(files[0]))
            for f in files:
                self.settings.add_recent_file(f)
            self.viz_widget.load_files_direct(files)
            self.indices_widget.add_files_direct(files)
            self.status_message.setText(f"Loaded {len(files)} raster file(s)")
            self.status_indicator.setText("Files loaded")
            self.status_indicator.setStyleSheet("color: #009688; font-weight: 600; font-size: 11px;")
            self.status_indicator.setVisible(True)

    def _on_open_folder(self):
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", self.settings.get_last_dir()
        )
        if folder:
            self.settings.set_last_dir(folder)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_recent_file_selected(self, file_path):
        if os.path.exists(file_path):
            self.viz_widget.load_files_direct([file_path])
            self.indices_widget.add_files_direct([file_path])
            self.status_message.setText(f"Opened: {os.path.basename(file_path)}")
        else:
            QMessageBox.warning(self, "File Not Found",
                               f"The file '{file_path}' no longer exists.")
            files = self.settings.get_recent_files()
            files = [f for f in files if f != file_path]
            self.settings.qs.setValue("recent_files", json.dumps(files))

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _toggle_panel(self, index):
        splitter = self.findChild(QSplitter, "mainSplitter")
        if splitter and index < splitter.count():
            widget = splitter.widget(index)
            widget.setVisible(not widget.isVisible())

    def _reset_layout(self):
        splitter = self.findChild(QSplitter, "mainSplitter")
        if splitter:
            total = splitter.width()
            w = total // splitter.count()
            splitter.setSizes([w, w, w])
            for i in range(splitter.count()):
                splitter.widget(i).setVisible(True)
            self.toggle_indices.setChecked(True)
            self.toggle_raoq.setChecked(True)
            self.toggle_viz.setChecked(True)
            self.status_message.setText("Layout reset")

    def _switch_theme(self, theme):
        """Switch between light and dark themes."""
        self.current_theme = theme
        self.settings.set_theme(theme)
        self.apply_theme(theme)
        self.light_theme_action.setChecked(theme == "light")
        self.dark_theme_action.setChecked(theme == "dark")
        self.status_message.setText(f"Theme switched to {theme.title()}")

    def _run_indices(self):
        self.status_message.setText("Configure Indices settings and click RUN")

    def _run_raoq(self):
        self.status_message.setText("Configure Rao's Q settings and click RUN")

    def _stop_processing(self):
        if hasattr(self.indices_widget, 'stop_processing'):
            self.indices_widget.stop_processing()
        if hasattr(self.raoq_widget, 'stop_processing'):
            self.raoq_widget.stop_processing()
        self.status_message.setText("Processing stopped")
        self.status_indicator.setText("Stopped")
        self.status_indicator.setStyleSheet("color: #e74c3c; font-weight: 600; font-size: 11px;")
        self.status_indicator.setVisible(True)

    def _show_about(self):
        dialog = AboutDialog(self)
        dialog.exec()

    # ------------------------------------------------------------------
    # Window state
    # ------------------------------------------------------------------

    def restore_state(self):
        geom = self.settings.get_geometry()
        if geom:
            self.restoreGeometry(geom)
        self.showFullScreen()

        QTimer.singleShot(100, self._do_restore_splitter)

    def _do_restore_splitter(self):
        state = self.settings.get_splitter_state()
        if state:
            splitter = self.findChild(QSplitter, "mainSplitter")
            if splitter:
                splitter.restoreState(state)

    def save_state(self):
        self.settings.set_geometry(self.saveGeometry())
        splitter = self.findChild(QSplitter, "mainSplitter")
        if splitter:
            self.settings.set_splitter_state(splitter.saveState())
        self.settings.set("window_state", self.saveState())

    def closeEvent(self, event):
        confirm = self.settings.get("confirm_exit", "true")
        if confirm == "true":
            reply = QMessageBox.question(
                self, "Exit",
                "Are you sure you want to exit PaRaVis?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        auto_save = self.settings.get("auto_save_layout", "true")
        if auto_save == "true":
            self.save_state()

        event.accept()

    def apply_theme(self, theme="light"):
        """Apply the selected stylesheet."""
        theme_file = "light_teal.qss" if theme == "light" else "dark_teal.qss"
        style_path = os.path.join(os.path.dirname(__file__), "theme", theme_file)
        if os.path.exists(style_path):
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith(('.tif', '.tiff', '.img', '.dat')):
                    files.append(path)
        if files:
            self.viz_widget.load_files_direct(files)
            self.indices_widget.add_files_direct(files)
            for f in files:
                self.settings.add_recent_file(f)
            self.status_message.setText(f"Dropped {len(files)} file(s)")
