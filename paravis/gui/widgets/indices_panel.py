"""
Exact port of v1 IndicesWidget from indices_calc.py.
"""
import os
import json

import spyndex
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
    QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
    QMessageBox, QProgressBar, QCheckBox, QFrame, QTableView,
    QHeaderView, QSizePolicy, QAbstractItemView, QDialog,
)
from PySide6.QtCore import Qt, QSortFilterProxyModel
from PySide6.QtGui import QFont

from paravis.gui.models.index_table_model import IndicesTableModel
from paravis.gui.models.proxy_model import IndexTableProxyModel
from paravis.gui.dialogs.band_mapping import BandMappingDialog
from paravis.gui.dialogs.constants_editor import ConstantsEditorDialog
from paravis.gui.dialogs.index_table import IndexTableExpandDialog
from paravis.core.indices import get_default_band_mapping


# --- v1 constants ---
DEFAULT_SCALE_DENOM = 250.0
DEFAULT_TILE_SIZE = 512


def get_available_bands(band_mapping):
    """Get set of available band codes from mapping (v1 helper)."""
    return set(band_mapping.values())


def get_index_info(idx_name, constants_override, band_mapping):
    """Get detailed information about an index (v1 exact port)."""
    try:
        idx_obj = spyndex.indices[idx_name]
        required_bands = getattr(idx_obj, "bands", [])
        if required_bands is None:
            required_bands = []
        formula = getattr(idx_obj, "formula", "N/A")
        available_bands = get_available_bands(band_mapping)
        computable = True
        for band in required_bands:
            if (band not in available_bands
                    and band not in constants_override
                    and band not in spyndex.constants):
                computable = False
                break
        return {
            "name": idx_name,
            "required_bands": required_bands,
            "formula": formula,
            "computable": computable,
        }
    except Exception:
        return {
            "name": idx_name,
            "required_bands": [],
            "formula": "N/A",
            "computable": False,
        }


class IndicesWidget(QWidget):
    """v1 IndicesWidget — exact port."""

    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.worker = None
        self.indices_all = sorted(list(spyndex.indices.keys()))
        self.constants_override = {}
        self.band_mapping = get_default_band_mapping().copy()
        self.expand_dialog = None
        self.init_ui()
        self.update_indices_table()

    # ------------------------------------------------------------------
    # UI setup (v1 exact)
    # ------------------------------------------------------------------

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("Indices")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet("background-color: #c0c0c0; margin: 4px 0;")
        line.setFixedHeight(1)
        layout.addWidget(line)

        # --- Input group ---
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(4)

        input_row = QHBoxLayout()
        self.band_status_label = QLabel("Band mapping: Not configured")
        self.band_status_label.setStyleSheet("color: #888888;")
        input_row.addWidget(self.band_status_label)

        config_bands_btn = QPushButton("Configure")
        config_bands_btn.clicked.connect(self.configure_band_mapping)
        input_row.addWidget(config_bands_btn)

        input_row.addSpacing(12)

        self.add_files_btn = QPushButton("Add TIFFs...")
        self.add_files_btn.clicked.connect(self.add_files)
        input_row.addWidget(self.add_files_btn)

        self.clear_files_btn = QPushButton("Clear")
        self.clear_files_btn.clicked.connect(self.clear_files)
        input_row.addWidget(self.clear_files_btn)
        input_layout.addLayout(input_row)

        self.files_label = QLabel("No files selected")
        self.files_label.setStyleSheet(
            "border: 1px solid #d0d0d0; border-radius: 4px; padding: 6px;"
        )
        self.files_label.setWordWrap(True)
        input_layout.addWidget(self.files_label)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- Index Selection group ---
        indices_group = QGroupBox("Index Selection")
        indices_layout = QVBoxLayout()

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by index name...")
        self.search_box.textChanged.connect(self.filter_indices_table)
        search_layout.addWidget(self.search_box, 1)

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_indices)
        search_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_none_indices)
        search_layout.addWidget(self.select_none_btn)

        self.show_only_computable = QCheckBox("Show only computable")
        self.show_only_computable.stateChanged.connect(self.filter_indices_table)
        search_layout.addWidget(self.show_only_computable)

        self.expand_btn = QPushButton("\u2b36")
        self.expand_btn.setToolTip("Open index table in expanded window")
        self.expand_btn.setFixedSize(28, 28)
        self.expand_btn.setStyleSheet("""
            QPushButton { border: 1px solid #b0b0b0; border-radius: 4px;
                         background-color: #ffffff; font-size: 14px; padding: 0; }
            QPushButton:hover { background-color: #e0f2f1; border-color: #009688; }
        """)
        self.expand_btn.clicked.connect(self._expand_indices_table)
        search_layout.addWidget(self.expand_btn)

        indices_layout.addLayout(search_layout)

        # Table view
        self.indices_table = QTableView()
        self.indices_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.indices_table.setAlternatingRowColors(True)
        self.indices_table.setSortingEnabled(True)
        self.indices_table.horizontalHeader().setStretchLastSection(True)
        self.indices_table.verticalHeader().setVisible(False)
        self.indices_table.setColumnWidth(0, 8)
        self.indices_table.setColumnWidth(1, 129)
        self.indices_table.setColumnWidth(2, 100)
        self.indices_table.setColumnWidth(3, 171)
        self.indices_table.setColumnWidth(4, 257)
        self.indices_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.indices_table.setMinimumHeight(220)

        indices_layout.addWidget(self.indices_table, 0)

        self.indices_table.clicked.connect(self._toggle_checkbox)
        self._last_checked_row = None

        # Selection info
        selection_layout = QHBoxLayout()
        self.selected_count_label = QLabel("Selected: 0 indices")
        self.selected_count_label.setStyleSheet("color: #009688; font-weight: 600;")
        selection_layout.addWidget(self.selected_count_label)

        self.available_bands_label = QLabel("No bands configured")
        self.available_bands_label.setStyleSheet("color: #009688; font-size: 10px;")
        selection_layout.addWidget(self.available_bands_label, 1)

        self.computable_count_label = QLabel("Computable: 0 / 0")
        self.computable_count_label.setStyleSheet("color: #555555;")
        selection_layout.addWidget(self.computable_count_label)

        indices_layout.addLayout(selection_layout)

        indices_group.setLayout(indices_layout)
        layout.addWidget(indices_group)

        # --- Output Settings group ---
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()

        out_folder_layout = QHBoxLayout()
        out_folder_layout.addWidget(QLabel("Output:"))
        self.out_folder_edit = QLineEdit()
        self.out_folder_edit.setPlaceholderText("Output folder...")
        out_folder_layout.addWidget(self.out_folder_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output)
        out_folder_layout.addWidget(browse_btn)

        out_folder_layout.addWidget(QLabel("Date:"))
        self.date_pattern_edit = QLineEdit()
        self.date_pattern_edit.setPlaceholderText("Regex (e.g. \\d{8})")
        self.date_pattern_edit.setMaximumWidth(140)
        out_folder_layout.addWidget(self.date_pattern_edit)

        edit_constants_btn = QPushButton("Constants")
        edit_constants_btn.clicked.connect(self.edit_constants)
        out_folder_layout.addWidget(edit_constants_btn)
        output_layout.addLayout(out_folder_layout)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        options_layout.addWidget(self.workers_spin, 1)

        options_layout.addWidget(QLabel("Tile:"))
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(128, 4096)
        self.tile_spin.setSingleStep(128)
        self.tile_spin.setValue(DEFAULT_TILE_SIZE)
        options_layout.addWidget(self.tile_spin, 1)

        options_layout.addWidget(QLabel("Scale:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1, 100000)
        self.scale_spin.setValue(DEFAULT_SCALE_DENOM)
        self.scale_spin.setToolTip(
            "Divide raw pixel values by this number to get reflectance"
            " (not applied to SAR bands)"
        )
        options_layout.addWidget(self.scale_spin, 1)

        output_layout.addLayout(options_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Progress group ---
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

        self.expand_log_btn = QPushButton("▼ Expand Log")
        self.expand_log_btn.setCheckable(True)
        self.expand_log_btn.toggled.connect(self._toggle_log_expand)
        run_layout.addWidget(self.expand_log_btn, 1)

        progress_layout.addLayout(run_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # --- Log group ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        self.log_display.setMinimumHeight(80)
        self.log_display.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        self.log_group = log_group
        log_group.setMaximumHeight(180)
        layout.addWidget(log_group)

        # Store original log heights for toggling
        self._log_compact_max = 150
        self._log_compact_group_max = 180
        self._log_expanded_max = 10000
        self._log_window = None
        self._log_window_text = None
        self._progress_start_time = None

        # Push everything to the top (like Sections 2 & 3)
        layout.addStretch(1)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public helpers (v1 exact)
    # ------------------------------------------------------------------

    def add_files_direct(self, files):
        """Add files programmatically (v1 style)."""
        if files:
            self.selected_files.extend(files)
            self.files_label.setText(f"{len(self.selected_files)} file(s) selected")
            self.log(f"\u2705 Added {len(files)} file(s)")

    def load_files_direct(self, file_paths):
        """Replace file list (e.g. command-line)."""
        self.selected_files = list(file_paths)
        self.files_label.setText(f"{len(self.selected_files)} file(s) selected")
        self.log(f"\u2705 Loaded {len(file_paths)} file(s)")

    # ------------------------------------------------------------------
    # Band mapping (v1 exact)
    # ------------------------------------------------------------------

    def configure_band_mapping(self):
        """Open band mapping dialog (v1 exact)."""
        current_selections = set(self.get_selected_indices())

        dialog = BandMappingDialog(self.band_mapping, self)
        if dialog.exec() == QMessageBox.DialogCode.Accepted:
            self.band_mapping = dialog.get_mapping()
            available = get_available_bands(self.band_mapping)

            band_list = ", ".join(sorted(available))
            self.band_status_label.setText(
                f"Band mapping: {len(available)} band(s) configured"
            )
            self.available_bands_label.setText(f"Available bands: {band_list}")

            self.log(f"\u2705 Band mapping updated: {len(available)} bands configured")
            self.log(f"   Bands: {band_list}")

            self.update_indices_table()
            self.restore_selections(current_selections)

    # ------------------------------------------------------------------
    # Indices table (v1 exact)
    # ------------------------------------------------------------------

    def update_indices_table(self):
        """Update the indices table with current band mapping (v1 exact)."""
        indices_data = []
        computable_count = 0

        for idx_name in self.indices_all:
            idx_info = get_index_info(idx_name, self.constants_override, self.band_mapping)
            indices_data.append(idx_info)
            if idx_info["computable"]:
                computable_count += 1

        self.indices_model = IndicesTableModel(indices_data, self)

        self.proxy_model = IndexTableProxyModel(self)
        self.proxy_model.setSourceModel(self.indices_model)

        self.indices_table.setModel(self.proxy_model)

        # Re-apply column widths after setModel (v1 style)
        self.indices_table.setColumnWidth(0, 8)
        self.indices_table.setColumnWidth(1, 129)
        self.indices_table.setColumnWidth(2, 100)
        self.indices_table.setColumnWidth(3, 171)
        self.indices_table.setColumnWidth(4, 257)

        try:
            self.indices_model.dataChanged.disconnect(self.on_indices_data_changed)
        except (RuntimeError, TypeError):
            pass
        self.indices_model.dataChanged.connect(self.on_indices_data_changed)

        self.computable_count_label.setText(
            f"Computable: {computable_count} / {len(self.indices_all)}"
        )

        self.filter_indices_table()

    def restore_selections(self, selected_indices):
        """Restore previously selected indices (v1 exact)."""
        if not selected_indices:
            return

        selected_set = set(selected_indices)

        for row in range(self.proxy_model.rowCount()):
            model_idx = self.proxy_model.mapToSource(self.proxy_model.index(row, 0))
            source_row = model_idx.row()
            idx_info = self.indices_model.indices_data[source_row]
            if idx_info["name"] in selected_set:
                if idx_info["computable"]:
                    self.indices_model.checked_rows.add(source_row)
                else:
                    self.log(
                        f"\u26a0\ufe0f Index '{idx_info['name']}' is no longer computable"
                        " with current band mapping"
                    )

        if self.indices_model.checked_rows:
            top_left = self.indices_model.index(0, 0)
            bottom_right = self.indices_model.index(
                self.indices_model.rowCount() - 1, 0
            )
            self.indices_model.dataChanged.emit(
                top_left, bottom_right, [Qt.CheckStateRole]
            )

    def _expand_indices_table(self):
        """Open the index table in a larger dialog (v1 exact)."""
        dialog = IndexTableExpandDialog(self, self)
        dialog.exec()

    def filter_indices_table(self):
        """Filter table by search text and computable filter (v1 exact)."""
        search_text = self.search_box.text()
        self.proxy_model.setFilterText(search_text)

        if self.show_only_computable.isChecked():
            source_model = self.indices_model
            for row in range(self.proxy_model.rowCount()):
                source_row = self.proxy_model.mapToSource(
                    self.proxy_model.index(row, 0)
                ).row()
                is_computable = source_model.indices_data[source_row]["computable"]
                self.indices_table.setRowHidden(row, not is_computable)
        else:
            for row in range(self.proxy_model.rowCount()):
                self.indices_table.setRowHidden(row, False)

    def _toggle_checkbox(self, index):
        """Toggle checkbox on row click (v1 exact)."""
        source_index = self.proxy_model.mapToSource(index)
        proxy_row = index.row()

        from PySide6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()

        if modifiers == Qt.ShiftModifier and self._last_checked_row is not None:
            current_clicked = self.indices_model.data(
                source_index, Qt.CheckStateRole
            )
            target = (
                Qt.Unchecked if current_clicked == Qt.Checked else Qt.Checked
            )
            anchor_proxy = self._last_checked_row
            clicked_proxy = proxy_row
            start_proxy = min(anchor_proxy, clicked_proxy)
            end_proxy = max(anchor_proxy, clicked_proxy)
            for rp in range(start_proxy, end_proxy + 1):
                if self.indices_table.isRowHidden(rp):
                    continue
                proxy_idx = self.proxy_model.index(rp, 0)
                src_idx = self.proxy_model.mapToSource(proxy_idx)
                self.indices_model.setData(src_idx, target, Qt.CheckStateRole)
        else:
            current = self.indices_model.data(source_index, Qt.CheckStateRole)
            new_state = (
                Qt.Unchecked if current == Qt.Checked else Qt.Checked
            )
            self.indices_model.setData(source_index, new_state, Qt.CheckStateRole)
            self._last_checked_row = proxy_row

    def on_indices_data_changed(self, top_left, bottom_right, roles):
        """Update selected count when checkbox data changes (v1 exact)."""
        if Qt.CheckStateRole in roles:
            self.update_selected_count()

    def update_selected_count(self):
        """Update selected indices count label (v1 exact)."""
        count = len(self.indices_model.checked_rows)
        self.selected_count_label.setText(f"Selected: {count} indices")

    def get_selected_indices(self):
        """Get list of selected index names (v1 exact)."""
        selected_indices = []
        for row in range(self.proxy_model.rowCount()):
            source_row = self.proxy_model.mapToSource(
                self.proxy_model.index(row, 0)
            ).row()
            if source_row in self.indices_model.checked_rows:
                model_idx = self.indices_model.index(source_row, 1)
                index_name = self.indices_model.data(model_idx, Qt.DisplayRole)
                selected_indices.append(index_name)
        return selected_indices

    def select_all_indices(self):
        """Select all indices (v1 exact)."""
        for row in range(self.proxy_model.rowCount()):
            source_row = self.proxy_model.mapToSource(
                self.proxy_model.index(row, 0)
            ).row()
            self.indices_model.checked_rows.add(source_row)
        top_left = self.indices_model.index(0, 0)
        bottom_right = self.indices_model.index(
            self.indices_model.rowCount() - 1, 0
        )
        self.indices_model.dataChanged.emit(
            top_left, bottom_right, [Qt.CheckStateRole]
        )
        self.update_selected_count()
        self.log("\u2705 Selected all indices")

    def select_none_indices(self):
        """Clear all selections (v1 exact)."""
        self.indices_model.checked_rows.clear()
        top_left = self.indices_model.index(0, 0)
        bottom_right = self.indices_model.index(
            self.indices_model.rowCount() - 1, 0
        )
        self.indices_model.dataChanged.emit(
            top_left, bottom_right, [Qt.CheckStateRole]
        )
        self.update_selected_count()
        self.log("\u2705 Cleared all selections")

    def edit_constants(self):
        """Open constants editor dialog (v1 exact)."""
        current_selections = set(self.get_selected_indices())

        dialog = ConstantsEditorDialog(self.constants_override, self)
        if dialog.exec() == QMessageBox.DialogCode.Accepted:
            self.constants_override = dialog.get_constants()
            if self.constants_override:
                self.log(
                    f"\u2705 Constants updated: {json.dumps(self.constants_override)}"
                )
            else:
                self.log("\u2705 Constants reset to defaults")

            self.update_indices_table()
            self.restore_selections(current_selections)

    # ------------------------------------------------------------------
    # File management (v1 exact)
    # ------------------------------------------------------------------

    def log(self, message):
        self.log_display.append(message)
        self.log_display.ensureCursorVisible()
        if self._log_window_text is not None:
            self._log_window_text.append(message)
            self._log_window_text.ensureCursorVisible()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select GeoTIFFs", "", "TIFF Files (*.tif *.tiff)"
        )
        if files:
            self.add_files_direct(files)

    def clear_files(self):
        self.selected_files = []
        self.files_label.setText("No files selected")
        self.log("\u2705 Cleared all files")

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.out_folder_edit.setText(folder)
            self.log(f"\u2705 Output folder: {folder}")

    # ------------------------------------------------------------------
    # Processing (v1 exact)
    # ------------------------------------------------------------------

    def run_processing(self):
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "Please select input files!")
            return

        indices = self.get_selected_indices()
        if not indices:
            QMessageBox.warning(
                self, "Warning", "Please select at least one index!"
            )
            return

        out_root = self.out_folder_edit.text().strip()
        if not out_root:
            QMessageBox.warning(
                self, "Warning", "Please select output folder!"
            )
            return

        if not self.band_mapping:
            reply = QMessageBox.question(
                self, "Band Mapping Required",
                "No band mapping configured. Would you like to configure it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.configure_band_mapping()
                return
            else:
                QMessageBox.warning(
                    self, "Warning",
                    "Band mapping is required for processing!"
                )
                return

        date_pattern = self.date_pattern_edit.text().strip()
        if not date_pattern:
            date_pattern = None

        from dataclasses import dataclass

        @dataclass
        class JobConfig:
            indices: list
            out_root: str
            constants_override: dict
            band_mapping: dict
            max_workers: int
            scale_denom: float
            tile_size: int
            date_format: str

        cfg = JobConfig(
            indices=indices,
            out_root=out_root,
            constants_override=self.constants_override,
            band_mapping=self.band_mapping,
            max_workers=self.workers_spin.value(),
            scale_denom=self.scale_spin.value(),
            tile_size=self.tile_spin.value(),
            date_format=date_pattern,
        )

        import os as _os
        _os.makedirs(out_root, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self._progress_start_time = None

        self.log("\n" + "=" * 60)
        self.log("\U0001f680 Starting Indices processing")
        self.log("=" * 60)
        self.log(f"Files: {len(self.selected_files)}")
        self.log(f"Indices: {len(indices)}")
        self.log(f"Output: {out_root}")
        self.log(f"Workers: {cfg.max_workers}")
        self.log(f"Scale denominator: {cfg.scale_denom}")
        self.log(f"Band mapping: {json.dumps(cfg.band_mapping, indent=2)}")
        if self.constants_override:
            self.log(
                f"Constants override: {json.dumps(self.constants_override, indent=2)}"
            )
        else:
            self.log("Constants: Using defaults")
        self.log("=" * 60 + "\n")

        from paravis.workers.index_worker import IndicesBatchWorker
        self.worker = IndicesBatchWorker(self.selected_files, cfg)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Stop", "Stop processing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.log("\n\u26a0\ufe0f Stopping...")

    def update_progress(self, current, total):
        if total > 0:
            from time import time as now
            if self._progress_start_time is None:
                self._progress_start_time = now()

            pct = (current / total) * 100
            self.progress_bar.setValue(int(pct))

            elapsed = now() - self._progress_start_time
            if elapsed > 0 and current > 0:
                speed = current / elapsed
                self.status_label.setText(
                    f"Tiles: {current:,} / {total:,} ({pct:.1f}%)"
                    f"  |  {speed:.0f} tiles/s"
                )
            else:
                self.status_label.setText(
                    f"Tiles: {current:,} / {total:,} ({pct:.1f}%)"
                )

    def on_finished(self, success, message):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if success:
            self.status_label.setText("Completed!")
            self.log(f"\n\u2705 {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Failed!")
            self.log(f"\n\u274c {message}")
            QMessageBox.critical(self, "Error", message)

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
            self._log_window.finished.connect(lambda: self.expand_log_btn.setChecked(False))
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

    def clear_log(self):
        self.log_display.clear()


__all__ = ["IndicesWidget"]
