"""
Band mapping configuration dialog.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialogButtonBox,
)
from PySide6.QtCore import Qt

from paravis.core.indices import get_default_band_mapping
from paravis.gui.models.delegates import CodeDelegate


# Spectral code descriptions
SPECTRAL_CODES = {
    "A": "Aerosol / Coastal",
    "B": "Blue",
    "G": "Green",
    "R": "Red",
    "N": "Near IR (NIR)",
    "S1": "Shortwave IR 1 (SWIR1)",
    "S2": "Shortwave IR 2 (SWIR2)",
    "T": "Thermal",
    "VV": "Sentinel-1 VV",
    "VH": "Sentinel-1 VH",
    "RE1": "Red Edge 1",
    "RE2": "Red Edge 2",
    "RE3": "Red Edge 3",
    "N2": "NIR narrow",
    "WV": "Water Vapor",
    "SWIR_CIRRUS": "SWIR Cirrus",
}
CODE_LIST = sorted(SPECTRAL_CODES.keys())


class BandMappingDialog(QDialog):
    """Dialog for configuring band-to-spectral mapping."""

    def __init__(self, current_mapping, parent=None):
        super().__init__(parent)
        self.current_mapping = current_mapping.copy()
        self.setWindowTitle("Band Mapping Configuration")
        self.setMinimumSize(550, 400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Band", "Code", "Description"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        delegate = CodeDelegate(CODE_LIST, self.table)
        self.table.setItemDelegateForColumn(1, delegate)

        # Populate with up to 15 bands
        for band_num in range(1, 16):
            row = self.table.rowCount()
            self.table.insertRow(row)

            band_item = QTableWidgetItem(str(band_num))
            band_item.setFlags(band_item.flags() & ~Qt.ItemIsEditable)
            band_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, band_item)

            code_item = QTableWidgetItem("")
            code_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, code_item)

            desc_item = QTableWidgetItem("")
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, desc_item)

        # Set default mapping
        defaults = get_default_band_mapping()
        for band_num in range(1, 16):
            row = band_num - 1
            current_code = self.current_mapping.get(band_num, "")
            if not current_code:
                current_code = defaults.get(band_num, "")
            if current_code:
                self.table.item(row, 1).setText(current_code)
                self._update_desc_for_row(row)

        self.table.itemChanged.connect(self._on_code_changed)
        self.table.resizeRowsToContents()
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()

        landsat_btn = QPushButton("Landsat 8/9")
        landsat_btn.clicked.connect(self.reset_landsat_defaults)
        btn_layout.addWidget(landsat_btn)

        sentinel2_btn = QPushButton("Sentinel-2")
        sentinel2_btn.clicked.connect(self.reset_sentinel2_defaults)
        btn_layout.addWidget(sentinel2_btn)

        sentinel1_btn = QPushButton("Sentinel-1")
        sentinel1_btn.clicked.connect(self.reset_sentinel1_defaults)
        btn_layout.addWidget(sentinel1_btn)

        btn_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _on_code_changed(self, item):
        if item.column() == 1:
            self._update_desc_for_row(item.row())

    def _update_desc_for_row(self, row):
        code_item = self.table.item(row, 1)
        desc_item = self.table.item(row, 2)
        if code_item and desc_item:
            code = code_item.text().strip()
            if code in SPECTRAL_CODES:
                desc_item.setText(SPECTRAL_CODES[code])
            elif code == "":
                desc_item.setText("(unused)")
            else:
                desc_item.setText(f"Custom: {code}")

    def _reset_defaults(self, defaults):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            band_num = row + 1
            code_item = self.table.item(row, 1)
            if code_item is None:
                continue
            if band_num in defaults:
                code_item.setText(defaults[band_num])
            else:
                code_item.setText("")
            self._update_desc_for_row(row)
        self.table.blockSignals(False)

    def reset_landsat_defaults(self):
        """v1: reset to Landsat 8/9 defaults."""
        self._reset_defaults({
            1: "A", 2: "B", 3: "G", 4: "R", 5: "N",
            6: "S1", 7: "S2", 8: "T",
        })

    def reset_sentinel2_defaults(self):
        """v1: reset to Sentinel-2 defaults."""
        self._reset_defaults({
            1: "A", 2: "B", 3: "G", 4: "R", 5: "RE1",
            6: "RE2", 7: "RE3", 8: "N2", 9: "WV",
            10: "S1", 11: "S2",
        })

    def reset_sentinel1_defaults(self):
        """v1: reset to Sentinel-1 defaults."""
        self._reset_defaults({1: "VV", 2: "VH"})

    def get_mapping(self):
        """Get the configured band mapping dict."""
        mapping = {}
        for row in range(self.table.rowCount()):
            code_item = self.table.item(row, 1)
            code = code_item.text().strip() if code_item else ""
            if code:
                mapping[row + 1] = code
        return mapping


__all__ = ["BandMappingDialog"]
