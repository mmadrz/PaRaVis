"""
Constants editor dialog — view/edit default constants for spectral indices.

Faithful port of v1 ConstantsEditorDialog with Reset All / Clear All buttons,
tooltips on every item, and proper int/float conversion in get_constants().
"""
import spyndex

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialogButtonBox,
)
from PySide6.QtCore import Qt


class ConstantsEditorDialog(QDialog):
    """Dialog for viewing and overriding spyndex constants."""

    def __init__(self, current_constants, parent=None):
        super().__init__(parent)
        self.current_constants = current_constants.copy()
        self.setWindowTitle("Constants Editor - Indices")
        self.setGeometry(200, 200, 1000, 700)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        info_label = QLabel(
            "📋 Edit Constants for Spectral Indices\n"
            "Double-click cells in 'Custom Value' column to edit. "
            "Leave empty to use default values.\n"
            "Constants are case-sensitive and affect index calculations."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #00695c; font-size: 10px; padding: 5px; border-radius: 3px;")
        layout.addWidget(info_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Constant", "Default Value", "Description", "Custom Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)

        for name, const in spyndex.constants.items():
            default = getattr(const, "default", "N/A")
            desc = getattr(const, "description", "No description")
            custom = self.current_constants.get(name, "")

            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setToolTip(f"Constant name: {name}")
            self.table.setItem(row, 0, name_item)

            default_item = QTableWidgetItem(str(default))
            default_item.setFlags(default_item.flags() & ~Qt.ItemIsEditable)
            default_item.setToolTip(f"Default value: {default}")
            self.table.setItem(row, 1, default_item)

            desc_item = QTableWidgetItem(str(desc))
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
            desc_item.setToolTip(str(desc))
            self.table.setItem(row, 2, desc_item)

            custom_item = QTableWidgetItem(str(custom) if custom else "")
            custom_item.setToolTip("Enter custom value (leave empty for default)")
            self.table.setItem(row, 3, custom_item)

        self.table.resizeRowsToContents()
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset All to Default")
        reset_btn.clicked.connect(self._reset_all)
        btn_layout.addWidget(reset_btn)

        clear_btn = QPushButton("Clear All Custom Values")
        clear_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _reset_all(self):
        """Clear all custom values, reverting to defaults."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 3)
            if item:
                item.setText("")

    def _clear_all(self):
        """Alias for reset_all (v1 had both buttons)."""
        self._reset_all()

    def get_constants(self):
        """Return the edited constants dict with proper int/float conversion."""
        constants = {}
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            custom_item = self.table.item(row, 3)
            if name_item and custom_item:
                val = custom_item.text().strip()
                if val:
                    try:
                        constants[name_item.text()] = float(val)
                    except ValueError:
                        constants[name_item.text()] = val
        return constants


__all__ = ["ConstantsEditorDialog"]
