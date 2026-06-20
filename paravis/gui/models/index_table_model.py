"""
IndicesTableModel — QAbstractItemModel for the spectral index table.

Exact port of v1 IndicesTableModel from indices_calc.py.
Uses list-of-dicts internally (like v1) for compatibility.
"""
from typing import List, Set

from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex
from PySide6.QtGui import QColor

from paravis.core.indices import is_index_computable


class IndicesTableModel(QAbstractItemModel):
    """Qt item model for displaying spectral indices in a table (v1 exact port)."""

    HEADERS = ["✓", "Index", "Computable", "Bands Required", "Formula"]

    def __init__(self, indices_data, parent=None):
        super().__init__(parent)
        self.indices_data = indices_data  # list of dicts like v1: {name, required_bands, formula, computable}
        self.checked_rows: Set[int] = set()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, child):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        return len(self.indices_data)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        if row >= len(self.indices_data):
            return None

        idx = self.indices_data[row]

        if role == Qt.CheckStateRole and col == 0:
            return Qt.Checked if row in self.checked_rows else Qt.Unchecked

        if role == Qt.DisplayRole:
            if col == 1:
                return idx["name"]
            elif col == 2:
                return "✓" if idx["computable"] else "✗"
            elif col == 3:
                bands = idx.get("required_bands", [])
                return ", ".join(sorted(bands)) if bands else "None"
            elif col == 4:
                formula = idx.get("formula", "N/A")
                if len(formula) > 100:
                    formula = formula[:97] + "..."
                return formula

        elif role == Qt.ToolTipRole:
            if col == 1:
                bands = idx.get("required_bands", [])
                return (f"Index: {idx['name']}\n"
                        f"Required bands: {', '.join(sorted(bands)) if bands else 'None'}")
            elif col == 2:
                return ("This index can be computed with current band mapping"
                        if idx["computable"] else "Missing required bands or constants")
            elif col == 3:
                bands = idx.get("required_bands", [])
                return (f"Required bands: {', '.join(sorted(bands)) if bands else 'No bands required (constants only)'}")
            elif col == 4:
                return idx.get("formula", "No formula available")

        elif role == Qt.ForegroundRole:
            if col == 2:
                return QColor(0, 150, 0) if idx["computable"] else QColor(200, 0, 0)

        elif role == Qt.TextAlignmentRole:
            if col == 0 or col == 2:
                return Qt.AlignmentFlag.AlignCenter

        return None

    def setData(self, index, value, role=Qt.CheckStateRole):
        if role == Qt.CheckStateRole and index.column() == 0:
            row = index.row()
            if value == Qt.Checked:
                self.checked_rows.add(row)
            else:
                self.checked_rows.discard(row)
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True
        return False

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsEnabled
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def _is_computable(self, idx) -> bool:
        return idx["computable"]

    def select_all(self):
        """Check all rows (v1 style)."""
        self.checked_rows = set(range(len(self.indices_data)))
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self.indices_data) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [Qt.CheckStateRole])

    def select_none(self):
        """Uncheck all rows (v1 style)."""
        self.checked_rows.clear()
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self.indices_data) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [Qt.CheckStateRole])
