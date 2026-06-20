"""
IndexTableProxyModel — QSortFilterProxyModel for text-search only.

Faithful port of v1 IndexTableProxyModel with setFilterText().
Computable filtering is done by the widget via setRowHidden (v1 style).
"""
from PySide6.QtCore import Qt, QSortFilterProxyModel


class IndexTableProxyModel(QSortFilterProxyModel):
    """Proxy model that filters the index table by text search (v1 style).

    - Searches the index name column (column 1) case-insensitively.
    - Computable filtering is done by the widget via setRowHidden.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_text = ""

    def setFilterText(self, text: str):
        """Update the search filter and invalidate."""
        self.filter_text = text.lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent) -> bool:
        model = self.sourceModel()
        if model is None:
            return True

        # Text filter on column 1 (Index name) — v1 style
        if self.filter_text:
            name = model.index(source_row, 1).data(Qt.DisplayRole).lower()
            if self.filter_text not in name:
                return False

        return True
