"""
IndexTableExpandDialog — Expanded index table (v1 exact port).

Takes indices_widget as first parameter (v1 pattern), shares its proxy model.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTableView, QHeaderView, QCheckBox, QStatusBar, QAbstractItemView,
)
from PySide6.QtCore import Qt


class IndexTableExpandDialog(QDialog):
    """Full-screen dialog for browsing and selecting spectral indices.

    v1 exact: takes indices_widget as first param, accesses its proxy_model directly.
    """

    def __init__(self, indices_widget, parent=None):
        super().__init__(parent)
        self.indices_widget = indices_widget
        self.setWindowTitle("Spectral Index Reference — Full Catalog")
        self.setMinimumSize(900, 600)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # v1: Shift+click range selection support
        self._last_checked_row = -1

        self._setup_ui()
        self._init_models()
        self._sync_labels()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        header = QLabel(
            "Spectral Index Reference — Full Catalog\n"
            "Search, browse, and select indices for computation."
        )
        header.setStyleSheet(
            "font-size: 13px; color: #00695c; padding: 4px;"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to filter indices by name...")
        self.search_box.textChanged.connect(self._on_search)
        search_layout.addWidget(self.search_box, 1)

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        search_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self._select_none)
        search_layout.addWidget(self.select_none_btn)

        self.show_computable_cb = QCheckBox("Show only computable")
        self.show_computable_cb.toggled.connect(self._on_filter_changed)
        search_layout.addWidget(self.show_computable_cb)

        layout.addLayout(search_layout)

        # Table view
        self.table = QTableView()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 8)
        self.table.setColumnWidth(1, 129)
        self.table.setColumnWidth(2, 171)
        self.table.setColumnWidth(3, 257)
        self.table.setColumnWidth(4, 69)
        # v1: click toggle only on col 0
        self.table.clicked.connect(self._on_click)
        layout.addWidget(self.table, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.selected_count_label = QLabel("Selected: 0")
        self.available_bands_label = QLabel("Bands: \u2014")
        self.computable_count_label = QLabel("Computable: 0")
        self.status_bar.addWidget(self.selected_count_label)
        self.status_bar.addWidget(self.available_bands_label)
        self.status_bar.addPermanentWidget(self.computable_count_label)
        layout.addWidget(self.status_bar)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setProperty("class", "primary")
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _init_models(self):
        """Share the proxy model from indices_widget (v1 pattern)."""
        self.proxy_model = self.indices_widget.proxy_model
        self.source_model = self.indices_widget.indices_model
        self.table.setModel(self.proxy_model)

        try:
            self.source_model.dataChanged.disconnect(self._on_data_changed)
        except (RuntimeError, TypeError):
            pass
        self.source_model.dataChanged.connect(self._on_data_changed)

    # ------------------------------------------------------------------
    # v1 slots
    # ------------------------------------------------------------------

    def _on_search(self, text):
        """v1: search + filter."""
        self.proxy_model.setFilterFixedString(text)
        self._update_row_visibility()

    def _on_filter_changed(self, checked):
        """v1: show only computable changed."""
        self._update_row_visibility()

    def _update_row_visibility(self):
        """v1: row hiding based on computable."""
        source_model = self.source_model
        show_comp_only = self.show_computable_cb.isChecked()
        for row in range(self.proxy_model.rowCount()):
            source_row = self.proxy_model.mapToSource(
                self.proxy_model.index(row, 0)
            ).row()
            hidden = False
            if show_comp_only:
                hidden = not source_model.indices_data[source_row]["computable"]
            self.table.setRowHidden(row, hidden)
        self._sync_labels()

    def _select_all(self):
        """v1: select all (delegates to indices_widget)."""
        self.indices_widget.select_all_indices()
        self._sync_labels()

    def _select_none(self):
        """v1: select none (delegates to indices_widget)."""
        self.indices_widget.select_none_indices()
        self._sync_labels()

    def _on_click(self, index):
        """v1: toggle checkbox only on column 0, with Shift+click."""
        if index.column() != 0:
            return

        proxy_row = index.row()
        src_index = self.proxy_model.mapToSource(index)

        from PySide6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()

        if (modifiers == Qt.ShiftModifier
                and self.indices_widget._last_checked_row is not None):
            current = self.source_model.data(src_index, Qt.CheckStateRole)
            target = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            anchor_proxy = self.indices_widget._last_checked_row
            clicked_proxy = proxy_row
            start_proxy = min(anchor_proxy, clicked_proxy)
            end_proxy = max(anchor_proxy, clicked_proxy)
            for rp in range(start_proxy, end_proxy + 1):
                if self.table.isRowHidden(rp):
                    continue
                p_idx = self.proxy_model.index(rp, 0)
                s_idx = self.proxy_model.mapToSource(p_idx)
                self.source_model.setData(s_idx, target, Qt.CheckStateRole)
        else:
            current = self.source_model.data(src_index, Qt.CheckStateRole)
            new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            self.source_model.setData(src_index, new_state, Qt.CheckStateRole)
            self.indices_widget._last_checked_row = proxy_row

    def _on_data_changed(self, top_left, bottom_right, roles):
        """Sync labels when model data changes."""
        if Qt.CheckStateRole in roles:
            self._sync_labels()

    def _sync_labels(self):
        """v1: update status labels from indices_widget."""
        count = len(self.source_model.checked_rows)
        total = len(self.source_model.indices_data)
        n_computable = sum(
            1 for idx in self.source_model.indices_data if idx["computable"]
        )
        self.selected_count_label.setText(f"Selected: {count}")
        self.available_bands_label.setText(
            self.indices_widget.available_bands_label.text()
        )
        self.computable_count_label.setText(f"Computable: {n_computable} / {total}")

    def closeEvent(self, event):
        """v1: disconnect dataChanged signal."""
        try:
            self.source_model.dataChanged.disconnect(self._on_data_changed)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)

        self._update_status()

    def _update_status(self):
        if self.source_model is not None:
            n_selected = len(self.source_model.checked_rows)
            n_computable = sum(1 for idx in self.source_model.indices_data
                               if self.source_model._is_computable(idx))
            self.selected_count_label.setText(f"Selected: {n_selected}")
            self.computable_count_label.setText(f"Computable: {n_computable}")
            if hasattr(self.source_model, 'band_mapping') and self.source_model.band_mapping:
                codes = list(self.source_model.band_mapping.values())
                self.available_bands_label.setText(f"Bands: {', '.join(codes)}")
            else:
                self.available_bands_label.setText("Bands: —")


__all__ = ["IndexTableExpandDialog"]
