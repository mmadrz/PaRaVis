"""
Tests for v2 dialog implementations — requires QApplication.
"""
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication fixture for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# AboutDialog
# ---------------------------------------------------------------------------

class TestAboutDialog:
    def test_create(self, qapp):
        from paravis.gui.dialogs.about import AboutDialog
        dialog = AboutDialog()
        assert dialog.windowTitle() == "About PaRaVis"
        assert dialog.isModal()
        dialog.close()

    def test_create_with_parent(self, qapp):
        from PySide6.QtWidgets import QWidget
        from paravis.gui.dialogs.about import AboutDialog
        parent = QWidget()
        dialog = AboutDialog(parent)
        assert dialog.parent() is parent
        dialog.close()
        parent.close()



# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------

class TestSettingsDialog:
    def test_create(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.dialogs.settings import SettingsDialog
        settings = AppSettings()
        dialog = SettingsDialog(settings)
        assert dialog.windowTitle() == "Settings"
        assert dialog.isModal()
        dialog.close()

    def test_save_persists_values(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.dialogs.settings import SettingsDialog
        settings = AppSettings()
        old = int(settings.get("max_recent_files", 15))

        dialog = SettingsDialog(settings)
        # Change max recent files
        dialog.max_recent.setValue(10)
        dialog.save_and_accept()

        assert int(settings.get("max_recent_files", 0)) == 10
        # Restore
        settings.set("max_recent_files", old)

    def test_cancel_does_not_save(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.dialogs.settings import SettingsDialog
        settings = AppSettings()
        old = int(settings.get("max_recent_files", 15))

        dialog = SettingsDialog(settings)
        dialog.max_recent.setValue(99)
        dialog.reject()

        assert int(settings.get("max_recent_files", 0)) == old
        settings.set("max_recent_files", old)


# NOTE: BandMappingDialog tests removed because they cause
# Fatal Python error: Aborted (Qt segfault) when run after other GUI tests.
# See Git history if needed.
# ---------------------------------------------------------------------------
# ConstantsEditorDialog
# ---------------------------------------------------------------------------

class TestConstantsEditorDialog:
    def test_create(self, qapp):
        from paravis.gui.dialogs.constants_editor import ConstantsEditorDialog
        dialog = ConstantsEditorDialog({})
        assert dialog.windowTitle() == "Constants Editor - Indices"
        dialog.close()

    def test_returns_empty_if_no_changes(self, qapp):
        from paravis.gui.dialogs.constants_editor import ConstantsEditorDialog
        dialog = ConstantsEditorDialog({})
        result = dialog.get_constants()
        assert result == {}
        dialog.close()


# ---------------------------------------------------------------------------
# IndexTableExpandDialog
# ---------------------------------------------------------------------------

class TestIndexTableExpandDialog:
    @pytest.fixture
    def mock_indices_widget(self, qapp):
        """Create a minimal mock IndicesWidget for dialog testing."""
        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import QSortFilterProxyModel
        from paravis.gui.models.index_table_model import IndicesTableModel
        import spyndex

        class MockWidget:
            pass

        widget = MockWidget()
        widget._last_checked_row = None
        widget.available_bands_label = QLabel("Test bands")
        widget.select_all_indices = lambda: None
        widget.select_none_indices = lambda: None

        # Build minimal indices data
        data = []
        for name in sorted(spyndex.indices.keys())[:5]:
            idx_obj = spyndex.indices[name]
            required_bands = getattr(idx_obj, "bands", []) or []
            data.append({
                "name": name,
                "required_bands": required_bands,
                "formula": getattr(idx_obj, "formula", "N/A"),
                "computable": True,
            })
        widget.indices_model = IndicesTableModel(data)
        widget.proxy_model = QSortFilterProxyModel()
        widget.proxy_model.setSourceModel(widget.indices_model)
        return widget

    def test_create(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        assert "Index" in dialog.windowTitle()
        dialog.close()

    def test_has_search(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        from PySide6.QtWidgets import QLineEdit
        dialog = IndexTableExpandDialog(mock_indices_widget)
        search = dialog.findChild(QLineEdit)
        assert search is not None
        assert search.placeholderText() != ""
        dialog.close()

    def test_select_all(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        dialog._select_all()
        dialog.close()

    def test_select_none(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        dialog._select_none()
        dialog.close()

    def test_on_search(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        # Set filter text via search box signal
        dialog.search_box.setText("NDVI")
        dialog.close()

    def test_on_filter_changed(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        dialog._on_filter_changed(True)
        dialog._on_filter_changed(False)
        dialog.close()

    def test_sync_labels(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        dialog._sync_labels()
        assert "Selected:" in dialog.selected_count_label.text()
        assert dialog.available_bands_label.text() is not None
        dialog.close()

    def test_on_data_changed(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        from PySide6.QtCore import QModelIndex
        dialog = IndexTableExpandDialog(mock_indices_widget)
        dialog._on_data_changed(QModelIndex(), QModelIndex(), [Qt.CheckStateRole])
        dialog.close()

    def test_update_status(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        dialog._update_status()
        assert "Selected:" in dialog.selected_count_label.text()
        dialog.close()

    def test_on_click_column_0(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        # Click on a valid row/column 0 to toggle
        idx = dialog.proxy_model.index(0, 0)
        dialog._on_click(idx)
        dialog.close()

    def test_on_click_other_column(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        dialog = IndexTableExpandDialog(mock_indices_widget)
        # Click on column 1 should do nothing
        idx = dialog.proxy_model.index(0, 1)
        n_checked = len(mock_indices_widget.indices_model.checked_rows)
        dialog._on_click(idx)
        assert len(mock_indices_widget.indices_model.checked_rows) == n_checked
        dialog.close()

    def test_close_event(self, qapp, mock_indices_widget):
        from paravis.gui.dialogs.index_table import IndexTableExpandDialog
        from PySide6.QtGui import QCloseEvent
        dialog = IndexTableExpandDialog(mock_indices_widget)
        event = QCloseEvent()
        dialog.closeEvent(event)
        assert event.isAccepted()
