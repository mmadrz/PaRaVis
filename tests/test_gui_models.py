"""
Tests for paravis.gui.models — Qt item models, delegates, proxy models.

Requires QApplication (pytest-qt). Run with:
    pytest tests/test_gui_models.py -v --cov=paravis.gui.models
"""
import spyndex
import pytest
from PySide6.QtWidgets import QApplication, QComboBox
from PySide6.QtCore import Qt, QModelIndex, QAbstractItemModel


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_indices_data(band_mapping, constants_override=None):
    """Build a list of index dicts matching v1 format."""
    consts = constants_override or {}
    available_bands = set(band_mapping.values())
    data = []
    for name in sorted(spyndex.indices.keys()):
        idx_obj = spyndex.indices[name]
        required_bands = getattr(idx_obj, "bands", []) or []
        formula = getattr(idx_obj, "formula", "N/A")
        computable = True
        for band in required_bands:
            if (band not in available_bands
                    and band not in consts
                    and band not in spyndex.constants):
                computable = False
                break
        data.append({
            "name": name,
            "required_bands": required_bands,
            "formula": formula,
            "computable": computable,
        })
    return data


@pytest.fixture
def landsat_mapping():
    return {1: "A", 2: "B", 3: "G", 4: "R", 5: "N", 6: "S1", 7: "S2", 8: "T"}


@pytest.fixture
def landsat_data(landsat_mapping):
    return _build_indices_data(landsat_mapping)


# ---------------------------------------------------------------------------
# IndicesTableModel
# ---------------------------------------------------------------------------

class TestIndicesTableModel:
    def test_create(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        assert model.rowCount() > 0
        assert model.columnCount() == 5

    def test_header_data(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        headers = ["✓", "Index", "Computable", "Bands Required", "Formula"]
        for col, expected in enumerate(headers):
            header = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
            assert header == expected

    def test_data_display(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        # Column 1 should show index name
        name = model.data(model.index(0, 1), Qt.DisplayRole)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_data_bands(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        bands = model.data(model.index(0, 3), Qt.DisplayRole)
        assert isinstance(bands, str)

    def test_check_state(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        idx = model.index(0, 0)
        # Initially unchecked
        assert model.data(idx, Qt.CheckStateRole) == Qt.Unchecked

    def test_set_data_check(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        idx = model.index(0, 0)
        # Check the row
        assert model.setData(idx, Qt.Checked, Qt.CheckStateRole) is True
        assert model.data(idx, Qt.CheckStateRole) == Qt.Checked
        assert 0 in model.checked_rows

    def test_set_data_uncheck(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        idx = model.index(0, 0)
        model.setData(idx, Qt.Checked, Qt.CheckStateRole)
        model.setData(idx, Qt.Unchecked, Qt.CheckStateRole)
        assert 0 not in model.checked_rows

    def test_set_data_non_zero_column(self, qapp, landsat_data):
        """setData on non-checkbox column should return False."""
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        idx = model.index(0, 1)
        assert model.setData(idx, "test", Qt.DisplayRole) is False

    def test_flags(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        # Column 0: enabled (v1 style — no ItemIsUserCheckable, toggle via clicked handler)
        flags_0 = model.flags(model.index(0, 0))
        assert flags_0 & Qt.ItemIsEnabled
        assert not (flags_0 & Qt.ItemIsUserCheckable)
        # Column 1: enabled + selectable
        flags_1 = model.flags(model.index(0, 1))
        assert flags_1 & Qt.ItemIsEnabled
        assert flags_1 & Qt.ItemIsSelectable

    def test_invalid_index(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        invalid = QModelIndex()
        assert model.data(invalid) is None
        assert model.flags(invalid) == Qt.NoItemFlags

    def test_index_out_of_range(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        assert model.index(9999, 0).isValid() is False

    def test_parent_returns_invalid(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        idx = model.index(0, 0)
        assert model.parent(idx) == QModelIndex()

    def test_tooltip(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        tip = model.data(model.index(0, 2), Qt.ToolTipRole)
        assert tip is not None

    def test_foreground_color(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        color = model.data(model.index(0, 2), Qt.ForegroundRole)
        assert color is not None

    def test_select_all(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        model.select_all()
        assert len(model.checked_rows) == model.rowCount()

    def test_select_none(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        model.select_all()
        model.select_none()
        assert len(model.checked_rows) == 0

    def test_data_with_constants(self, qapp):
        """Test model with constants that affect computability."""
        from paravis.gui.models.index_table_model import IndicesTableModel
        mapping = {4: "R", 5: "N", 2: "B", 3: "G"}
        data = _build_indices_data(mapping, {"L": 0.5})
        model = IndicesTableModel(data)
        # Should have rows
        assert model.rowCount() > 0
        # Some indices should be computable with constants
        n_computable = sum(1 for idx in data if idx["computable"])
        assert n_computable > 0


# ---------------------------------------------------------------------------
# CodeDelegate
# ---------------------------------------------------------------------------

class TestCodeDelegate:
    def test_create(self, qapp):
        from paravis.gui.models.delegates import CodeDelegate
        delegate = CodeDelegate(["A", "B", "G", "R", "N"])
        assert delegate.codes == ["A", "B", "G", "R", "N"]

    def test_create_editor(self, qapp):
        from paravis.gui.models.delegates import CodeDelegate
        delegate = CodeDelegate(["A", "B", "G"])
        editor = delegate.createEditor(None, None, None)
        assert isinstance(editor, QComboBox)
        # v1: addItem("") first, then addItem(code, code) for each
        assert editor.count() == 4  # empty + 3 codes
        assert editor.currentText() == ""  # first item is empty
        # Empty item has no user data (None)
        assert editor.itemData(0) is None
        assert editor.itemData(1) == "A"
        assert editor.itemData(2) == "B"
        assert editor.itemData(3) == "G"
        editor.deleteLater()

    def test_set_editor_data(self, qapp):
        from paravis.gui.models.delegates import CodeDelegate
        from PySide6.QtCore import QAbstractItemModel, QModelIndex

        delegate = CodeDelegate(["A", "B", "G"])
        editor = delegate.createEditor(None, None, None)

        # Create a simple model to test with
        model = _SimpleDelegateModel()
        idx = model.index(0, 0)
        delegate.setEditorData(editor, idx)
        assert editor.currentText() == "B"
        editor.deleteLater()

    def test_set_model_data(self, qapp):
        from paravis.gui.models.delegates import CodeDelegate
        delegate = CodeDelegate(["A", "B", "G"])
        editor = delegate.createEditor(None, None, None)
        editor.setCurrentText("G")

        model = _SimpleDelegateModel()
        idx = model.index(0, 0)
        delegate.setModelData(editor, model, idx)
        assert model.data(idx, Qt.DisplayRole) == "G"
        editor.deleteLater()

    def test_update_editor_geometry(self, qapp):
        from paravis.gui.models.delegates import CodeDelegate
        from PySide6.QtWidgets import QStyleOptionViewItem
        delegate = CodeDelegate(["A", "B"])
        editor = delegate.createEditor(None, None, None)
        option = QStyleOptionViewItem()
        delegate.updateEditorGeometry(editor, option, None)
        editor.deleteLater()


class _SimpleDelegateModel(QAbstractItemModel):
    """Helper model for delegate tests."""
    def __init__(self):
        super().__init__()
        self._data = ["B"]

    def index(self, row, column, parent=QModelIndex()):
        return self.createIndex(row, column)

    def parent(self, child):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        return 1

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return self._data[index.row()]
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and index.isValid():
            self._data[index.row()] = value
            return True
        return False

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsEditable


# ---------------------------------------------------------------------------
# IndexTableProxyModel
# ---------------------------------------------------------------------------

class TestIndexTableProxyModel:
    def test_create(self, qapp):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        proxy = IndexTableProxyModel()
        assert proxy.filter_text == ""

    def test_set_filter_text(self, qapp):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        proxy = IndexTableProxyModel()
        proxy.setFilterText("NDVI")
        assert proxy.filter_text == "ndvi"

    def test_filter_accepts_all_when_no_text(self, qapp, landsat_data):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        from paravis.gui.models.index_table_model import IndicesTableModel
        source = IndicesTableModel(landsat_data)
        proxy = IndexTableProxyModel()
        proxy.setSourceModel(source)
        assert proxy.filterAcceptsRow(0, QModelIndex()) is True

    def test_filter_empty_text(self, qapp, landsat_data):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        from paravis.gui.models.index_table_model import IndicesTableModel
        source = IndicesTableModel(landsat_data)
        proxy = IndexTableProxyModel()
        proxy.setSourceModel(source)
        proxy.setFilterText("")
        assert proxy.filterAcceptsRow(0, QModelIndex()) is True

    def test_filter_with_text(self, qapp, landsat_data):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        from paravis.gui.models.index_table_model import IndicesTableModel
        source = IndicesTableModel(landsat_data)
        proxy = IndexTableProxyModel()
        proxy.setSourceModel(source)
        # Filter for a known index name
        proxy.setFilterText("NDVI")
        # Check if any row passes filter
        passed = any(proxy.filterAcceptsRow(r, QModelIndex())
                    for r in range(source.rowCount()))
        assert passed

    def test_filter_no_match(self, qapp, landsat_data):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        from paravis.gui.models.index_table_model import IndicesTableModel
        source = IndicesTableModel(landsat_data)
        proxy = IndexTableProxyModel()
        proxy.setSourceModel(source)
        proxy.setFilterText("ZZZZ_NOT_A_REAL_INDEX")
        # No row should pass
        for r in range(source.rowCount()):
            assert not proxy.filterAcceptsRow(r, QModelIndex())

    def test_filter_no_source_model(self, qapp):
        from paravis.gui.models.proxy_model import IndexTableProxyModel
        proxy = IndexTableProxyModel()
        # Without source model, should accept all
        assert proxy.filterAcceptsRow(0, QModelIndex()) is True


class TestCodeDelegateMore:
    def test_set_editor_data_empty_model(self, qapp):
        """Test setEditorData with model that returns None."""
        from paravis.gui.models.delegates import CodeDelegate
        from PySide6.QtCore import QAbstractItemModel, QModelIndex

        delegate = CodeDelegate(["A", "B"])
        editor = delegate.createEditor(None, None, None)

        class EmptyModel(QAbstractItemModel):
            def index(self, row, col, parent=QModelIndex()):
                return self.createIndex(row, col)
            def parent(self, child):
                return QModelIndex()
            def rowCount(self, parent=QModelIndex()):
                return 1
            def columnCount(self, parent=QModelIndex()):
                return 1
            def data(self, index, role=Qt.DisplayRole):
                return None
            def flags(self, index):
                return Qt.ItemIsEnabled

        model = EmptyModel()
        idx = model.index(0, 0)
        delegate.setEditorData(editor, idx)
        # Should not crash, editor text should be empty
        assert editor.currentText() == ""
        editor.deleteLater()

    def test_set_model_data_empty_editor(self, qapp):
        from paravis.gui.models.delegates import CodeDelegate
        from PySide6.QtCore import QAbstractItemModel, QModelIndex

        delegate = CodeDelegate(["A", "B"])
        editor = delegate.createEditor(None, None, None)
        editor.setCurrentText("B")

        model = _SimpleDelegateModel()
        idx = model.index(0, 0)
        delegate.setModelData(editor, model, idx)
        assert model.data(idx, Qt.DisplayRole) == "B"
        editor.deleteLater()


class TestIndicesTableModelMore:
    def test_data_invalid_index(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        from PySide6.QtCore import QModelIndex
        assert model.data(QModelIndex()) is None

    def test_data_out_of_range(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        idx = model.index(99999, 0)
        assert model.data(idx) is None

    def test_text_alignment(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        # Column 0 and 2 should have center alignment
        align_0 = model.data(model.index(0, 0), Qt.TextAlignmentRole)
        align_2 = model.data(model.index(0, 2), Qt.TextAlignmentRole)
        assert align_0 == Qt.AlignmentFlag.AlignCenter
        assert align_2 == Qt.AlignmentFlag.AlignCenter
        # Column 1 should not have alignment
        align_1 = model.data(model.index(0, 1), Qt.TextAlignmentRole)
        assert align_1 is None

    def test_tooltip_all_columns(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        for col in range(5):
            tip = model.data(model.index(0, col), Qt.ToolTipRole)
            if col > 0:  # Column 0 (checkbox) has no tooltip
                assert tip is not None

    def test_formula_truncation(self, qapp):
        """Test that long formulas are truncated."""
        from paravis.gui.models.index_table_model import IndicesTableModel
        long_formula = "A" * 200
        data = [{
            "name": "TEST",
            "required_bands": ["R", "N"],
            "formula": long_formula,
            "computable": True,
        }]
        model = IndicesTableModel(data)
        formula = model.data(model.index(0, 4), Qt.DisplayRole)
        assert len(formula) == 100  # 97 + "..."
        assert formula.endswith("...")

    def test_header_data_invalid(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        # Vertical header should return None
        hdr = model.headerData(0, Qt.Vertical, Qt.DisplayRole)
        assert hdr is None

    def test_flags_non_checkable(self, qapp, landsat_data):
        from paravis.gui.models.index_table_model import IndicesTableModel
        model = IndicesTableModel(landsat_data)
        # Column 0 should not have ItemIsUserCheckable (v1 style)
        flags_0 = model.flags(model.index(0, 0))
        assert not (flags_0 & Qt.ItemIsUserCheckable)
