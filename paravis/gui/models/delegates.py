"""
CodeDelegate — QStyledItemDelegate for spectral code dropdowns (v1 exact port).
"""
from PySide6.QtWidgets import QStyledItemDelegate, QComboBox
from PySide6.QtCore import Qt


class CodeDelegate(QStyledItemDelegate):
    """v1: QComboBox editor with addItem("") + addItem(code, code), findData."""

    def __init__(self, codes, parent=None):
        super().__init__(parent)
        self.codes = codes

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItem("")
        for code in self.codes:
            combo.addItem(code, code)
        return combo

    def setEditorData(self, editor, index):
        current = index.data(Qt.DisplayRole) or ""
        idx = editor.findData(current)
        if idx >= 0:
            editor.setCurrentIndex(idx)
        else:
            editor.setEditText(current)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)
