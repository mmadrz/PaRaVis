"""
Recent files menu — dynamic menu showing recently opened files.
"""
import os
from typing import Callable, Optional

from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction

from paravis.utils.settings import AppSettings


class RecentFilesMenu(QMenu):
    """Dynamic menu showing recently opened raster files.

    Parameters
    ----------
    title : str
        Menu title.
    settings : AppSettings
        Settings instance for loading/saving recent files.
    on_file_selected : Callable[[str], None], optional
        Callback when a file is selected from the menu.
    """

    def __init__(
        self,
        title: str,
        settings: AppSettings,
        on_file_selected: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        super().__init__(title, parent)
        self.settings = settings
        self.on_file_selected = on_file_selected
        self.rebuild()

    def rebuild(self):
        """Rebuild the menu from stored recent files."""
        self.clear()
        files = self.settings.get_recent_files()

        if not files:
            action = self.addAction("(No recent files)")
            action.setEnabled(False)
            return

        for i, file_path in enumerate(files[:10]):
            filename = os.path.basename(file_path)
            action = self.addAction(f"{i + 1}. {filename}")
            action.setData(file_path)
            action.setToolTip(file_path)
            action.triggered.connect(
                lambda checked, path=file_path: self._on_selected(path)
            )

        self.addSeparator()
        clear_action = self.addAction("Clear Recent Files")
        clear_action.triggered.connect(self._clear_recent)

    def _on_selected(self, file_path: str):
        if self.on_file_selected:
            self.on_file_selected(file_path)

    def _clear_recent(self):
        self.settings.clear_recent_files()
        self.rebuild()
