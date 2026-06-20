"""
Persistent application settings using QSettings (INI-style).

Thin wrapper around QSettings for saving/restoring window state,
recent files, and user preferences.
"""
import json
from pathlib import Path


class AppSettings:
    """Persistent application settings.

    Stores window geometry, splitter state, recent files, and theme
    preferences via QSettings.
    """

    def __init__(self, app_name: str = "PaRaVis", org_name: str = "RaoQ"):
        from PySide6.QtCore import QSettings
        self.qs = QSettings(org_name, app_name)

    # ------------------------------------------------------------------
    # Generic get / set
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):
        """Get a setting value."""
        return self.qs.value(key, default)

    def set(self, key: str, value):
        """Set a setting value."""
        self.qs.setValue(key, value)

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def get_recent_files(self):
        """Get the list of recently opened files."""
        files = self.qs.value("recent_files", [])
        if isinstance(files, str):
            files = json.loads(files)
        return files if isinstance(files, list) else []

    def add_recent_file(self, file_path: str):
        """Add a file path to the recent files list."""
        files = self.get_recent_files()
        files = [f for f in files if f != file_path]
        files.insert(0, file_path)
        files = files[:15]
        self.qs.setValue("recent_files", json.dumps(files))
        return files

    def clear_recent_files(self):
        """Clear the recent files list."""
        self.qs.remove("recent_files")

    # ------------------------------------------------------------------
    # Window state
    # ------------------------------------------------------------------

    def get_geometry(self):
        """Get saved window geometry."""
        return self.qs.value("window_geometry")

    def set_geometry(self, geom):
        """Save window geometry."""
        self.qs.setValue("window_geometry", geom)

    def get_splitter_state(self):
        """Get saved splitter state."""
        return self.qs.value("splitter_state")

    def set_splitter_state(self, state):
        """Save splitter state."""
        self.qs.setValue("splitter_state", state)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def get_last_dir(self) -> str:
        """Get the last used directory."""
        return self.qs.value("last_dir", str(Path.home()))

    def set_last_dir(self, path: str):
        """Set the last used directory."""
        self.qs.setValue("last_dir", path)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def get_theme(self) -> str:
        """Get the current theme name."""
        return self.qs.value("theme", "light")

    def set_theme(self, theme: str):
        """Set the current theme name."""
        self.qs.setValue("theme", theme)
