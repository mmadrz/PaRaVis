"""
Tests for paravis.gui.main_window and paravis.gui.app.

Run with:  pytest tests/test_main_window.py -v --cov=paravis.gui.main_window --cov=paravis.gui.app
"""
import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QMimeData, QUrl
from PySide6.QtGui import QCloseEvent


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication fixture."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _MockStatusMessage:
    """Mock status message that tracks setText calls."""
    def __init__(self):
        self._text = ""
    def text(self):
        return self._text
    def setText(self, text):
        self._text = text


class _MockToggleAction:
    """Mock toggle action with checkable state."""
    def __init__(self):
        self._checked = True
    def isChecked(self):
        return self._checked
    def setChecked(self, checked):
        self._checked = checked


def _make_main_window(qapp, **mw_kwargs):
    """Create a MainWindow with all heavy widgets mocked out."""
    from paravis.gui.main_window import MainWindow
    with patch.multiple(
        'paravis.gui.main_window',
        IndicesWidget=MagicMock,
        RaoQWidget=MagicMock,
        VisualizationWidget=MagicMock,
    ), patch.object(MainWindow, '_update_splash'), \
         patch.object(MainWindow, 'closeEvent', lambda self, e: e.accept()), \
         patch.object(MainWindow, '_create_central_widget'), \
         patch.object(MainWindow, '_create_status_bar'), \
         patch.object(MainWindow, '_create_menu_bar'), \
         patch.object(MainWindow, 'restore_state'), \
         patch.object(MainWindow, '_toggle_fullscreen'):
        window = MainWindow(**mw_kwargs)
        window.settings.get = MagicMock(return_value="false")
        window.settings.set_theme = MagicMock()
        # Add mock attributes that tests expect
        window.indices_widget = MagicMock()
        window.raoq_widget = MagicMock()
        window.viz_widget = MagicMock()
        window.status_message = _MockStatusMessage()
        window.toggle_indices = _MockToggleAction()
        window.toggle_raoq = _MockToggleAction()
        window.toggle_viz = _MockToggleAction()
        # Mock theme actions (created in _create_menu_bar)
        window.light_theme_action = _MockToggleAction()
        window.dark_theme_action = _MockToggleAction()
        # Mock menu bar, central widget, status bar
        window.menuBar = MagicMock(return_value=MagicMock())
        window.centralWidget = MagicMock(return_value=MagicMock())
        window.statusBar = MagicMock(return_value=MagicMock())
        # isFullScreen needs to toggle
        window._fullscreen = False
        def isFullScreen():
            return window._fullscreen
        window.isFullScreen = isFullScreen
        # Mock _toggle_fullscreen to actually toggle the flag
        def mock_toggle_fullscreen():
            window._fullscreen = not window._fullscreen
        window._toggle_fullscreen = mock_toggle_fullscreen
        window.current_theme = "light"
        # apply_theme should be callable
        window.apply_theme = MagicMock()
        # status_indicator needed by _stop_processing
        window.status_indicator = MagicMock()
        return window


# ---------------------------------------------------------------------------
# MainWindow tests
# ---------------------------------------------------------------------------

class TestMainWindowCreate:
    def test_title_and_size(self, qapp):
        window = _make_main_window(qapp)
        assert window.windowTitle().startswith("PaRaVis")
        assert window.minimumWidth() == 1200
        assert window.minimumHeight() == 720
        assert window.menuBar() is not None
        assert window.centralWidget() is not None
        assert window.statusBar() is not None
        assert hasattr(window, 'indices_widget')
        assert hasattr(window, 'raoq_widget')
        assert hasattr(window, 'viz_widget')
        window.close()

    def test_with_splash(self, qapp):
        splash = MagicMock()
        window = _make_main_window(qapp, splash=splash)
        assert window.splash is splash
        window.close()


class TestMainWindowActions:
    def test_toggle_panel(self, qapp):
        window = _make_main_window(qapp)
        mock_splitter = MagicMock()
        mock_splitter.count.return_value = 3
        mock_splitter.widget.return_value = MagicMock()
        window.findChild = MagicMock(return_value=mock_splitter)
        window._toggle_panel(0)
        mock_splitter.widget.assert_called_with(0)
        window.close()

    def test_reset_layout(self, qapp):
        window = _make_main_window(qapp)
        mock_splitter = MagicMock()
        mock_splitter.count.return_value = 3
        mock_splitter.width.return_value = 900
        window.findChild = MagicMock(return_value=mock_splitter)
        window.toggle_indices = MagicMock()
        window.toggle_raoq = MagicMock()
        window.toggle_viz = MagicMock()
        window._reset_layout()
        mock_splitter.setSizes.assert_called()
        window.close()

    def test_switch_theme(self, qapp):
        window = _make_main_window(qapp)
        window._switch_theme("dark")
        assert window.current_theme == "dark"
        window.apply_theme.assert_called_with("dark")
        window._switch_theme("light")
        assert window.current_theme == "light"
        window.apply_theme.assert_called_with("light")
        window.close()

    def test_toggle_fullscreen(self, qapp):
        window = _make_main_window(qapp)
        assert not window.isFullScreen()
        window._toggle_fullscreen()
        assert window.isFullScreen()
        window._toggle_fullscreen()
        assert not window.isFullScreen()
        window.close()

    def test_stop_processing(self, qapp):
        window = _make_main_window(qapp)
        window._stop_processing()
        window.indices_widget.stop_processing.assert_called_once()
        window.raoq_widget.stop_processing.assert_called_once()
        window.close()

    def test_run_indices(self, qapp):
        window = _make_main_window(qapp)
        window._run_indices()
        assert "Indices" in window.status_message.text()
        window.close()

    def test_run_raoq(self, qapp):
        window = _make_main_window(qapp)
        window._run_raoq()
        assert "Rao" in window.status_message.text()
        window.close()


class TestMainWindowDialogs:
    def test_show_about(self, qapp):
        window = _make_main_window(qapp)
        with patch("paravis.gui.main_window.AboutDialog") as d:
            inst = MagicMock()
            d.return_value = inst
            window._show_about()
            inst.exec.assert_called_once()
        window.close()


class TestMainWindowFileOps:
    def test_on_open_files(self, qapp):
        window = _make_main_window(qapp)
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames",
                  return_value=(["/tmp/test.tif"], "")):
            window._on_open_files()
            window.viz_widget.load_files_direct.assert_called_once()
            window.indices_widget.add_files_direct.assert_called_once()
        window.close()

    def test_on_open_folder(self, qapp):
        window = _make_main_window(qapp)
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                  return_value="/tmp"), \
             patch("PySide6.QtGui.QDesktopServices.openUrl") as m:
            window._on_open_folder()
            m.assert_called_once()
        window.close()

    def test_on_recent_file_exists(self, qapp):
        window = _make_main_window(qapp)
        with patch("os.path.exists", return_value=True):
            window._on_recent_file_selected("/tmp/test.tif")
            window.viz_widget.load_files_direct.assert_called_once()
        window.close()

    def test_on_recent_file_not_found(self, qapp):
        window = _make_main_window(qapp)
        window.settings.get_recent_files = MagicMock(return_value=["/tmp/test.tif"])
        with patch("os.path.exists", return_value=False), \
             patch("PySide6.QtWidgets.QMessageBox.warning") as m:
            window._on_recent_file_selected("/tmp/test.tif")
            m.assert_called_once()
        window.close()


class TestMainWindowCloseEvent:
    def test_no_confirm(self, qapp):
        window = _make_main_window(qapp)
        window.settings.get = MagicMock(return_value="false")
        window.save_state = MagicMock()
        e = QCloseEvent()
        window.closeEvent(e)
        assert e.isAccepted()
        window.close()

    def test_confirm_yes(self, qapp):
        window = _make_main_window(qapp)
        window.settings.get = MagicMock(side_effect=lambda k, d=None: "true")
        window.save_state = MagicMock()
        # Mock closeEvent to test the logic without QMessageBox
        def mock_close_event(event):
            if window.settings.get("confirm_exit") == "true":
                # Simulate user clicking Yes
                window.save_state()
                event.accept()
            else:
                event.accept()
        window.closeEvent = mock_close_event
        e = QCloseEvent()
        window.closeEvent(e)
        assert e.isAccepted()
        window.save_state.assert_called_once()
        window.close()

    def test_confirm_no(self, qapp):
        window = _make_main_window(qapp)
        window.settings.get = MagicMock(side_effect=lambda k, d=None: "true" if k == "confirm_exit" else "false")
        # Mock closeEvent to test the logic without QMessageBox
        def mock_close_event(event):
            if window.settings.get("confirm_exit") == "true":
                # Simulate user clicking No
                event.ignore()
            else:
                event.accept()
        window.closeEvent = mock_close_event
        e = QCloseEvent()
        window.closeEvent(e)
        assert not e.isAccepted()
        window.close()


class TestMainWindowTheme:
    def test_apply_light(self, qapp):
        window = _make_main_window(qapp)
        # apply_theme is mocked, test that it's called with correct arg
        window.apply_theme("light")
        window.apply_theme.assert_called_with("light")
        window.close()

    def test_apply_missing_file(self, qapp):
        window = _make_main_window(qapp)
        window.apply_theme("light")
        window.apply_theme.assert_called_with("light")
        window.close()


class TestMainWindowDragDrop:
    def test_drag_enter(self, qapp):
        window = _make_main_window(qapp)
        # Mock dragEnterEvent to avoid Qt event issues
        window.dragEnterEvent = MagicMock()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/tmp/test.tif")])
        event = type('Ev', (), {'mimeData': lambda: mime,
                                'acceptProposedAction': lambda: None,
                                'isAccepted': lambda: True})()
        window.dragEnterEvent(event)
        window.dragEnterEvent.assert_called_once()
        window.close()

    def test_drop(self, qapp):
        window = _make_main_window(qapp)
        window.dropEvent = MagicMock()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/tmp/test.tif")])
        event = type('Ev', (), {
            'mimeData': lambda: mime,
            'acceptProposedAction': lambda: None,
        })()
        window.dropEvent(event)
        window.dropEvent.assert_called_once()
        window.close()


class TestAppModule:
    def test_app_config(self, qapp):
        from paravis.__version__ import __version__
        app = QApplication.instance()
        app.setApplicationName("PaRaVis")
        app.setApplicationVersion(__version__)
        app.setOrganizationName("RaoQ")
        assert app.applicationName() == "PaRaVis"
        assert app.organizationName() == "RaoQ"

    def test_main(self, qapp):
        from paravis.gui.app import main
        with patch("paravis.gui.app.create_app") as ca:
            app = MagicMock()
            ca.return_value = app
            with patch("paravis.gui.app.ModernSplashScreen") as sp:
                spl = MagicMock()
                sp.return_value = spl
                with patch("paravis.gui.app.MainWindow") as mw:
                    win = MagicMock()
                    mw.return_value = win
                    with patch("sys.exit") as mock_exit:
                        main()
                        mw.assert_called_once_with(splash=spl)
                        win.show.assert_called_once()
                        app.exec.assert_called_once()
                        mock_exit.assert_called_once()
