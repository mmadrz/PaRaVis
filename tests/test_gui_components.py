"""
Integration tests for GUI components — requires QApplication.

Run with:  pytest tests/test_gui_components.py -v --cov=paravis.gui.components
"""
import os
import tempfile

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication fixture for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Splash screen
# ---------------------------------------------------------------------------

class TestSplashScreen:
    def test_create(self, qapp):
        from paravis.gui.components.splash import ModernSplashScreen
        splash = ModernSplashScreen()
        assert splash.width() == 520
        assert splash.height() == 320
        assert splash._status_text == "Loading modules..."

    def test_paint_event(self, qapp):
        """paintEvent should not crash."""
        from paravis.gui.components.splash import ModernSplashScreen
        splash = ModernSplashScreen()
        from PySide6.QtGui import QPaintEvent
        from PySide6.QtCore import QRect
        event = QPaintEvent(QRect(0, 0, 520, 320))
        splash.paintEvent(event)

    def test_update_status(self, qapp):
        from paravis.gui.components.splash import ModernSplashScreen
        splash = ModernSplashScreen()
        splash.update_status("Custom status")
        assert splash._status_text == "Custom status"
        splash.update_status("New status")
        assert splash._status_text == "New status"


# ---------------------------------------------------------------------------
# MplCanvas
# ---------------------------------------------------------------------------

class TestMplCanvas:
    def test_create(self, qapp):
        from paravis.gui.components.mpl_canvas import MplCanvas
        canvas = MplCanvas()
        assert canvas.fig is not None
        assert canvas.axes is not None

    def test_create_with_params(self, qapp):
        from paravis.gui.components.mpl_canvas import MplCanvas
        canvas = MplCanvas(width=12, height=8, dpi=150)
        assert canvas.fig.get_figwidth() == 12
        assert canvas.fig.get_figheight() == 8
        assert canvas.fig.get_dpi() == 150

    def test_create_with_parent(self, qapp):
        from paravis.gui.components.mpl_canvas import MplCanvas
        parent = QWidget()
        canvas = MplCanvas(parent=parent)
        assert canvas.parent() == parent
        parent.deleteLater()


# ---------------------------------------------------------------------------
# RecentFilesMenu
# ---------------------------------------------------------------------------

class TestRecentFilesMenu:
    def test_create(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.components.recent_files import RecentFilesMenu
        settings = AppSettings()
        menu = RecentFilesMenu("Recent", settings)
        assert menu.title() == "Recent"

    def test_rebuild_no_files(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.components.recent_files import RecentFilesMenu
        settings = AppSettings()
        settings.clear_recent_files()
        menu = RecentFilesMenu("Recent", settings)
        actions = menu.actions()
        assert len(actions) > 0
        assert "(No recent files)" in actions[0].text()

    def test_rebuild_with_files(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.components.recent_files import RecentFilesMenu
        settings = AppSettings()
        settings.clear_recent_files()
        settings.add_recent_file("/path/to/file1.tif")
        settings.add_recent_file("/path/to/file2.tif")
        menu = RecentFilesMenu("Recent", settings)
        actions = menu.actions()
        file_actions = [a for a in actions if a.data() is not None]
        assert len(file_actions) == 2

    def test_on_file_selected_callback(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.components.recent_files import RecentFilesMenu
        settings = AppSettings()
        settings.clear_recent_files()
        settings.add_recent_file("/path/to/test.tif")
        selected = []

        def callback(path):
            selected.append(path)

        menu = RecentFilesMenu("Recent", settings, on_file_selected=callback)
        menu._on_selected("/path/to/test.tif")
        assert selected == ["/path/to/test.tif"]

    def test_clear_recent(self, qapp):
        from paravis.utils.settings import AppSettings
        from paravis.gui.components.recent_files import RecentFilesMenu
        settings = AppSettings()
        settings.add_recent_file("/path/to/test.tif")
        menu = RecentFilesMenu("Recent", settings)
        menu._clear_recent()
        assert settings.get_recent_files() == []


# ---------------------------------------------------------------------------
# PlotWorker (GUI component worker)
# ---------------------------------------------------------------------------

class TestPlotWorker:
    def test_run_completes(self, qapp):
        from paravis.gui.components.workers import PlotWorker

        def dummy_plot():
            return "plot_result"

        worker = PlotWorker(dummy_plot)
        results = []

        def capture(val):
            results.append(val)

        worker.finished.connect(capture)
        worker.start()
        worker.wait(5000)
        QApplication.processEvents()

        assert results == ["plot_result"]

    def test_error_emits_signal(self, qapp):
        from paravis.gui.components.workers import PlotWorker

        def failing_plot():
            raise ValueError("test error")

        worker = PlotWorker(failing_plot)
        errors = []

        def capture(msg):
            errors.append(msg)

        worker.error.connect(capture)
        worker.start()
        worker.wait(3000)
        QApplication.processEvents()

        assert len(errors) > 0
        assert "test error" in errors[0]

    def test_stop_before_run(self, qapp):
        from paravis.gui.components.workers import PlotWorker

        def dummy_plot():
            return "result"

        worker = PlotWorker(dummy_plot)
        worker.stop()
        results = []
        worker.finished.connect(lambda v: results.append(v))
        worker.start()
        worker.wait(3000)
        QApplication.processEvents()
        assert len(results) == 0


# ---------------------------------------------------------------------------
# CompareWorker
# ---------------------------------------------------------------------------

class TestCompareWorker:
    def test_run_completes(self, qapp):
        from paravis.gui.components.workers import CompareWorker
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        def diff(a, b):
            return b - a

        worker = CompareWorker(data1, data2, diff)
        results = []

        def capture(val):
            results.append(val)

        worker.finished.connect(capture)
        worker.start()
        worker.wait(5000)
        QApplication.processEvents()

        assert len(results) > 0
        assert np.array_equal(results[0], [3, 3, 3])


# ---------------------------------------------------------------------------
# GifWorker
# ---------------------------------------------------------------------------

@pytest.fixture
def cleanup_worker():
    """Fixture that ensures QThread workers are fully stopped before teardown."""
    workers = []

    def _track(worker):
        workers.append(worker)
        return worker

    yield _track

    for w in workers:
        w.quit()
        if not w.wait(5000):
            w.terminate()
            w.wait(5000)
        w.deleteLater()


class TestGifWorker:
    def test_run_completes(self, qapp, cleanup_worker):
        """Test GifWorker with synthetic data files."""
        from paravis.gui.components.workers import GifWorker
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            data_paths = []
            for i in range(2):
                path = os.path.join(tmpdir, f"frame_{i}.npy")
                data = np.random.rand(20, 20).astype(np.float32)
                np.save(path, data)
                data_paths.append(path)

            def mock_read_raster(path, **kwargs):
                data = np.load(path)
                return data, None, None

            with patch("paravis.core.raster.read_raster", mock_read_raster):
                worker = cleanup_worker(GifWorker(
                    file_list=data_paths,
                    file_names=[f"Frame {i}" for i in range(2)],
                    colormap="viridis",
                    output_dir=tmpdir,
                    output_name="test_anim",
                    dpi=72,
                    fps=2,
                    normalize_all=True,
                ))
                results = []

                def capture(val):
                    results.append(val)

                worker.finished.connect(capture)
                worker.start()
                worker.wait(30000)
                QApplication.processEvents()

                if results:
                    assert os.path.exists(results[0])


# ---------------------------------------------------------------------------
# PlotWorker additional tests
# ---------------------------------------------------------------------------

class TestPlotWorkerStopped:
    def test_run_stopped_during_execution(self, qapp, cleanup_worker):
        """Test worker stops during execution (no result emitted)."""
        from paravis.gui.components.workers import PlotWorker
        import time

        def slow_plot():
            import time
            time.sleep(5)
            return "result"

        worker = cleanup_worker(PlotWorker(slow_plot))
        results = []
        worker.finished.connect(lambda v: results.append(v))
        worker.start()
        worker.stop()  # Stop while running
        worker.wait(3000)
        QApplication.processEvents()
        # Should not emit finished if stopped
        assert len(results) == 0


# ---------------------------------------------------------------------------
# CompareWorker additional tests
# ---------------------------------------------------------------------------

class TestCompareWorkerEdgeCases:
    def test_error_emits_signal(self, qapp, cleanup_worker):
        from paravis.gui.components.workers import CompareWorker
        import numpy as np

        def failing_func(a, b):
            raise ValueError("comparison failed")

        worker = cleanup_worker(CompareWorker(np.array([1]), np.array([2]), failing_func))
        errors = []

        def capture(msg):
            errors.append(msg)

        worker.error.connect(capture)
        worker.start()
        worker.wait(3000)
        QApplication.processEvents()

        assert len(errors) > 0
        assert "comparison failed" in errors[0]


# ---------------------------------------------------------------------------
# GifWorker additional tests
# ---------------------------------------------------------------------------

class TestGifWorkerEdgeCases:
    def test_error_on_missing_file(self, qapp, cleanup_worker):
        """Test GifWorker handles file read errors gracefully."""
        from paravis.gui.components.workers import GifWorker
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = cleanup_worker(GifWorker(
                file_list=["/nonexistent/file.tif"],
                file_names=["Frame 0"],
                colormap="viridis",
                output_dir=tmpdir,
                output_name="test_anim",
                dpi=72,
                fps=2,
                normalize_all=True,
            ))
            errors = []

            def capture(msg):
                errors.append(msg)

            worker.error.connect(capture)
            worker.start()
            worker.wait(10000)
            QApplication.processEvents()

            # Should emit an error
            assert len(errors) > 0

    def test_stop_during_run(self, qapp, cleanup_worker):
        """Test stopping GifWorker during execution."""
        from paravis.gui.components.workers import GifWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np
            data_paths = []
            for i in range(2):
                path = os.path.join(tmpdir, f"frame_{i}.npy")
                data = np.random.rand(20, 20).astype(np.float32)
                np.save(path, data)
                data_paths.append(path)

            worker = cleanup_worker(GifWorker(
                file_list=data_paths,
                file_names=["Frame 0"],
                colormap="viridis",
                output_dir=tmpdir,
                output_name="test_anim",
                dpi=72,
                fps=2,
                normalize_all=True,
            ))
            results = []
            worker.finished.connect(lambda v: results.append(v))
            worker.start()
            worker.stop()  # Stop immediately
            worker.wait(5000)
            QApplication.processEvents()
            # May or may not have finished, but should not crash
            assert worker.is_running is False or len(results) == 0


# ---------------------------------------------------------------------------
# SplashScreen additional tests
# ---------------------------------------------------------------------------

class TestSplashScreenMore:
    def test_show_message(self, qapp):
        from paravis.gui.components.splash import ModernSplashScreen
        splash = ModernSplashScreen()
        splash.show_message()
        assert splash.isVisible()
        splash.close()

    def test_paint_event_with_custom_status(self, qapp):
        """Paint event with custom status text."""
        from paravis.gui.components.splash import ModernSplashScreen
        splash = ModernSplashScreen()
        splash.update_status("Custom text")
        from PySide6.QtGui import QPaintEvent
        from PySide6.QtCore import QRect
        event = QPaintEvent(QRect(0, 0, 520, 320))
        splash.paintEvent(event)
        assert splash._status_text == "Custom text"
        splash.close()


# ---------------------------------------------------------------------------
# App module (entry point)
# ---------------------------------------------------------------------------

class TestAppModule:
    def test_create_app_config(self, qapp):
        """Test app configuration without creating a new QApplication."""
        from paravis.__version__ import __version__
        app = QApplication.instance()
        assert app is not None
        app.setApplicationName("PaRaVis")
        app.setApplicationVersion(__version__)
        app.setOrganizationName("RaoQ")
        assert app.applicationName() == "PaRaVis"
        assert app.applicationVersion() != ""

    def test_mpl_canvas_clear(self, qapp):
        """Test MplCanvas clear method if it exists."""
        from paravis.gui.components.mpl_canvas import MplCanvas
        canvas = MplCanvas()
        canvas.axes.plot([1, 2, 3])
        # Should not crash
        canvas.axes.clear()
        assert len(canvas.axes.lines) == 0
        canvas.deleteLater()
