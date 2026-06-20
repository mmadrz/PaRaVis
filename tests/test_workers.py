"""
Tests for paravis.workers — background worker threads.

Requires QApplication (pytest-qt). Run with:
    pytest tests/test_workers.py -v --cov=paravis.workers
"""
import os
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication fixture."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestBaseWorker:
    def test_create_worker(self, qapp):
        from paravis.workers.base_worker import BaseWorker
        worker = BaseWorker()
        assert worker.is_running is True
        assert worker.is_paused is False

    def test_stop(self, qapp):
        from paravis.workers.base_worker import BaseWorker
        worker = BaseWorker()
        worker.stop()
        assert worker.is_running is False

    def test_pause_resume(self, qapp):
        from paravis.workers.base_worker import BaseWorker
        worker = BaseWorker()
        worker.pause()
        assert worker.is_paused is True
        worker.resume()
        assert worker.is_paused is False

    def test_emit_progress(self, qapp):
        from paravis.workers.base_worker import BaseWorker
        worker = BaseWorker()
        results = []

        def capture(val):
            results.append(val)

        worker.progress.connect(capture)
        worker.emit_progress(42)
        assert results == [42]

    def test_stop_twice_no_error(self, qapp):
        from paravis.workers.base_worker import BaseWorker
        worker = BaseWorker()
        worker.stop()
        worker.stop()  # second stop should not raise
        assert worker.is_running is False


class TestIndexComputeWorker:
    def test_run_completes(self, qapp):
        from paravis.workers.index_worker import IndexComputeWorker
        data = np.zeros((6, 10, 10), dtype=np.float32)
        data[3] = 0.2
        data[4] = 0.8
        mapping = {4: "R", 5: "N"}

        worker = IndexComputeWorker(data, mapping, ["NDVI"])
        results = []
        errors = []

        def capture(result):
            results.append(result)

        def capture_error(msg):
            errors.append(msg)

        worker.finished.connect(capture)
        worker.error.connect(capture_error)
        worker.start()
        worker.wait(10000)
        QApplication.processEvents()

        if errors:
            pytest.fail(f"Worker error: {errors[0]}")
        assert len(results) > 0
        assert "NDVI" in results[0]

    def test_run_with_constants(self, qapp):
        from paravis.workers.index_worker import IndexComputeWorker
        data = np.zeros((6, 10, 10), dtype=np.float32)
        data[3] = 0.2
        data[4] = 0.8
        mapping = {4: "R", 5: "N"}

        worker = IndexComputeWorker(data, mapping, ["NDVI"])
        results = []
        errors = []

        def capture(result):
            results.append(result)

        def capture_error(msg):
            errors.append(msg)

        worker.finished.connect(capture)
        worker.error.connect(capture_error)
        worker.start()
        worker.wait(10000)
        QApplication.processEvents()

        if errors:
            pytest.fail(f"Worker error: {errors[0]}")
        assert len(results) > 0

    def test_run_cancelled(self, qapp):
        """Stop before starting should emit error."""
        from paravis.workers.index_worker import IndexComputeWorker
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}

        worker = IndexComputeWorker(data, mapping, ["NDVI"])
        error_results = []

        def capture_error(msg):
            error_results.append(msg)

        worker.error.connect(capture_error)
        worker.stop()  # Stop before running
        worker.start()
        worker.wait(3000)
        QApplication.processEvents()

        # Should have emitted an error since we stopped
        assert len(error_results) > 0


class TestRaoQWorker:
    def test_run_cpu(self, qapp, tmp_path):
        from paravis.workers.raoq_worker import RaoQWorker
        import rasterio
        from rasterio.transform import from_origin

        # Create a small test raster
        data = np.random.rand(16, 16).astype(np.float32)
        transform = from_origin(0, 16, 1, 1)
        raster_path = str(tmp_path / "test_band.tif")
        with rasterio.open(
            raster_path, 'w', driver='GTiff', height=16, width=16,
            count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
        ) as dst:
            dst.write(data, 1)

        output_path = str(tmp_path / "output.tif")

        finished_results = []
        error_results = []

        def on_finished(success, msg):
            finished_results.append((success, msg))

        def on_log(msg):
            pass  # ignore logs in test

        worker = RaoQWorker(
            raster_paths=[raster_path],
            output_path=output_path,
            distance_m="euclidean",
            window=3,
            na_tolerance=1.0,
            block_size=1024,
            num_workers=1,
            p_minkowski=2,
            use_gpu=False,
            simplify=2,
        )
        worker.log_signal.connect(on_log)
        worker.finished_signal.connect(on_finished)
        worker.start()
        worker.wait(15000)
        QApplication.processEvents()

        if not finished_results:
            pytest.fail("Worker did not emit finished signal")
        success, msg = finished_results[0]
        assert success, f"Worker failed: {msg}"
        assert os.path.exists(output_path), "Output file was not created"


class TestBaseWorkerSignals:
    def test_finished_signal(self, qapp):
        """Test that finished signal works through inheritance."""
        from paravis.workers.base_worker import BaseWorker

        class TestWorker(BaseWorker):
            def run(self):
                self.finished.emit("done")

        worker = TestWorker()
        results = []

        def capture(val):
            results.append(val)

        worker.finished.connect(capture)
        worker.start()
        worker.wait(3000)
        QApplication.processEvents()

        assert results == ["done"]

    def test_error_signal(self, qapp):
        from paravis.workers.base_worker import BaseWorker

        class ErrorWorker(BaseWorker):
            def run(self):
                self.error.emit("test error")

        worker = ErrorWorker()
        errors = []

        def capture(msg):
            errors.append(msg)

        worker.error.connect(capture)
        worker.start()
        worker.wait(3000)
        QApplication.processEvents()

        assert errors == ["test error"]
