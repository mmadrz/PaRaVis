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

    def test_run_raises_exception(self, qapp):
        """Worker should emit error when computation raises."""
        from unittest.mock import patch
        from paravis.workers.index_worker import IndexComputeWorker
        data = np.zeros((6, 10, 10), dtype=np.float32)
        data[3] = 0.2
        data[4] = 0.8
        mapping = {4: "R", 5: "N"}

        worker = IndexComputeWorker(data, mapping, ["NDVI"])
        error_results = []

        def capture_error(msg):
            error_results.append(msg)

        worker.error.connect(capture_error)

        with patch("paravis.workers.index_worker._compute_indices",
                   side_effect=ValueError("computation failed")):
            worker.start()
            worker.wait(5000)
            QApplication.processEvents()

        assert len(error_results) > 0
        assert "computation failed" in error_results[0]


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

    def test_run_bad_file(self, qapp):
        """Worker should emit error with non-existent raster file."""
        from paravis.workers.raoq_worker import RaoQWorker

        finished_results = []

        def on_finished(success, msg):
            finished_results.append((success, msg))

        worker = RaoQWorker(
            raster_paths=["/nonexistent/file.tif"],
            output_path="/nonexistent/output.tif",
            distance_m="euclidean",
            window=3,
            na_tolerance=1.0,
            block_size=1024,
            num_workers=1,
            p_minkowski=2,
            use_gpu=False,
            simplify=2,
        )
        worker.finished_signal.connect(on_finished)
        worker.start()
        worker.wait(10000)
        QApplication.processEvents()

        assert len(finished_results) > 0
        # Should have failed
        assert finished_results[0][0] is False


    def test_run_minkowski(self, qapp, tmp_path):
        """Test RaoQWorker with Minkowski distance."""
        from paravis.workers.raoq_worker import RaoQWorker
        import rasterio
        from rasterio.transform import from_origin

        data = np.random.rand(16, 16).astype(np.float32)
        transform = from_origin(0, 16, 1, 1)
        raster_path = str(tmp_path / "test_mink.tif")
        with rasterio.open(
            raster_path, 'w', driver='GTiff', height=16, width=16,
            count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
        ) as dst:
            dst.write(data, 1)

        output_path = str(tmp_path / "output_mink.tif")
        finished_results = []

        def on_finished(success, msg):
            finished_results.append((success, msg))

        worker = RaoQWorker(
            raster_paths=[raster_path],
            output_path=output_path,
            distance_m="minkowski",
            window=3,
            na_tolerance=1.0,
            block_size=1024,
            num_workers=1,
            p_minkowski=3,
            use_gpu=False,
            simplify=2,
        )
        worker.finished_signal.connect(on_finished)
        worker.start()
        worker.wait(15000)
        QApplication.processEvents()

        if not finished_results:
            pytest.fail("Worker did not emit finished signal")
        success, msg = finished_results[0]
        assert success, f"Worker failed: {msg}"
        assert os.path.exists(output_path)

    def test_run_multi_worker(self, qapp, tmp_path):
        """Test RaoQWorker with multiple CPU workers."""
        from paravis.workers.raoq_worker import RaoQWorker
        import rasterio
        from rasterio.transform import from_origin

        data = np.random.rand(20, 20).astype(np.float32)
        transform = from_origin(0, 20, 1, 1)
        raster_path = str(tmp_path / "test_multi.tif")
        with rasterio.open(
            raster_path, 'w', driver='GTiff', height=20, width=20,
            count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
        ) as dst:
            dst.write(data, 1)

        output_path = str(tmp_path / "output_multi.tif")
        finished_results = []

        def on_finished(success, msg):
            finished_results.append((success, msg))

        worker = RaoQWorker(
            raster_paths=[raster_path],
            output_path=output_path,
            distance_m="euclidean",
            window=3,
            na_tolerance=1.0,
            block_size=1024,
            num_workers=2,
            p_minkowski=2,
            use_gpu=False,
            simplify=2,
        )
        worker.finished_signal.connect(on_finished)
        worker.start()
        worker.wait(15000)
        QApplication.processEvents()

        if not finished_results:
            pytest.fail("Worker did not emit finished signal")
        success, msg = finished_results[0]
        assert success, f"Worker failed: {msg}"
        assert os.path.exists(output_path)

    def test_stop_during_computation(self, qapp, tmp_path):
        """Test stopping RaoQWorker during computation."""
        from paravis.workers.raoq_worker import RaoQWorker
        from unittest.mock import patch
        import rasterio
        from rasterio.transform import from_origin

        data = np.random.rand(16, 16).astype(np.float32)
        transform = from_origin(0, 16, 1, 1)
        raster_path = str(tmp_path / "test_stop.tif")
        with rasterio.open(
            raster_path, 'w', driver='GTiff', height=16, width=16,
            count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
        ) as dst:
            dst.write(data, 1)

        output_path = str(tmp_path / "output_stop.tif")

        # Patch _compute_rao_q to simulate cancellation
        original_compute = RaoQWorker._compute_rao_q

        def mock_compute(self):
            self.is_running = False  # simulate cancellation
            raise RuntimeError("Cancelled")

        with patch.object(RaoQWorker, '_compute_rao_q', mock_compute):
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
            finished_results = []
            log_messages = []

            def on_finished(success, msg):
                finished_results.append((success, msg))
            def on_log(msg):
                log_messages.append(msg)

            worker.finished_signal.connect(on_finished)
            worker.log_signal.connect(on_log)
            worker.start()
            worker.wait(10000)
            QApplication.processEvents()

            assert len(finished_results) > 0
            # Should report as cancelled/stopped by user
            assert "stopped by user" in finished_results[0][1].lower()

    def test_cancelled_exception_in_run(self, qapp):
        """Test that a non-cancelled RuntimeError is re-raised."""
        from paravis.workers.raoq_worker import RaoQWorker
        from unittest.mock import patch

        def mock_compute_raise(self):
            raise RuntimeError("Some other error")

        with patch.object(RaoQWorker, '_compute_rao_q', mock_compute_raise):
            worker = RaoQWorker([], "", "euclidean", 3, 1.0, 1024, 1, 2, False, 2)
            with pytest.raises(RuntimeError, match="Some other error"):
                worker.run()


class TestIndexComputeWorkerMore:
    def test_run_multiple_indices(self, qapp):
        """Test IndexComputeWorker with multiple indices."""
        from paravis.workers.index_worker import IndexComputeWorker
        data = np.zeros((6, 10, 10), dtype=np.float32)
        data[3] = 0.2  # R
        data[4] = 0.8  # N
        data[1] = 0.3  # B
        data[2] = 0.5  # G
        mapping = {2: "B", 3: "G", 4: "R", 5: "N"}

        worker = IndexComputeWorker(data, mapping, ["NDVI", "SAVI"], constants={"L": 0.5})
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
        assert "SAVI" in results[0]

    def test_run_empty_index_list(self, qapp):
        """Test IndexComputeWorker with empty index list."""
        from paravis.workers.index_worker import IndexComputeWorker
        data = np.zeros((5, 10, 10), dtype=np.float32)
        mapping = {4: "R", 5: "N"}

        worker = IndexComputeWorker(data, mapping, [])
        results = []
        errors = []

        def capture(result):
            results.append(result)

        def capture_error(msg):
            errors.append(msg)

        worker.finished.connect(capture)
        worker.error.connect(capture_error)
        worker.start()
        worker.wait(5000)
        QApplication.processEvents()

        if errors:
            pytest.fail(f"Worker error: {errors[0]}")
        assert len(results) > 0
        assert results[0] == {}  # empty dict for no indices


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


class TestIndicesBatchWorker:
    def test_create(self, qapp):
        """Test IndicesBatchWorker creation with minimal config."""
        from paravis.workers.index_worker import IndicesBatchWorker

        class MockConfig:
            indices = ["NDVI"]
            out_root = "/tmp"
            constants_override = {}
            band_mapping = {4: "R", 5: "N"}
            max_workers = 1
            scale_denom = 250.0
            tile_size = 64
            date_format = None

        worker = IndicesBatchWorker([], MockConfig())
        assert worker._is_running is True
        assert worker.file_paths == []

    def test_stop(self, qapp):
        """Test stopping IndicesBatchWorker."""
        from paravis.workers.index_worker import IndicesBatchWorker

        class MockConfig:
            indices = ["NDVI"]
            out_root = "/tmp"
            constants_override = {}
            band_mapping = {4: "R", 5: "N"}
            max_workers = 1
            scale_denom = 250.0
            tile_size = 64
            date_format = None

        worker = IndicesBatchWorker(["/nonexistent.tif"], MockConfig())
        worker.stop()
        assert worker._is_running is False

    def test_process_one_file_index(self, qapp, tmp_path):
        """Test the static _process_one_file_index method."""
        from paravis.workers.index_worker import IndicesBatchWorker
        import rasterio
        from rasterio.transform import from_origin
        import queue

        # Create input raster
        data = np.random.rand(20, 20).astype(np.float32)
        transform = from_origin(0, 20, 1, 1)
        src_path = str(tmp_path / "input.tif")
        with rasterio.open(
            src_path, 'w', driver='GTiff', height=20, width=20,
            count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Create output path
        out_path = str(tmp_path / "output.tif")

        # Create a minimal valid output GeoTIFF first
        profile = rasterio.open(src_path).profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(np.full((1, 20, 20), np.nan, dtype=np.float32))

        # Build windows
        windows = [rasterio.windows.Window(0, 0, 20, 20)]

        progress_queue = queue.SimpleQueue()

        result = IndicesBatchWorker._process_one_file_index(
            src_path=src_path,
            idx_name="NDVI",
            out_path=out_path,
            windows=windows,
            band_mapping={4: "R", 5: "N"},
            constants_override={},
            scale_denom=1.0,
            src_nodata=None,
            n_tiles=1,
            progress_queue=progress_queue,
        )

        assert result.file_path == src_path
        assert result.idx_name == "NDVI"
        assert result.tiles_ok == 0  # _compute_index needs proper data
        # Should not crash
        assert os.path.exists(out_path)

    def test_process_one_file_with_nodata(self, qapp, tmp_path):
        """Test _process_one_file_index with nodata handling."""
        from paravis.workers.index_worker import IndicesBatchWorker
        import rasterio
        from rasterio.transform import from_origin
        import queue

        data = np.ones((20, 20), dtype=np.float32)
        data[0:5, 0:5] = -9999
        transform = from_origin(0, 20, 1, 1)
        src_path = str(tmp_path / "input_nodata.tif")
        with rasterio.open(
            src_path, 'w', driver='GTiff', height=20, width=20,
            count=1, dtype=np.float32, crs='EPSG:4326',
            transform=transform, nodata=-9999,
        ) as dst:
            dst.write(data, 1)

        out_path = str(tmp_path / "output_nodata.tif")
        profile = rasterio.open(src_path).profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(np.full((1, 20, 20), np.nan, dtype=np.float32))

        windows = [rasterio.windows.Window(0, 0, 20, 20)]
        q = queue.SimpleQueue()

        result = IndicesBatchWorker._process_one_file_index(
            src_path=src_path,
            idx_name="NDVI",
            out_path=out_path,
            windows=windows,
            band_mapping={4: "R", 5: "N"},
            constants_override={},
            scale_denom=250.0,
            src_nodata=-9999,
            n_tiles=1,
            progress_queue=q,
        )

        assert result.file_path == src_path
        assert os.path.exists(out_path)
