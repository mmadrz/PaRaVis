"""
Tests for paravis.gui.widgets — module-level helpers + widget creation.

Run with:  pytest tests/test_gui_widgets.py -v --cov=paravis.gui.widgets
"""
import os
import tempfile

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication fixture for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =========================================================================
# Helper: normalize_data (viz_panel.py)
# =========================================================================

class TestNormalizeData:
    def test_normalize_basic(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = normalize_data(data)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[-1], 1.0)

    def test_normalize_with_nan(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
        result = normalize_data(data)
        assert np.isnan(result[2])
        assert np.isclose(np.nanmin(result), 0.0)
        assert np.isclose(np.nanmax(result), 1.0)

    def test_normalize_all_same(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.full(10, 42.0, dtype=np.float32)
        result = normalize_data(data)
        assert np.all(result == 0.0)

    def test_normalize_empty(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.array([], dtype=np.float32)
        result = normalize_data(data)
        assert result.size == 0

    def test_normalize_all_nan(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.full(10, np.nan, dtype=np.float32)
        result = normalize_data(data)
        assert np.all(result == 0.0)

    def test_normalize_integer(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.array([0, 100, 200], dtype=np.int32)
        result = normalize_data(data)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 0.5)
        assert np.isclose(result[2], 1.0)

    def test_normalize_2d(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
        result = normalize_data(data)
        assert result.shape == (2, 2)
        assert np.isclose(result[0, 0], 0.0)
        assert np.isclose(result[1, 1], 1.0)

    def test_normalize_negative_values(self, qapp):
        from paravis.gui.widgets.viz_panel import normalize_data
        data = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
        result = normalize_data(data)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 0.5)
        assert np.isclose(result[2], 1.0)


# =========================================================================
# Helper: decimal_degrees_to_dms (viz_panel.py)
# =========================================================================

class TestDecimalDegreesToDMS:
    def test_positive(self, qapp):
        from paravis.gui.widgets.viz_panel import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(45.5)
        assert d == 45
        assert m == 30
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_negative(self, qapp):
        from paravis.gui.widgets.viz_panel import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(-45.5)
        assert d == -45
        assert m == -30
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_zero(self, qapp):
        from paravis.gui.widgets.viz_panel import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(0.0)
        assert d == 0
        assert m == 0
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_precise(self, qapp):
        from paravis.gui.widgets.viz_panel import decimal_degrees_to_dms
        # 48°51'29.1348"
        d, m, s = decimal_degrees_to_dms(48.858093)
        assert d == 48
        assert m == 51
        assert s == pytest.approx(29.1348, abs=0.01)


# =========================================================================
# Helper: get_cmap_with_nan (viz_panel.py)
# =========================================================================

class TestGetCmapWithNan:
    def test_valid_cmap(self, qapp):
        from paravis.gui.widgets.viz_panel import get_cmap_with_nan, MATPLOTLIB_AVAILABLE
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        cmap = get_cmap_with_nan("viridis")
        assert cmap is not None
        # Should be a Colormap with bad color set
        import matplotlib as mpl
        assert isinstance(cmap, mpl.colors.Colormap)

    def test_invalid_cmap_returns_name(self, qapp):
        from paravis.gui.widgets.viz_panel import get_cmap_with_nan, MATPLOTLIB_AVAILABLE
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        result = get_cmap_with_nan("nonexistent_cmap_xyz")
        assert result == "nonexistent_cmap_xyz"

    def test_custom_bad_color(self, qapp):
        from paravis.gui.widgets.viz_panel import get_cmap_with_nan, MATPLOTLIB_AVAILABLE
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        cmap = get_cmap_with_nan("viridis", bad_color="red", bad_alpha=0.5)
        assert cmap is not None


# =========================================================================
# Helper: get_available_bands (indices_panel.py)
# =========================================================================

class TestGetAvailableBands:
    def test_basic(self, qapp):
        from paravis.gui.widgets.indices_panel import get_available_bands
        mapping = {1: "A", 2: "B", 3: "G", 4: "R", 5: "N"}
        bands = get_available_bands(mapping)
        assert bands == {"A", "B", "G", "R", "N"}

    def test_empty(self, qapp):
        from paravis.gui.widgets.indices_panel import get_available_bands
        bands = get_available_bands({})
        assert bands == set()

    def test_duplicates(self, qapp):
        from paravis.gui.widgets.indices_panel import get_available_bands
        mapping = {1: "R", 2: "R", 3: "N"}
        bands = get_available_bands(mapping)
        assert bands == {"R", "N"}


# =========================================================================
# Helper: get_index_info (indices_panel.py)
# =========================================================================

class TestGetIndexInfo:
    def test_known_index(self, qapp):
        from paravis.gui.widgets.indices_panel import get_index_info
        info = get_index_info("NDVI", {}, {4: "R", 5: "N"})
        assert info["name"] == "NDVI"
        assert "N" in info["required_bands"]
        assert "R" in info["required_bands"]
        assert info["computable"] is True
        assert "formula" in info

    def test_unknown_index(self, qapp):
        from paravis.gui.widgets.indices_panel import get_index_info
        info = get_index_info("NONEXISTENT_INDEX", {}, {})
        assert info["name"] == "NONEXISTENT_INDEX"
        assert info["computable"] is False

    def test_missing_bands_not_computable(self, qapp):
        from paravis.gui.widgets.indices_panel import get_index_info
        # NDVI needs N and R, but we only provide R
        info = get_index_info("NDVI", {}, {4: "R"})
        assert info["computable"] is False

    def test_with_constants(self, qapp):
        from paravis.gui.widgets.indices_panel import get_index_info
        # SAVI needs bands N, R and constant L
        info = get_index_info("SAVI", {"L": 0.5}, {4: "R", 5: "N"})
        assert info["name"] == "SAVI"
        assert info["computable"] is True

    def test_missing_constant(self, qapp):
        from paravis.gui.widgets.indices_panel import get_index_info
        # SAVI needs L constant, not provided
        info = get_index_info("SAVI", {}, {4: "R", 5: "N"})
        # L has a spyndex default of 0.5, so it's computable anyway
        assert info["computable"] is True


# =========================================================================
# Widget creation: IndicesWidget
# =========================================================================

class TestIndicesWidget:
    def test_create(self, qapp):
        from paravis.gui.widgets.indices_panel import IndicesWidget
        widget = IndicesWidget()
        assert widget.windowTitle() == ""  # no title set, it's a panel
        assert widget.indices_all is not None
        assert len(widget.indices_all) > 0
        assert hasattr(widget, 'band_mapping')
        assert hasattr(widget, 'constants_override')
        widget.close()

    def test_initial_band_mapping_default(self, qapp):
        from paravis.gui.widgets.indices_panel import IndicesWidget
        widget = IndicesWidget()
        assert len(widget.band_mapping) == 8
        assert widget.band_mapping[4] == "R"
        assert widget.band_mapping[5] == "N"
        widget.close()

    def test_update_indices_table(self, qapp):
        from paravis.gui.widgets.indices_panel import IndicesWidget
        widget = IndicesWidget()
        assert hasattr(widget, 'indices_model')
        assert widget.indices_model.rowCount() > 0
        widget.close()

    def test_get_available_bands_label(self, qapp):
        from paravis.gui.widgets.indices_panel import IndicesWidget
        widget = IndicesWidget()
        assert hasattr(widget, 'available_bands_label')
        assert widget.available_bands_label is not None
        widget.close()


# =========================================================================
# Widget creation: RaoQWidget
# =========================================================================

class TestRaoQWidget:
    def test_create(self, qapp):
        from paravis.gui.widgets.raoq_panel import RaoQWidget
        widget = RaoQWidget()
        assert widget.windowTitle() == ""  # it's a panel
        assert hasattr(widget, 'tab_widget')
        assert hasattr(widget, 'input_files')
        assert hasattr(widget, 'output_folder')
        assert hasattr(widget, 'sys_profile')
        widget.close()

    def test_has_tabs(self, qapp):
        from paravis.gui.widgets.raoq_panel import RaoQWidget
        widget = RaoQWidget()
        assert widget.tab_widget.count() >= 2  # Single + Batch
        widget.close()

    def test_input_files_initially_empty(self, qapp):
        from paravis.gui.widgets.raoq_panel import RaoQWidget
        widget = RaoQWidget()
        assert widget.input_files == []
        widget.close()

    def test_has_log_display(self, qapp):
        from paravis.gui.widgets.raoq_panel import RaoQWidget
        widget = RaoQWidget()
        assert hasattr(widget, 'log_display')
        widget.close()


# =========================================================================
# BatchProcessingManager
# =========================================================================

class TestBatchProcessingManager:
    def test_create(self, qapp):
        from paravis.gui.widgets.raoq_panel import BatchProcessingManager
        manager = BatchProcessingManager([], {})
        assert manager.is_running is True
        assert manager.current_worker is None

    def test_empty_job_list(self, qapp):
        """Running with no jobs should immediately finish."""
        from paravis.gui.widgets.raoq_panel import BatchProcessingManager
        manager = BatchProcessingManager([], {"output_folder": "/tmp"})
        results = []

        def capture(success, msg):
            results.append((success, msg))

        manager.finished_signal.connect(capture)
        manager.start()
        manager.wait(5000)
        QApplication.processEvents()

        assert len(results) > 0

    def test_stop_during_run(self, qapp):
        """Stop signal should be respected during processing."""
        from paravis.gui.widgets.raoq_panel import BatchProcessingManager
        manager = BatchProcessingManager([], {"output_folder": "/tmp"})
        manager.stop()
        assert manager.is_running is False


# =========================================================================
# VisualizationWidget (basic creation)
# =========================================================================

class TestVisualizationWidget:
    def test_create(self, qapp):
        from paravis.gui.widgets.viz_panel import VisualizationWidget
        widget = VisualizationWidget()
        assert hasattr(widget, 'plot_tabs')
        assert hasattr(widget, 'file_dict')
        assert widget.file_dict == {}
        assert hasattr(widget, 'progress_bar')
        widget.close()

    def test_file_dict_initially_empty(self, qapp):
        from paravis.gui.widgets.viz_panel import VisualizationWidget
        widget = VisualizationWidget()
        assert len(widget.file_dict) == 0
        widget.close()

    def test_has_plot_tabs(self, qapp):
        from paravis.gui.widgets.viz_panel import VisualizationWidget
        widget = VisualizationWidget()
        assert widget.plot_tabs.count() >= 4  # Individual, Stats, Diff, Split, TimeSeries
        widget.close()

    def test_has_load_button(self, qapp):
        from paravis.gui.widgets.viz_panel import VisualizationWidget
        widget = VisualizationWidget()
        assert hasattr(widget, 'load_btn')
        assert widget.load_btn is not None
        widget.close()

    def test_has_file_combo(self, qapp):
        from paravis.gui.widgets.viz_panel import VisualizationWidget
        widget = VisualizationWidget()
        assert hasattr(widget, 'file_combo')
        widget.close()


# =========================================================================
# FileSelectionDialog
# =========================================================================

class TestFileSelectionDialog:
    def test_create_empty(self, qapp):
        from paravis.gui.widgets.viz_panel import FileSelectionDialog
        dialog = FileSelectionDialog(file_dict={})
        assert dialog.windowTitle() is not None
        dialog.close()

    def test_with_files(self, qapp):
        from paravis.gui.widgets.viz_panel import FileSelectionDialog
        dialog = FileSelectionDialog(file_dict={"/tmp/a.tif": "/tmp/a.tif", "/tmp/b.tif": "/tmp/b.tif"})
        assert dialog.windowTitle() is not None
        assert len(dialog.selected_files) == 0  # nothing selected yet
        dialog.close()


# =========================================================================
# read_raster_memory_efficient
# =========================================================================

class TestReadRasterMemoryEfficient:
    def test_read_small_raster(self, qapp):
        """Test reading a small synthetic raster."""
        from paravis.gui.widgets.viz_panel import read_raster_memory_efficient
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp_path = f.name

        try:
            data = np.random.rand(50, 50).astype(np.float32)
            transform = from_origin(0, 50, 1, 1)
            with rasterio.open(
                tmp_path, 'w', driver='GTiff', height=50, width=50,
                count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
            ) as dst:
                dst.write(data, 1)

            result, transform_out, crs_out = read_raster_memory_efficient(tmp_path)
            assert result.shape == (50, 50)
            assert transform_out is not None
            assert crs_out is not None
        finally:
            os.unlink(tmp_path)

    def test_read_with_nodata(self, qapp):
        """Test reading a raster with nodata values."""
        from paravis.gui.widgets.viz_panel import read_raster_memory_efficient
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp_path = f.name

        try:
            data = np.ones((50, 50), dtype=np.float32)
            data[0:10, 0:10] = -9999  # nodata area
            transform = from_origin(0, 50, 1, 1)
            with rasterio.open(
                tmp_path, 'w', driver='GTiff', height=50, width=50,
                count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
                nodata=-9999,
            ) as dst:
                dst.write(data, 1)

            result, _, _ = read_raster_memory_efficient(tmp_path)
            assert np.isnan(result[0, 0])  # nodata → NaN
            assert result[20, 20] == 1.0   # valid pixel preserved
        finally:
            os.unlink(tmp_path)

    def test_large_raster_downsampled(self, qapp):
        """Test that large rasters get downsampled."""
        from paravis.gui.widgets.viz_panel import read_raster_memory_efficient
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a raster larger than max_pixels
            data = np.random.rand(2000, 2000).astype(np.float32)
            transform = from_origin(0, 2000, 1, 1)
            with rasterio.open(
                tmp_path, 'w', driver='GTiff', height=2000, width=2000,
                count=1, dtype=np.float32, crs='EPSG:4326', transform=transform,
            ) as dst:
                dst.write(data, 1)

            result, _, _ = read_raster_memory_efficient(tmp_path, max_pixels=500000)
            # Should be downsampled
            assert result.size < 2000 * 2000
            assert result.size > 0
        finally:
            os.unlink(tmp_path)
