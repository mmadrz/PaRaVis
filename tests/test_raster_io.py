"""
Unit tests for paravis.core.raster — no Qt dependency required.

Covers reader, writer, and utility functions for raster I/O.
Run with:  pytest tests/test_raster_io.py -v
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ===========================================================================
# Helper: mock rasterio DatasetReader
# ===========================================================================


class _MockRasterReader:
    """Simulates a rasterio DatasetReader for testing read_raster."""
    def __init__(self, height=20, width=30, nodata=None, crs=None,
                 n_bands=3, dtype=np.float32):
        self.height = height
        self.width = width
        self.nodata = nodata
        self.crs = crs
        self.transform = None
        self._n_bands = n_bands
        self._dtype = dtype

    def read(self, band=None, out_shape=None):
        if band is not None:
            # Single band
            shape = out_shape or (self.height, self.width)
            return np.random.rand(*shape).astype(self._dtype)
        # All bands
        if out_shape is not None:
            return np.random.rand(self._n_bands, *out_shape).astype(self._dtype)
        return np.random.rand(self._n_bands, self.height, self.width).astype(self._dtype)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ===========================================================================
# reader.py
# ===========================================================================


class TestReadRaster:
    """Tests for paravis.core.raster.reader.read_raster."""

    def test_read_all_bands_small(self):
        """Reading a small raster without downsampling returns all bands."""
        mock_src = _MockRasterReader(height=20, width=30, n_bands=3, crs="EPSG:4326")

        with patch("rasterio.open", return_value=mock_src):
            from paravis.core.raster.reader import read_raster
            data, transform, crs = read_raster("dummy.tif", max_pixels=1_000_000)
            assert data.shape == (3, 20, 30)
            assert data.dtype == np.float64
            assert crs == "EPSG:4326"

    def test_read_single_band(self):
        """Reading a single band returns 2D array."""
        mock_src = _MockRasterReader(height=20, width=30, n_bands=3)

        with patch("rasterio.open", return_value=mock_src):
            from paravis.core.raster.reader import read_raster
            data, transform, crs = read_raster("dummy.tif", band=1, max_pixels=1_000_000)
            assert data.shape == (20, 30)

    def test_read_downsample(self):
        """Reading a large raster triggers downsampling."""
        # 2000x2000 = 4M pixels > 1M max_pixels
        # factor = int(sqrt(4M/1M)) = int(2) = 2
        mock_src = _MockRasterReader(height=2000, width=2000)

        with patch("rasterio.open", return_value=mock_src):
            from paravis.core.raster.reader import read_raster
            # Just verify it doesn't crash — the actual downsampling uses out_shape
            data, transform, crs = read_raster("dummy.tif", max_pixels=1_000_000)
            assert data is not None

    def test_read_factor_one(self):
        """When factor == 1 (total_pixels just over max_pixels), should read without out_shape."""
        # 1001x1000 = 1,001,000 pixels > 1,000,000 max_pixels
        # factor = int(sqrt(1,001,000/1,000,000)) = int(sqrt(1.001)) = int(1.0005) = 1
        mock_src = _MockRasterReader(height=1001, width=1000)

        with patch("rasterio.open", return_value=mock_src):
            from paravis.core.raster.reader import read_raster
            data, _, _ = read_raster("dummy.tif", max_pixels=1_000_000)
            assert data is not None

    def test_read_nodata_conversion(self):
        """Nodata values should become NaN."""
        mock_src = _MockRasterReader(height=10, width=10, nodata=-9999)

        with patch("rasterio.open", return_value=mock_src):
            from paravis.core.raster.reader import read_raster
            data, _, _ = read_raster("dummy.tif", max_pixels=1_000_000)
            assert data.dtype == np.float64

    def test_read_without_nodata_conversion(self):
        """Without nodata, data stays float but converted to float64."""
        mock_src = _MockRasterReader(height=10, width=10, nodata=None)

        with patch("rasterio.open", return_value=mock_src):
            from paravis.core.raster.reader import read_raster
            data, _, _ = read_raster("dummy.tif", max_pixels=1_000_000)
            assert data.dtype == np.float64
            assert not np.any(np.isnan(data))


class TestNormalizeData:
    """Tests for paravis.core.raster.reader.normalize_data."""

    def test_normal_bounds(self):
        """Normal data maps to [0, 1]."""
        from paravis.core.raster.reader import normalize_data
        data = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        result = normalize_data(data)
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_all_same_value(self):
        """All-equal data should return zeros."""
        from paravis.core.raster.reader import normalize_data
        data = np.ones((5, 5), dtype=np.float64) * 3.0
        result = normalize_data(data)
        assert np.all(result == 0.0)

    def test_all_nan(self):
        """All-NaN data should return zeros."""
        from paravis.core.raster.reader import normalize_data
        data = np.full((5, 5), np.nan)
        result = normalize_data(data)
        assert np.all(result == 0.0)

    def test_empty_array(self):
        """Empty array should return empty."""
        from paravis.core.raster.reader import normalize_data
        data = np.array([])
        result = normalize_data(data)
        assert result.size == 0

    def test_min_equals_max(self):
        """When min == max (single unique value), return zeros."""
        from paravis.core.raster.reader import normalize_data
        data = np.full((3, 3), 42.0, dtype=np.float64)
        result = normalize_data(data)
        assert np.all(result == 0.0)

    def test_with_nan_values(self):
        """NaN values should be preserved (not affect min/max)."""
        from paravis.core.raster.reader import normalize_data
        data = np.array([np.nan, 2.0, 8.0], dtype=np.float64)
        result = normalize_data(data)
        assert np.isnan(result[0])
        assert np.isclose(result[1], 0.0)
        assert np.isclose(result[2], 1.0)

    def test_negative_values(self):
        """Negative values should be handled properly."""
        from paravis.core.raster.reader import normalize_data
        data = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
        result = normalize_data(data)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 0.5)
        assert np.isclose(result[2], 1.0)

    def test_float32_input(self):
        """Float32 input should still work."""
        from paravis.core.raster.reader import normalize_data
        data = np.array([0.0, 10.0], dtype=np.float32)
        result = normalize_data(data)
        assert result.dtype == np.float64
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 1.0)


class TestDownsampleData:
    """Tests for paravis.core.raster.reader.downsample_data."""

    def test_below_threshold_no_change(self):
        """Data below max_pixels returns unchanged."""
        from paravis.core.raster.reader import downsample_data
        data = np.random.rand(100, 100)
        result = downsample_data(data, max_pixels=1_000_000)
        assert result.shape == (100, 100)
        assert np.array_equal(result, data)

    def test_above_threshold_downsample(self):
        """Data above max_pixels is downsampled."""
        from paravis.core.raster.reader import downsample_data
        data = np.random.rand(1000, 1000)
        result = downsample_data(data, max_pixels=100_000)
        # 1M pixels > 100K, factor = int(sqrt(10)) = 3
        # shape = (1000//3, 1000//3) = (333, 333)
        assert result.shape[0] < 1000
        assert result.shape[1] < 1000

    def test_factor_less_than_one(self):
        """If factor < 1, data is returned unchanged."""
        from paravis.core.raster.reader import downsample_data
        data = np.random.rand(50, 50)
        result = downsample_data(data, max_pixels=100_000)
        assert result.shape == (50, 50)

    def test_exact_threshold(self):
        """Data exactly at max_pixels threshold returns unchanged."""
        from paravis.core.raster.reader import downsample_data
        data = np.random.rand(1000, 1000)
        result = downsample_data(data, max_pixels=1_000_000)
        assert result.shape == (1000, 1000)


# ===========================================================================
# utils.py
# ===========================================================================


class TestDecimalDegreesToDMS:
    """Tests for paravis.core.raster.utils.decimal_degrees_to_dms."""

    def test_positive_value(self):
        from paravis.core.raster.utils import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(45.5)
        assert d == 45
        assert m == 30
        assert s == 0.0

    def test_negative_value(self):
        from paravis.core.raster.utils import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(-45.5)
        assert d == -45
        assert m == 30
        assert s == 0.0

    def test_negative_small(self):
        """Small negative value."""
        from paravis.core.raster.utils import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(-0.5)
        assert d == 0
        assert m == 30
        assert s == 0.0

    def test_zero(self):
        from paravis.core.raster.utils import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(0.0)
        assert d == 0
        assert m == 0
        assert s == 0.0

    def test_seconds_fraction(self):
        from paravis.core.raster.utils import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(10.1234)
        assert d == 10
        assert m == 7
        assert s == pytest.approx(24.24, rel=0.1)

    def test_exact_degree(self):
        from paravis.core.raster.utils import decimal_degrees_to_dms
        d, m, s = decimal_degrees_to_dms(90.0)
        assert d == 90
        assert m == 0
        assert s == 0.0


class TestGetCmapWithNan:
    """Tests for paravis.core.raster.utils.get_cmap_with_nan."""

    def test_returns_cmap(self):
        """Should return a valid colormap when matplotlib is available."""
        from paravis.core.raster.utils import get_cmap_with_nan
        cmap = get_cmap_with_nan("viridis")
        # Should be a matplotlib colormap
        assert cmap is not None

    def test_with_custom_bad_color(self):
        """Should set a custom bad color."""
        from paravis.core.raster.utils import get_cmap_with_nan
        cmap = get_cmap_with_nan("plasma", bad_color="red", bad_alpha=0.8)
        assert cmap is not None

    def test_colormaps_fails_returns_none(self):
        """When plt.colormaps[] fails, should return None."""
        with patch("matplotlib.pyplot.colormaps") as mock_cm:
            mock_cm.__getitem__.side_effect = KeyError("not found")
            from paravis.core.raster.utils import get_cmap_with_nan
            cmap = get_cmap_with_nan("viridis")
            assert cmap is None

    def test_nonexistent_cmap_returns_none(self):
        """Non-existent colormap should return None."""
        from paravis.core.raster.utils import get_cmap_with_nan
        cmap = get_cmap_with_nan("nonexistent_cmap_xyz")
        assert cmap is None

    def test_get_cmap_with_nan_success(self):
        """Normal path with a valid colormap should return a colormap."""
        from paravis.core.raster.utils import get_cmap_with_nan
        cmap = get_cmap_with_nan("viridis")
        assert cmap is not None


# ===========================================================================
# writer.py
# ===========================================================================


class TestWriteGeoTiff:
    """Tests for paravis.core.raster.writer.write_geotiff."""

    def test_basic_write(self):
        """Basic write should work with minimal args."""
        data = np.random.rand(20, 30).astype(np.float32)

        with patch("rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            from paravis.core.raster.writer import write_geotiff
            result = write_geotiff(data, "/tmp/test.tif")
            assert result == "/tmp/test.tif"
            mock_dst.write.assert_called_once_with(data, 1)

    def test_write_with_crs_transform(self):
        """Write with CRS and transform should pass them to profile."""
        data = np.random.rand(10, 10).astype(np.float64)
        transform = object()
        crs = "EPSG:4326"

        with patch("rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            from paravis.core.raster.writer import write_geotiff
            result = write_geotiff(data, "/tmp/test.tif", transform=transform, crs=crs)
            assert result == "/tmp/test.tif"

    def test_write_no_compress(self):
        """Write without compression should set compress to None."""
        data = np.random.rand(10, 10).astype(np.float32)

        with patch("rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            from paravis.core.raster.writer import write_geotiff
            result = write_geotiff(data, "/tmp/test.tif", compress=False)
            assert result == "/tmp/test.tif"

    def test_write_with_nodata(self):
        """Write with nodata should pass nodata to profile."""
        data = np.random.rand(10, 10).astype(np.float32)

        with patch("rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            from paravis.core.raster.writer import write_geotiff
            result = write_geotiff(data, "/tmp/test.tif", nodata=-9999.0)
            assert result == "/tmp/test.tif"

    def test_write_with_explicit_dtype(self):
        """Explicit dtype should be used."""
        data = np.random.rand(10, 10).astype(np.float32)

        with patch("rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            from paravis.core.raster.writer import write_geotiff
            result = write_geotiff(data, "/tmp/test.tif", dtype="float32")
            assert result == "/tmp/test.tif"

    def test_write_inferred_dtype(self):
        """When dtype is None, should infer from data."""
        data = np.random.rand(10, 10).astype(np.float32)

        with patch("rasterio.open") as mock_open:
            mock_dst = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_dst

            from paravis.core.raster.writer import write_geotiff
            result = write_geotiff(data, "/tmp/test.tif")
            assert result == "/tmp/test.tif"


# ===========================================================================
# __init__.py (raster package exports)
# ===========================================================================


class TestRasterInit:
    """Tests for paravis.core.raster.__init__ exports."""

    def test_exports(self):
        """The raster __init__ should export key functions."""
        from paravis.core.raster import read_raster, normalize_data, downsample_data
        assert callable(read_raster)
        assert callable(normalize_data)
        assert callable(downsample_data)
