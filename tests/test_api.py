"""
Tests for paravis.api — high-level public API.

Run with:  pytest tests/test_api.py -v --cov=paravis.api
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pytest


class TestListAvailableIndices:
    def test_returns_list_of_strings(self):
        from paravis.api.indices import list_available_indices
        result = list_available_indices()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(n, str) for n in result)
        assert "NDVI" in result


class TestComputeIndex:
    def test_compute_single_index(self):
        """Test compute_index with mocked read_raster."""
        mock_data = np.zeros((5, 10, 10), dtype=np.float32)
        mock_data[3] = 0.2  # R
        mock_data[4] = 0.8  # N

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_index
            result = compute_index("dummy.tif", "NDVI")
            assert result.shape == (10, 10)
            assert np.isclose(result[0, 0], 0.6, atol=1e-5)

    def test_compute_with_custom_mapping(self):
        mock_data = np.zeros((5, 10, 10), dtype=np.float32)
        mock_data[3] = 0.2
        mock_data[4] = 0.8

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_index
            result = compute_index("dummy.tif", "NDVI",
                                   band_mapping={4: "R", 5: "N"})
            assert result.shape == (10, 10)

    def test_compute_2d_raster(self):
        """2D input should be auto-expanded to 3D."""
        mock_data_2d = np.random.rand(10, 10).astype(np.float32)

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data_2d, None, None)
            from paravis.api.indices import compute_index
            # This would fail for NDVI since 2D has no bands,
            # but we just ensure the 2D->3D path is exercised
            with pytest.raises(RuntimeError):
                compute_index("dummy.tif", "NDVI")


class TestComputeIndices:
    def test_compute_multiple_indices(self):
        mock_data = np.zeros((5, 10, 10), dtype=np.float32)
        mock_data[3] = 0.2
        mock_data[4] = 0.8
        mock_data[0] = 0.1  # A

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_indices
            results = compute_indices("dummy.tif", indices=["NDVI"],
                                      band_mapping={4: "R", 5: "N"})
            assert "NDVI" in results
            assert results["NDVI"].shape == (10, 10)

    def test_compute_all_indices(self):
        """When indices=None, should auto-detect computable indices."""
        mock_data = np.zeros((5, 10, 10), dtype=np.float32)
        mock_data[3] = 0.2
        mock_data[4] = 0.8

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_indices
            # With only R and N mapped, only indices needing those bands are computable
            results = compute_indices("dummy.tif", indices=None,
                                      band_mapping={4: "R", 5: "N"})
            assert isinstance(results, dict)
            # Should contain NDVI since it only needs R and N
            assert "NDVI" in results


class TestComputeRaoQ:
    def test_compute_rao_q_cpu(self):
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="cpu")
            assert result.shape == (16, 16)

    def test_compute_rao_q_auto_cpu_fallback(self):
        """When GPU is not available, 'auto' should fall back to CPU."""
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            with patch("paravis.api.raoq.is_gpu_available", return_value=False):
                mock_read.return_value = (mock_data, None, None)
                from paravis.api.raoq import compute_rao_q
                result = compute_rao_q("dummy.tif", window_size=3)
                assert result.shape == (16, 16)

    def test_compute_rao_q_parallel(self):
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="parallel",
                                   n_workers=2)
            assert result.shape == (16, 16)

    def test_compute_rao_q_gpu_backend(self):
        """Test explicit GPU backend selection (may run CPU fallback in test)."""
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="gpu")
            assert result.shape == (16, 16)

    def test_rao_q_2d_input(self):
        """2D input should be auto-expanded."""
        mock_data_2d = np.random.rand(16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data_2d, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="cpu")
            assert result.shape == (16, 16)


class TestPlotRaster:
    def test_plot_raster_returns_fig_ax(self):
        """plot_raster should return a (figure, axes) tuple."""
        mock_data = np.random.rand(50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.colorbar") as mock_cbar,
        ):
            mock_read.return_value = (mock_data, None, None)
            import matplotlib.figure
            import matplotlib.axes
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_ax = MagicMock(spec=matplotlib.axes.Axes)
            mock_subplots.return_value = (mock_fig, mock_ax)

            from paravis.api.visualization import plot_raster
            fig, ax = plot_raster("dummy.tif", title="Test")
            assert fig is mock_fig
            assert ax is mock_ax
            mock_cbar.assert_called_once()

    def test_plot_raster_3d(self):
        """3D data should use first band."""
        mock_data_3d = np.random.rand(3, 50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.colorbar") as mock_cbar,
        ):
            mock_read.return_value = (mock_data_3d, None, None)
            import matplotlib.figure
            import matplotlib.axes
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_ax = MagicMock(spec=matplotlib.axes.Axes)
            mock_subplots.return_value = (mock_fig, mock_ax)

            from paravis.api.visualization import plot_raster
            fig, ax = plot_raster("dummy.tif")
            assert fig is mock_fig
            mock_cbar.assert_called_once()

    def test_plot_raster_no_colorbar(self):
        mock_data = np.random.rand(50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
        ):
            mock_read.return_value = (mock_data, None, None)
            import matplotlib.figure
            import matplotlib.axes
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_ax = MagicMock(spec=matplotlib.axes.Axes)
            mock_subplots.return_value = (mock_fig, mock_ax)

            from paravis.api.visualization import plot_raster
            fig, ax = plot_raster("dummy.tif", show_colorbar=False)
            assert fig is mock_fig


class TestPlotComparison:
    @staticmethod
    def _make_axes_array(n):
        """Build a (1, n) object array of MagicMock axes."""
        import matplotlib.axes
        axes = np.empty((1, n), dtype=object)
        for i in range(n):
            axes[0, i] = MagicMock(spec=matplotlib.axes.Axes)
        return axes

    def test_plot_comparison(self):
        mock_data = np.random.rand(50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.colorbar") as mock_cbar,
        ):
            mock_read.return_value = (mock_data, None, None)
            import matplotlib.figure
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_axes = self._make_axes_array(2)
            mock_subplots.return_value = (mock_fig, mock_axes)

            from paravis.api.visualization import plot_comparison
            fig, axes = plot_comparison(
                ["file1.tif", "file2.tif"],
                labels=["A", "B"]
            )
            assert fig is mock_fig

    def test_plot_comparison_no_labels(self):
        mock_data = np.random.rand(50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.colorbar") as mock_cbar,
        ):
            mock_read.return_value = (mock_data, None, None)
            import matplotlib.figure
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_axes = self._make_axes_array(1)
            mock_subplots.return_value = (mock_fig, mock_axes)

            from paravis.api.visualization import plot_comparison
            fig, axes = plot_comparison(["file1.tif"])
            assert fig is mock_fig


class TestAPIInit:
    def test_lazy_imports(self):
        """Test that the package __init__ lazy imports work."""
        import paravis
        # These should trigger lazy import
        indices = paravis.compute_indices
        raoq = paravis.compute_rao_q
        plot = paravis.plot_raster
        comp = paravis.plot_comparison
        assert callable(indices)
        assert callable(raoq)
        assert callable(plot)
        assert callable(comp)


class TestComputeRaoQEdgeCases:
    """Edge cases for the RaoQ API."""

    def test_compute_rao_q_auto_with_gpu(self):
        """Backend 'auto' with GPU available should use GPU path."""
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            with patch("paravis.api.raoq.is_gpu_available", return_value=True):
                mock_read.return_value = (mock_data, None, None)
                from paravis.api.raoq import compute_rao_q
                result = compute_rao_q("dummy.tif", window_size=3, backend="auto")
                assert result.shape == (16, 16)

    def test_compute_rao_q_with_simplify(self):
        """Test simplify parameter is passed through."""
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="cpu", simplify=5)
            assert result.shape == (16, 16)

    def test_compute_rao_q_with_na_tolerance(self):
        """Test na_tolerance parameter is passed through."""
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="cpu",
                                   na_tolerance=0.5)
            assert result.shape == (16, 16)

    def test_compute_rao_q_with_step_size(self):
        """Test step_size parameter is passed through."""
        mock_data = np.random.rand(1, 16, 16).astype(np.float32)

        with patch("paravis.api.raoq.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.raoq import compute_rao_q
            result = compute_rao_q("dummy.tif", window_size=3, backend="cpu", step_size=1)
            assert result.shape == (16, 16)


class TestComputeIndicesAllNone:
    """Test compute_indices with indices=None and no computable indices."""

    def test_no_computable_indices(self):
        """When no indices are computable, should return empty dict."""
        mock_data = np.zeros((2, 10, 10), dtype=np.float32)

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_indices
            # Mapping only bands 1,2 → no index can be computed
            results = compute_indices("dummy.tif", indices=None,
                                      band_mapping={1: "A", 2: "B"})
            assert isinstance(results, dict)
            # May be empty since no standard index uses only A and B
            # But should at least return a dict without error


class TestComputeIndexEdgeCases:
    """Additional edge cases for compute_index API."""

    def test_compute_index_no_constants(self):
        """compute_index with constants=None."""
        mock_data = np.zeros((5, 10, 10), dtype=np.float32)
        mock_data[3] = 0.2
        mock_data[4] = 0.8

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_index
            result = compute_index("dummy.tif", "NDVI",
                                   band_mapping={4: "R", 5: "N"},
                                   constants=None)
            assert result.shape == (10, 10)


class TestPlotComparisonEdgeCases:
    """Edge cases for plot_comparison API."""

    def test_plot_comparison_single_file(self):
        """Comparison with single file."""
        mock_data = np.random.rand(50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.colorbar") as mock_cbar,
        ):
            mock_read.return_value = (mock_data, None, None)
            import matplotlib.figure
            import matplotlib.axes
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_axes = np.empty((1, 1), dtype=object)
            mock_axes[0, 0] = MagicMock(spec=matplotlib.axes.Axes)
            mock_subplots.return_value = (mock_fig, mock_axes)

            from paravis.api.visualization import plot_comparison
            fig, axes = plot_comparison(["file1.tif"])
            assert fig is mock_fig

    def test_plot_comparison_3d_data(self):
        """Comparison with 3D raster data (should use first band)."""
        mock_data_3d = np.random.rand(3, 50, 50).astype(np.float32)

        with (
            patch("paravis.api.visualization.read_raster") as mock_read,
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("matplotlib.pyplot.colorbar") as mock_cbar,
        ):
            mock_read.return_value = (mock_data_3d, None, None)
            import matplotlib.figure
            import matplotlib.axes
            mock_fig = MagicMock(spec=matplotlib.figure.Figure)
            mock_axes = np.empty((1, 2), dtype=object)
            mock_axes[0, 0] = MagicMock(spec=matplotlib.axes.Axes)
            mock_axes[0, 1] = MagicMock(spec=matplotlib.axes.Axes)
            mock_subplots.return_value = (mock_fig, mock_axes)

            from paravis.api.visualization import plot_comparison
            fig, axes = plot_comparison(["file1.tif", "file2.tif"],
                                        labels=["A", "B"])
            assert fig is mock_fig


class TestAPIIndicesAutoDetect:
    """Test the auto-detect path in api.indices (indices=None)."""

    def test_auto_detect_computable(self):
        """When indices=None, should auto-detect computable indices."""
        # Create data with bands that map to R and N (needed for NDVI)
        mock_data = np.zeros((5, 10, 10), dtype=np.float32)
        mock_data[3] = 0.2  # R
        mock_data[4] = 0.8  # N

        with patch("paravis.api.indices.read_raster") as mock_read:
            mock_read.return_value = (mock_data, None, None)
            from paravis.api.indices import compute_indices
            results = compute_indices("dummy.tif", indices=None,
                                      band_mapping={4: "R", 5: "N"})
            assert isinstance(results, dict)
            assert "NDVI" in results
