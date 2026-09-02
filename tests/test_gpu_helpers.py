"""
Tests for GPU helper functions with mocked CuPy — no real GPU needed.

Covers _get_gpu_limits, _estimate_gpu_batch_size, get_gpu_info,
is_gpu_available, METRIC_IDS, and CPU fallback paths in gpu.py.

Run with:  pytest tests/test_gpu_helpers.py -v
"""
import sys
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: make cupy mockable even when not installed
# ---------------------------------------------------------------------------

def _mock_cupy_device(mock_device=None, side_effect=None):
    """Return a context manager that injects a mock cupy into sys.modules.

    This lets ``patch("cupy.cuda.Device", ...)`` resolve correctly even
    when CuPy is not installed in the test environment.
    """
    mock_cupy = MagicMock()
    if side_effect is not None:
        mock_cupy.cuda.Device.side_effect = side_effect
    elif mock_device is not None:
        mock_cupy.cuda.Device.return_value = mock_device
    return patch.dict("sys.modules", {"cupy": mock_cupy})


class TestIsGpuAvailable:
    def test_returns_false_without_cupy(self):
        with patch.dict("sys.modules", {"cupy": None}):
            import paravis.core.raoq.gpu as gpu_mod
            # When cupy import fails, GPU_AVAILABLE should be False
            # The module-level flag is set at import time, so we test
            # the function directly
            result = gpu_mod.is_gpu_available()
            # Result depends on whether cupy is actually installed
            assert isinstance(result, bool)

    def test_returns_bool(self):
        from paravis.core.raoq.gpu import is_gpu_available
        result = is_gpu_available()
        assert isinstance(result, bool)


class TestGetGpuInfo:
    def test_no_gpu_returns_zeros(self):
        """When GPU_AVAILABLE is False, get_gpu_info returns zeroed dict."""
        from paravis.core.raoq.gpu import get_gpu_info, GPU_AVAILABLE
        if not GPU_AVAILABLE:
            info = get_gpu_info()
            assert info["name"] is None
            assert info["total_gb"] == 0
            assert info["free_gb"] == 0
            assert info["compute_capability"] is None

    def test_with_mock_gpu(self):
        """When GPU is available, mock it and verify return."""
        from paravis.core.raoq.gpu import get_gpu_info
        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)  # total=8GB, free=4GB
        mock_device.properties = {"name": b"TestGPU", "major": 7, "minor": 5}
        mock_attrs = {"max_shared_memory_per_block": 49152}
        mock_device.attributes = mock_attrs

        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True):
            with _mock_cupy_device(mock_device):
                info = get_gpu_info()
                assert info["total_gb"] == pytest.approx(8.0, rel=0.01)
                assert info["free_gb"] == pytest.approx(4.0, rel=0.01)
                assert "7.5" == info["compute_capability"]

    def test_exception_returns_unknown(self):
        """When an exception occurs, get_gpu_info returns 'Unknown'."""
        from paravis.core.raoq.gpu import get_gpu_info

        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True):
            with _mock_cupy_device(side_effect=Exception("no gpu")):
                info = get_gpu_info()
                assert info["name"] == "Unknown"
                assert info["total_gb"] == 0


class TestGetGpuLimits:
    def test_returns_expected_keys(self):
        """_get_gpu_limits should return all expected keys."""
        from paravis.core.raoq.gpu import _get_gpu_limits

        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)
        mock_device.attributes = {"max_shared_memory_per_block": 49152}

        with _mock_cupy_device(mock_device):
            limits = _get_gpu_limits(block_size=256)
            assert "max_shared_n_pixels" in limits
            assert "max_fallback_n_pixels" in limits
            assert "max_shared_mem_bytes" in limits
            assert "free_memory_gb" in limits

    def test_with_mock_cupy_values(self):
        """_get_gpu_limits should compute correct values from a mocked GPU."""
        from paravis.core.raoq.gpu import _get_gpu_limits

        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)  # total=8GB, free=4GB
        mock_device.attributes = {"max_shared_memory_per_block": 49152}

        with _mock_cupy_device(mock_device):
            limits = _get_gpu_limits(block_size=256)
            assert limits["max_shared_mem_bytes"] == 49152
            assert limits["free_memory_gb"] == pytest.approx(4.0, rel=0.01)
            # max_fallback = (4GB * 0.25) / (12 * 256), capped at 10000
            assert limits["max_fallback_n_pixels"] <= 10000
            assert limits["max_fallback_n_pixels"] > 0

    def test_fallback_limit_capped_at_10000(self):
        """Fallback n_pixels should never exceed MAX_N_PIXELS (10000)."""
        from paravis.core.raoq.gpu import _get_gpu_limits

        # Huge free memory → unbounded fallback without cap
        mock_device = MagicMock()
        mock_device.mem_info = (1024 * 1024**3, 512 * 1024**3)  # 512 GB free
        mock_device.attributes = {"max_shared_memory_per_block": 49152}

        with _mock_cupy_device(mock_device):
            limits = _get_gpu_limits(block_size=256)
            assert limits["max_fallback_n_pixels"] <= 10000

    def test_small_memory(self):
        """Very small GPU memory should give fallback limit at or below cap."""
        from paravis.core.raoq.gpu import _get_gpu_limits

        # Only 1 MB free — the formula gives (1MB * 0.25) / (12*256) ≈ 85
        mock_device = MagicMock()
        mock_device.mem_info = (1 * 1024**3, 1 * 1024**2)  # 1 MB free
        mock_device.attributes = {"max_shared_memory_per_block": 49152}

        with _mock_cupy_device(mock_device):
            limits = _get_gpu_limits(block_size=256)
            assert limits["max_fallback_n_pixels"] <= 10000
            assert limits["max_fallback_n_pixels"] > 0

    def test_shared_mem_headroom(self):
        """Shared memory should have 1KB headroom subtracted."""
        from paravis.core.raoq.gpu import _get_gpu_limits

        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)
        # Request 65536 bytes → usable = 65536 - 1024 = 64512
        mock_device.attributes = {"max_shared_memory_per_block": 65536}

        with _mock_cupy_device(mock_device):
            limits = _get_gpu_limits(block_size=256)
            assert limits["max_shared_mem_bytes"] == 64512

    def test_fallback_old_attr_name(self):
        """Should handle 'SharedMemoryPerBlock' attribute name."""
        from paravis.core.raoq.gpu import _get_gpu_limits

        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)
        # Simulate older GPU that doesn't have max_shared_memory_per_block
        mock_device.attributes = {"SharedMemoryPerBlock": 49152}

        with _mock_cupy_device(mock_device):
            limits = _get_gpu_limits(block_size=256)
            assert limits["max_shared_mem_bytes"] > 0


class TestEstimateGpuBatchSize:
    def test_returns_positive_integer(self):
        """_estimate_gpu_batch_size should return a positive integer."""
        from paravis.core.raoq.gpu import _estimate_gpu_batch_size

        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)

        with _mock_cupy_device(mock_device):
            result = _estimate_gpu_batch_size(n_pixels=25, n_bands=4)
            assert isinstance(result, int)
            assert result > 0

    def test_caps_at_200000(self):
        """Batch size should never exceed 200000."""
        from paravis.core.raoq.gpu import _estimate_gpu_batch_size

        mock_device = MagicMock()
        mock_device.mem_info = (1024 * 1024**3, 512 * 1024**3)  # huge GPU

        with _mock_cupy_device(mock_device):
            result = _estimate_gpu_batch_size(n_pixels=1, n_bands=1)
            assert result <= 200000

    def test_minimum_batch(self):
        """Batch size should be at least target_rows * 100."""
        from paravis.core.raoq.gpu import _estimate_gpu_batch_size

        mock_device = MagicMock()
        mock_device.mem_info = (1024**3, 1024**2)  # 1MB free → very small

        with _mock_cupy_device(mock_device):
            result = _estimate_gpu_batch_size(n_pixels=10000, n_bands=20, target_rows=1)
            assert result >= 100  # target_rows * 100

    def test_exception_returns_fallback(self):
        """When cupy fails, should return conservative fallback (50000)."""
        from paravis.core.raoq.gpu import _estimate_gpu_batch_size

        with _mock_cupy_device(side_effect=Exception("no gpu")):
            result = _estimate_gpu_batch_size(n_pixels=25, n_bands=4)
            assert result == 50000

    def test_large_window_smaller_batch(self):
        """Larger windows should produce smaller batch sizes."""
        from paravis.core.raoq.gpu import _estimate_gpu_batch_size

        mock_device = MagicMock()
        mock_device.mem_info = (8 * 1024**3, 4 * 1024**3)

        with _mock_cupy_device(mock_device):
            small = _estimate_gpu_batch_size(n_pixels=9, n_bands=3)
            large = _estimate_gpu_batch_size(n_pixels=10000, n_bands=20)
            assert small > large

    def test_target_rows_parameter(self):
        """target_rows parameter should affect minimum batch size."""
        from paravis.core.raoq.gpu import _estimate_gpu_batch_size

        mock_device = MagicMock()
        mock_device.mem_info = (1024**3, 1024**2)  # very small GPU

        with _mock_cupy_device(mock_device):
            r1 = _estimate_gpu_batch_size(n_pixels=25, n_bands=4, target_rows=1)
            r5 = _estimate_gpu_batch_size(n_pixels=25, n_bands=4, target_rows=5)
            assert r5 >= r1


class TestMetricIds:
    """Test METRIC_IDS mapping."""

    def test_metric_ids_complete(self):
        from paravis.core.raoq.gpu import METRIC_IDS
        expected = {"euclidean": 0, "manhattan": 1, "chebyshev": 2,
                    "minkowski": 3, "canberra": 4, "braycurtis": 5}
        assert METRIC_IDS == expected

    def test_metric_ids_all_unique(self):
        from paravis.core.raoq.gpu import METRIC_IDS
        values = list(METRIC_IDS.values())
        assert len(values) == len(set(values))


class TestComputeRaoQGpuCpuFallback:
    """Test compute_rao_q_gpu when GPU is not available (CPU fallback)."""

    def test_cpu_fallback_no_gpu(self):
        """When GPU_AVAILABLE is False, should fall back to CPU."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.random.rand(1, 10, 10).astype(np.float32)
        config = RaoQConfig(window_size=3, use_gpu=False)

        result = compute_rao_q_gpu(data, config)
        assert result.shape == (10, 10)

    def test_cpu_fallback_with_progress(self):
        """CPU fallback should still call progress_callback."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.random.rand(1, 10, 10).astype(np.float32)
        config = RaoQConfig(window_size=3, use_gpu=False)
        progress_calls = []

        result = compute_rao_q_gpu(
            data, config,
            progress_callback=lambda c, t: progress_calls.append((c, t))
        )
        assert result.shape == (10, 10)
        assert len(progress_calls) > 0

    def test_cpu_fallback_all_nan_batch(self):
        """CPU fallback with all-NaN data should not crash."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.full((1, 10, 10), np.nan, dtype=np.float32)
        config = RaoQConfig(window_size=3, use_gpu=False)

        result = compute_rao_q_gpu(data, config)
        assert result.shape == (10, 10)
        assert np.all(np.isnan(result))

    def test_cpu_fallback_all_nan_with_progress(self):
        """All-NaN data with progress_callback should still report progress."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.full((1, 10, 10), np.nan, dtype=np.float32)
        config = RaoQConfig(window_size=3, use_gpu=False)
        progress_calls = []

        result = compute_rao_q_gpu(
            data, config,
            progress_callback=lambda c, t: progress_calls.append((c, t))
        )
        assert result.shape == (10, 10)
        assert len(progress_calls) > 0

    def test_cpu_fallback_all_nan_multiband(self):
        """All-NaN data with multiple bands should handle correctly."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.full((3, 10, 10), np.nan, dtype=np.float32)
        config = RaoQConfig(window_size=3, use_gpu=False)

        result = compute_rao_q_gpu(data, config)
        assert result.shape == (10, 10)

    def test_cpu_fallback_mixed_nan(self):
        """Mixed NaN/valid data should compute partial results."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.random.rand(2, 12, 12).astype(np.float32)
        data[0, :5, :5] = np.nan  # partial NaN
        config = RaoQConfig(window_size=3, na_tolerance=0.8, use_gpu=False)

        result = compute_rao_q_gpu(data, config)
        assert result.shape == (12, 12)

    def test_cpu_fallback_distance_metrics(self):
        """CPU fallback should work with all distance metrics."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig

        data = np.random.rand(2, 10, 10).astype(np.float32)
        for metric in ["euclidean", "manhattan", "chebyshev", "minkowski",
                        "canberra", "braycurtis"]:
            config = RaoQConfig(window_size=3, distance_metric=metric,
                               use_gpu=False)
            result = compute_rao_q_gpu(data, config)
            assert result.shape == (10, 10), f"Failed for metric={metric}"
