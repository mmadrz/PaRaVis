"""
Unit tests for paravis.core.raoq — no Qt dependency required.

Run with:  pytest tests/test_core_raoq.py -v
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from paravis.core.raoq import compute_rao_q, compute_rao_q_parallel
from paravis.core.raoq.models import RaoQConfig, RaoQResult

# Check if CuPy is available for GPU tests
try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


class TestRaoQConfig:
    def test_defaults(self):
        cfg = RaoQConfig()
        assert cfg.window_size == 15
        assert cfg.step_size == 1
        assert cfg.na_tolerance == 0.3

    def test_custom_values(self):
        cfg = RaoQConfig(window_size=5, step_size=2, na_tolerance=0.1,
                        n_workers=4, distance_metric='manhattan',
                        p_minkowski=3, simplify=1, block_size=512,
                        gpu_batch_size=100000)
        assert cfg.window_size == 5
        assert cfg.distance_metric == 'manhattan'
        assert cfg.p_minkowski == 3
        assert cfg.simplify == 1
        assert cfg.block_size == 512
        assert cfg.gpu_batch_size == 100000

    def test_mutable(self):
        cfg = RaoQConfig()
        cfg.window_size = 7
        assert cfg.window_size == 7


class TestRaoQResult:
    def test_create(self):
        import numpy as np
        data = np.random.rand(10, 10).astype(np.float32)
        result = RaoQResult(data=data, window_size=15, step_size=1,
                           computation_time=1.5, n_windows=100)
        assert result.data.shape == (10, 10)
        assert result.computation_time == 1.5
        assert result.n_windows == 100
        assert result.window_size == 15

    def test_defaults(self):
        import numpy as np
        data = np.random.rand(10, 10).astype(np.float32)
        result = RaoQResult(data=data, window_size=15, step_size=1)
        assert result.computation_time == 0.0
        assert result.n_windows == 0
        assert result.window_size == 15
        assert result.step_size == 1


class TestComputeRaoQ:
    def test_output_shape(self):
        data = np.random.rand(1, 20, 20).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        assert result.shape == (20, 20)
        assert result.dtype == np.float32

    def test_constant_input(self):
        """Constant input should give Rao's Q = 0 (or close)."""
        data = np.ones((1, 10, 10), dtype=np.float32) * 0.5
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        # Border pixels are NaN (partial window); center should be 0
        center = result[1:-1, 1:-1]
        assert np.allclose(center, 0.0, atol=1e-6)

    def test_two_distinct_values(self):
        """Two distinct values half-and-half should give non-zero."""
        data = np.zeros((1, 10, 10), dtype=np.float32)
        data[:, :5, :] = 0.0
        data[:, 5:, :] = 1.0
        result = compute_rao_q(data, RaoQConfig(window_size=5))
        # Some windows will be mixed → non-zero Rao's Q
        assert np.nanmax(result) > 0

    def test_species_abundance_formula(self):
        """Verify the full double-sum species-abundance formula.

        For a window with known species, manually compute:
            Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ
        and compare against the engine output.
        """
        from paravis.core.raoq.engine import _compute_rao_q_window
        # Create a window: 3 species of 3 bands, 9 pixels total
        # Species A (4 pixels): [0.0, 0.0, 0.0]
        # Species B (3 pixels): [1.0, 0.0, 0.0]
        # Species C (2 pixels): [0.0, 1.0, 0.0]
        window = np.zeros((9, 3), dtype=np.float32)
        window[:4] = [0.0, 0.0, 0.0]    # species A
        window[4:7] = [1.0, 0.0, 0.0]   # species B
        window[7:9] = [0.0, 1.0, 0.0]   # species C
        valid = np.ones(9, dtype=bool)

        q = _compute_rao_q_window(window, valid, na_tolerance=0.3,
                                  distance_metric="euclidean")

        # Manual computation:
        # n = 9, p_A = 4/9, p_B = 3/9, p_C = 2/9
        # d(A,B) = sqrt((1-0)^2 + 0 + 0) = 1.0
        # d(A,C) = sqrt(0 + (1-0)^2 + 0) = 1.0
        # d(B,C) = sqrt(1^2 + 1^2 + 0) = sqrt(2) ≈ 1.41421356
        # Q = d(A,B)*p_A*p_B + d(B,A)*p_B*p_A + d(A,C)*p_A*p_C + d(C,A)*p_C*p_A
        #   + d(B,C)*p_B*p_C + d(C,B)*p_C*p_B
        #   = 2*(1.0*4/9*3/9 + 1.0*4/9*2/9 + sqrt(2)*3/9*2/9)
        p_a, p_b, p_c = 4/9, 3/9, 2/9
        d_ab, d_ac, d_bc = 1.0, 1.0, np.sqrt(2)
        expected = (d_ab * p_a * p_b + d_ab * p_b * p_a +
                    d_ac * p_a * p_c + d_ac * p_c * p_a +
                    d_bc * p_b * p_c + d_bc * p_c * p_b)

        assert np.isclose(q, expected, atol=1e-6), f"Expected {expected}, got {q}"

    def test_progress_callback(self):
        """Test progress callback is called."""
        data = np.random.rand(1, 10, 10).astype(np.float32)
        progress_values = []

        def progress_callback(current, total):
            progress_values.append((current, total))

        result = compute_rao_q(data, RaoQConfig(window_size=3),
                              progress_callback=progress_callback)
        assert len(progress_values) > 0
        assert progress_values[-1][0] == progress_values[-1][1]  # 100% at end

    def test_multi_band_input(self):
        """Test with multiple bands."""
        data = np.random.rand(3, 10, 10).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        assert result.shape == (10, 10)

    def test_simplify_rounding(self):
        """Test simplify parameter affects precision."""
        data = np.random.rand(1, 8, 8).astype(np.float32)
        result_no_simplify = compute_rao_q(data, RaoQConfig(window_size=3, simplify=5))
        result_simplified = compute_rao_q(data, RaoQConfig(window_size=3, simplify=0))
        assert result_no_simplify.shape == result_simplified.shape


class TestRaoQParallel:
    def test_matches_serial(self):
        data = np.random.rand(1, 16, 16).astype(np.float32)
        cfg = RaoQConfig(window_size=3, n_workers=2)
        serial = compute_rao_q(data, cfg)
        parallel = compute_rao_q_parallel(data, cfg)
        assert np.allclose(serial, parallel, equal_nan=True)

    def test_default_config(self):
        """Test compute_rao_q with config=None to cover default branch."""
        data = np.random.rand(1, 10, 10).astype(np.float32)
        result = compute_rao_q(data, None)
        assert result.shape == (10, 10)

    def test_parallel_default_config(self):
        """Test compute_rao_q_parallel with config=None."""
        data = np.random.rand(1, 10, 10).astype(np.float32)
        result = compute_rao_q_parallel(data, None)
        assert result.shape == (10, 10)

    def test_parallel_with_progress(self):
        """Test parallel with progress callback."""
        data = np.random.rand(1, 12, 12).astype(np.float32)
        progress_values = []

        def callback(cur, tot):
            progress_values.append((cur, tot))

        result = compute_rao_q_parallel(data, RaoQConfig(window_size=3, n_workers=2),
                                        progress_callback=callback)
        assert result.shape == (12, 12)
        assert len(progress_values) > 0


class TestRaoQGpu:
    def test_gpu_available(self):
        """is_gpu_available should return a bool."""
        from paravis.core.raoq.gpu import is_gpu_available
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_gpu_info(self):
        """get_gpu_info should return a dict without crashing."""
        from paravis.core.raoq.gpu import get_gpu_info
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "total_gb" in info

    def test_compute_rao_q_gpu(self):
        """Test GPU compute (may fall back to CPU if no GPU)."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        from paravis.core.raoq.models import RaoQConfig
        data = np.random.rand(1, 10, 10).astype(np.float32)
        result = compute_rao_q_gpu(data, RaoQConfig(window_size=3))
        assert result.shape == (10, 10)

    def test_compute_rao_q_gpu_default_config(self):
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        data = np.random.rand(1, 10, 10).astype(np.float32)
        result = compute_rao_q_gpu(data)
        assert result.shape == (10, 10)

    def test_compute_rao_q_gpu_with_progress(self):
        """GPU compute with progress callback."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        data = np.random.rand(1, 10, 10).astype(np.float32)
        progress_values = []

        def callback(cur, tot):
            progress_values.append((cur, tot))

        result = compute_rao_q_gpu(data, RaoQConfig(window_size=3),
                                   progress_callback=callback)
        assert result.shape == (10, 10)
        assert len(progress_values) > 0


class TestRaoQWindow:
    """Direct tests for the internal _compute_rao_q_window function."""

    def test_all_valid_pixels(self):
        """All pixels valid should produce a finite value."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 3
        window_data = np.random.rand(n_pixels, n_bands).astype(np.float32)
        valid_mask = np.ones(n_pixels, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, na_tolerance=0.3)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_all_nan_pixels(self):
        """All NaN pixels should return NaN."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 3
        window_data = np.full((n_pixels, n_bands), np.nan)
        valid_mask = np.zeros(n_pixels, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, na_tolerance=0.3)
        assert np.isnan(result)

    def test_mostly_nan_above_tolerance(self):
        """More NaN than tolerance should return NaN."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 1
        window_data = np.full((n_pixels, n_bands), np.nan)
        window_data[0, 0] = 1.0
        window_data[1, 0] = 2.0  # only 2 valid out of 9 → 22% < 70% → return NaN
        valid_mask = ~np.isnan(window_data.flatten())
        result = _compute_rao_q_window(window_data, valid_mask, na_tolerance=0.3)
        assert np.isnan(result)

    def test_less_than_two_valid(self):
        """Fewer than 2 valid pixels should return NaN."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 1
        window_data = np.full((n_pixels, n_bands), np.nan)
        window_data[0, 0] = 1.0  # only 1 valid pixel
        valid_mask = ~np.isnan(window_data.flatten())
        result = _compute_rao_q_window(window_data, valid_mask, na_tolerance=0.9)
        assert np.isnan(result)

    def test_constant_pixels(self):
        """All pixels have same value → Rao's Q should be 0."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 3
        window_data = np.ones((n_pixels, n_bands), dtype=np.float32) * 0.5
        valid_mask = np.ones(n_pixels, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, na_tolerance=0.3)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_two_distinct_values(self):
        """Two clusters should give positive Rao's Q."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 1
        window_data = np.zeros((n_pixels, n_bands), dtype=np.float32)
        window_data[:5, 0] = 0.0  # 5 pixels = 0
        window_data[5:, 0] = 1.0  # 4 pixels = 1
        valid_mask = np.ones(n_pixels, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, na_tolerance=0.3)
        assert result > 0.0

    def test_three_species_full_double_sum(self):
        """Verify the full double-sum formula with 3 species and known distances.

        Creates a window with 3 unique profiles (species), manually computes
        Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ using all 6 distance metrics, and compares
        against the engine output.
        """
        from paravis.core.raoq.engine import _compute_rao_q_window

        # 3 species, 3 bands, 10 pixels total
        # Species A (5 pixels): [0.0, 0.0, 0.0]
        # Species B (3 pixels): [2.0, 0.0, 0.0]
        # Species C (2 pixels): [0.0, 3.0, 0.0]
        n_pixels = 10
        window = np.zeros((n_pixels, 3), dtype=np.float32)
        window[:5] = [0.0, 0.0, 0.0]    # species A (5)
        window[5:8] = [2.0, 0.0, 0.0]   # species B (3)
        window[8:10] = [0.0, 3.0, 0.0]  # species C (2)
        valid = np.ones(n_pixels, dtype=bool)

        p_a, p_b, p_c = 5/10, 3/10, 2/10

        # Pre-compute inter-species distances for all 6 metrics
        metrics_data = {
            "euclidean": {
                # d(A,B) = sqrt(4+0+0) = 2, d(A,C) = sqrt(0+9+0) = 3, d(B,C) = sqrt(4+9+0) = sqrt(13)
                "dists": [2.0, 3.0, np.sqrt(13)],
            },
            "manhattan": {
                # |2-0|+|0-0|+|0-0| = 2, |0-0|+|3-0|+|0-0| = 3, |2-0|+|0-3|+|0-0| = 5
                "dists": [2.0, 3.0, 5.0],
            },
            "chebyshev": {
                # max(|2|,0,0) = 2, max(0,|3|,0) = 3, max(|2|,|3|,0) = 3
                "dists": [2.0, 3.0, 3.0],
            },
            "minkowski": {
                # p=3: (|2|^3)^(1/3)=2, (|3|^3)^(1/3)=3, (|2|^3+|3|^3)^(1/3) = (8+27)^(1/3) = 35^(1/3)
                "dists": [2.0, 3.0, 35**(1/3)],
                "p": 3,
            },
            "canberra": {
                # d(A,B) = |2|/(0+2) = 1, d(A,C) = |3|/(0+3) = 1
                # d(B,C) = |2-0|/(2+0) + |0-3|/(0+3) = 2/2 + 3/3 = 2
                "dists": [1.0, 1.0, 2.0],
            },
            "braycurtis": {
                # d(A,B) = 2/(2+0+0) = 1, d(A,C) = 3/(0+3+0) = 1
                # d(B,C) = (|2-0|+|0-3|+|0-0|)/(2+0+0+3+0+0) = 5/5 = 1
                "dists": [1.0, 1.0, 1.0],
            },
        }

        for metric, data in metrics_data.items():
            d_ab, d_ac, d_bc = data["dists"]
            extra = {"p_minkowski": data.get("p", 2)}
            q = _compute_rao_q_window(
                window, valid, na_tolerance=0.3,
                distance_metric=metric, **extra
            )
            # Expected: Q = 2*(d_AB*p_A*p_B + d_AC*p_A*p_C + d_BC*p_B*p_C)
            expected = 2 * (d_ab * p_a * p_b + d_ac * p_a * p_c + d_bc * p_b * p_c)
            assert np.isclose(q, expected, atol=1e-5), \
                f"{metric}: expected {expected:.6f}, got {q:.6f}"

    def test_pairwise_distances_all_metrics(self):
        """Test _compute_pairwise_distances produces correct outputs for all 6 metrics."""
        from paravis.core.raoq.engine import _compute_pairwise_distances

        # 3 profiles, 3 bands
        profiles = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ], dtype=np.float32)

        # Expected upper-triangular distances (0,1), (0,2), (1,2)
        test_cases = [
            ("euclidean", {}, [2.0, 3.0, np.sqrt(13)]),
            ("manhattan", {}, [2.0, 3.0, 5.0]),
            ("chebyshev", {}, [2.0, 3.0, 3.0]),
            ("minkowski", {"p": 3}, [2.0, 3.0, 35**(1/3)]),
            ("canberra", {}, [1.0, 1.0, 2.0]),
            ("braycurtis", {}, [1.0, 1.0, 1.0]),
        ]

        for metric, kwargs, expected in test_cases:
            dists = _compute_pairwise_distances(profiles, metric=metric, **kwargs)
            assert np.allclose(dists, expected, atol=1e-6), \
                f"{metric}: expected {expected}, got {dists}"


class TestRaoQEdgeCases:
    """Edge cases for compute_rao_q."""

    def test_all_nan_input(self):
        """All-NaN input should produce all-NaN result."""
        data = np.full((1, 10, 10), np.nan, dtype=np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        assert np.all(np.isnan(result))

    def test_single_row(self):
        """Single row raster should work."""
        data = np.random.rand(1, 1, 20).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        assert result.shape == (1, 20)

    def test_single_column(self):
        """Single column raster should work."""
        data = np.random.rand(1, 20, 1).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        assert result.shape == (20, 1)

    def test_large_simplify(self):
        """Large simplify value should still work (just more precision)."""
        data = np.random.rand(1, 8, 8).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=3, simplify=10))
        assert result.shape == (8, 8)
        assert not np.all(np.isnan(result))

    def test_simplify_zero(self):
        """simplify=0 means truncate to integers."""
        data = np.random.rand(1, 8, 8).astype(np.float32)
        data[0, 0, 0] = 1.234567
        result_high = compute_rao_q(data, RaoQConfig(window_size=3, simplify=6))
        result_low = compute_rao_q(data, RaoQConfig(window_size=3, simplify=0))
        assert result_high.shape == result_low.shape == (8, 8)

    def test_window_size_same_as_image(self):
        """Window size same as image should work (single window)."""
        data = np.random.rand(1, 5, 5).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=5))
        assert result.shape == (5, 5)

    def test_window_size_larger_than_image(self):
        """Window size larger than image should still run (all padded)."""
        data = np.random.rand(1, 3, 3).astype(np.float32)
        result = compute_rao_q(data, RaoQConfig(window_size=5))
        assert result.shape == (3, 3)
        assert np.all(np.isnan(result))  # All padded


class TestProcessChunkDirect:
    """Direct tests for the _process_chunk helper function."""

    def test_process_chunk_basic(self):
        """Call _process_chunk directly with valid args."""
        import paravis.core.raoq.engine as _eng
        data = np.random.rand(1, 10, 10).astype(np.float32)
        cfg = RaoQConfig(window_size=3)
        row_range = [2, 3, 4]
        _eng._shared_raster_data = data
        args = (cfg, row_range)
        rows, chunk = _eng._process_chunk(args)
        _eng._shared_raster_data = None
        assert len(rows) == 3
        assert chunk.shape == (3, 10)

    def test_process_chunk_all_nan(self):
        """_process_chunk with all-NaN data."""
        import paravis.core.raoq.engine as _eng
        data = np.full((1, 10, 10), np.nan, dtype=np.float32)
        cfg = RaoQConfig(window_size=3)
        row_range = [0, 1]
        _eng._shared_raster_data = data
        args = (cfg, row_range)
        rows, chunk = _eng._process_chunk(args)
        _eng._shared_raster_data = None
        assert np.all(np.isnan(chunk))


class TestRaoQGpuEdgeCases:
    """Advanced GPU module edge cases that require mocking."""

    def test_compute_rao_q_gpu_no_gpu_fallback(self):
        """When GPU is not available, should fall back to CPU."""
        from paravis.core.raoq import gpu as gpu_module
        with patch.object(gpu_module, 'GPU_AVAILABLE', False):
            data = np.random.rand(1, 10, 10).astype(np.float32)
            result = gpu_module.compute_rao_q_gpu(data, RaoQConfig(window_size=3))
            assert result.shape == (10, 10)

    def test_get_gpu_info_no_gpu(self):
        """get_gpu_info should return empty info when no GPU."""
        from paravis.core.raoq import gpu as gpu_module
        with patch.object(gpu_module, 'GPU_AVAILABLE', False):
            info = gpu_module.get_gpu_info()
            assert info == {"name": None, "total_gb": 0, "free_gb": 0, "compute_capability": None}

    def test_cpu_fallback_no_config(self):
        """CPU fallback with config=None should work."""
        from paravis.core.raoq import gpu as gpu_module
        with patch.object(gpu_module, 'GPU_AVAILABLE', False):
            data = np.random.rand(1, 10, 10).astype(np.float32)
            result = gpu_module.compute_rao_q_gpu(data, None)
            assert result.shape == (10, 10)

    def test_compute_rao_q_gpu_all_nan(self):
        """GPU compute with all-NaN input should produce all-NaN result."""
        from paravis.core.raoq.gpu import compute_rao_q_gpu
        data = np.full((1, 10, 10), np.nan, dtype=np.float32)
        result = compute_rao_q_gpu(data, RaoQConfig(window_size=3))
        assert np.all(np.isnan(result))

    @pytest.mark.skipif(not HAVE_CUPY, reason="CuPy not installed")
    def test_get_gpu_info_with_exception(self):
        """get_gpu_info should handle exceptions during GPU query."""
        from paravis.core.raoq import gpu as gpu_module
        with patch.object(gpu_module, 'GPU_AVAILABLE', True):
            with patch('cupy.cuda.Device', side_effect=Exception("No device")):
                info = gpu_module.get_gpu_info()
                assert info == {"name": "Unknown", "total_gb": 0, "free_gb": 0,
                               "compute_capability": None}

    @pytest.mark.skipif(not HAVE_CUPY, reason="CuPy not installed")
    def test_get_gpu_info_success(self):
        """get_gpu_info with GPU available should return device info."""
        from paravis.core.raoq import gpu as gpu_module
        with patch.object(gpu_module, 'GPU_AVAILABLE', True):
            mock_device = MagicMock()
            mock_device.mem_info = [8 * 1024**3, 4 * 1024**3]  # 8GB total, 4GB free
            mock_device.properties = {
                "name": b"Test GPU",
                "major": 8,
                "minor": 6,
            }
            mock_device.attributes = {}
            with patch('cupy.cuda.Device', return_value=mock_device):
                info = gpu_module.get_gpu_info()
                assert info["name"] == "Test GPU"
                assert info["total_gb"] == 8.0
                assert info["free_gb"] == 4.0
                assert info["compute_capability"] == "8.6"


class TestRaoQSpecies:
    """Tests for the species-abundance approach to Rao's Q.

    The correct ecological formula is:
        Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ

    where unique spectral profiles = "species" and pᵢ = countᵢ / total.
    """

    # ── _compute_rao_q_window (unit) ──────────────────────────────────

    def test_single_species_returns_zero(self):
        """A single unique profile → no diversity → Q = 0."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        n_pixels = 9
        n_bands = 2
        window_data = np.ones((n_pixels, n_bands), dtype=np.float32) * 0.5
        valid_mask = np.ones(n_pixels, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, 0.3)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_two_species_equal_abundance(self):
        """Two unique profiles with equal counts in a 1-band window."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # 8 pixels: 4 of value 0.0, 4 of value 1.0
        window_data = np.array([[0.0], [0.0], [0.0], [0.0],
                                [1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        valid_mask = np.ones(8, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, 0.3)
        # d = |0-1| = 1.0, p₀ = p₁ = 0.5
        # Q = 2 × 1.0 × 0.5 × 0.5 = 0.5
        assert np.isclose(result, 0.5, atol=1e-6)

    def test_three_species_uneven_abundance(self):
        """Three unique profiles with different frequencies."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # 7 pixels total:
        #   value 0.0 appears 4 times  → p₀ = 4/7
        #   value 1.0 appears 2 times  → p₁ = 2/7
        #   value 2.0 appears 1 time   → p₂ = 1/7
        values = [0.0] * 4 + [1.0] * 2 + [2.0] * 1
        window_data = np.array(values, dtype=np.float32).reshape(-1, 1)
        valid_mask = np.ones(7, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, 0.3)
        # d₀₁ = 1, d₀₂ = 2, d₁₂ = 1
        # Q = 2 × [d₀₁×p₀×p₁ + d₀₂×p₀×p₂ + d₁₂×p₁×p₂]
        #   = 2 × [1×(4/7)(2/7) + 2×(4/7)(1/7) + 1×(2/7)(1/7)]
        #   = 2 × (8/49 + 8/49 + 2/49) = 2 × 18/49 = 36/49
        p0, p1, p2 = 4 / 7, 2 / 7, 1 / 7
        expected = 2 * (1 * p0 * p1 + 2 * p0 * p2 + 1 * p1 * p2)
        assert np.isclose(result, expected, atol=1e-6)

    def test_multi_band_species(self):
        """Species defined across multiple spectral bands."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # 4 pixels × 2 bands — 3 unique profiles
        window_data = np.array([[0, 0], [0, 0],    # species A × 2
                                [1, 1],              # species B × 1
                                [2, 2]], dtype=np.float32)  # species C × 1
        valid_mask = np.ones(4, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, 0.3)
        # d_A_B = sqrt(1² + 1²) = sqrt(2)
        # d_A_C = sqrt(2² + 2²) = sqrt(8) = 2*sqrt(2)
        # d_B_C = sqrt(1² + 1²) = sqrt(2)
        # p_A = 2/4 = 0.5, p_B = 0.25, p_C = 0.25
        # Q = 2 × [√2×0.5×0.25 + 2√2×0.5×0.25 + √2×0.25×0.25]
        sqrt2 = np.sqrt(2)
        pA, pB, pC = 0.5, 0.25, 0.25
        expected = 2 * (sqrt2 * pA * pB + 2 * sqrt2 * pA * pC + sqrt2 * pB * pC)
        assert np.isclose(result, expected, atol=1e-6)

    def test_all_pixels_unique_equals_equal_weight(self):
        """When every pixel is a unique species, Q matches the equal-weight formula.

        Equal-weight: Q = 2/n² × Σ_{i<j} dᵢⱼ
        Species-based with all unique: pᵢ = 1/n for each of n species
        → Q = 2 × Σ_{i<j} dᵢⱼ × (1/n) × (1/n) = 2/n² × Σ_{i<j} dᵢⱼ  ✓
        """
        from paravis.core.raoq.engine import _compute_rao_q_window
        rng = np.random.default_rng(42)
        n_pixels = 8
        n_bands = 2
        window_data = rng.random((n_pixels, n_bands)).astype(np.float32)
        valid_mask = np.ones(n_pixels, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, 0.3)

        # Compute equal-weight formula directly
        total_dist = 0.0
        for i in range(n_pixels):
            diffs = window_data[i + 1:] - window_data[i]
            total_dist += np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
        expected = 2.0 * total_dist / (n_pixels * n_pixels)

        assert np.isclose(result, expected, atol=1e-6)

    def test_identical_and_unique_mixed(self):
        """Mix of duplicate and unique profiles."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # 6 pixels: 3 of value 1, 3 of value 2
        window_data = np.array([[1], [1], [1],
                                [2], [2], [2]], dtype=np.float32)
        valid_mask = np.ones(6, dtype=bool)
        result = _compute_rao_q_window(window_data, valid_mask, 0.3)
        # Two species, equal abundance → Q = 0.5
        assert np.isclose(result, 0.5, atol=1e-6)

    # ── Integration via compute_rao_q ─────────────────────────────────

    def test_integration_two_clusters(self):
        """compute_rao_q with two distinct halves should give positive Q."""
        data = np.zeros((1, 10, 10), dtype=np.float32)
        data[:, :5, :] = 0.0
        data[:, 5:, :] = 1.0
        result = compute_rao_q(data, RaoQConfig(window_size=5))
        assert np.nanmax(result) > 0

    def test_integration_all_same(self):
        """Homogeneous input → Q ≈ 0 everywhere."""
        data = np.ones((1, 10, 10), dtype=np.float32) * 0.5
        result = compute_rao_q(data, RaoQConfig(window_size=3))
        center = result[1:-1, 1:-1]
        assert np.allclose(center, 0.0, atol=1e-6)

    # ── Distance metrics ────────────────────────────────────────────────

    def test_manhattan_distance(self):
        """Manhattan distance gives different values than Euclidean."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # Two species at values 0 and 2 in 1 band
        # Euclidean: d = 2, Manhattan: d = 2 → same for 1D
        window_data = np.array([[0.0], [2.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result_eucl = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="euclidean"
        )
        result_man = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="manhattan"
        )
        assert np.isclose(result_eucl, result_man, atol=1e-6)

        # In 2D, Euclidean and Manhattan differ:
        window_data = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result_eucl = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="euclidean"
        )
        result_man = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="manhattan"
        )
        # Euclidean: sqrt(3²+4²) = 5, Manhattan: 3+4 = 7
        # Both have p₁=p₂=0.5, so Q = 2 × d × 0.5 × 0.5 = d × 0.5
        assert np.isclose(result_eucl, 2.5, atol=1e-6)   # 5 × 0.5
        assert np.isclose(result_man, 3.5, atol=1e-6)    # 7 × 0.5

    def test_chebyshev_distance(self):
        """Chebyshev distance = max(|Δb|) across bands."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        window_data = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="chebyshev"
        )
        # Chebyshev: max(3, 4) = 4, Q = 2 × 4 × 0.5 × 0.5 = 2.0
        assert np.isclose(result, 2.0, atol=1e-6)

    def test_minkowski_p1_equals_manhattan(self):
        """Minkowski with p=1 is Manhattan distance."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        window_data = np.array([[0.0, 0.0, 0.0],
                                [1.0, 2.0, 3.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result_mink = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="minkowski", p_minkowski=1
        )
        result_man = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="manhattan"
        )
        assert np.isclose(result_mink, result_man, atol=1e-6)

    def test_minkowski_p2_equals_euclidean(self):
        """Minkowski with p=2 is Euclidean distance."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        window_data = np.array([[0.0, 0.0, 0.0],
                                [1.0, 2.0, 3.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result_mink = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="minkowski", p_minkowski=2
        )
        result_eucl = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="euclidean"
        )
        assert np.isclose(result_mink, result_eucl, atol=1e-6)

    # ── New distance metrics ────────────────────────────────────────────

    def test_canberra_distance(self):
        """Canberra distance is a weighted L1, different from Manhattan."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # Two profiles: (1,1) vs (2,2)
        # Canberra per band: |1-2|/(1+2) + |1-2|/(1+2) = 1/3 + 1/3 = 2/3
        # Manhattan per band: |1-2| + |1-2| = 2
        window_data = np.array([[1.0, 1.0],
                                [2.0, 2.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result_can = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="canberra"
        )
        result_man = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="manhattan"
        )
        assert np.isclose(result_can, 2.0 / 3.0 * 0.5, atol=1e-6)  # Q = d × 0.5
        assert not np.isclose(result_can, result_man, atol=1e-6)  # different from Manhattan

    def test_braycurtis_distance(self):
        """Bray-Curtis is bounded [0, 1], ecological dissimilarity."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        # Two identical profiles → dist=0 → Q=0
        window_data = np.array([[5.0, 5.0],
                                [5.0, 5.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="braycurtis"
        )
        assert np.isclose(result, 0.0, atol=1e-6)

        # Two profiles with no overlap → dist=1
        window_data = np.array([[5.0, 0.0],
                                [0.0, 5.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="braycurtis"
        )
        # num = |5-0|+|0-5| = 10, denom = (5+0)+(0+5) = 10, d = 1
        # Q = 2 × 1 × 0.5 × 0.5 = 0.5
        assert np.isclose(result, 0.5, atol=1e-6)

        # Partial overlap
        window_data = np.array([[4.0, 1.0],
                                [1.0, 4.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="braycurtis"
        )
        # num = |4-1|+|1-4| = 6, denom = (4+1)+(1+4) = 10, d = 0.6
        # Q = 2 × 0.6 × 0.5 × 0.5 = 0.3
        assert np.isclose(result, 0.3, atol=1e-6)

    def test_new_metrics_differ_from_euclidean(self):
        """All new metrics produce values different from Euclidean for same data."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        window_data = np.array([[1.0, 0.0],
                                [0.0, 1.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        result_eucl = _compute_rao_q_window(
            window_data, valid_mask, 0.3, distance_metric="euclidean"
        )
        for metric in ("canberra", "braycurtis"):
            result = _compute_rao_q_window(
                window_data, valid_mask, 0.3, distance_metric=metric
            )
            assert not np.isclose(result, result_eucl, atol=1e-6), \
                f"{metric} should differ from Euclidean"

    def test_invalid_metric_raises(self):
        """Unknown metric should raise ValueError."""
        from paravis.core.raoq.engine import _compute_rao_q_window
        window_data = np.array([[0.0], [1.0]], dtype=np.float32)
        valid_mask = np.ones(2, dtype=bool)
        with pytest.raises(ValueError, match="Unknown distance metric"):
            _compute_rao_q_window(
                window_data, valid_mask, 0.3, distance_metric="this_metric_does_not_exist"
            )
