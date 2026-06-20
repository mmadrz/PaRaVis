"""
Test that CPU and GPU Rao's Q implementations produce identical results
under various random conditions and parameter combinations.

Also tests pure CuPy fallback (without CUDA kernel) vs CPU for equivalence.

Run with:  pytest tests/test_cpu_gpu_equivalence.py -v
"""
import sys
import os
import math
import random
import traceback
from pathlib import Path

import numpy as np
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paravis.core.raoq.models import RaoQConfig
from paravis.core.raoq.engine import compute_rao_q as cpu_compute

# GPU imports (will gracefully skip if unavailable)
try:
    from paravis.core.raoq.gpu import (
        compute_rao_q_gpu as gpu_compute,
        is_gpu_available,
        GPU_BACKEND,
        CUSTOM_KERNEL_AVAILABLE,
        GPU_AVAILABLE,
    )
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = is_gpu_available()
except ImportError:
    HAS_CUPY = False
    HAS_GPU = False
    GPU_BACKEND = None
    CUSTOM_KERNEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Random test generator
# ---------------------------------------------------------------------------

METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis"]


def generate_random_raster(height, width, n_bands, seed=None, nan_frac=0.0, value_scale=1.0):
    """Generate a random 3D raster array (n_bands, height, width)."""
    if seed is not None:
        np.random.seed(seed)
    data = np.random.randn(n_bands, height, width).astype(np.float32) * value_scale

    # Add some duplicate profiles (same value repeated) to test species grouping
    if n_bands >= 3 and height * width > 10:
        src_idx = np.random.randint(0, height * width, size=max(2, height * width // 20))
        dst_idx = np.random.randint(0, height * width, size=max(2, height * width // 10))
        h, w = data.shape[1:]
        for s in src_idx:
            si, sj = divmod(s, w)
            for d in dst_idx:
                di, dj = divmod(d, w)
                data[:, di, dj] = data[:, si, sj]

    # Add NaN values
    if nan_frac > 0:
        nan_mask = np.random.random((height, width)) < nan_frac
        data[:, nan_mask] = np.nan

    return data


def get_config_params():
    """Yield combinations of parameters to test."""
    # Small raster, few bands, various metrics
    for metric in METRICS:
        for ws in [3, 5, 7]:
            yield {
                "height": 20, "width": 25, "n_bands": 3,
                "window_size": ws, "metric": metric,
                "nan_frac": 0.0, "value_scale": 1.0,
            }

    # Medium raster with NaN
    for nan_frac in [0.05, 0.15]:
        for metric in ["euclidean", "braycurtis"]:
            yield {
                "height": 30, "width": 30, "n_bands": 4,
                "window_size": 5, "metric": metric,
                "nan_frac": nan_frac, "value_scale": 2.0,
            }

    # Test with duplicate profiles (species grouping)
    yield {
        "height": 15, "width": 20, "n_bands": 5,
        "window_size": 7, "metric": "euclidean",
        "nan_frac": 0.0, "value_scale": 1.0,
    }

    # Minkowski with different p values
    for p in [1.5, 3.0, 5.0]:
        yield {
            "height": 20, "width": 20, "n_bands": 3,
            "window_size": 5, "metric": "minkowski",
            "nan_frac": 0.0, "value_scale": 1.0,
            "extra": {"p_minkowski": p},
        }

    # Non-square raster
    yield {
        "height": 40, "width": 15, "n_bands": 3,
        "window_size": 5, "metric": "manhattan",
        "nan_frac": 0.1, "value_scale": 0.5,
    }

    # Lots of bands
    yield {
        "height": 16, "width": 16, "n_bands": 10,
        "window_size": 3, "metric": "euclidean",
        "nan_frac": 0.0, "value_scale": 100.0,
    }


# ===========================================================================
# Pytest-style test class
# ===========================================================================


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestCpuGpuEquivalence:
    """Parameterized CPU vs GPU equivalence tests using pytest."""

    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("window_size", [3, 5, 7])
    def test_various_metrics_and_windows(self, metric, window_size):
        """Compare CPU and GPU results for all metrics at various window sizes."""
        height, width, n_bands = 20, 25, 3
        seed = random.randint(0, 10000)

        config = RaoQConfig(
            window_size=window_size,
            distance_metric=metric,
            na_tolerance=0.5,
            simplify=0,
        )

        raster = generate_random_raster(
            height, width, n_bands,
            seed=seed, nan_frac=0.0, value_scale=1.0,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        # NaN positions must match
        assert np.array_equal(np.isnan(cpu_result), np.isnan(gpu_result)), \
            f"NaN positions differ for {metric}, ws={window_size}"

        # Values must be close (float32 precision)
        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True), \
            f"Values differ for {metric}, ws={window_size}"

    @pytest.mark.parametrize("nan_frac", [0.05, 0.15])
    def test_with_nan_values(self, nan_frac):
        """CPU and GPU must produce identical results with NaN pixels."""
        height, width, n_bands = 30, 30, 4
        seed = random.randint(0, 10000)

        config = RaoQConfig(
            window_size=5,
            distance_metric="euclidean",
            na_tolerance=0.5,
            simplify=0,
        )

        raster = generate_random_raster(
            height, width, n_bands,
            seed=seed, nan_frac=nan_frac, value_scale=2.0,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        assert np.array_equal(np.isnan(cpu_result), np.isnan(gpu_result))
        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)

    @pytest.mark.parametrize("p_val", [1.5, 3.0, 5.0])
    def test_minkowski_various_p(self, p_val):
        """Minkowski with different p exponents should match CPU."""
        height, width, n_bands = 20, 20, 3

        config = RaoQConfig(
            window_size=5,
            distance_metric="minkowski",
            p_minkowski=p_val,
            na_tolerance=0.5,
            simplify=0,
        )

        raster = generate_random_raster(
            height, width, n_bands, seed=42, nan_frac=0.0,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)

    def test_many_bands(self):
        """High number of bands should match."""
        height, width, n_bands = 16, 16, 10

        config = RaoQConfig(window_size=3, distance_metric="euclidean", simplify=0)
        raster = generate_random_raster(
            height, width, n_bands, seed=99, value_scale=100.0,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)

    def test_non_square_raster(self):
        """Non-square rasters should be handled correctly."""
        height, width, n_bands = 40, 15, 3

        config = RaoQConfig(
            window_size=5, distance_metric="manhattan",
            na_tolerance=0.5, simplify=0,
        )
        raster = generate_random_raster(
            height, width, n_bands, seed=777, nan_frac=0.1, value_scale=0.5,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)

    def test_duplicate_profiles(self):
        """Rasters with many duplicate spectral profiles (same species repeated)."""
        height, width, n_bands = 15, 20, 5

        config = RaoQConfig(window_size=7, distance_metric="euclidean", simplify=0)
        raster = generate_random_raster(
            height, width, n_bands, seed=333,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)

    def test_simplify_parameter(self):
        """With simplify=0, CPU and GPU should still match (both truncated)."""
        height, width, n_bands = 16, 16, 3

        config = RaoQConfig(window_size=3, distance_metric="euclidean", simplify=0)
        raster = generate_random_raster(
            height, width, n_bands, seed=555,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = gpu_compute(raster, config)

        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)


# ===========================================================================
# Pure CuPy fallback tests (no custom CUDA kernel)
# ===========================================================================


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestPureCuPyFallback:
    """Test the pure-CuPy fallback path (without custom CUDA kernel)."""

    def _run_pure_cupy(self, raster, config):
        """Run GPU computation but force pure-CuPy path."""
        from paravis.core.raoq import gpu as gpu_module
        orig_custom = gpu_module.CUSTOM_KERNEL_AVAILABLE
        orig_gpu = gpu_module.GPU_AVAILABLE
        try:
            gpu_module.CUSTOM_KERNEL_AVAILABLE = False
            gpu_module.GPU_AVAILABLE = True
            return gpu_compute(raster, config)
        finally:
            gpu_module.CUSTOM_KERNEL_AVAILABLE = orig_custom
            gpu_module.GPU_AVAILABLE = orig_gpu

    def test_pure_cupy_vs_cpu(self):
        """Pure CuPy fallback (without CUDA kernel) must match CPU."""
        height, width, n_bands = 15, 12, 3
        config = RaoQConfig(window_size=3, distance_metric="euclidean", simplify=0)
        raster = generate_random_raster(
            height, width, n_bands, seed=123,
        )

        cpu_result = cpu_compute(raster, config)
        gpu_result = self._run_pure_cupy(raster, config)

        assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)

    def test_pure_cupy_all_metrics(self):
        """Pure CuPy fallback with all distance metrics."""
        height, width, n_bands = 10, 12, 3
        for metric in METRICS:
            config = RaoQConfig(
                window_size=3, distance_metric=metric, simplify=0,
                p_minkowski=3 if metric == "minkowski" else 2,
            )
            raster = generate_random_raster(
                height, width, n_bands, seed=456,
            )

            cpu_result = cpu_compute(raster, config)
            gpu_result = self._run_pure_cupy(raster, config)

            assert np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True), \
                f"Pure CuPy failed for metric={metric}"


# ===========================================================================
# Legacy script entry point
# ===========================================================================


def compare():
    """Run CPU vs GPU comparisons over random test cases (original script)."""
    results = []
    passed = 0
    failed = 0
    skipped = 0

    print("=" * 70)
    print(f"  CPU vs GPU Rao's Q Equivalence Test")
    print(f"  GPU available: {HAS_GPU}")
    print(f"  GPU backend: {GPU_BACKEND}")
    print(f"  Custom CUDA kernel: {CUSTOM_KERNEL_AVAILABLE}")
    print("=" * 70)

    for idx, params in enumerate(get_config_params()):
        height = params["height"]
        width = params["width"]
        n_bands = params["n_bands"]
        ws = params["window_size"]
        metric = params["metric"]
        nan_frac = params["nan_frac"]
        value_scale = params["value_scale"]
        extra = params.get("extra", {})

        seed = random.randint(0, 10000) + idx

        config = RaoQConfig(
            window_size=ws,
            distance_metric=metric,
            na_tolerance=0.5,
            simplify=0,
            **extra,
        )

        raster = generate_random_raster(
            height, width, n_bands,
            seed=seed, nan_frac=nan_frac, value_scale=value_scale,
        )

        case_desc = (
            f"[{idx + 1:2d}]  "
            f"{height}×{width}×{n_bands}  "
            f"ws={ws}  {metric:12s}  "
            f"NaN={nan_frac:.0%}  "
            f"scale={value_scale:.0f}  "
            f"seed={seed}"
        )

        print(f"\n  {case_desc}")
        print(f"  {'─' * (len(case_desc) - 6)}")

        # CPU reference
        try:
            cpu_result = cpu_compute(raster, config)
            cpu_ok = True
        except Exception as e:
            cpu_result = None
            cpu_ok = False
            print(f"    ❌ CPU error: {e}")

        # GPU result
        gpu_ok = False
        gpu_result = None
        gpu_error = None

        if HAS_GPU:
            try:
                gpu_result = gpu_compute(raster, config)
                gpu_ok = True
            except Exception as e:
                gpu_error = e
                print(f"    ❌ GPU error: {e}")
        else:
            try:
                import cupy as cp
                from paravis.core.raoq import gpu as gpu_module
                orig_available = gpu_module.CUSTOM_KERNEL_AVAILABLE
                gpu_module.CUSTOM_KERNEL_AVAILABLE = False
                orig_gpu_available = gpu_module.GPU_AVAILABLE
                gpu_module.GPU_AVAILABLE = True
                try:
                    gpu_result = gpu_compute(raster, config)
                    gpu_ok = True
                finally:
                    gpu_module.CUSTOM_KERNEL_AVAILABLE = orig_available
                    gpu_module.GPU_AVAILABLE = orig_gpu_available
            except ImportError:
                print(f"    ⚠ Skipped — needs GPU or CuPy to run")
                skipped += 1
                results.append((idx, "SKIP"))
                continue

        if not cpu_ok or not gpu_ok:
            failed += 1
            results.append((idx, "FAIL"))
            continue

        # Compare
        cpu_nan = np.isnan(cpu_result)
        gpu_nan = np.isnan(gpu_result)
        nan_match = np.array_equal(cpu_nan, gpu_nan)

        if not nan_match:
            print(f"    ❌ NaN positions differ!")
            print(f"       CPU NaN count: {np.sum(cpu_nan)}, GPU NaN count: {np.sum(gpu_nan)}")
            failed += 1
            results.append((idx, "FAIL"))
            continue

        # Compare non-NaN values
        valid_mask = ~cpu_nan
        cpu_vals = cpu_result[valid_mask]
        gpu_vals = gpu_result[valid_mask]

        if len(cpu_vals) == 0:
            print(f"    ⚠ All NaN (all windows had insufficient valid pixels)")
            passed += 1
            results.append((idx, "PASS"))
            continue

        abs_diff = np.abs(cpu_vals - gpu_vals)
        rel_diff = abs_diff / (np.abs(cpu_vals) + 1e-15)

        max_abs = np.max(abs_diff)
        max_rel = np.max(rel_diff)
        mean_abs = np.mean(abs_diff)
        mean_rel = np.mean(rel_diff)

        atol = 1e-5
        rtol = 1e-4
        close_mask = abs_diff < atol
        close_mask |= rel_diff < rtol
        n_close = np.sum(close_mask)
        n_total = len(cpu_vals)

        if n_close == n_total:
            print(f"    ✅ PASS  (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")
            passed += 1
            results.append((idx, "PASS"))
        else:
            print(f"    ❌ FAIL  ({n_total - n_close}/{n_total} values outside tolerance)")
            print(f"       max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
            print(f"       mean_abs={mean_abs:.2e}, mean_rel={mean_rel:.2e}")
            bad = ~close_mask
            bad_indices = np.where(bad.ravel())[0][:5]
            for bi in bad_indices:
                print(f"       CPU={cpu_vals.ravel()[bi]:.6f}  GPU={gpu_vals.ravel()[bi]:.6f}  diff={abs_diff.ravel()[bi]:.2e}")
            failed += 1
            results.append((idx, "FAIL"))

    print()
    print("=" * 70)
    print(f"  SUMMARY")
    print(f"  Total tests: {passed + failed + skipped}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")
    print(f"  Skipped:     {skipped}")
    print("=" * 70)

    return failed == 0


def quick_test():
    """Run a single random test with detailed output."""
    np.random.seed(42)
    height, width, n_bands = 10, 12, 3
    ws = 5

    raster = generate_random_raster(height, width, n_bands, seed=123, nan_frac=0.1)

    config = RaoQConfig(window_size=ws, distance_metric="euclidean", na_tolerance=0.5)

    print(f"Raster shape: {raster.shape} ({height}×{width}×{n_bands})")
    print(f"Window size: {ws}")
    print(f"Distance metric: euclidean")
    print()

    cpu_result = cpu_compute(raster, config)
    print(f"CPU result ({cpu_result.shape}):")
    print(np.round(cpu_result, 4))
    print()

    if HAS_GPU:
        gpu_result = gpu_compute(raster, config)
        print(f"GPU result ({gpu_result.shape}):")
        print(np.round(gpu_result, 4))
        print()

        diff = np.abs(cpu_result - gpu_result)
        print(f"Max absolute difference: {np.max(diff):.2e}")
        print(f"Mean absolute difference: {np.mean(diff):.2e}")
        print(f"All close (1e-5): {np.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-4, equal_nan=True)}")
    else:
        print("(GPU not available on this machine)")


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        quick_test()
    else:
        success = compare()
        sys.exit(0 if success else 1)
