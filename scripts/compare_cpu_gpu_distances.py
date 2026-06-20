#!/usr/bin/env python3
"""Compare CPU vs GPU Rao's Q results for all distance metrics.

Generates random multi-spectral data, computes Rao's Q with each
distance metric on both CPU and GPU, and checks numerical agreement.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from paravis.core.raoq.models import RaoQConfig
from paravis.core.raoq.engine import compute_rao_q
from paravis.core.raoq.gpu import compute_rao_q_gpu, GPU_AVAILABLE, GPU_BACKEND

# ── Test data ────────────────────────────────────────────────────────────
# Small 3-band image with some repeated spectral profiles (to exercise
# the species-abundance deduplication).
rng = np.random.default_rng(42)
N_BANDS, H, W = 3, 20, 20

# Mix of random and repeated profiles
data = rng.normal(0.5, 0.2, (N_BANDS, H, W)).astype(np.float32)
# Add a few constant patches (repeated profiles)
data[:, 0:5, 0:5] = 0.3   # all same
data[:, 5:10, 5:10] = 0.7  # all same

METRICS = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",      # uses p=2 by default → should match euclidean
    "canberra",
    "braycurtis",
]

# Also test Minkowski with p=1 (should match manhattan) and p=3
EXTRA_CONFIGS = [
    ("minkowski_p1", RaoQConfig(distance_metric="minkowski", p_minkowski=1)),
    ("minkowski_p3", RaoQConfig(distance_metric="minkowski", p_minkowski=3)),
]


def main():
    print(f"GPU available: {GPU_AVAILABLE}  ({GPU_BACKEND})")
    print(f"Data shape: {data.shape} ({N_BANDS} bands, {H}×{W})")
    print()

    all_ok = True

    # ── Standard metrics ─────────────────────────────────────────────────
    for metric in METRICS:
        config = RaoQConfig(window_size=5, distance_metric=metric, na_tolerance=0.3)
        cpu_result = compute_rao_q(data, config)
        gpu_result = compute_rao_q_gpu(data, config)

        match = np.allclose(cpu_result, gpu_result, atol=1e-5, equal_nan=True)
        max_diff = np.nanmax(np.abs(cpu_result - gpu_result)) if cpu_result.size else 0.0
        status = "✓" if match else "✗"
        print(f"  {status} {metric:20s}  CPU≈GPU: {match:5}  max|Δ|={max_diff:.3e}")

        if not match:
            all_ok = False

    # ── Extra Minkowski configs ──────────────────────────────────────────
    for name, config in EXTRA_CONFIGS:
        cpu_result = compute_rao_q(data, config)
        gpu_result = compute_rao_q_gpu(data, config)

        match = np.allclose(cpu_result, gpu_result, atol=1e-5, equal_nan=True)
        max_diff = np.nanmax(np.abs(cpu_result - gpu_result)) if cpu_result.size else 0.0
        status = "✓" if match else "✗"
        print(f"  {status} {name:20s}  CPU≈GPU: {match:5}  max|Δ|={max_diff:.3e}")

        if not match:
            all_ok = False

    print()
    if all_ok:
        print("✅ ALL metrics match between CPU and GPU!")
    else:
        print("❌ Some metrics differ between CPU and GPU — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
