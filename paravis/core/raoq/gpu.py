"""
GPU-accelerated Rao's Q computation via CuPy and custom CUDA kernel.

Provides GPU detection and optimized computation for NVIDIA GPUs.
"""
from typing import Callable, Optional, Tuple
import time

import numpy as np

from .models import RaoQConfig, RaoQResult

# ---------------------------------------------------------------------------
# GPU availability detection
# ---------------------------------------------------------------------------

GPU_AVAILABLE = False
GPU_BACKEND: Optional[str] = None
CUSTOM_KERNEL_AVAILABLE = False

try:
    import cupy as cp

    GPU_AVAILABLE = True
    GPU_BACKEND = "CuPy"

    # Try compiling the custom CUDA kernel
    try:
        raoq_kernel_code = """
        // ---- Shared-memory block kernel (for small n_bands) --------------
        // Uses one CUDA block per window. Emulates the CPU species-abundance
        // approach: identifies unique spectral profiles ("species"), counts
        // their frequencies, computes pairwise distances between species,
        // and computes Q = (2 / n²) * Σ_{i<j} d_ij * count_i * count_j.
        // Shared memory needed: 4*(2+3*n_pixels) + 4*n_pixels*n_bands bytes.
        //
        // metric_id: 0=euclidean 1=manhattan 2=chebyshev 3=minkowski
        //            4=canberra  5=braycurtis
        // -------------------------------------------------------------------

        extern "C" __global__
        void compute_raoq_kernel(
            const float* __restrict__ windows,
            const bool* __restrict__ valid_masks,
            int n_windows,
            int n_pixels,
            int n_bands,
            float* __restrict__ results,
            float na_tolerance,
            int metric_id,
            float p_param
        ) {
            int window_idx = blockIdx.x;
            if (window_idx >= n_windows) return;

            int tid = threadIdx.x;
            int base = window_idx * n_pixels;

            // Shared memory layout  (dynamic, sized at launch):
            //   s_int[0]                = valid_count
            //   s_int[1]                = n_species
            //   s_int[2..n_pixels+1]    = valid_indices
            //   s_int[n_pixels+2..2*n_pixels+1] = species_rep
            //   s_int[2*n_pixels+2..3*n_pixels+1] = species_count
            //   s_float[3*n_pixels+2..] = pixel_data  (vc * n_bands floats)
            extern __shared__ int s_int[];
            int* s_valid_count = &s_int[0];
            int* s_n_species = &s_int[1];
            int* s_valid_indices = &s_int[2];
            int* s_species_rep = &s_int[2 + n_pixels];
            int* s_species_count = &s_int[2 + 2 * n_pixels];
            float* s_pixel_data = (float*)&s_int[2 + 3 * n_pixels];

            // ---- Pass 1 — thread 0 counts valid pixels -------------------
            if (tid == 0) {
                int vc = 0;
                for (int i = 0; i < n_pixels; i++) {
                    if (valid_masks[base + i]) {
                        s_valid_indices[vc++] = i;
                    }
                }
                *s_valid_count = vc;
            }
            __syncthreads();

            int vc = *s_valid_count;
            float valid_ratio = (float)vc / n_pixels;
            if (valid_ratio < (1.0f - na_tolerance) || vc < 2) {
                if (tid == 0) results[window_idx] = nanf("");
                return;
            }

            // ---- Pass 2 — all threads load pixel data into shared memory --
            int offset = tid;
            while (offset < vc) {
                int idx = s_valid_indices[offset];
                int src_base = window_idx * n_pixels * n_bands + idx * n_bands;
                for (int b = 0; b < n_bands; b++) {
                    s_pixel_data[offset * n_bands + b] = windows[src_base + b];
                }
                offset += blockDim.x;
            }
            __syncthreads();

            // ---- Pass 3 — thread 0 identifies unique species --------------
            if (tid == 0) {
                int ns = 0;
                for (int i = 0; i < vc; i++) {
                    int found = -1;
                    for (int s = 0; s < ns; s++) {
                        int rep = s_species_rep[s];
                        int match = 1;
                        for (int b = 0; b < n_bands; b++) {
                            if (s_pixel_data[i * n_bands + b] != s_pixel_data[rep * n_bands + b]) {
                                match = 0;
                                break;
                            }
                        }
                        if (match) {
                            found = s;
                            break;
                        }
                    }
                    if (found >= 0) {
                        s_species_count[found]++;
                    } else {
                        s_species_rep[ns] = i;
                        s_species_count[ns] = 1;
                        ns++;
                    }
                }
                *s_n_species = ns;
            }
            __syncthreads();

            int n_species = *s_n_species;
            if (n_species < 2) {
                if (tid == 0) results[window_idx] = 0.0f;
                return;
            }

            // ---- Pass 4 — all threads compute species-pair distances ------
            float partial = 0.0f;
            int i_per_thread = (n_species + blockDim.x - 1) / blockDim.x;
            int start_i = tid * i_per_thread;
            int end_i = min(start_i + i_per_thread, n_species);

            for (int i = start_i; i < end_i; i++) {
                int rep_i = s_species_rep[i];
                float ci = (float)s_species_count[i];
                for (int j = i + 1; j < n_species; j++) {
                    int rep_j = s_species_rep[j];
                    float cj = (float)s_species_count[j];
                    float dist = 0.0f;
                    #define A s_pixel_data[rep_i * n_bands + b]
                    #define B s_pixel_data[rep_j * n_bands + b]

                    if (metric_id == 0) {        // euclidean
                        for (int b = 0; b < n_bands; b++) {
                            float diff = A - B;
                            dist += diff * diff;
                        }
                        dist = sqrtf(dist);
                    } else if (metric_id == 1) {  // manhattan
                        for (int b = 0; b < n_bands; b++) {
                            dist += fabsf(A - B);
                        }
                    } else if (metric_id == 2) {  // chebyshev
                        float max_val = 0.0f;
                        for (int b = 0; b < n_bands; b++) {
                            max_val = fmaxf(max_val, fabsf(A - B));
                        }
                        dist = max_val;
                    } else if (metric_id == 3) {  // minkowski
                        for (int b = 0; b < n_bands; b++) {
                            dist += powf(fabsf(A - B), p_param);
                        }
                        dist = powf(dist, 1.0f / p_param);
                    } else if (metric_id == 4) {  // canberra
                        for (int b = 0; b < n_bands; b++) {
                            float a = A, bv = B;
                            float denom = fabsf(a) + fabsf(bv);
                            if (denom > 1e-15f) dist += fabsf(a - bv) / denom;
                        }
                    } else if (metric_id == 5) {  // braycurtis
                        float num = 0.0f, denom = 0.0f;
                        for (int b = 0; b < n_bands; b++) {
                            float a = A, bv = B;
                            num += fabsf(a - bv);
                            denom += fabsf(a) + fabsf(bv);
                        }
                        dist = (denom > 1e-15f) ? num / denom : 0.0f;
                    }

                    #undef A
                    #undef B
                    partial += dist * ci * cj;
                }
            }

            // ---- Warp-shuffle reduction (all threads participate) ----------
            for (int offset_s = 16; offset_s > 0; offset_s >>= 1) {
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset_s);
            }

            if ((tid & 31) == 0) {
                s_int[tid >> 5] = __float_as_int(partial);
            }
            __syncthreads();

            if (tid == 0) {
                float total = 0.0f;
                int n_warps = (blockDim.x + 31) / 32;
                for (int w = 0; w < n_warps; w++) {
                    total += __int_as_float(s_int[w]);
                }
                // Rao's Q = 2 * Σ_{i<j} d_ij * count_i * count_j / n²
                results[window_idx] = 2.0f * total / (float)(vc * vc);
            }
        }

        // ---- Per-thread fallback kernel (for high n_bands / large ws) -----
        // Uses one thread per window with on-the-fly validity checks.
        // Emulates the CPU species-abundance approach using global memory
        // reads (no shared memory needed).
        //
        // metric_id: 0=euclidean 1=manhattan 2=chebyshev 3=minkowski
        //            4=canberra  5=braycurtis
        // -------------------------------------------------------------------

        extern "C" __global__
        void compute_raoq_kernel_fallback(
            const float* __restrict__ windows,
            const bool* __restrict__ valid_masks,
            int n_windows,
            int n_pixels,
            int n_bands,
            float* __restrict__ results,
            float na_tolerance,
            int metric_id,
            float p_param
        ) {
            int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (window_idx >= n_windows) return;

            int base = window_idx * n_pixels;

            // Pass 1 — count valid pixels and build index list
            int valid_indices[225];  // max for 15×15 window
            int valid_count = 0;
            for (int i = 0; i < n_pixels; i++) {
                if (valid_masks[base + i]) {
                    valid_indices[valid_count++] = i;
                }
            }

            float valid_ratio = (float)valid_count / n_pixels;
            if (valid_ratio < (1.0f - na_tolerance) || valid_count < 2) {
                results[window_idx] = nanf("");
                return;
            }

            // Pass 2 — identify unique species by comparing profiles
            // species_rep[s] = pixel index (0..n_pixels-1) of species s representative
            // species_count[s] = number of pixels in species s
            int species_rep[225];
            int species_count[225];
            int n_species = 0;

            for (int vi = 0; vi < valid_count; vi++) {
                int i = valid_indices[vi];
                int off_i = window_idx * n_pixels * n_bands + i * n_bands;
                int found = -1;
                for (int s = 0; s < n_species; s++) {
                    int rep = species_rep[s];
                    int off_rep = window_idx * n_pixels * n_bands + rep * n_bands;
                    int match = 1;
                    for (int b = 0; b < n_bands; b++) {
                        if (windows[off_i + b] != windows[off_rep + b]) {
                            match = 0;
                            break;
                        }
                    }
                    if (match) {
                        found = s;
                        break;
                    }
                }
                if (found >= 0) {
                    species_count[found]++;
                } else {
                    species_rep[n_species] = i;
                    species_count[n_species] = 1;
                    n_species++;
                }
            }

            if (n_species < 2) {
                results[window_idx] = 0.0f;
                return;
            }

            // Pass 3 — pairwise distances between unique species
            float total = 0.0f;
            for (int i = 0; i < n_species; i++) {
                int rep_i = species_rep[i];
                float ci = (float)species_count[i];
                int off_i = window_idx * n_pixels * n_bands + rep_i * n_bands;
                for (int j = i + 1; j < n_species; j++) {
                    int rep_j = species_rep[j];
                    float cj = (float)species_count[j];
                    int off_j = window_idx * n_pixels * n_bands + rep_j * n_bands;
                    float dist = 0.0f;
                    #define A windows[off_i + b]
                    #define B windows[off_j + b]

                    if (metric_id == 0) {        // euclidean
                        for (int b = 0; b < n_bands; b++) {
                            float diff = A - B;
                            dist += diff * diff;
                        }
                        dist = sqrtf(dist);
                    } else if (metric_id == 1) {  // manhattan
                        for (int b = 0; b < n_bands; b++) {
                            dist += fabsf(A - B);
                        }
                    } else if (metric_id == 2) {  // chebyshev
                        float max_val = 0.0f;
                        for (int b = 0; b < n_bands; b++) {
                            max_val = fmaxf(max_val, fabsf(A - B));
                        }
                        dist = max_val;
                    } else if (metric_id == 3) {  // minkowski
                        for (int b = 0; b < n_bands; b++) {
                            dist += powf(fabsf(A - B), p_param);
                        }
                        dist = powf(dist, 1.0f / p_param);
                    } else if (metric_id == 4) {  // canberra
                        for (int b = 0; b < n_bands; b++) {
                            float a = A, bv = B;
                            float denom = fabsf(a) + fabsf(bv);
                            if (denom > 1e-15f) dist += fabsf(a - bv) / denom;
                        }
                    } else if (metric_id == 5) {  // braycurtis
                        float num = 0.0f, denom = 0.0f;
                        for (int b = 0; b < n_bands; b++) {
                            float a = A, bv = B;
                            num += fabsf(a - bv);
                            denom += fabsf(a) + fabsf(bv);
                        }
                        dist = (denom > 1e-15f) ? num / denom : 0.0f;
                    }

                    #undef A
                    #undef B
                    total += dist * ci * cj;
                }
            }
            // Rao's Q = 2 * Σ_{i<j} d_ij * count_i * count_j / n²
            results[window_idx] = 2.0f * total / (float)(valid_count * valid_count);
        }
        """
        # Compile both kernels
        cp.RawKernel(raoq_kernel_code, "compute_raoq_kernel")
        cp.RawKernel(raoq_kernel_code, "compute_raoq_kernel_fallback")
        CUSTOM_KERNEL_AVAILABLE = True
    except Exception:
        CUSTOM_KERNEL_AVAILABLE = False

except ImportError:
    pass


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def get_gpu_info() -> dict:
    """Get GPU device information.

    Returns
    -------
    dict
        Keys: 'name', 'total_gb', 'free_gb', 'compute_capability'
    """
    if not GPU_AVAILABLE:
        return {"name": None, "total_gb": 0, "free_gb": 0, "compute_capability": None}

    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = device.attributes
        total = device.mem_info[0] / (1024 ** 3)
        free = device.mem_info[1] / (1024 ** 3)
        name = device.properties.get("name", b"Unknown").decode()
        cc = f"{device.properties.get('major', 0)}.{device.properties.get('minor', 0)}"
        return {"name": name, "total_gb": total, "free_gb": free, "compute_capability": cc}
    except Exception:
        return {"name": "Unknown", "total_gb": 0, "free_gb": 0, "compute_capability": None}


# Map metric name → integer ID for the CUDA kernels
METRIC_IDS = {
    "euclidean": 0,
    "manhattan": 1,
    "chebyshev": 2,
    "minkowski": 3,
    "canberra": 4,
    "braycurtis": 5,
}


def compute_rao_q_gpu(
    raster_data: np.ndarray,
    config: Optional[RaoQConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute Rao's Q on GPU using CuPy.

    Supports all distance metrics: euclidean, manhattan, chebyshev,
    minkowski, canberra, braycurtis.

    Falls back to CPU implementation if no GPU is available.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    config : RaoQConfig, optional
        Configuration.
    progress_callback : Callable[[int, int], None], optional
        Called with (current, total) after each row is processed.

    Returns
    -------
    np.ndarray
        2D array of Rao's Q values.
    """
    if config is None:
        config = RaoQConfig()

    # Fall back to CPU if no GPU available
    if not GPU_AVAILABLE:
        from .engine import compute_rao_q
        return compute_rao_q(raster_data, config, progress_callback=progress_callback)

    metric_id = METRIC_IDS.get(config.distance_metric, 0)
    p_param = np.float32(config.p_minkowski)

    import cupy as cp

    # ---- Clean slate: free GPU memory from any previous run ---------------
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    n_bands, height, width = raster_data.shape
    ws = config.window_size
    half = ws // 2
    n_pixels = ws * ws

    # Pad
    padded = np.pad(
        raster_data,
        ((0, 0), (half, half), (half, half)),
        mode="constant",
        constant_values=np.nan,
    )

    total_windows = height * width
    result = np.full((height, width), np.nan, dtype=np.float32)

    # Truncate input data to simplify precision (no rounding)
    fac = 10 ** config.simplify
    padded = np.trunc(padded * fac) / fac

    # NOTE: CuPy RawKernel has a bug where Python float scalars are
    # received as 0.0 on the GPU. Always use np.float32 for scalar args.
    na_tol = np.float32(config.na_tolerance)

    if CUSTOM_KERNEL_AVAILABLE:
        # ---- Select kernel based on shared-memory requirements ------------
        # Shared-memory kernel is faster but needs room for all pixel data.
        # For many bands / large windows we fall back to a per-thread kernel.
        smem_needed = 4 * (2 + 3 * n_pixels) + 4 * n_pixels * n_bands
        use_shared_mem_kernel = smem_needed <= 101376  # GPU max opt-in (99 KB)

        if use_shared_mem_kernel:
            kernel = cp.RawKernel(raoq_kernel_code, "compute_raoq_kernel")
            block_size = 256
            shared_mem = smem_needed
            # Opt in to >48 KB shared memory if needed
            if smem_needed > kernel.max_dynamic_shared_size_bytes:
                kernel.max_dynamic_shared_size_bytes = smem_needed
        else:
            kernel = cp.RawKernel(raoq_kernel_code, "compute_raoq_kernel_fallback")
            block_size = 256
            shared_mem = 0  # no shared memory

    # ---- Batched row processing -------------------------------------------
    # Process multiple rows per GPU launch to reduce launch overhead,
    # improve GPU utilisation, and reduce Python-side loop overhead.
    # gpu_batch_size is the target number of windows per batch — convert
    # to rows so that each batch has roughly that many windows.
    # CRITICAL: Each batch transfers ONLY the strip it needs to the GPU,
    # preventing OOM on very large rasters (e.g. 7+ GB full array).
    batch_rows = max(1, config.gpu_batch_size // width)
    batch_rows = min(batch_rows, height)  # clamp to image height

    try:
        for batch_start in range(0, height, batch_rows):
            batch_end = min(batch_start + batch_rows, height)
            n_rows = batch_end - batch_start

            # ---- Transfer ONLY the strip needed for this batch to GPU --------
            # We need rows [batch_start, batch_end + ws - 1] because the
            # sliding window at the last row of the batch needs (ws-1)/2
            # pixels of padding below it.
            d_slice = cp.asarray(
                padded[:, batch_start:batch_end + ws - 1, :],
                dtype=cp.float32,
            )

            # ---- Extract ALL windows on GPU for this batch --------------------
            n_total = n_rows * width

            d_5d = cp.lib.stride_tricks.sliding_window_view(
                d_slice, (ws, ws), axis=(1, 2)
            )
            # Reshape to (n_windows, n_pixels * n_bands) — this creates a
            # contiguous copy; d_slice and d_5d can be freed afterward.
            d_windows_flat = d_5d.transpose(1, 2, 3, 4, 0).reshape(
                n_total, n_pixels * n_bands
            )
            # Free d_slice and d_5d — they're no longer needed and holding
            # GPU memory that can be reused for the next batch.
            d_slice = None
            d_5d = None

            # Per-pixel NaN mask & zero-out — all on GPU
            d_valid = ~cp.any(
                cp.isnan(d_windows_flat.reshape(n_total, n_pixels, n_bands)),
                axis=2,
            )
            d_windows = cp.nan_to_num(d_windows_flat, nan=0.0)

            # Allocate output buffer
            d_batch_results = cp.zeros(n_total, dtype=cp.float32)

            # ---- Launch kernel ------------------------------------------------
            if CUSTOM_KERNEL_AVAILABLE:
                if use_shared_mem_kernel:
                    # One block per window
                    grid = (n_total,)
                else:
                    # One thread per window
                    grid = ((n_total + block_size - 1) // block_size,)
                kernel(
                    grid, (block_size,),
                    (d_windows, d_valid, n_total, n_pixels, n_bands,
                     d_batch_results, na_tol, metric_id, p_param),
                    shared_mem=shared_mem,
                )
            else:
                # Pure CuPy fallback — species-abundance per window (matches CPU)
                d_w3d = d_windows.reshape(n_total, n_pixels, n_bands)
                d_v2d = d_valid.reshape(n_total, n_pixels)
                for i in range(n_total):
                    valid_idx = cp.where(d_v2d[i])[0]
                    nv = len(valid_idx)
                    valid_ratio = nv / n_pixels
                    if nv < 2 or valid_ratio < (1.0 - config.na_tolerance):
                        d_batch_results[i] = cp.nan
                        continue
                    sub = d_w3d[i][valid_idx]

                    # Find unique spectral profiles (species) and frequencies
                    unique_profiles, counts = cp.unique(sub, axis=0, return_counts=True)
                    n_species = len(unique_profiles)
                    if n_species < 2:
                        d_batch_results[i] = 0.0
                        continue

                    # Relative abundances
                    p = counts.astype(cp.float64) / nv

                    # Full pairwise distance matrix between unique profiles
                    a = unique_profiles[:, None, :]   # (n_species, 1, n_bands)
                    b = unique_profiles[None, :, :]   # (1, n_species, n_bands)
                    diffs = a - b                     # (n_species, n_species, n_bands)

                    if metric_id == 0:  # euclidean
                        dists = cp.sqrt(cp.sum(diffs ** 2, axis=-1))
                    elif metric_id == 1:  # manhattan
                        dists = cp.sum(cp.abs(diffs), axis=-1)
                    elif metric_id == 2:  # chebyshev
                        dists = cp.max(cp.abs(diffs), axis=-1)
                    elif metric_id == 3:  # minkowski
                        dists = cp.sum(cp.abs(diffs) ** p_param, axis=-1) ** (1.0 / p_param)
                    elif metric_id == 4:  # canberra
                        denom = cp.abs(a) + cp.abs(b)
                        denom = cp.where(denom < 1e-15, 1.0, denom)
                        dists = cp.sum(cp.abs(diffs) / denom, axis=-1)
                    elif metric_id == 5:  # braycurtis
                        num = cp.sum(cp.abs(diffs), axis=-1)
                        denom = cp.sum(cp.abs(a) + cp.abs(b), axis=-1)
                        dists = cp.where(denom < 1e-15, 0.0, num / denom)
                    else:
                        dists = cp.sqrt(cp.sum(diffs ** 2, axis=-1))

                    # Q = Σᵢ Σⱼ dᵢⱼ × pᵢ × pⱼ
                    p_outer = p[:, None] * p[None, :]  # (n_species, n_species)
                    d_batch_results[i] = cp.sum(dists * p_outer)

            # ---- Copy back ----------------------------------------------------
            batch_result = cp.asnumpy(d_batch_results).reshape(n_rows, width)
            result[batch_start:batch_end, :] = batch_result

            if progress_callback is not None:
                progress_callback(batch_end * width, total_windows)

        return result
    finally:
        # Drop all GPU array references so CuPy's pool can free the blocks
        d_windows_flat = None
        d_windows = None
        d_valid = None
        d_batch_results = None
        d_slice = None
        d_5d = None
        d_w3d = None
        d_v2d = None
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
