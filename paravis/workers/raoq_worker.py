"""
Background worker for Rao's Q computation — v1 faithful port.

Uses the v2 computation engine internally but exposes v1's exact
constructor signature, signals, and progress reporting semantics.
"""
import os
import gc
import traceback

import numpy as np

from PySide6.QtCore import QThread, Signal

from paravis.core.raoq.gpu import GPU_AVAILABLE, GPU_BACKEND, CUSTOM_KERNEL_AVAILABLE


class RaoQWorker(QThread):
    """Worker thread for Rao's Q diversity computation (v1 parity).

    Parameters
    ----------
    raster_paths : list of str
        Paths to input raster files (one per spectral band).
    output_path : str
        Path for the output raster file.
    distance_m : str
        Distance metric ('euclidean', 'manhattan', 'chebyshev', 'minkowski').
    window : int
        Moving window size (odd number).
    na_tolerance : float
        Maximum fraction of NA pixels allowed (0.0 to 1.0).
    block_size : int
        Number of windows per processing block.
    num_workers : int
        Number of CPU worker processes.
    p_minkowski : int
        Minkowski p parameter (only used if distance_m == 'minkowski').
    use_gpu : bool
        Whether to use GPU acceleration.
    simplify : int
        Number of decimal places to truncate to (0 = no truncation — full float32 precision).
    """

    log_signal = Signal(str)
    progress_signal = Signal(int, int)  # current, total
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, raster_paths, output_path, distance_m, window, na_tolerance,
                 block_size, num_workers, p_minkowski, use_gpu=False, simplify=2):
        super().__init__()
        self.raster_paths = raster_paths
        self.output_path = output_path
        self.distance_m = distance_m
        self.window = window
        self.na_tolerance = na_tolerance
        self.block_size = block_size
        self.num_workers = num_workers
        self.p_minkowski = p_minkowski
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.simplify = simplify
        self.is_running = True

    def log(self, message):
        self.log_signal.emit(message)

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self._compute_rao_q()
        except RuntimeError as e:
            if "Cancelled" in str(e):
                self.log("\n⏹️ Processing stopped by user")
                self.finished_signal.emit(False, "Processing stopped by user")
                return
            raise
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            self.log(traceback.format_exc())
            self.finished_signal.emit(False, str(e))
            return

        if not self.is_running:
            self.finished_signal.emit(False, "Processing stopped by user")

    def _compute_rao_q(self):
        """Compute Rao's Q using v2 engine with row-by-row progress."""
        self.log(f"\n{'='*60}")
        if self.use_gpu:
            self.log(f"🚀 STARTING RAO'S Q CALCULATION (GPU MODE)")
            self.log(f"🎮 Backend: {GPU_BACKEND}")
            if GPU_BACKEND == "CuPy" and CUSTOM_KERNEL_AVAILABLE:
                self.log(f"⚡ Custom CUDA kernel available")
        else:
            self.log(f"🚀 STARTING RAO'S Q CALCULATION (CPU MODE)")
            self.log(f"💻 Using {self.num_workers} CPU cores")
        self.log(f"{'='*60}")
        self.log(f"Input files: {len(self.raster_paths)}")
        self.log(f"Output: {self.output_path}")
        self.log(f"Distance: {self.distance_m}")
        self.log(f"Window: {self.window}")
        self.log(f"NA Tolerance: {self.na_tolerance*100:.0f}%")
        self.log(f"Simplify: {self.simplify} decimals")
        self.log(f"Block size: {self.block_size} windows")

        # Read rasters
        self.log("📂 Reading raster data...")
        try:
            import rasterio
            datasets = [rasterio.open(p) for p in self.raster_paths]
            profile = datasets[0].profile
            height = datasets[0].height
            width = datasets[0].width
            n_bands = len(datasets)
            self.log(f"   Raster: {width}×{height}, {n_bands} bands")

            # Read all bands into a 3D array (bands, height, width)
            # and convert nodata → NaN for every band
            data = np.zeros((n_bands, height, width), dtype=np.float32)
            for i, ds in enumerate(datasets):
                band_data = ds.read(1).astype(np.float32)
                nodata = ds.nodata
                if nodata is not None and np.isfinite(nodata):
                    nodata_count = int(np.isclose(band_data, nodata).sum())
                    if nodata_count:
                        self.log(f"   Band {i+1}: nodata={nodata}, masking {nodata_count} pixel(s)")
                        band_data[np.isclose(band_data, nodata)] = np.nan
                data[i] = band_data
                ds.close()

            # Log final NaN statistics
            total_nan = np.isnan(data).sum()
            total_pixels = height * width * n_bands
            nan_pct = 100.0 * total_nan / total_pixels
            self.log(f"   Total NaN: {total_nan:,} / {total_pixels:,} ({nan_pct:.1f}%)")
        except Exception as e:
            self.log(f"❌ Failed to read rasters: {e}")
            raise

        total_windows = height * width
        self.log(f"   Total windows: {total_windows:,}")
        self.log("")

        # Build config
        from paravis.core.raoq.models import RaoQConfig

        config = RaoQConfig(
            window_size=self.window,
            step_size=1,
            na_tolerance=self.na_tolerance,
            n_workers=self.num_workers,
            tile_size=1024,
            use_gpu=self.use_gpu,
            gpu_batch_size=50000,
            cpu_batch_size=self.block_size,
            distance_metric=self.distance_m,
            p_minkowski=self.p_minkowski,
            simplify=self.simplify,
        )

        if not self.is_running:
            return

        # Process using the appropriate engine
        try:
            if self.use_gpu:
                self.log("⚡ Launching GPU computation...")
                from paravis.core.raoq.gpu import compute_rao_q_gpu
                result = compute_rao_q_gpu(
                    data, config,
                    progress_callback=lambda cur, tot:
                        self.progress_signal.emit(min(cur, total_windows), total_windows),
                )
                self.progress_signal.emit(total_windows, total_windows)
            else:
                if self.num_workers > 1:
                    self.log(f"⚡ Launching CPU parallel computation ({self.num_workers} workers)...")
                    from paravis.core.raoq.engine import compute_rao_q_parallel
                    config.n_workers = self.num_workers
                    result = compute_rao_q_parallel(
                        data, config,
                        progress_callback=lambda cur, tot:
                            self.progress_signal.emit(min(cur, total_windows), total_windows),
                    )
                    self.progress_signal.emit(total_windows, total_windows)
                else:
                    self.log("⚡ Launching CPU computation (single-threaded)...")
                    from paravis.core.raoq.engine import compute_rao_q

                    # Use compute_rao_q directly — it's already vectorised
                    # with sliding_window_view per row
                    def _progress(cur, tot):
                        self.progress_signal.emit(min(cur, total_windows), total_windows)
                        if not self.is_running:
                            raise RuntimeError("Cancelled")

                    result = compute_rao_q(data, config, progress_callback=_progress)

                self.progress_signal.emit(total_windows, total_windows)

            if not self.is_running:
                return

            # Write output
            self.log("\n💾 Writing output raster...")
            profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            with rasterio.open(self.output_path, 'w', **profile) as dst:
                dst.write(result.astype(np.float32), 1)

            self.log(f"✅ Output saved to: {self.output_path}")
            self.finished_signal.emit(True, f"Rao's Q computation completed! Processed {total_windows:,} windows.")

        except Exception as e:
            self.log(f"\n❌ Computation error: {e}")
            self.log(traceback.format_exc())
            self.finished_signal.emit(False, str(e))
