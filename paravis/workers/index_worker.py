"""
Background workers for spectral index computation.

- IndexComputeWorker: Compute indices from in-memory numpy arrays (for API/tests).
- IndicesBatchWorker: Batch-process raster files with progress signals (for GUI).
"""
import os
import math
import queue
import concurrent.futures
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from PySide6.QtCore import QThread, Signal

import spyndex

from paravis.core.indices import (
    compute_index as _compute_index,
    compute_indices as _compute_indices,
    is_index_computable,
    get_default_constants,
)
from paravis.core.raster import write_geotiff
from .base_worker import BaseWorker


@dataclass
class FileResult:
    """Structured result returned by _process_single_file."""
    file_path: str
    success: bool
    log_messages: List[str] = field(default_factory=list)
    saved_files: List[str] = field(default_factory=list)
    error: str = ""


class IndexComputeWorker(BaseWorker):
    """Worker thread for computing spectral indices.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array (n_bands, height, width).
    band_mapping : Dict[int, str]
        Band number to spectral code mapping.
    index_names : List[str]
        Names of indices to compute.
    constants : Dict[str, float], optional
        Constant overrides.
    """

    def __init__(
        self,
        raster_data: np.ndarray,
        band_mapping: Dict[int, str],
        index_names: List[str],
        constants: Optional[Dict[str, float]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.raster_data = raster_data
        self.band_mapping = band_mapping
        self.index_names = index_names
        self.constants = constants or {}

    def run(self):
        """Execute index computation."""
        try:
            results: Dict[str, np.ndarray] = {}
            total = len(self.index_names)
            for i, name in enumerate(self.index_names):
                if not self.is_running:
                    self.error.emit("Cancelled by user")
                    return

                result = _compute_indices(
                    self.raster_data,
                    self.band_mapping,
                    [name],
                    self.constants,
                )
                if name in result:
                    results[name] = result[name]

                self.emit_progress(int((i + 1) / total * 100))

            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class IndicesBatchWorker(QThread):
    """Worker thread for batch index computation on raster files.

    Submits **every (file, index) pair** as an independent task to a
    thread pool — both multiple files and multiple indices are processed
    in parallel (up to ``max_workers``).  Each task:

    * opens one source file,
    * iterates over tiles (``tile_size`` × ``tile_size``),
    * reads, masks, scales, and computes **one index** per tile,
    * writes the result to its dedicated output GeoTIFF.

    Memory usage is bounded by ``tile_size² × n_bands`` per task.

    **Progress** is reported **per tile** — the bar updates smoothly as
    each individual tile finishes, regardless of which file or index it
    belongs to.

    **Output** GeoTIFFs use LZW compression with float32 prediction,
    internal tiling, and sparse-OK mode to minimise file size.

    Parameters
    ----------
    file_paths : list of str
        Paths to raster files to process.
    config : JobConfig (duck-typed)
        An object with the following attributes:
            indices : list of str
                Names of indices to compute.
            out_root : str
                Output directory.
            constants_override : dict
                Constant overrides for index formulas.
            band_mapping : dict
                Band-number to spectral-code mapping.
            max_workers : int
                Max number of parallel tasks (files × indices).
            scale_denom : float
                Divide raw pixel values by this number (reflectance scaling).
            tile_size : int
                Tile dimension (height & width) for block processing.
            date_format : str or None
                Optional regex to extract a date from the filename.
    """

    log_signal = Signal(str)
    progress_signal = Signal(int, int)  # current, total
    finished_signal = Signal(bool, str)  # success, message

    def __init__(
        self,
        file_paths: List[str],
        config,
        parent=None,
    ):
        super().__init__(parent)
        self.file_paths = file_paths
        self.config = config
        self._is_running = True
        # Thread-safe counter + queue for tile-level progress
        self._tiles_done: int = 0
        self._tiles_lock = Lock()
        self._progress_queue: "queue.SimpleQueue[int]" = queue.SimpleQueue()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def log(self, message: str):
        """Emit a log message via the log_signal."""
        self.log_signal.emit(message)

    def stop(self):
        """Request the worker to stop."""
        self._is_running = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Build all (file, index) tasks and process them in parallel."""
        n_workers = max(1, getattr(self.config, "max_workers", 1))
        tile_size = max(64, getattr(self.config, "tile_size", 512))
        total_files = len(self.file_paths)

        self.log(
            f"Processing {total_files} file(s), "
            f"{len(self.config.indices)} index(es) "
            f"({n_workers} worker(s), tile={tile_size}×{tile_size})..."
        )

        # --------------------------------------------------------------
        # 1. Gather metadata for every file, create output GeoTIFFs
        # --------------------------------------------------------------
        tasks: List[dict] = []          # kwargs for _process_one_file_index
        all_logs: List[str] = []
        add_log = all_logs.append
        grand_total_tiles = 0

        for fp in self.file_paths:
            base = os.path.splitext(os.path.basename(fp))[0]

            try:
                src = rasterio.open(fp)
            except Exception as exc:
                add_log(f"\n📂 {os.path.basename(fp)} — SKIP: {exc}")
                continue

            h, w = src.height, src.width
            nodata_val = src.nodata
            n_bands = src.count
            profile = src.profile

            # ---- Optimised output profile ----
            profile.update(
                dtype=rasterio.float32,
                count=1,
                nodata=np.nan,
                compress='lzw',
                predictor=3,                 # floating-point prediction
                tiled=True,
                blockxsize=tile_size,
                blockysize=tile_size,
                BIGTIFF='IF_NEEDED',
                sparse_ok=True,
            )

            add_log(
                f"\n📂 {os.path.basename(fp)} — "
                f"{w}×{h}, {n_bands} band(s), nodata={nodata_val}"
            )

            # Check each index for computability and log per-index warnings
            default_consts = get_default_constants()
            merged_consts = {**default_consts, **self.config.constants_override}
            available_bands = set(self.config.band_mapping.values())

            for idx_name in self.config.indices:
                if idx_name not in spyndex.indices:
                    add_log(f"   ⚠ '{idx_name}': unknown index — skipping")
                    continue
                idx = spyndex.indices[idx_name]
                required_bands = getattr(idx, "bands", [])
                required_consts = getattr(idx, "constants", [])

                missing_bands = [b for b in required_bands if b not in available_bands]
                missing_consts = []
                for c in required_consts:
                    if c in self.config.constants_override:
                        if self.config.constants_override[c] is not None:
                            continue
                    spyndex_c = spyndex.constants.get(c)
                    if spyndex_c is not None and spyndex_c.default is not None:
                        continue
                    missing_consts.append(c)

                if missing_bands:
                    add_log(f"   ⚠ '{idx_name}': missing band(s) {missing_bands}")
                if missing_consts:
                    add_log(f"   ⚠ '{idx_name}': missing constant(s) {missing_consts} (set them in constants editor)")

            # Build tile list
            windows: List[Window] = []
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    win_w = min(tile_size, w - x)
                    win_h = min(tile_size, h - y)
                    windows.append(Window(x, y, win_w, win_h))

            n_tiles = len(windows)

            # One task per (file, index) — skip indices that aren't computable
            tasks_before = len(tasks)
            for idx_name in self.config.indices:
                if not is_index_computable(
                    idx_name, self.config.constants_override,
                    self.config.band_mapping,
                ):
                    add_log(f"   ➖ '{idx_name}': not computable — skipped")
                    continue
                out_path = os.path.join(
                    self.config.out_root, f"{base}_{idx_name}.tif"
                )
                try:
                    with rasterio.open(out_path, "w", **profile) as dst:
                        pass
                except Exception as exc:
                    add_log(
                        f"   ❌ Cannot create {os.path.basename(out_path)}: {exc}"
                    )
                    continue
                tasks.append(dict(
                    src_path=fp,
                    idx_name=idx_name,
                    out_path=out_path,
                    windows=windows,
                    band_mapping=self.config.band_mapping,
                    constants_override=self.config.constants_override,
                    scale_denom=self.config.scale_denom,
                    src_nodata=nodata_val,
                    n_tiles=n_tiles,
                ))

            grand_total_tiles += n_tiles * (len(tasks) - tasks_before)
            src.close()

        # Emit gathered logs
        for msg in all_logs:
            self.log_signal.emit(msg)

        if not tasks:
            self.finished_signal.emit(False, "No valid files to process.")
            return

        self.log_signal.emit(
            f"   {len(tasks)} task(s), "
            f"{grand_total_tiles} tile(s) total — starting..."
        )
        self.log_signal.emit("")

        # --------------------------------------------------------------
        # 2. Submit all tasks to thread pool
        # --------------------------------------------------------------
        self._tiles_done = 0
        task_results: List["IndicesBatchWorker._TaskResult"] = []

        # Drain progress queue and emit signal
        def _flush_progress():
            drained = 0
            while not self._progress_queue.empty():
                try:
                    self._progress_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained:
                with self._tiles_lock:
                    self._tiles_done += drained
                    self.progress_signal.emit(
                        min(self._tiles_done, grand_total_tiles),
                        grand_total_tiles,
                    )

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as pool:

                future_map = {
                    pool.submit(
                        IndicesBatchWorker._process_one_file_index,
                        progress_queue=self._progress_queue,
                        **{k: v for k, v in task_kw.items()
                           if k != 'progress_queue'},
                    ): task_kw
                    for task_kw in tasks
                }

                # Use wait(FIRST_COMPLETED, timeout) so we can drain the
                # progress queue periodically (every ~200 ms).
                pending = set(future_map)
                while pending and self._is_running:
                    done_set, pending = concurrent.futures.wait(
                        pending, timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    _flush_progress()

                    for future in done_set:
                        task_kw = future_map[future]
                        try:
                            tres = future.result()
                            task_results.append(tres)
                        except Exception as exc:
                            task_results.append(
                                IndicesBatchWorker._TaskResult(
                                    file_path=task_kw["src_path"],
                                    idx_name=task_kw["idx_name"],
                                    tiles_ok=0,
                                    tiles_fail=task_kw["n_tiles"],
                                    error=str(exc),
                                )
                            )

                if not self._is_running:
                    for f in future_map:
                        f.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    self.finished_signal.emit(
                        False, "Processing stopped by user"
                    )
                    return

                # Final drain after all tasks complete
                _flush_progress()

        except Exception as exc:
            self.log_signal.emit(f"\n❌ ERROR: {exc}")
            self.finished_signal.emit(False, str(exc))
            return

        # --------------------------------------------------------------
        # 3. Summarise — per-task (file+index), not per-file
        # --------------------------------------------------------------
        self.log_signal.emit("")
        self.log_signal.emit("─── RESULTS ────────────────────────────────────────")

        tasks_ok = 0
        tasks_fail = 0
        failed_names: List[str] = []

        for tres in task_results:
            short = os.path.basename(tres.file_path)
            if tres.tiles_ok > 0:
                tasks_ok += 1
                self.log_signal.emit(
                    f"   ✅ '{tres.idx_name}' ({short}): "
                    f"{tres.tiles_ok} tile(s)"
                    + (f", {tres.tiles_fail} failed" if tres.tiles_fail else "")
                )
            else:
                tasks_fail += 1
                err_msg = tres.error or "Unknown error"
                failed_names.append(f"'{tres.idx_name}' ({short})")
                self.log_signal.emit(
                    f"   ❌ {failed_names[-1]}: {err_msg}"
                )

        self.log_signal.emit("")
        msg_parts = []
        if tasks_ok:
            msg_parts.append(f"{tasks_ok} task(s) OK")
        if tasks_fail:
            msg_parts.append(f"{tasks_fail} task(s) failed")
            msg_parts.append("Failed: " + ", ".join(failed_names))
        all_ok = tasks_fail == 0
        self.finished_signal.emit(
            all_ok, "; ".join(msg_parts) if msg_parts else "Done"
        )

    # ------------------------------------------------------------------
    # Single (file, index) task — runs in thread-pool thread
    # ------------------------------------------------------------------

    @dataclass
    class _TaskResult:
        """Result for one (file, index) task."""
        file_path: str
        idx_name: str
        tiles_ok: int
        tiles_fail: int
        error: str = ""

    @staticmethod
    def _process_one_file_index(
        src_path: str,
        idx_name: str,
        out_path: str,
        windows: List[Window],
        band_mapping: Dict[int, str],
        constants_override: Dict[str, float],
        scale_denom: float,
        src_nodata: Optional[float],
        n_tiles: int,  # pylint: disable=unused-argument
        progress_queue: "queue.SimpleQueue[int]" = None,
    ) -> "_TaskResult":
        """Compute one index for all tiles of one raster.

        After each tile the function pushes a token to ``progress_queue``
        so the main thread can emit tile-level progress.

        Runs in a thread-pool thread — MUST NOT emit Qt signals.
        """
        tiles_ok = 0
        tiles_fail = 0
        first_error: Optional[str] = None
        report = (progress_queue is not None)

        for window in windows:
            try:
                with rasterio.open(src_path) as src:
                    data = src.read(window=window).astype(np.float32)

                if src_nodata is not None and np.isfinite(src_nodata):
                    data[data == src_nodata] = np.nan

                if scale_denom > 1:
                    data = data / scale_denom

                result = _compute_index(
                    data, band_mapping, idx_name,
                    constants=constants_override,
                )

                if np.any(np.isfinite(result)):
                    with rasterio.open(out_path, "r+") as dst:
                        dst.write(
                            result[np.newaxis, :, :].astype(np.float32),
                            window=window,
                        )

                tiles_ok += 1

            except Exception as exc:
                if first_error is None:
                    first_error = str(exc).split('\n')[0].strip()
                tiles_fail += 1

            # --- Report per-tile progress ---
            if report:
                progress_queue.put(1)

        error_msg = ""
        if first_error:
            error_msg = (f"{first_error} "
                         f"({tiles_fail} tile(s) failed, {tiles_ok} ok)")
        elif tiles_ok == 0:
            error_msg = f"All {tiles_fail} tile(s) failed — no valid data"

        return IndicesBatchWorker._TaskResult(
            file_path=src_path,
            idx_name=idx_name,
            tiles_ok=tiles_ok,
            tiles_fail=tiles_fail,
            error=error_msg,
        )
