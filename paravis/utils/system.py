"""
System profiler — detect CPU, RAM, and GPU hardware.

Pure Python; Qt-free hardware detection.
"""
import math
import os
import platform
from multiprocessing import cpu_count
from typing import Dict, Optional


class SystemProfiler:
    """Detect and report system hardware information."""

    @staticmethod
    def get_cpu_info() -> Dict[str, object]:
        """Get CPU information.

        Returns
        -------
        dict with keys: 'physical_cores', 'logical_cores', 'architecture'
        """
        logical = cpu_count()
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            arch = platform.machine()
        except ImportError:
            physical = logical
            arch = platform.machine()

        return {
            "physical_cores": physical or logical,
            "logical_cores": logical,
            "architecture": arch,
        }

    @staticmethod
    def get_ram_info() -> Dict[str, object]:
        """Get RAM information.

        Returns
        -------
        dict with keys: 'total_gb', 'available_gb', 'percent_used'
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "percent_used": mem.percent,
            }
        except ImportError:
            return {"total_gb": 0, "available_gb": 0, "percent_used": 0}

    @staticmethod
    def get_gpu_info() -> Dict[str, object]:
        """Get GPU information.

        Returns
        -------
        dict with keys: 'available', 'backend', 'name', 'total_gb', 'free_gb',
                         'cuda_version', 'custom_kernel'
        """
        from paravis.core.raoq.gpu import GPU_AVAILABLE, GPU_BACKEND, CUSTOM_KERNEL_AVAILABLE

        info: Dict[str, object] = {
            "available": GPU_AVAILABLE,
            "backend": GPU_BACKEND,
            "name": None,
            "total_gb": 0,
            "free_gb": 0,
            "cuda_version": None,
            "custom_kernel": CUSTOM_KERNEL_AVAILABLE,
        }

        if GPU_BACKEND == "CuPy":
            try:
                import cupy as cp
                device = cp.cuda.Device()
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                info["total_gb"] = total_mem / (1024 ** 3)
                info["free_gb"] = free_mem / (1024 ** 3)
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                info["name"] = props.get("name", b"Unknown").decode()
                try:
                    info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
                except Exception:
                    pass
            except Exception:
                pass
        elif GPU_BACKEND == "Numba CUDA":
            try:
                from numba import cuda
                gpu = cuda.get_current_device()
                info["total_gb"] = gpu.total_memory / (1024 ** 3)
                info["free_gb"] = info["total_gb"] * 0.85
                info["name"] = gpu.name
            except Exception:
                pass

        return info

    @classmethod
    def get_auto_config(cls, n_bands_guess=3, window_guess=5) -> Dict:
        """Recommend optimal settings based on detected hardware.

        Parameters
        ----------
        n_bands_guess : int
            Expected number of raster bands.
        window_guess : int
            Expected window size for Rao's Q.

        Returns
        -------
        dict with recommended values for all configurable parameters.
        """
        gpu = cls.get_gpu_info()
        cpu = cls.get_cpu_info()
        ram = cls.get_ram_info()

        logical = cpu.get("logical_cores", 4)
        gpu_workers = max(1, logical - 2)
        cpu_workers = max(1, logical - 1 if logical > 1 else 1)

        # Block size: target ~40% of available RAM
        bytes_per_block = n_bands_guess * window_guess * window_guess * 4
        avail_gb = ram.get("available_gb", 4)
        target_bytes = avail_gb * 0.40 * (1024 ** 3)
        max_windows = max(1, int(target_bytes / max(1, bytes_per_block)))
        block = int(math.sqrt(max_windows))
        block = max(256, min(block, 10000))
        block = ((block + 63) // 128) * 128  # round to multiple of 128
        block = max(128, min(block, 10000))

        # GPU batch estimate
        est_batch = 100000
        if gpu.get("available") and gpu.get("total_gb", 0) > 0:
            free_gb = gpu.get("total_gb", 8) * 0.85
            bytes_per_win = window_guess * window_guess * n_bands_guess * 4
            est_batch = int(free_gb * (1024 ** 3) / max(1, bytes_per_win))
            est_batch = min(est_batch, 50_000_000)

        return {
            "gpu": gpu,
            "cpu": cpu,
            "ram": ram,
            "recommended": {
                "use_gpu": gpu.get("available", False),
                "gpu_workers": gpu_workers,
                "cpu_workers": cpu_workers,
                "block_size": block,
                "est_batch_size": est_batch,
            },
            "optimal_workers": cpu_workers,
            "optimal_window": 15,
            "optimal_block": block,
            "est_batch_size": est_batch,
        }

    @classmethod
    def summary_string(cls, n_bands_guess=3, window_guess=5) -> str:
        """Return a human-readable system summary for display in the UI."""
        cfg = cls.get_auto_config(n_bands_guess, window_guess)
        gpu = cfg["gpu"]
        cpu = cfg["cpu"]
        ram = cfg["ram"]
        rec = cfg.get("recommended", cfg)

        lines = []
        lines.append("═══ System Profile ═══")

        # GPU
        if gpu.get("available"):
            backend = gpu.get("backend", "GPU") or "GPU"
            gpu_line = f"GPU: {backend}"
            total = gpu.get("total_gb", 0)
            free = gpu.get("free_gb", 0)
            if total > 0:
                gpu_line += f" | {total:.1f} GB total, {free:.1f} GB free"
            if gpu.get("custom_kernel"):
                gpu_line += " | CUDA Kernel ✓"
            lines.append(gpu_line)
        else:
            lines.append("GPU: Not available")

        # CPU
        lines.append(f"CPU: {cpu.get('logical_cores', '?')} logical cores")

        # RAM
        lines.append(f"RAM: {ram.get('total_gb', 0):.1f} GB total, {ram.get('available_gb', 0):.1f} GB available")

        # Recommendations
        lines.append("── Recommended Settings ──")
        mode = "GPU" if rec.get("use_gpu", False) else "CPU"
        wk = rec.get("gpu_workers", 4) if rec.get("use_gpu", False) else rec.get("cpu_workers", 4)
        blk = rec.get("block_size", 256)
        lines.append(f"Mode: {mode}  |  Workers: {wk}  |  Block: {blk}")

        if gpu.get("available"):
            batch = rec.get("est_batch_size", 100000)
            batch_str = f"{batch:,}" if batch < 1e6 else f"{batch/1e6:.1f}M"
            lines.append(f"Est. GPU batch: {batch_str} windows")

        lines.append("═══════════════════════")

        return "\n".join(lines)

    @staticmethod
    def get_os_info() -> Dict[str, str]:
        """Get operating system information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        }

    def get_full_report(self) -> Dict[str, object]:
        """Get a complete system report."""
        return {
            "os": self.get_os_info(),
            "cpu": self.get_cpu_info(),
            "ram": self.get_ram_info(),
            "gpu": self.get_gpu_info(),
        }

    def print_report(self):
        """Print a human-readable system report."""
        report = self.get_full_report()
        print("=" * 50)
        print("  PaRaVis System Report")
        print("=" * 50)
        os_info = report["os"]
        print(f"  OS:      {os_info['system']} {os_info['release']}")
        cpu_info = report["cpu"]
        print(f"  CPU:     {cpu_info['logical_cores']} logical cores ({cpu_info['architecture']})")
        ram_info = report["ram"]
        print(f"  RAM:     {ram_info['total_gb']:.1f} GB total")
        gpu_info = report["gpu"]
        if gpu_info["available"]:
            print(f"  GPU:     {gpu_info['name']} ({gpu_info['total_gb']:.0f} GB)")
        else:
            print(f"  GPU:     Not available")
        print("=" * 50)
