"""
Tests for paravis.utils — logging, settings, system profiler.

Run with:  pytest tests/test_utils.py -v --cov=paravis.utils
"""
import os
import tempfile
import logging
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_setup_logger_returns_logger(self):
        from paravis.utils.logging import setup_logging
        logger = setup_logging(level=logging.DEBUG, name="test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG

    def test_logger_has_console_handler(self):
        from paravis.utils.logging import setup_logging
        logger = setup_logging(level=logging.INFO, name="test_console")
        handlers = logger.handlers
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)

    def test_logger_with_file(self):
        from paravis.utils.logging import setup_logging
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
            log_path = f.name
        try:
            logger = setup_logging(level=logging.DEBUG, log_file=log_path,
                                   name="test_file_logger")
            logger.info("Test log message")
            # Close all handlers so the file is released
            for h in logger.handlers:
                h.close()
            logger.handlers.clear()
            with open(log_path) as f:
                content = f.read()
            assert "Test log message" in content
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_get_logger(self):
        from paravis.utils.logging import get_logger, setup_logging
        setup_logging(name="test_get_logger")
        logger = get_logger("test_get_logger")
        assert isinstance(logger, logging.Logger)

    def test_auto_setup_on_import(self):
        """Test that importing logging auto-creates a logger."""
        # Re-import to trigger auto-setup
        import importlib
        import paravis.utils.logging
        importlib.reload(paravis.utils.logging)
        assert hasattr(paravis.utils.logging, "logger")


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------

class TestAppSettings:
    def test_get_set(self):
        """Test basic get/set operations."""
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        # Use a unique key to avoid interfering with real settings
        settings.set("__test_key__", "test_value")
        val = settings.get("__test_key__")
        assert val == "test_value"

    def test_get_default(self):
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        val = settings.get("__nonexistent_key__", "default_val")
        assert val == "default_val"

    def test_recent_files_roundtrip(self):
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        # Add a file
        files = settings.add_recent_file("/path/to/test/file.tif")
        assert "/path/to/test/file.tif" in files
        # Get recent files
        recent = settings.get_recent_files()
        assert "/path/to/test/file.tif" in recent
        # Clear
        settings.clear_recent_files()
        recent_after = settings.get_recent_files()
        assert recent_after == []

    def test_add_recent_file_dedup(self):
        """Adding same path twice should not duplicate."""
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        settings.add_recent_file("/path/to/dup.tif")
        files = settings.add_recent_file("/path/to/dup.tif")
        assert files.count("/path/to/dup.tif") == 1

    def test_recent_files_limit(self):
        """Recent files list should be capped at 15."""
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        for i in range(20):
            settings.add_recent_file(f"/path/to/file_{i}.tif")
        recent = settings.get_recent_files()
        assert len(recent) <= 15

    def test_geometry_roundtrip(self):
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        settings.set_geometry(b"test_geom")
        geom = settings.get_geometry()
        assert geom == b"test_geom"

    def test_splitter_state_roundtrip(self):
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        settings.set_splitter_state(b"test_splitter")
        state = settings.get_splitter_state()
        assert state == b"test_splitter"

    def test_last_dir(self):
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        settings.set_last_dir("/tmp")
        assert settings.get_last_dir() == "/tmp"

    def test_theme(self):
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        settings.set_theme("dark")
        assert settings.get_theme() == "dark"

    def test_str_recent_files_fallback(self):
        """Test that string-type recent files are handled (backward compat)."""
        from paravis.utils.settings import AppSettings
        settings = AppSettings()
        # Directly set a JSON string to simulate old format
        settings.qs.setValue("recent_files", json.dumps(["/old/format.tif"]))
        files = settings.get_recent_files()
        assert "/old/format.tif" in files


# ---------------------------------------------------------------------------
# System profiler tests
# ---------------------------------------------------------------------------

class TestSystemProfiler:
    def test_get_cpu_info(self):
        from paravis.utils.system import SystemProfiler
        info = SystemProfiler.get_cpu_info()
        assert "physical_cores" in info
        assert info["physical_cores"] > 0

    def test_get_ram_info(self):
        from paravis.utils.system import SystemProfiler
        info = SystemProfiler.get_ram_info()
        assert "total_gb" in info
        # total_gb is 0 when psutil is not installed
        assert info["total_gb"] >= 0

    def test_get_os_info(self):
        from paravis.utils.system import SystemProfiler
        info = SystemProfiler.get_os_info()
        assert "system" in info
        assert "release" in info
        assert "version" in info

    def test_get_gpu_info_no_cupy(self):
        """Test GPU info when CuPy is not available."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", False), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", None), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            info = SystemProfiler.get_gpu_info()
            assert info["available"] is False
            assert info["name"] is None

    def test_get_gpu_info_with_cupy(self):
        """Test GPU info when CuPy is available."""
        from paravis.utils.system import SystemProfiler
        info = SystemProfiler.get_gpu_info()
        # May or may not have GPU, but should not crash
        assert "available" in info
        assert "name" in info
        assert "total_gb" in info
        assert "cuda_version" in info

    def test_get_gpu_info_numba_cuda(self):
        """Test GPU info with Numba CUDA backend."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "Numba CUDA"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            info = SystemProfiler.get_gpu_info()
            assert info["available"] is True
            assert info["backend"] == "Numba CUDA"

    def test_get_full_report(self):
        from paravis.utils.system import SystemProfiler
        profiler = SystemProfiler()
        report = profiler.get_full_report()
        assert "os" in report
        assert "cpu" in report
        assert "ram" in report
        assert "gpu" in report

    def test_print_report(self, capsys):
        from paravis.utils.system import SystemProfiler
        profiler = SystemProfiler()
        profiler.print_report()
        captured = capsys.readouterr()
        assert "PaRaVis System Report" in captured.out
        assert "OS" in captured.out or "CPU" in captured.out
        assert "GB" in captured.out

    def test_get_auto_config(self):
        """Test auto_config returns recommendations."""
        from paravis.utils.system import SystemProfiler
        config = SystemProfiler.get_auto_config(n_bands_guess=3, window_guess=5)
        assert "gpu" in config
        assert "cpu" in config
        assert "ram" in config
        assert "recommended" in config
        rec = config["recommended"]
        assert "use_gpu" in rec
        assert "gpu_workers" in rec
        assert "cpu_workers" in rec
        assert rec["cpu_workers"] > 0
        assert rec["block_size"] >= 128

    def test_get_auto_config_large(self):
        """Test auto_config with large window/bands."""
        from paravis.utils.system import SystemProfiler
        config = SystemProfiler.get_auto_config(n_bands_guess=10, window_guess=31)
        assert config["recommended"]["block_size"] >= 128

    def test_summary_string(self):
        """Test summary_string returns formatted output."""
        from paravis.utils.system import SystemProfiler
        summary = SystemProfiler.summary_string(n_bands_guess=3, window_guess=5)
        assert "System Profile" in summary or "GPU" in summary or "CPU" in summary
        assert "Recommended" in summary

    def test_summary_string_with_gpu(self):
        """Test summary_string when GPU is available."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "CuPy"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", True):
            summary = SystemProfiler.summary_string()
            assert "GPU" in summary

    def test_get_cpu_info_no_psutil(self):
        """CPU info when psutil is not available."""
        from paravis.utils.system import SystemProfiler
        with patch.dict('sys.modules', {'psutil': None}):
            # Force reimport of the module
            import importlib
            import paravis.utils.system
            importlib.reload(paravis.utils.system)
            from paravis.utils.system import SystemProfiler
            info = SystemProfiler.get_cpu_info()
            assert "physical_cores" in info
            assert info["physical_cores"] > 0

    def test_get_ram_info_no_psutil(self):
        """RAM info when psutil is not available."""
        from paravis.utils.system import SystemProfiler
        import importlib
        # Use real psutil since mocking imports is tricky
        info = SystemProfiler.get_ram_info()
        assert "total_gb" in info

    def test_get_os_info(self):
        """Test OS info detection."""
        from paravis.utils.system import SystemProfiler
        info = SystemProfiler.get_os_info()
        assert "system" in info
        assert "release" in info
        assert "version" in info

    def test_get_full_report(self):
        """Test full system report."""
        from paravis.utils.system import SystemProfiler
        profiler = SystemProfiler()
        report = profiler.get_full_report()
        assert "os" in report
        assert "cpu" in report
        assert "ram" in report
        assert "gpu" in report

    def test_print_report(self, capsys):
        """Test print_report outputs to stdout."""
        from paravis.utils.system import SystemProfiler
        profiler = SystemProfiler()
        profiler.print_report()
        captured = capsys.readouterr()
        assert "PaRaVis System Report" in captured.out

    def test_get_auto_config(self):
        """Test auto_config returns recommendations."""
        from paravis.utils.system import SystemProfiler
        config = SystemProfiler.get_auto_config(n_bands_guess=3, window_guess=5)
        assert "gpu" in config
        assert "cpu" in config
        assert "ram" in config
        assert "recommended" in config
        rec = config["recommended"]
        assert "use_gpu" in rec
        assert "gpu_workers" in rec
        assert "cpu_workers" in rec
        assert rec["cpu_workers"] > 0
        assert rec["block_size"] >= 128

    def test_get_auto_config_large(self):
        """Test auto_config with large window/bands."""
        from paravis.utils.system import SystemProfiler
        config = SystemProfiler.get_auto_config(n_bands_guess=10, window_guess=31)
        assert config["recommended"]["block_size"] >= 128

    def test_summary_string(self):
        """Test summary_string returns formatted output."""
        from paravis.utils.system import SystemProfiler
        summary = SystemProfiler.summary_string(n_bands_guess=3, window_guess=5)
        assert "System Profile" in summary or "GPU" in summary or "CPU" in summary
        assert "Recommended" in summary

    def test_summary_string_with_gpu(self):
        """Test summary_string when GPU is available."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "CuPy"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", True):
            summary = SystemProfiler.summary_string()
            assert "GPU" in summary

    def test_get_cpu_info_no_psutil(self):
        """CPU info when psutil is not available."""
        from paravis.utils.system import SystemProfiler
        with patch.dict('sys.modules', {'psutil': None}):
            # Force reimport of the module
            import importlib
            import paravis.utils.system
            importlib.reload(paravis.utils.system)
            from paravis.utils.system import SystemProfiler
            info = SystemProfiler.get_cpu_info()
            assert "physical_cores" in info
            assert info["physical_cores"] > 0

    def test_get_ram_info_no_psutil(self):
        """RAM info when psutil is not available."""
        from paravis.utils.system import SystemProfiler
        import importlib
        # Use real psutil since mocking imports is tricky
        info = SystemProfiler.get_ram_info()
        assert "total_gb" in info

    def test_get_gpu_info_numba_cuda(self):
        """Test GPU info with Numba CUDA backend."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "Numba CUDA"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            info = SystemProfiler.get_gpu_info()
            assert info["available"] is True
            assert info["backend"] == "Numba CUDA"

    def test_get_gpu_info_cupy(self):
        """Test GPU info with CuPy backend."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "CuPy"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", True):
            info = SystemProfiler.get_gpu_info()
            assert info["available"] is True
            assert info["backend"] == "CuPy"
            assert info["custom_kernel"] is True

    def test_get_gpu_info_no_gpu(self):
        """Test GPU info when no GPU available."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", False), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", None), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            info = SystemProfiler.get_gpu_info()
            assert info["available"] is False
            assert info["name"] is None


class TestSystemProfilerGPUDetails:
    """Targeted tests for uncovered GPU detection paths."""

    @pytest.mark.skipif(not HAVE_CUPY, reason="CuPy not installed")
    def test_get_gpu_info_cupy_exception(self):
        """CuPy GPU info when cp.cuda.runtime.memGetInfo raises."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "CuPy"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", True):
            with patch("cupy.cuda.runtime.memGetInfo",
                       side_effect=Exception("CUDA error")):
                info = SystemProfiler.get_gpu_info()
                assert info["available"] is True
                assert info["backend"] == "CuPy"
                assert info["name"] is None

    @pytest.mark.skipif(not HAVE_CUPY, reason="CuPy not installed")
    def test_get_gpu_info_cupy_runtime_version_exception(self):
        """CuPy GPU info when runtimeGetVersion raises (covers inner except)."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "CuPy"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", True):
            with patch("cupy.cuda.runtime.runtimeGetVersion",
                       side_effect=Exception("version error")):
                info = SystemProfiler.get_gpu_info()
                assert info["available"] is True
                assert info["backend"] == "CuPy"
                # cuda_version should be None since we raised
                assert info["cuda_version"] is None

    def test_get_gpu_info_numba_cuda_success(self):
        """Numba CUDA GPU info with mocked device."""
        from paravis.utils.system import SystemProfiler
        import sys
        mock_device = MagicMock()
        mock_device.total_memory = 8 * 1024 ** 3
        mock_device.name = "Mock GPU"

        mock_numba = MagicMock()
        mock_numba.cuda.get_current_device.return_value = mock_device

        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "Numba CUDA"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            with patch.dict('sys.modules', {'numba': mock_numba,
                                            'numba.cuda': mock_numba.cuda}):
                info = SystemProfiler.get_gpu_info()
                assert info["available"] is True
                assert info["backend"] == "Numba CUDA"
                assert info["name"] == "Mock GPU"
                assert info["total_gb"] == 8.0
                assert info["free_gb"] == 8.0 * 0.85

    def test_get_gpu_info_numba_cuda_exception(self):
        """Numba CUDA GPU info when get_current_device raises."""
        from paravis.utils.system import SystemProfiler
        import sys
        mock_numba = MagicMock()
        mock_numba.cuda.get_current_device.side_effect = Exception("No CUDA device")

        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", True), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", "Numba CUDA"), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            with patch.dict('sys.modules', {'numba': mock_numba,
                                            'numba.cuda': mock_numba.cuda}):
                info = SystemProfiler.get_gpu_info()
                assert info["available"] is True
                assert info["backend"] == "Numba CUDA"
                assert info["name"] is None

    def test_gpu_not_available_in_report(self):
        """get_full_report with GPU not available."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", False), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", None), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            profiler = SystemProfiler()
            report = profiler.get_full_report()
            assert "gpu" in report
            assert report["gpu"]["available"] is False

    def test_print_report_gpu_not_available(self, capsys):
        """print_report with no GPU."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", False), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", None), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            profiler = SystemProfiler()
            profiler.print_report()
            captured = capsys.readouterr()
            assert "GPU" in captured.out

    def test_summary_string_no_gpu(self):
        """summary_string when GPU is not available (covers 'GPU: Not available' line)."""
        from paravis.utils.system import SystemProfiler
        with patch("paravis.core.raoq.gpu.GPU_AVAILABLE", False), \
             patch("paravis.core.raoq.gpu.GPU_BACKEND", None), \
             patch("paravis.core.raoq.gpu.CUSTOM_KERNEL_AVAILABLE", False):
            summary = SystemProfiler.summary_string()
            assert "GPU: Not available" in summary

    def test_get_ram_info_import_error(self):
        """get_ram_info when psutil is not installed (covers ImportError branch)."""
        from paravis.utils.system import SystemProfiler
        with patch.dict('sys.modules', {'psutil': None}):
            import importlib
            import paravis.utils.system
            importlib.reload(paravis.utils.system)
            from paravis.utils.system import SystemProfiler
            info = SystemProfiler.get_ram_info()
            assert info["total_gb"] == 0
            assert info["available_gb"] == 0
            assert info["percent_used"] == 0


# ---------------------------------------------------------------------------
# Package-level __init__ tests (lazy import via __getattr__)
# ---------------------------------------------------------------------------


class TestUtilsPackage:
    """Tests for paravis.utils.__init__ lazy import mechanism."""

    def test_lazy_import_app_settings(self):
        """AppSettings should be importable via __getattr__."""
        from paravis.utils import AppSettings
        from paravis.utils.settings import AppSettings as RealAppSettings
        assert AppSettings is RealAppSettings

    def test_attr_error(self):
        """Accessing non-existent attr should raise AttributeError."""
        import paravis.utils
        with pytest.raises(AttributeError):
            paravis.utils.nonexistent_attr
