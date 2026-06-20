"""PaRaVis shared utilities."""

from .system import SystemProfiler


def __getattr__(name):
    """Lazy import AppSettings to avoid requiring PySide6 at module level."""
    if name == "AppSettings":
        from .settings import AppSettings
        return AppSettings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AppSettings", "SystemProfiler"]
