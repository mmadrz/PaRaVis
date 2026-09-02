"""
Application entry point — QApplication factory and main().

Usage:
    python -m paravis.gui.app
    # or after pip install:  paravis
"""
import ctypes
import ctypes.util
import os
import sys

from PySide6.QtCore import QLibraryInfo, Qt
from PySide6.QtWidgets import QApplication

from paravis.__version__ import __version__
from .main_window import MainWindow
from .components.splash import ModernSplashScreen


def _suppress_gdkpixbuf_warnings():
    """Silence harmless GdkPixbuf-CRITICAL noise from GTK on Linux.

    These warnings (e.g. "gdk_pixbuf_new_from_bytes: assertion ... failed")
    are emitted by the GTK pixbuf loader when Qt renders widgets through a
    graphics effect (e.g. the blinking logo). They are non-fatal and purely
    cosmetic, but they clutter the terminal. We install a GLib log handler
    that drops only the GdkPixbuf-CRITICAL messages.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        glib = ctypes.CDLL(ctypes.util.find_library("glib-2.0") or "libglib-2.0.so.0")

        G_LOG_LEVEL_CRITICAL = 1 << 3  # G_LOG_LEVEL_CRITICAL
        G_LOG_LEVEL_ERROR = 1 << 4     # G_LOG_LEVEL_ERROR

        # gboolean (*GLogFunc)(const gchar *log_domain, GLogLevelFlags log_level,
        #                      const gchar *message, gpointer user_data)
        LOG_FUNC = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_void_p,
        )

        @LOG_FUNC
        def _log_handler(log_domain, log_level, message, user_data):
            # Drop GdkPixbuf-CRITICAL messages; keep everything else.
            if log_domain and b"GdkPixbuf" in log_domain:
                return 0  # handled / suppressed
            return 1  # not handled -> let GLib print it

        glib.g_log_set_handler(
            ctypes.c_char_p(b"GdkPixbuf"),
            G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_ERROR,
            _log_handler,
            None,
        )
    except Exception:
        # If we can't set the handler, just let the warnings print.
        pass


def create_app() -> QApplication:
    """Create and configure the QApplication instance.

    Returns
    -------
    QApplication
        Configured application instance.
    """
    # Force XCB on Linux
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    # Silence harmless GdkPixbuf-CRITICAL noise from GTK (Linux only).
    _suppress_gdkpixbuf_warnings()

    # Point Qt at the PySide6-bundled plugins so it doesn't pick up a
    # conflicting conda/system Qt installation (e.g. conda qtbase 6.9.x
    # vs pip PySide6 6.11.x), which causes "Could not find the Qt platform
    # plugin 'xcb'" due to a version mismatch.
    if not os.environ.get("QT_PLUGIN_PATH"):
        try:
            plugins_path = QLibraryInfo.path(
                QLibraryInfo.LibraryPath.PluginsPath
            )
            if plugins_path and os.path.isdir(plugins_path):
                os.environ["QT_PLUGIN_PATH"] = plugins_path
        except Exception:
            # If we can't resolve the plugins path, fall back to the
            # default behaviour.
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("PaRaVis")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("RaoQ")
    app.setStyle("Fusion")

    return app


def main():
    """Main entry point — run the PaRaVis GUI."""
    app = create_app()

    # Splash
    splash = ModernSplashScreen()
    splash.show_message()

    # Main window
    window = MainWindow(splash=splash)
    window.show()

    # Close splash
    if splash.isVisible():
        splash.close()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
