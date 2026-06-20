"""
Application entry point — QApplication factory and main().

Usage:
    python -m paravis.gui.app
    # or after pip install:  paravis
"""
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from paravis.__version__ import __version__
from .main_window import MainWindow
from .components.splash import ModernSplashScreen


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
