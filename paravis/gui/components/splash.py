"""
Modern splash screen — frameless QWidget with custom painting.
"""
import os

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QFontDatabase


class ModernSplashScreen(QWidget):
    """Custom splash screen displayed during application startup."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(520, 320)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._status_text = "Loading modules..."
        self._bg_color = QColor(0, 150, 136)
        self._stripe_color = QColor(0, 77, 64)

        # Load logo
        logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "logo.png")
        self._logo = QPixmap(logo_path)
        if self._logo.isNull():
            self._logo = None

    def paintEvent(self, event):
        """Paint the splash screen."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)

        # Shadow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 40))
        painter.drawRoundedRect(rect.translated(0, 2), 16, 16)

        # Background
        painter.setBrush(self._bg_color)
        painter.drawRoundedRect(rect, 16, 16)

        # Stripe
        stripe_rect = rect.adjusted(0, 0, 0, -rect.height() + 6)
        painter.setBrush(self._stripe_color)
        painter.drawRoundedRect(stripe_rect, 16, 16)
        painter.drawRect(stripe_rect.adjusted(0, 3, 0, 0))

        # Title
        painter.setPen(QColor(255, 255, 255))
        title_font = QFont("Segoe UI", 22, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(rect.adjusted(30, 40, -30, -100),
                         Qt.AlignmentFlag.AlignLeft, "PaRaVis")

        # Subtitle
        sub_font = QFont("Segoe UI", 12, QFont.Weight.Normal)
        painter.setFont(sub_font)
        painter.setPen(QColor(200, 230, 225))
        painter.drawText(rect.adjusted(30, 85, -30, -80),
                         Qt.AlignmentFlag.AlignLeft,
                         "Indices · Rao's Q · Visualization")

        # Version
        version_font = QFont("Segoe UI", 9, QFont.Weight.Normal)
        painter.setFont(version_font)
        painter.setPen(QColor(180, 210, 205))
        painter.drawText(rect.adjusted(30, 115, -30, -60),
                         Qt.AlignmentFlag.AlignLeft, "v2.0.0")

        # Line
        line_y = rect.bottom() - 100
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawLine(30, line_y, rect.right() - 30, line_y)

        # Logo
        if self._logo is not None:
            scaled = self._logo.scaled(100, 60,
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
            logo_x = rect.right() - scaled.width() - 30
            logo_y = line_y - scaled.height() - 8
            painter.drawPixmap(logo_x, logo_y, scaled)

        # Status
        painter.setPen(QColor(255, 255, 255))
        msg_font = QFont("Segoe UI", 10, QFont.Weight.Light)
        painter.setFont(msg_font)
        painter.drawText(rect.adjusted(30, 20, -30, -30),
                         Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                         self._status_text)
        painter.end()

    def show_message(self):
        """Show the splash screen."""
        self.show()
        QApplication.processEvents()

    def update_status(self, message: str):
        """Update the status text and repaint."""
        self._status_text = message
        self.update()
        QApplication.processEvents()
