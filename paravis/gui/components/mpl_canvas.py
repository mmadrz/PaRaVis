"""
Matplotlib canvas widget for embedding plots in the GUI.
"""
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MplCanvas(FigureCanvas):
    """Simple Matplotlib canvas widget for PySide6.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    width : int
        Figure width in inches.
    height : int
        Figure height in inches.
    dpi : int
        Figure resolution.
    """

    def __init__(self, parent=None, width: int = 8, height: int = 6, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
