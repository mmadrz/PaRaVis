"""
GUI-level worker threads for plot generation and file processing.

These wrap the core workers with GUI-specific features (progress bars,
status updates).
"""
from PySide6.QtCore import QThread, Signal

from paravis.workers.base_worker import BaseWorker


class PlotWorker(BaseWorker):
    """Worker for generating matplotlib plots in a background thread.

    Parameters
    ----------
    plot_func : callable
        Function that generates the plot (takes no args, returns a figure).
    """

    def __init__(self, plot_func, parent=None):
        super().__init__(parent)
        self.plot_func = plot_func

    def run(self):
        """Execute the plot function."""
        try:
            result = self.plot_func()
            if self.is_running:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class CompareWorker(BaseWorker):
    """Worker for computing raster differences/comparisons."""

    def __init__(self, data1, data2, func, parent=None):
        super().__init__(parent)
        self.data1 = data1
        self.data2 = data2
        self.func = func

    def run(self):
        try:
            result = self.func(self.data1, self.data2)
            if self.is_running:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class GifWorker(BaseWorker):
    """Worker for generating animated GIFs from a list of rasters."""

    def __init__(self, file_list, file_names, colormap, output_dir,
                 output_name, dpi, fps=10, normalize_all=True, parent=None):
        super().__init__(parent)
        self.file_list = file_list
        self.file_names = file_names
        self.colormap = colormap
        self.output_dir = output_dir
        self.output_name = output_name
        self.dpi = dpi
        self.fps = fps
        self.normalize_all = normalize_all

    def run(self):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import numpy as np
            from paravis.core.raster import read_raster, normalize_data

            fig, ax = plt.subplots(figsize=(10, 8))
            im = None
            cbar = None

            def update(frame_idx):
                nonlocal im, cbar
                path = self.file_list[frame_idx]
                data, transform, _ = read_raster(path)
                if data.ndim == 3:
                    data = data[0]
                if self.normalize_all:
                    data = normalize_data(data)

                num_rows, num_cols = data.shape
                x = np.arange(0, num_cols) * transform.a + transform.c
                y = np.arange(0, num_rows) * transform.e + transform.f

                if im is None:
                    im = ax.imshow(
                        np.ma.masked_invalid(data),
                        cmap=self.colormap,
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin="upper",
                        animated=True,
                    )
                    # Add colorbar on first frame
                    cbar = fig.colorbar(im, label="Value", pad=0.01)
                    cbar.ax.get_yaxis().label.set_fontsize(12)

                    # Remove axis ticks and labels (no lat/long)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                else:
                    im.set_array(data)

                ax.set_title(self.file_names[frame_idx])
                return [im]

            anim = animation.FuncAnimation(
                fig, update, frames=len(self.file_list),
                interval=1000 // self.fps, blit=True
            )
            output_path = f"{self.output_dir}/{self.output_name}.gif"
            anim.save(output_path, writer="pillow", fps=self.fps, dpi=self.dpi)
            plt.close(fig)

            if self.is_running:
                self.finished.emit(output_path)
        except Exception as exc:
            self.error.emit(str(exc))
