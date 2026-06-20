"""
Base worker thread — reusable QThread with start/stop/pause and signals.
"""
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker


class BaseWorker(QThread):
    """Abstract base class for all background workers.

    Signals
    -------
    finished : object
        Emitted when work completes successfully.
    error : str
        Emitted on failure with error message.
    progress : int
        Emitted with progress percentage (0-100).
    """

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._is_running = True
        self._is_paused = False

    def stop(self):
        """Request the worker to stop."""
        with QMutexLocker(self._mutex):
            self._is_running = False

    def pause(self):
        """Pause the worker."""
        with QMutexLocker(self._mutex):
            self._is_paused = True

    def resume(self):
        """Resume a paused worker."""
        with QMutexLocker(self._mutex):
            self._is_paused = False

    @property
    def is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._is_running

    @property
    def is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._is_paused

    def emit_progress(self, value: int):
        """Thread-safe progress emission."""
        self.progress.emit(value)
