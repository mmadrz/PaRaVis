"""
paravis.workers — Background QThread workers for async processing.

All thread workers follow the BaseWorker interface:
    - start() / stop() / pause()
    - Signals: finished, error, progress
"""
