"""
Data models for Rao's Q computation configuration and results.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RaoQConfig:
    """Configuration for Rao's Q computation."""
    window_size: int = 15
    step_size: int = 1
    na_tolerance: float = 0.3
    n_workers: int = 4
    tile_size: int = 1024
    use_gpu: bool = False
    gpu_batch_size: int = 50000
    cpu_batch_size: int = 10000
    distance_metric: str = "euclidean"
    p_minkowski: int = 2
    simplify: int = 2  # truncate to N decimal places (0=integers, no rounding)
    block_size: int = 1024


@dataclass
class RaoQResult:
    """Result of a Rao's Q computation."""
    data: np.ndarray
    window_size: int
    step_size: int
    computation_time: float = 0.0
    gpu_used: bool = False
    n_windows: int = 0
