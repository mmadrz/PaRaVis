"""Rao's Q diversity computation — CPU and GPU backends."""

from .engine import compute_rao_q, compute_rao_q_parallel
from .models import RaoQConfig, RaoQResult

__all__ = [
    "compute_rao_q",
    "compute_rao_q_parallel",
    "RaoQConfig",
    "RaoQResult",
]
