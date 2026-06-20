"""
Pytest fixtures for PaRaVis tests.

Provides sample raster data and configurations.
"""
import numpy as np
import pytest


@pytest.fixture
def sample_raster_3d():
    """Return a small 3-band raster (3, 10, 10) with known values."""
    np.random.seed(42)
    data = np.random.rand(3, 10, 10).astype(np.float32)
    return data


@pytest.fixture
def sample_raster_2d():
    """Return a single-band raster (10, 10)."""
    np.random.seed(42)
    data = np.random.rand(10, 10).astype(np.float32)
    return data


@pytest.fixture
def landsat_mapping():
    """Landsat 8/9 band mapping."""
    return {1: "A", 2: "B", 3: "G", 4: "R", 5: "N", 6: "S1", 7: "S2", 8: "T"}


@pytest.fixture
def sample_constants():
    """Common constant overrides for testing."""
    return {"L": 0.5}
