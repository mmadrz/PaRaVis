"""
Tests for paravis.core.models — shared data models.

Run with:  pytest tests/test_core_models.py -v --cov=paravis.core.models
"""
import numpy as np
import pytest


class TestRasterProfile:
    def test_create_defaults(self):
        from paravis.core.models import RasterProfile
        profile = RasterProfile(path="/test.tif", shape=(100, 200))
        assert profile.path == "/test.tif"
        assert profile.shape == (100, 200)
        assert profile.crs is None
        assert profile.transform is None
        assert profile.dtype is None
        assert profile.nodata is None
        assert profile.size_mb == 0.0

    def test_create_full(self):
        from paravis.core.models import RasterProfile
        profile = RasterProfile(
            path="/test.tif",
            shape=(100, 200),
            crs="EPSG:4326",
            transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            dtype="float32",
            nodata=-9999.0,
            size_mb=0.5,
        )
        assert profile.crs == "EPSG:4326"
        assert profile.dtype == "float32"
        assert profile.nodata == -9999.0
        assert profile.size_mb == 0.5

    def test_mutable(self):
        """Dataclass fields should be mutable."""
        from paravis.core.models import RasterProfile
        profile = RasterProfile(path="/test.tif", shape=(10, 10))
        profile.size_mb = 2.5
        assert profile.size_mb == 2.5


class TestWindowConfig:
    def test_defaults(self):
        from paravis.core.models import WindowConfig
        cfg = WindowConfig()
        assert cfg.window_size == 15
        assert cfg.step_size == 1
        assert cfg.na_tolerance == 0.3
        assert cfg.max_pixels == 500_000

    def test_custom_values(self):
        from paravis.core.models import WindowConfig
        cfg = WindowConfig(window_size=5, step_size=2, na_tolerance=0.1, max_pixels=1000)
        assert cfg.window_size == 5
        assert cfg.step_size == 2
        assert cfg.na_tolerance == 0.1
        assert cfg.max_pixels == 1000

    def test_mutable(self):
        from paravis.core.models import WindowConfig
        cfg = WindowConfig()
        cfg.window_size = 7
        assert cfg.window_size == 7


class TestSpectralIndex:
    """Tests for SpectralIndex data model."""

    def test_create_defaults(self):
        from paravis.core.indices.models import SpectralIndex
        idx = SpectralIndex(name="TESTVI", formula="N-R", bands=["N", "R"])
        assert idx.name == "TESTVI"
        assert idx.formula == "N-R"
        assert idx.bands == ["N", "R"]
        assert idx.constants == []
        assert idx.reference == ""
        assert idx.long_name == ""
        assert idx.platform == "generic"
        assert idx.computable is False

    def test_create_full(self):
        from paravis.core.indices.models import SpectralIndex
        idx = SpectralIndex(
            name="EVI",
            formula="G*(N-R)/(N+C1*R-C2*B+L)",
            bands=["N", "R", "B"],
            constants=["L", "G", "C1", "C2"],
            reference="doi:10.1016/S0034-4257(96)00122-7",
            long_name="Enhanced Vegetation Index",
            platform="MODIS",
            computable=True,
        )
        assert idx.constants == ["L", "G", "C1", "C2"]
        assert idx.reference.startswith("doi:")
        assert idx.platform == "MODIS"
        assert idx.computable


class TestBandMappingModel:
    """Tests for BandMapping data model (separate from indices tests)."""

    def test_create_defaults(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping()
        assert bm.mapping == {}
        assert bm.name == "Custom"

    def test_landsat_8_9_classmethod(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.landsat_8_9()
        assert bm.name == "Landsat 8/9"
        assert bm.get_code(4) == "R"

    def test_sentinel2_classmethod(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.sentinel2()
        assert bm.name == "Sentinel-2"
        assert bm.get_code(4) == "R"
        assert bm.get_code(8) == "N2"

    def test_sentinel1_classmethod(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.sentinel1()
        assert bm.name == "Sentinel-1"
        assert bm.get_code(1) == "VV"
        assert bm.get_code(2) == "VH"


class TestRaoQModels:
    """Tests for RaoQ data models."""

    def test_rao_q_config_defaults(self):
        from paravis.core.raoq.models import RaoQConfig
        cfg = RaoQConfig()
        assert cfg.window_size == 15
        assert cfg.step_size == 1
        assert cfg.na_tolerance == 0.3
        assert cfg.n_workers == 4
        assert cfg.tile_size == 1024
        assert cfg.use_gpu is False
        assert cfg.gpu_batch_size == 50000
        assert cfg.distance_metric == "euclidean"
        assert cfg.p_minkowski == 2
        assert cfg.simplify == 2
        assert cfg.block_size == 1024

    def test_rao_q_config_custom(self):
        from paravis.core.raoq.models import RaoQConfig
        cfg = RaoQConfig(window_size=7, simplify=1, use_gpu=True, gpu_batch_size=100000)
        assert cfg.window_size == 7
        assert cfg.simplify == 1
        assert cfg.use_gpu is True
        assert cfg.gpu_batch_size == 100000

    def test_rao_q_result_defaults(self):
        import numpy as np
        from paravis.core.raoq.models import RaoQResult
        data = np.random.rand(5, 5).astype(np.float32)
        result = RaoQResult(data=data, window_size=15, step_size=1)
        assert result.computation_time == 0.0
        assert result.gpu_used is False
        assert result.n_windows == 0

    def test_rao_q_result_full(self):
        import numpy as np
        from paravis.core.raoq.models import RaoQResult
        data = np.random.rand(10, 10).astype(np.float32)
        result = RaoQResult(
            data=data,
            window_size=5,
            step_size=2,
            computation_time=3.14,
            gpu_used=True,
            n_windows=100,
        )
        assert result.window_size == 5
        assert result.step_size == 2
        assert result.computation_time == 3.14
        assert result.gpu_used is True
        assert result.n_windows == 100