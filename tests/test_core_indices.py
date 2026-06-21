"""
Unit tests for paravis.core.indices — no Qt dependency required.

Run with:  pytest tests/test_core_indices.py -v
"""
import numpy as np
import pytest
from unittest.mock import patch

from paravis.core.indices import (
    get_available_indices,
    is_index_computable,
    compute_index,
    compute_indices,
    get_default_band_mapping,
)
from paravis.core.indices.constants import get_default_constants, merge_constants
from paravis.core.indices.registry import register_index, list_custom_indices


class TestGetAvailableIndices:
    def test_returns_list(self):
        indices = get_available_indices()
        assert len(indices) > 0

    def test_contains_ndvi(self):
        names = [idx.name for idx in get_available_indices()]
        assert "NDVI" in names


class TestIsIndexComputable:
    def test_ndvi_computable(self, landsat_mapping):
        assert is_index_computable("NDVI", {}, landsat_mapping) is True

    def test_ndvi_not_computable_no_nir(self):
        mapping = {1: "A", 2: "B", 3: "G", 4: "R"}  # missing N
        assert is_index_computable("NDVI", {}, mapping) is False

    def test_unknown_index(self):
        assert is_index_computable("NONEXISTENT", {}, {}) is False

    def test_constant_with_none_override(self):
        """A required constant with override=None should check spyndex default.

        If spyndex also has default=None, is_index_computable returns False.
        """
        import spyndex
        # Find an index that requires a constant
        const_name = None
        for name in spyndex.indices:
            idx = spyndex.indices[name]
            if getattr(idx, "constants", []):
                const_name = getattr(idx, "constants", [])[0]
                break

        if const_name is not None:
            # Override the constant with None — should fall through to spyndex
            result = is_index_computable("SAVI", {const_name: None}, {4: "R", 5: "N"})
            # If spyndex has a non-None default, SAVI is computable
            # If spyndex has None default, it's not
            spyndex_const = spyndex.constants.get(const_name)
            if spyndex_const is not None and spyndex_const.default is not None:
                assert result is True
            else:
                assert result is False

    def test_constant_not_in_override_with_spyndex_none_default(self):
        """When a required constant is not provided and spyndex default
        is None, is_index_computable should return False."""
        import spyndex
        # spyndex.constants is a frozen Box; patch the module-level reference
        mock_consts = {"L": type('MockConst', (), {'default': None})()}
        with patch("spyndex.constants", mock_consts):
            result = is_index_computable("SAVI", {}, {4: "R", 5: "N"})
            assert result is False

    def test_savi_with_all_bands(self):
        """SAVI requires bands L, N, R; with all three it's computable."""
        result = is_index_computable("SAVI", {}, {3: "L", 4: "R", 5: "N"})
        assert result is True

    def test_savi_missing_l_band(self):
        """SAVI requires L band; without it, not computable."""
        result = is_index_computable("SAVI", {}, {4: "R", 5: "N"})
        assert result is False

    def test_constant_path_with_mocked_index(self):
        """Test the constants loop by adding a mock index with constants
        attribute (no stock spyndex index has it in this version)."""
        import spyndex

        # Spyndex indices is a frozen Box — unfreeze to add a mock index
        spyndex.indices._box_config['frozen_box'] = False

        mock_idx = type('MockIdx', (), {
            'bands': ['R', 'N'],
            'constants': ['L'],
        })()
        mock_name = '_TEST_MOCK_VI'
        spyndex.indices[mock_name] = mock_idx

        try:
            # Band L not in constants_override AND spyndex.constants has
            # L with default=1.0 → should be True
            result = is_index_computable(mock_name, {}, {4: "R", 5: "N"})
            assert result is True

            # Override L with a value → should be True
            result = is_index_computable(mock_name, {"L": 0.5}, {4: "R", 5: "N"})
            assert result is True

            # Override L with None → spyndex default is 1.0 (not None) → True
            result = is_index_computable(mock_name, {"L": None}, {4: "R", 5: "N"})
            assert result is True
        finally:
            del spyndex.indices[mock_name]
            spyndex.indices._box_config['frozen_box'] = True

    def test_constant_path_missing_band_and_constant(self):
        """When both a required band AND constant are missing,
        band check fails first."""
        import spyndex

        spyndex.indices._box_config['frozen_box'] = False

        mock_idx = type('MockIdx', (), {
            'bands': ['R', 'N', 'L'],
            'constants': ['X'],
        })()
        mock_name = '_TEST_MOCK_VI2'
        spyndex.indices[mock_name] = mock_idx

        try:
            result = is_index_computable(mock_name, {}, {4: "R", 5: "N"})
            assert result is False  # L band missing
        finally:
            del spyndex.indices[mock_name]
            spyndex.indices._box_config['frozen_box'] = True


class TestComputeIndex:
    def test_ndvi_output_shape(self, sample_raster_3d, landsat_mapping):
        # sample_raster_3d has bands [0,1,2], mapping expects bands 1-8
        # NDVI needs R(4) and N(5) — those bands don't exist in sample
        # We need to create a proper test raster
        data = np.random.rand(5, 10, 10).astype(np.float32)  # 5 bands
        result = compute_index(data, landsat_mapping, "NDVI")
        assert result.shape == (10, 10)
        assert result.dtype == np.float32

    def test_ndvi_values(self):
        # R=0.2, N=0.8  =>  NDVI = (0.8-0.2)/(0.8+0.2) = 0.6
        data = np.zeros((5, 1, 1), dtype=np.float32)
        data[3] = 0.2  # R
        data[4] = 0.8  # N
        mapping = {4: "R", 5: "N"}
        result = compute_index(data, mapping, "NDVI")
        assert np.isclose(result[0, 0], 0.6, atol=1e-6)

    def test_invalid_index_raises(self, sample_raster_3d):
        with pytest.raises(RuntimeError):
            compute_index(sample_raster_3d, {}, "INVALID")


class TestComputeIndices:
    def test_multiple_indices(self, landsat_mapping):
        data = np.random.rand(5, 10, 10).astype(np.float32)
        consts = {"L": 0.5}
        results = compute_indices(data, landsat_mapping, ["NDVI", "SAVI"], constants=consts)
        assert "NDVI" in results
        assert "SAVI" in results
        assert results["NDVI"].shape == (10, 10)


class TestConstants:
    def test_get_default_constants(self):
        consts = get_default_constants()
        assert isinstance(consts, dict)
        assert len(consts) > 0

    def test_merge_constants(self):
        defaults = {"L": 0.5, "K": 1.0}
        merged = merge_constants(defaults, {"L": 0.3})
        assert merged["L"] == 0.3
        assert merged["K"] == 1.0

    def test_merge_removes_none(self):
        defaults = {"L": 0.5}
        merged = merge_constants(defaults, {"L": None})
        assert "L" not in merged


class TestCustomRegistry:
    def test_register_and_list(self):
        @register_index(name="TEST_VI", bands=["N", "R"])
        def test_vi(nir, red):
            return nir - red

        custom = list_custom_indices()
        assert "TEST_VI" in custom
        assert custom["TEST_VI"]["bands"] == ["N", "R"]


class TestComputeIndicesDask:
    def test_dask_compute_single(self):
        """Test dask-accelerated compute with single index."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_indices_dask(data, mapping, ["NDVI"], tile_size=8)
        assert "NDVI" in result
        assert result["NDVI"].shape == (10, 10)

    def test_dask_compute_multiple(self):
        """Test dask with multiple indices."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 20, 20).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_indices_dask(data, mapping, ["NDVI", "SAVI"],
                                      constants={"L": 0.5}, tile_size=8)
        assert "NDVI" in result
        assert "SAVI" in result

    def test_dask_empty_names(self):
        """Test dask with empty index list."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 10, 10).astype(np.float32)
        result = compute_indices_dask(data, {}, [], tile_size=8)
        assert result == {}

    def test_dask_unknown_indices(self):
        """Test dask with only unknown indices."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 10, 10).astype(np.float32)
        result = compute_indices_dask(data, {}, ["INVALID_123"], tile_size=8)
        assert result == {}

    def test_dask_with_num_workers(self):
        """Test dask with explicit num_workers."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 16, 16).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_indices_dask(data, mapping, ["NDVI"],
                                      num_workers=2, tile_size=8)
        assert "NDVI" in result


class TestComputeIndicesBatchFallback:
    def test_batch_fallback_on_error(self):
        """When batch spyndex call fails, should fall back to one-by-one."""
        from paravis.core.indices.engine import compute_indices
        import spyndex

        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}

        # Mock spyndex.computeIndex to fail on batch call
        original_compute = spyndex.computeIndex
        call_count = [0]

        def failing_compute(indices, **params):
            call_count[0] += 1
            if call_count[0] == 1 and isinstance(indices, list) and len(indices) > 0:
                raise ValueError("Batch failed")
            return original_compute(indices, **params)

        with patch.object(spyndex, 'computeIndex', side_effect=failing_compute):
            results = compute_indices(data, mapping, ["NDVI"], constants={"L": 0.5})
            assert "NDVI" in results
            assert results["NDVI"].shape == (10, 10)


class TestComputeIndexEdgeCases:
    def test_compute_with_none_constants(self):
        """Test compute_index with None constants (should use defaults)."""
        from paravis.core.indices import compute_index
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_index(data, mapping, "NDVI", constants=None)
        assert result.shape == (10, 10)

    def test_compute_with_some_constants_none(self):
        """Test compute_index with some None-valued constants."""
        from paravis.core.indices import compute_index
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        # Passing None for a constant should be harmless
        result = compute_index(data, mapping, "NDVI", constants={"L": None})
        assert result.shape == (10, 10)

    def test_compute_with_extra_bands(self):
        """Mapping bands beyond data shape should be ignored."""
        from paravis.core.indices import compute_index
        data = np.random.rand(3, 10, 10).astype(np.float32)
        mapping = {1: "B", 2: "G", 3: "R", 4: "N", 5: "S1"}  # band 4,5 > data.shape[0]
        # NDVI needs R(3) and N(4) - band 4 doesn't exist, so this should fail
        with pytest.raises(RuntimeError):
            compute_index(data, mapping, "NDVI")

    def test_empty_indices_returns_empty(self):
        """compute_indices with empty list returns empty dict."""
        from paravis.core.indices import compute_indices
        data = np.random.rand(5, 10, 10).astype(np.float32)
        results = compute_indices(data, {}, [])
        assert results == {}


class TestBandMapping:
    def test_landsat_8_9(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.landsat_8_9()
        assert bm.name == "Landsat 8/9"
        assert bm.mapping[4] == "R"
        assert bm.mapping[8] == "T"

    def test_sentinel2(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.sentinel2()
        assert bm.name == "Sentinel-2"
        assert bm.mapping[4] == "R"
        assert bm.mapping[8] == "N2"

    def test_sentinel1(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.sentinel1()
        assert bm.name == "Sentinel-1"
        assert bm.mapping[1] == "VV"
        assert bm.mapping[2] == "VH"

    def test_get_code(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.landsat_8_9()
        assert bm.get_code(4) == "R"
        assert bm.get_code(99) == ""

    def test_to_dict(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.landsat_8_9()
        d = bm.to_dict()
        assert d[4] == "R"
        assert isinstance(d, dict)


class TestComputeCustomIndex:
    def test_compute_custom_index(self):
        from paravis.core.indices.registry import register_index, list_custom_indices, compute_custom_index
        import numpy as np

        @register_index(name="MY_TEST_VI", bands=["R", "N"])
        def my_test_index(R, N):
            return N - R

        result = compute_custom_index("MY_TEST_VI", {"R": np.array(1.0), "N": np.array(2.0)})
        assert result == 1.0

    def test_compute_custom_index_not_found(self):
        from paravis.core.indices.registry import compute_custom_index
        with pytest.raises(KeyError, match="not found"):
            compute_custom_index("DOES_NOT_EXIST", {})


class TestComputeIndexWithConstants:
    def test_compute_index_with_constants(self):
        """Test compute_index with constant overrides."""
        from paravis.core.indices import compute_index
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_index(data, mapping, "NDVI", constants={"L": 0.5})
        assert result.shape == (10, 10)

    def test_compute_index_invalid_name(self):
        """Test compute_index with an invalid index name."""
        from paravis.core.indices import compute_index
        data = np.random.rand(3, 10, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="Failed to compute index"):
            compute_index(data, {}, "INVALID_INDEX_NAME")


class TestComputeIndicesSkipFailing:
    def test_skips_failing_indices(self):
        """compute_indices should skip failing indices without crashing."""
        from paravis.core.indices import compute_indices
        # Use 6 bands so mapping {4: "R", 5: "N"} can access band 4 and 5
        data = np.random.rand(6, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        # One valid index and one invalid
        results = compute_indices(data, mapping, ["NDVI", "INVALID_INDEX"])
        assert "NDVI" in results
        assert "INVALID_INDEX" not in results


class TestIsIndexComputableWithConstants:
    def test_missing_constant_makes_not_computable(self):
        """Index requiring a constant that isn't provided should not be computable."""
        from paravis.core.indices import is_index_computable
        # SAVI in spyndex has bands ['L', 'N', 'R'] — 'L' is treated as a band, not a constant
        # Without mapping 'L', SAVI should not be computable
        result_no_l = is_index_computable("SAVI", {}, {4: "R", 5: "N"})
        assert result_no_l is False
        # With mapping including 'L', SAVI should be computable
        result_with_l = is_index_computable("SAVI", {}, {4: "R", 5: "N", 9: "L"})
        assert result_with_l is True

    def test_constant_override_makes_computable(self):
        """A constant override can satisfy an index's constant requirement."""
        from paravis.core.indices import is_index_computable
        # Some indices have constants that are not in the band mapping
        # EVI needs bands: g, N, R, C1, C2, B, L — map them all
        # Note: spyndex treats these as 'bands', not 'constants'
        mapping = {1: "B", 2: "g", 3: "R", 4: "N", 5: "C1", 6: "C2", 7: "L"}
        result = is_index_computable("EVI", {}, mapping)
        assert result is True


class TestComputeIndicesSingleResult:
    """Test the path where compute_indices returns a single 2D array."""

    def test_single_index_returns_dict(self):
        """compute_indices with one valid name should return a dict with one entry."""
        from paravis.core.indices import compute_indices
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        results = compute_indices(data, mapping, ["NDVI"])
        assert isinstance(results, dict)
        assert "NDVI" in results
        assert results["NDVI"].shape == (10, 10)


class TestGetDefaultBandMapping:
    def test_returns_dict(self):
        from paravis.core.indices import get_default_band_mapping
        mapping = get_default_band_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) >= 8

    def test_contains_key_bands(self):
        from paravis.core.indices import get_default_band_mapping
        mapping = get_default_band_mapping()
        assert 1 in mapping
        assert mapping[4] == "R"
        assert mapping[5] == "N"


class TestComputeIndicesDaskEdgeCases:
    """Edge cases for dask-accelerated index computation."""

    def test_dask_with_constants_none(self):
        """dask compute with constants=None."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_indices_dask(data, mapping, ["NDVI"], constants=None, tile_size=8)
        assert "NDVI" in result

    def test_dask_with_some_constants_none(self):
        """dask compute with some None constants."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(5, 10, 10).astype(np.float32)
        mapping = {4: "R", 5: "N"}
        result = compute_indices_dask(data, mapping, ["NDVI"],
                                      constants={"L": None}, tile_size=8)
        assert "NDVI" in result

    def test_dask_all_bands_mapped(self):
        """dask with more bands mapped than data has (bands 4,5 > 3)."""
        from paravis.core.indices.engine import compute_indices_dask
        data = np.random.rand(3, 10, 10).astype(np.float32)  # only 3 bands
        mapping = {1: "A", 2: "B", 3: "G", 4: "R", 5: "N"}  # bands 4,5 > 3
        # NDVI needs R and N which aren't available → spyndex will fail
        with pytest.raises(RuntimeError):
            compute_indices_dask(data, mapping, ["NDVI"], tile_size=8)


class TestComputeIndicesEmptyInput:
    def test_empty_raster_data(self):
        """compute_indices with empty raster data."""
        from paravis.core.indices import compute_indices
        data = np.empty((0, 10, 10), dtype=np.float32)
        results = compute_indices(data, {}, ["NDVI"])
        assert results == {}


class TestSpectralIndexModel:
    """Tests for the SpectralIndex dataclass."""

    def test_default_creation(self):
        from paravis.core.indices.models import SpectralIndex
        idx = SpectralIndex(name="TEST", formula="N-R", bands=["N", "R"])
        assert idx.name == "TEST"
        assert idx.formula == "N-R"
        assert idx.bands == ["N", "R"]
        assert idx.constants == []
        assert idx.reference == ""
        assert idx.long_name == ""
        assert idx.platform == "generic"
        assert idx.computable is False

    def test_full_creation(self):
        from paravis.core.indices.models import SpectralIndex
        idx = SpectralIndex(
            name="NDVI",
            formula="(N-R)/(N+R)",
            bands=["N", "R"],
            constants=["L"],
            reference="https://",
            long_name="Normalized Difference Vegetation Index",
            platform="Landsat",
            computable=True,
        )
        assert idx.name == "NDVI"
        assert idx.computable is True
        assert idx.platform == "Landsat"


class TestBandMappingEdgeCases:
    """Edge cases for BandMapping class."""

    def test_empty_mapping(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping()
        assert bm.mapping == {}
        assert bm.name == "Custom"
        assert bm.get_code(1) == ""

    def test_to_dict_copy(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.landsat_8_9()
        d = bm.to_dict()
        d[4] = "X"  # modify dict
        assert bm.mapping[4] == "R"  # original unchanged

    def test_get_code_missing(self):
        from paravis.core.indices.models import BandMapping
        bm = BandMapping.landsat_8_9()
        assert bm.get_code(99) == ""


class TestRegisterIndexEdgeCases:
    """Edge cases for custom index registry."""

    def test_register_multiple(self):
        from paravis.core.indices.registry import register_index, list_custom_indices
        # Clean up before test
        import paravis.core.indices.registry as reg
        reg._custom_indices.clear()

        @register_index(name="IDX_A", bands=["R"])
        def idx_a(R):
            return R

        @register_index(name="IDX_B", bands=["N", "R"])
        def idx_b(N, R):
            return N - R

        all_indices = list_custom_indices()
        assert "IDX_A" in all_indices
        assert "IDX_B" in all_indices

    def test_register_with_description(self):
        from paravis.core.indices.registry import register_index, list_custom_indices
        import paravis.core.indices.registry as reg
        reg._custom_indices.clear()

        @register_index(name="DESC_TEST", bands=["R"], description="Test index")
        def desc_test(R):
            return R * 2

        result = list_custom_indices()
        assert result["DESC_TEST"]["description"] == "Test index"

    def test_compute_custom_with_kwargs(self):
        from paravis.core.indices.registry import register_index, compute_custom_index
        import numpy as np
        import paravis.core.indices.registry as reg
        reg._custom_indices.clear()

        @register_index(name="MULTI_BAND", bands=["R", "G", "B"])
        def multi_band(R, G, B):
            return (R + G + B) / 3.0

        result = compute_custom_index("MULTI_BAND", {
            "R": np.array(1.0),
            "G": np.array(2.0),
            "B": np.array(3.0),
        })
        assert result == 2.0
