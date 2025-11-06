"""
Round-trip tests for Sentinel-2 pydantic-zarr integrated models.

These tests verify that Sentinel-2 data can be:
1. Loaded from zarr stores using from_zarr()
2. Validated through Pydantic models
3. Written back to zarr using to_zarr()
4. Round-tripped without data loss

Note: All tests use MemoryStore() for fast in-memory storage.
"""

import numpy as np
import pytest
from tests.test_data_api.conftest import SENTINEL2_EXAMPLES
import zarr
from zarr.storage import MemoryStore
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v2 import GroupSpec

from eopf_geozarr.data_api.sentinel2 import (
    Sentinel2CoordinateArray,
    Sentinel2DataArray,
    Sentinel2MeasurementsGroup,
    Sentinel2ReflectanceGroup,
    Sentinel2ResolutionDataset,
    Sentinel2Root,
    Sentinel2RootAttrs,
)
from eopf_geozarr.data_api.geozarr.common import DatasetAttrs


class TestSentinel2DataArrayZarr:
    """Test Sentinel2DataArray with zarr integration."""

    def test_from_array_basic(self):
        """Test creating data array from numpy array."""
        data = np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16)
        array = Sentinel2DataArray.from_band("b02", data)

        assert array.shape == (1, 100, 100)
        assert array.attributes.long_name == "Band B02"
        assert array.attributes.standard_name == "toa_bidirectional_reflectance"

    def test_array_to_zarr_round_trip(self):
        """Test writing and reading array from zarr store."""
        data = np.random.randint(0, 10000, (1, 50, 50), dtype=np.uint16)
        array = Sentinel2DataArray.from_band("b02", data)

        # Use in-memory store
        store = MemoryStore()

        # Write to zarr
        zarr_array = array.to_zarr(store, path='')
        zarr_array[:] = data

        # Read back
        loaded_array = Sentinel2DataArray.from_zarr(zarr_array)

        # Verify
        assert loaded_array.shape == array.shape
        assert loaded_array.attributes.long_name == array.attributes.long_name
        np.testing.assert_array_equal(np.array(zarr_array), data)

    def test_coordinate_arrays(self):
        """Test creating coordinate arrays."""
        x_vals = np.arange(600000, 601000, 10, dtype=np.float64)
        y_vals = np.arange(5095490, 5094490, -10, dtype=np.float64)
        time_vals = np.array([np.datetime64("2025-01-13T10:33:09")])

        x_coord = Sentinel2CoordinateArray.create_x_coordinate(x_vals)
        y_coord = Sentinel2CoordinateArray.create_y_coordinate(y_vals)
        time_coord = Sentinel2CoordinateArray.create_time_coordinate(time_vals)

        assert x_coord.shape == (100,)
        assert x_coord.attributes.standard_name == "projection_x_coordinate"

        assert y_coord.shape == (100,)

        assert time_coord.shape == (1,)


class TestSentinel2ResolutionDataset:
    """Test Sentinel2ResolutionDataset with zarr integration."""

    @pytest.fixture
    def sample_resolution_dataset(self):
        """Create a sample resolution dataset with bands and coordinates."""
        # Create coordinates
        x_vals = np.arange(600000, 600100, 10, dtype=np.float64)  # 10 pixels
        y_vals = np.arange(5095490, 5095390, -10, dtype=np.float64)  # 10 pixels
        time_vals = np.array([np.datetime64("2025-01-13T10:33:09")])

        x_coord = Sentinel2CoordinateArray.create_x_coordinate(x_vals)
        y_coord = Sentinel2CoordinateArray.create_y_coordinate(y_vals)
        time_coord = Sentinel2CoordinateArray.create_time_coordinate(time_vals)

        # Create bands
        b02 = Sentinel2DataArray.from_band(
            "b02",
            np.random.randint(0, 10000, (1, 10, 10), dtype=np.uint16),
        )
        b03 = Sentinel2DataArray.from_band(
            "b03",
            np.random.randint(0, 10000, (1, 10, 10), dtype=np.uint16),
        )

        # Create dataset
        dataset = Sentinel2ResolutionDataset(
            attributes=DatasetAttrs(),
            members={
                "x": x_coord,
                "y": y_coord,
                "time": time_coord,
                "b02": b02,
                "b03": b03,
            },
            resolution_level="r10m",
        )

        return dataset

    def test_get_bands(self, sample_resolution_dataset):
        """Test extracting bands from dataset."""
        bands = sample_resolution_dataset.get_bands()
        assert "b02" in bands
        assert "b03" in bands
        assert len(bands) == 2

    def test_get_coordinates(self, sample_resolution_dataset):
        """Test extracting coordinates from dataset."""
        coords = sample_resolution_dataset.get_coordinates()
        assert "x" in coords
        assert "y" in coords
        assert "time" in coords
        assert len(coords) == 3

    def test_resolution_dataset_to_zarr_round_trip(self, sample_resolution_dataset):
        """Test writing and reading resolution dataset from zarr."""
        # Use in-memory store
        store = MemoryStore()

        # Write to zarr
        zarr_group = sample_resolution_dataset.to_zarr(store, path='')

        # Verify zarr structure
        assert isinstance(zarr_group, zarr.Group)
        assert "b02" in zarr_group
        assert "b03" in zarr_group
        assert "x" in zarr_group
        assert "y" in zarr_group
        assert "time" in zarr_group

        # Read back using generic GroupSpec first
        loaded_untyped = GroupSpec.from_zarr(zarr_group)
        assert loaded_untyped.members is not None
        assert len(loaded_untyped.members) == 5  # 2 bands + 3 coords

        # Try to reload as typed dataset
        loaded_dataset = Sentinel2ResolutionDataset.from_zarr(zarr_group)
        assert loaded_dataset.members is not None
        assert len(loaded_dataset.get_bands()) == 2
        assert len(loaded_dataset.get_coordinates()) == 3


class TestSentinel2ReflectanceGroup:
    """Test Sentinel2ReflectanceGroup with zarr integration."""

    @pytest.fixture
    def sample_reflectance_group(self):
        """Create a reflectance group with multiple resolutions."""
        # Create minimal 10m dataset
        x_vals_10m = np.arange(600000, 600100, 10, dtype=np.float64)
        y_vals_10m = np.arange(5095490, 5095390, -10, dtype=np.float64)
        time_vals = np.array([np.datetime64("2025-01-13")])

        r10m_dataset = Sentinel2ResolutionDataset(
            attributes=DatasetAttrs(),
            members={
                "x": Sentinel2CoordinateArray.create_x_coordinate(x_vals_10m),
                "y": Sentinel2CoordinateArray.create_y_coordinate(y_vals_10m),
                "time": Sentinel2CoordinateArray.create_time_coordinate(time_vals),
                "b02": Sentinel2DataArray.from_band(
                    "b02",
                    np.random.randint(0, 10000, (1, 10, 10), dtype=np.uint16),
                ),
            },
            resolution_level="r10m",
        )

        # Create reflectance group
        reflectance = Sentinel2ReflectanceGroup(
            attributes=DatasetAttrs(),
            members={"r10m": r10m_dataset},
        )

        return reflectance

    def test_validate_at_least_one_resolution(self, sample_reflectance_group):
        """Test validation requires at least one native resolution."""
        # Should pass with r10m
        assert sample_reflectance_group.members is not None
        assert "r10m" in sample_reflectance_group.members

        # Should fail with no resolutions
        with pytest.raises(ValueError, match="[Aa]t least one native resolution"):
            Sentinel2ReflectanceGroup(
                attributes=DatasetAttrs(),
                members={},
            )

    def test_list_resolutions(self, sample_reflectance_group):
        """Test listing available resolutions."""
        resolutions = sample_reflectance_group.list_resolutions()
        assert "r10m" in resolutions

    def test_get_resolution_dataset(self, sample_reflectance_group):
        """Test getting specific resolution dataset."""
        r10m = sample_reflectance_group.get_resolution_dataset("r10m")
        assert r10m is not None
        assert isinstance(r10m, Sentinel2ResolutionDataset)

        r20m = sample_reflectance_group.get_resolution_dataset("r20m")
        assert r20m is None  # Not present

    def test_reflectance_group_to_zarr_round_trip(self, sample_reflectance_group):
        """Test writing and reading reflectance group from zarr."""
        # Use in-memory store
        store = MemoryStore()

        # Write to zarr
        zarr_group = sample_reflectance_group.to_zarr(store, path='')

        # Verify structure
        assert isinstance(zarr_group, zarr.Group)
        assert "r10m" in zarr_group
        assert isinstance(zarr_group["r10m"], zarr.Group)
        assert "b02" in zarr_group["r10m"]

        # Read back
        loaded = Sentinel2ReflectanceGroup.from_zarr(zarr_group)
        assert loaded.members is not None
        assert "r10m" in loaded.members


class TestSentinel2Root:
    """Test complete Sentinel2Root with zarr integration."""

    @pytest.fixture
    def sample_s2_root(self):
        """Create a minimal but complete Sentinel-2 root structure."""
        # Create a simple dataset
        x_vals = np.arange(600000, 600050, 10, dtype=np.float64)
        y_vals = np.arange(5095490, 5095440, -10, dtype=np.float64)
        time_vals = np.array([np.datetime64("2025-01-13")])

        r10m_dataset = Sentinel2ResolutionDataset(
            attributes=DatasetAttrs(),
            members={
                "x": Sentinel2CoordinateArray.create_x_coordinate(x_vals),
                "y": Sentinel2CoordinateArray.create_y_coordinate(y_vals),
                "time": Sentinel2CoordinateArray.create_time_coordinate(time_vals),
                "b02": Sentinel2DataArray.from_band(
                    "b02",
                    np.random.randint(0, 10000, (1, 5, 5), dtype=np.uint16),
                ),
            },
            resolution_level="r10m",
        )

        reflectance = Sentinel2ReflectanceGroup(
            attributes=DatasetAttrs(),
            members={"r10m": r10m_dataset},
        )

        measurements = Sentinel2MeasurementsGroup(
            attributes=DatasetAttrs(),
            members={"reflectance": reflectance},
        )

        # Create root with STAC metadata
        root_attrs = Sentinel2RootAttrs(
            Conventions="CF-1.7",
            title="Test Sentinel-2",
            stac_discovery={
                "properties": {
                    "mission": "sentinel-2a",
                    "datetime": "2025-01-13T10:33:09",
                }
            },
        )

        root = Sentinel2Root(
            attributes=root_attrs,
            members={"measurements": measurements},
        )

        return root

    def test_root_structure_validation(self, sample_s2_root):
        """Test root structure validation."""
        assert sample_s2_root.members is not None
        assert "measurements" in sample_s2_root.members

    def test_measurements_property(self, sample_s2_root):
        """Test accessing measurements property."""
        measurements = sample_s2_root.measurements
        assert measurements is not None
        assert isinstance(measurements, Sentinel2MeasurementsGroup)

    def test_list_available_bands(self, sample_s2_root):
        """Test listing available bands."""
        bands = sample_s2_root.list_available_bands()
        assert "b02" in bands

    def test_get_band_info(self, sample_s2_root):
        """Test getting band information."""
        band_info = sample_s2_root.get_band_info("b02")
        assert band_info is not None
        assert band_info.native_resolution == 10

    def test_validate_geozarr_compliance(self, sample_s2_root):
        """Test GeoZarr compliance validation."""
        compliance = sample_s2_root.validate_geozarr_compliance()
        assert compliance["has_measurements"] is True
        assert compliance["has_bands"] is True
        assert compliance["has_valid_hierarchy"] is True
        assert compliance["has_coordinates"] is True

    def test_root_to_zarr_round_trip(self, sample_s2_root):
        """Test complete round-trip: model -> zarr -> model."""
        # Use in-memory store
        store = MemoryStore()

        # Write to zarr
        zarr_group = sample_s2_root.to_zarr(store, path='')

        # Verify zarr structure
        assert isinstance(zarr_group, zarr.Group)
        assert "measurements" in zarr_group
        assert "measurements/reflectance" in dict(zarr_group.members(max_depth=None))
        assert "measurements/reflectance/r10m" in dict(zarr_group.members(max_depth=None))

        # Read back as untyped first
        loaded_untyped = GroupSpec.from_zarr(zarr_group)
        assert loaded_untyped.members is not None

        # Try to reload as typed Sentinel2Root
        loaded_root = Sentinel2Root.from_zarr(zarr_group)
        assert loaded_root.measurements is not None
        assert loaded_root.measurements.reflectance is not None

        # Verify data integrity
        bands_before = sample_s2_root.list_available_bands()
        bands_after = loaded_root.list_available_bands()
        assert set(bands_before) == set(bands_after)

    def test_model_dump_round_trip(self, sample_s2_root):
        """Test model serialization round-trip."""
        # Dump to dict
        dumped = sample_s2_root.model_dump()
        assert isinstance(dumped, dict)
        assert "attributes" in dumped
        assert "members" in dumped

        # Reconstruct from dict (with tuplification)
        reconstructed_data = tuplify_json(dumped)
        reconstructed = Sentinel2Root(**reconstructed_data)

        # Verify structure is preserved
        assert reconstructed.measurements is not None
        assert len(reconstructed.list_available_bands()) == len(sample_s2_root.list_available_bands())

    def test_invalid_mission_rejected(self):
        """Test that non-Sentinel-2 mission is rejected."""
        # Create minimal structure with wrong mission
        x_vals = np.array([600000.0])
        r10m_dataset = Sentinel2ResolutionDataset(
            attributes=DatasetAttrs(),
            members={
                "x": Sentinel2CoordinateArray.create_x_coordinate(x_vals),
                "b02": Sentinel2DataArray.from_band(
                    "b02",
                    np.array([[[1000]]], dtype=np.uint16),
                ),
            },
        )

        reflectance = Sentinel2ReflectanceGroup(
            attributes=DatasetAttrs(),
            members={"r10m": r10m_dataset},
        )

        measurements = Sentinel2MeasurementsGroup(
            attributes=DatasetAttrs(),
            members={"reflectance": reflectance},
        )

        root_attrs = Sentinel2RootAttrs(
            stac_discovery={
                "properties": {
                    "mission": "sentinel-1a",  # Wrong mission!
                }
            }
        )

        with pytest.raises(ValueError, match="STAC.*must indicate Sentinel-2"):
            Sentinel2Root(
                attributes=root_attrs,
                members={"measurements": measurements},
            )


@pytest.mark.slow
def test_load_real_sentinel2_example(example_group):
    """Test loading the real sentinel_2.json example.

    This uses the fixture from conftest.py that loads the actual
    Sentinel-2 example metadata.
    """
    # Load as generic GroupSpec first
    generic_model = GroupSpec.from_zarr(example_group)
    assert generic_model.members is not None

    # Inspect structure
    flat = generic_model.to_flat()
    print(f"\nFound {len(flat)} items in flattened hierarchy")

    # Look for key paths
    paths = list(flat.keys())
    print(f"Sample paths: {paths[:10]}")

    # Check for expected Sentinel-2 structure
    s2_paths = [p for p in paths if "measurements" in p or "reflectance" in p or "quality" in p]
    print(f"Sentinel-2 related paths: {len(s2_paths)}")

    # This test verifies we can at least load the structure
    # Full typed loading would require the example to have actual array data
    assert len(flat) > 0

@pytest.mark.parametrize('example', SENTINEL2_EXAMPLES)
def test_sentinel2_roundtrip(example: dict[str, object]) -> None:
    model = Sentinel2Root(**example)
    assert model.to_dict() == example
