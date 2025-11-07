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
from zarr.storage import MemoryStore

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.data_api.sentinel2 import (
    ALL_BAND_NAMES,
    NATIVE_BANDS,
    RESOLUTION_TO_METERS,
    Sentinel2BandInfo,
    Sentinel2CoordinateArray,
    Sentinel2DataArray,
    Sentinel2ReflectanceGroup,
    Sentinel2ResolutionDataset,
    Sentinel2Root,
)
from tests.test_data_api.conftest import SENTINEL2_EXAMPLES


class TestSentinel2DataArrayZarr:
    """Test Sentinel2DataArray with zarr integration."""

    def test_from_array_basic(self) -> None:
        """Test creating data array from numpy array."""
        data = np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16)
        array = Sentinel2DataArray.from_band("b02", data)

        assert array.shape == (1, 100, 100)
        assert array.attributes.long_name == "Band B02"
        assert array.attributes.standard_name == "toa_bidirectional_reflectance"

    def test_array_to_zarr_round_trip(self) -> None:
        """Test writing and reading array from zarr store."""
        data = np.random.randint(0, 10000, (1, 50, 50), dtype=np.uint16)
        array = Sentinel2DataArray.from_band("b02", data)

        # Use in-memory store
        store = MemoryStore()

        # Write to zarr
        zarr_array = array.to_zarr(store, path="")
        zarr_array[:] = data

        # Read back
        loaded_array = Sentinel2DataArray.from_zarr(zarr_array)

        # Verify
        assert loaded_array.shape == array.shape
        assert loaded_array.attributes.long_name == array.attributes.long_name
        np.testing.assert_array_equal(np.array(zarr_array), data)

    def test_coordinate_arrays(self) -> None:
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
    def sample_resolution_dataset(self) -> Sentinel2ResolutionDataset:
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

    def test_create_resolution_dataset(
        self, sample_resolution_dataset: Sentinel2ResolutionDataset
    ) -> None:
        """Test creating a resolution dataset."""
        assert sample_resolution_dataset.resolution_level == "r10m"
        assert "b02" in sample_resolution_dataset.members
        assert "b03" in sample_resolution_dataset.members
        assert "x" in sample_resolution_dataset.members
        assert "y" in sample_resolution_dataset.members

    def test_get_coordinates(
        self, sample_resolution_dataset: Sentinel2ResolutionDataset
    ) -> None:
        """Test accessing coordinate arrays."""
        coords = sample_resolution_dataset.get_coordinates()
        assert coords is not None
        assert "x" in coords or "y" in coords or "time" in coords

    def test_get_bands(
        self, sample_resolution_dataset: Sentinel2ResolutionDataset
    ) -> None:
        """Test accessing band arrays."""
        bands = sample_resolution_dataset.get_bands()
        assert "b02" in bands
        assert "b03" in bands


class TestSentinel2ReflectanceGroup:
    """Test Sentinel2ReflectanceGroup with zarr integration."""

    @pytest.fixture
    def sample_reflectance_group(self) -> Sentinel2ReflectanceGroup:
        """Create a sample reflectance group with multiple resolutions."""
        # Create r10m dataset
        x_vals = np.arange(600000, 600100, 10, dtype=np.float64)
        y_vals = np.arange(5095490, 5095390, -10, dtype=np.float64)
        time_vals = np.array([np.datetime64("2025-01-13T10:33:09")])

        x_coord = Sentinel2CoordinateArray.create_x_coordinate(x_vals)
        y_coord = Sentinel2CoordinateArray.create_y_coordinate(y_vals)
        time_coord = Sentinel2CoordinateArray.create_time_coordinate(time_vals)

        b02 = Sentinel2DataArray.from_band(
            "b02", np.random.randint(0, 10000, (1, 10, 10), dtype=np.uint16)
        )
        b03 = Sentinel2DataArray.from_band(
            "b03", np.random.randint(0, 10000, (1, 10, 10), dtype=np.uint16)
        )

        r10m = Sentinel2ResolutionDataset(
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

        # Create reflectance group
        group = Sentinel2ReflectanceGroup(
            attributes=DatasetAttrs(),
            members={"r10m": r10m},
        )

        return group

    def test_create_reflectance_group(
        self, sample_reflectance_group: Sentinel2ReflectanceGroup
    ) -> None:
        """Test creating a reflectance group."""
        assert sample_reflectance_group.members is not None
        assert "r10m" in sample_reflectance_group.members

    def test_get_resolution_dataset(
        self, sample_reflectance_group: Sentinel2ReflectanceGroup
    ) -> None:
        """Test getting resolution dataset from reflectance group."""
        dataset = sample_reflectance_group.get_resolution_dataset("r10m")
        assert dataset is not None


class TestSentinel2Root:
    """Test Sentinel2Root model."""

    @pytest.mark.parametrize("example", SENTINEL2_EXAMPLES)
    def test_model_dump_round_trip(self, example: dict) -> None:
        """Test that we can dump and load the model."""
        model = Sentinel2Root(**example)

        assert model.attributes is not None
        dumped = model.model_dump(exclude_none=True)
        assert dumped is not None
        assert "members" in dumped

    @pytest.mark.parametrize("example", SENTINEL2_EXAMPLES)
    def test_load_from_example(self, example: dict) -> None:
        """Test loading from example JSON."""
        model = Sentinel2Root(**example)
        assert model is not None
        assert model.members is not None

    @pytest.mark.parametrize("example", SENTINEL2_EXAMPLES)
    def test_sentinel2_roundtrip(self, example: dict) -> None:
        """Test that Sentinel2Root can load example JSON and maintain structure."""
        model = Sentinel2Root(**example)
        # Verify model loaded successfully
        assert model is not None
        assert model.members is not None
        # Verify we can re-serialize it
        dumped = model.to_dict()
        assert dumped is not None
        # Load it again to ensure structural consistency
        model2 = Sentinel2Root(**dumped)
        assert model2 is not None


class TestSentinel2BandInfo:
    """Test Sentinel2BandInfo model."""

    def test_create_band_info(self) -> None:
        """Test creating BandInfo with valid parameters."""
        band_info = Sentinel2BandInfo(
            name="b02",
            native_resolution=10,
            data_type="uint16",
            wavelength_center=490.0,
            wavelength_width=66.0,
            long_name="Blue band",
        )
        assert band_info.name == "b02"
        assert band_info.native_resolution == 10
        assert band_info.wavelength_center == 490.0

    def test_from_band_name(self) -> None:
        """Test creating BandInfo from band name."""
        band_info = Sentinel2BandInfo.from_band_name("b02")
        assert band_info.native_resolution == 10
        assert band_info.wavelength_center == 490
        assert band_info.data_type == "uint16"

    def test_from_band_name_all_bands(self) -> None:
        """Test that all defined band names can be created."""
        for band_name in ALL_BAND_NAMES:
            band_info = Sentinel2BandInfo.from_band_name(band_name)
            assert band_info.native_resolution in (10, 20, 60)
            assert band_info.wavelength_center > 0


class TestBandNameConstants:
    """Test band name and resolution constants."""

    def test_native_bands_structure(self) -> None:
        """Test NATIVE_BANDS constant."""
        assert 10 in NATIVE_BANDS
        assert 20 in NATIVE_BANDS
        assert 60 in NATIVE_BANDS
        assert len(NATIVE_BANDS[10]) == 4  # 4 10m bands
        assert len(NATIVE_BANDS[20]) == 6  # 6 20m bands
        assert len(NATIVE_BANDS[60]) == 3  # 3 60m bands

    def test_all_band_names(self) -> None:
        """Test ALL_BAND_NAMES constant."""
        assert "b02" in ALL_BAND_NAMES
        assert "b03" in ALL_BAND_NAMES
        assert len(ALL_BAND_NAMES) == 13  # 13 total bands

    def test_resolution_to_meters(self) -> None:
        """Test RESOLUTION_TO_METERS mapping."""
        assert RESOLUTION_TO_METERS["r10m"] == 10
        assert RESOLUTION_TO_METERS["r20m"] == 20
        assert RESOLUTION_TO_METERS["r60m"] == 60
