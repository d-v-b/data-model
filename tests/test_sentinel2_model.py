"""
Tests for Sentinel-2 Pydantic data models.

This test module validates the declarative Pydantic models for Sentinel-2
data structure, ensuring proper validation, type checking, and compliance
with the EOPF hierarchy.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from eopf_geozarr.data_api.sentinel2 import (
    BandName,
    ResolutionLevel,
    Sentinel2BandInfo,
    Sentinel2CoordinateArray,
    Sentinel2DataArray,
    Sentinel2DataArrayAttrs,
    Sentinel2QualityGroup,
    Sentinel2ReflectanceGroup,
    Sentinel2ResolutionDataset,
    Sentinel2Root,
    Sentinel2RootAttrs,
)
from eopf_geozarr.data_api.geozarr.common import DatasetAttrs

class TestSentinel2BandInfo:
    """Test Sentinel2BandInfo model."""

    def test_create_band_info(self) -> None:
        """Test creating BandInfo with valid parameters."""
        band_info = Sentinel2BandInfo(
            name='b02',
            native_resolution=10,
            data_type="uint16",
            wavelength_center=490.0,
            wavelength_width=66.0,
            long_name="Blue band",
        )
        assert band_info.name == 'b02'
        assert band_info.native_resolution == 10
        assert band_info.wavelength_center == 490.0

    def test_from_band_name(self) -> None:
        """Test creating BandInfo from band name."""
        band_info = Sentinel2BandInfo.from_band_name('b02')
        assert band_info.native_resolution == 10
        assert band_info.wavelength_center == 490
        assert band_info.data_type == "uint16"

    def test_invalid_dtype(self) -> None:
        """Test that invalid data type raises ValidationError."""
        with pytest.raises(ValidationError):
            Sentinel2BandInfo(
                name=BandName.B02,
                native_resolution=10,
                data_type="invalid_dtype",
                wavelength_center=490.0,
                wavelength_width=66.0,
                long_name="Blue",
            )

    def test_negative_wavelength(self) -> None:
        """Test that negative wavelength raises ValidationError."""
        with pytest.raises(ValidationError):
            Sentinel2BandInfo(
                name=BandName.B02,
                native_resolution=10,
                data_type="uint16",
                wavelength_center=-490.0,
                wavelength_width=66.0,
                long_name="Blue",
            )


class TestCoordinateArray:
    """Test CoordinateArray model."""

    def test_x_coordinate(self) -> None:
        """Test X coordinate creation."""
        x_coord = CoordinateArray(
            name="x",
            values=[600000.0, 600010.0, 600020.0],
            units="m",
            standard_name="projection_x_coordinate",
            long_name="x coordinate of projection",
            axis="X",
        )
        assert x_coord.name == "x"
        assert len(x_coord.values) == 3
        assert x_coord.axis == "X"

    def test_invalid_values(self) -> None:
        """Test that invalid values raise ValidationError."""
        with pytest.raises(ValidationError):
            CoordinateArray(
                name="x",
                values="invalid",  # Should be array-like
                units="m",
                standard_name="projection_x_coordinate",
                long_name="x coordinate",
                axis="X",
            )


class TestSentinel2Coordinates:
    """Test Sentinel2Coordinates model."""

    @pytest.fixture
    def valid_crs(self) -> None:
        """Create valid CRS for testing."""
        return ProjAttrs(
            code="EPSG:32632",
            bbox=(600000.0, 5090000.0, 605490.0, 5095490.0),
            transform=(10.0, 0.0, 600000.0, 0.0, -10.0, 5095490.0),
            spatial_dimensions=("x", "y"),
        )

    def test_create_coordinates(self, valid_crs) -> None:
        """Test creating coordinate system."""
        # Create exactly 10m spacing: 549 pixels * 10m = 5490m
        x_vals = list(np.arange(600000, 600000 + 549 * 10, 10))
        y_vals = list(np.arange(5095490, 5095490 - 549 * 10, -10))

        coords = Sentinel2Coordinates(
            x=CoordinateArray(
                name="x",
                values=x_vals,
                units="m",
                standard_name="projection_x_coordinate",
                long_name="x coordinate",
                axis="X",
            ),
            y=CoordinateArray(
                name="y",
                values=y_vals,
                units="m",
                standard_name="projection_y_coordinate",
                long_name="y coordinate",
                axis="Y",
            ),
            time=CoordinateArray(
                name="time",
                values=[np.datetime64("2025-01-13T10:33:09")],
                units="seconds since 1970-01-01",
                standard_name="time",
                long_name="time",
                axis="T",
            ),
            crs=valid_crs,
            resolution_meters=10,
        )
        assert coords.resolution_meters == 10
        assert len(coords.x.values) == 549

    def test_empty_coordinates_fail(self, valid_crs) -> None:
        """Test that empty coordinate arrays raise ValidationError."""
        with pytest.raises(ValidationError):
            Sentinel2Coordinates(
                x=CoordinateArray(
                    name="x",
                    values=[],  # Empty
                    units="m",
                    standard_name="projection_x_coordinate",
                    long_name="x coordinate",
                    axis="X",
                ),
                y=CoordinateArray(
                    name="y",
                    values=[1.0],
                    units="m",
                    standard_name="projection_y_coordinate",
                    long_name="y coordinate",
                    axis="Y",
                ),
                time=CoordinateArray(
                    name="time",
                    values=[np.datetime64("2025-01-13")],
                    units="seconds since 1970-01-01",
                    standard_name="time",
                    long_name="time",
                    axis="T",
                ),
                crs=valid_crs,
                resolution_meters=10,
            )


class TestSentinel2DataArray:
    """Test Sentinel2DataArray model."""

    def test_create_data_array(self) -> None:
        """Test creating a data array."""
        attrs = DataArrayAttributes(
            long_name="Blue band",
            standard_name="toa_bidirectional_reflectance",
            units="1",
            grid_mapping="crs",
        )
        data_array = Sentinel2DataArray(
            name="b02",
            shape=(1, 549, 549),
            dtype="uint16",
            chunks=(1, 256, 256),
            attributes=attrs,
        )
        assert data_array.name == "b02"
        assert data_array.shape == (1, 549, 549)

    def test_invalid_shape(self):
        """Test that invalid shape raises ValidationError."""
        with pytest.raises(ValidationError):
            Sentinel2DataArray(
                name="b02",
                shape=(549, 549),  # Should be 3D
                dtype="uint16",
                attributes=DataArrayAttributes(long_name="Blue"),
            )

    def test_invalid_chunks(self) -> None:
        """Test that chunks larger than shape raise ValidationError."""
        with pytest.raises(ValidationError):
            Sentinel2DataArray(
                name="b02",
                shape=(1, 549, 549),
                dtype="uint16",
                chunks=(1, 1000, 1000),  # Larger than shape
                attributes=DataArrayAttributes(long_name="Blue"),
            )


class TestSentinel2ResolutionDataset:
    """Test Sentinel2ResolutionDataset model."""

    @pytest.fixture
    def valid_coords(self) -> None:
        """Create valid coordinates for testing."""
        crs = ProjAttrs(
            code="EPSG:32632",
            bbox=(600000.0, 5090000.0, 605490.0, 5095490.0),
            spatial_dimensions=("x", "y"),
        )
        return Sentinel2Coordinates(
            x=CoordinateArray(
                name="x",
                values=[600000.0, 600010.0],
                units="m",
                standard_name="projection_x_coordinate",
                long_name="x",
                axis="X",
            ),
            y=CoordinateArray(
                name="y",
                values=[5095490.0, 5095480.0],
                units="m",
                standard_name="projection_y_coordinate",
                long_name="y",
                axis="Y",
            ),
            time=CoordinateArray(
                name="time",
                values=[np.datetime64("2025-01-13")],
                units="seconds since 1970-01-01",
                standard_name="time",
                long_name="time",
                axis="T",
            ),
            crs=crs,
            resolution_meters=10,
        )

    def test_create_resolution_dataset(self, valid_coords) -> None:
        """Test creating a resolution dataset."""
        dataset = Sentinel2ResolutionDataset(
            resolution=ResolutionLevel.R10M,
            coordinates=valid_coords,
            bands={
                "b02": Sentinel2DataArray(
                    name="b02",
                    shape=(1, 2, 2),
                    dtype="uint16",
                    attributes=DataArrayAttributes(long_name="Blue"),
                )
            },
        )
        assert dataset.resolution == ResolutionLevel.R10M
        assert "b02" in dataset.bands

    def test_invalid_bands_for_resolution(self, valid_coords) -> None:
        """Test that invalid bands for resolution raise ValidationError."""
        with pytest.raises(ValidationError):
            Sentinel2ResolutionDataset(
                resolution=ResolutionLevel.R10M,
                coordinates=valid_coords,
                bands={
                    "b05": Sentinel2DataArray(  # B05 is 20m, not 10m
                        name="b05",
                        shape=(1, 2, 2),
                        dtype="uint16",
                        attributes=DataArrayAttributes(long_name="Red Edge"),
                    )
                },
            )


class TestSentinel2ReflectanceGroup:
    """Test Sentinel2ReflectanceGroup model."""

    def test_at_least_one_resolution_required(self) -> None:
        """Test that at least one native resolution is required."""
        with pytest.raises(ValidationError):
            Sentinel2ReflectanceGroup()  # No resolutions provided

    def test_get_all_bands(self) -> None:
        """Test getting all bands across resolutions."""
        crs = ProjAttrs(code="EPSG:32632", spatial_dimensions=("x", "y"))
        coords = Sentinel2Coordinates(
            x=CoordinateArray(
                name="x", values=[1.0], units="m", standard_name="projection_x_coordinate", long_name="x", axis="X"
            ),
            y=CoordinateArray(
                name="y", values=[1.0], units="m", standard_name="projection_y_coordinate", long_name="y", axis="Y"
            ),
            time=CoordinateArray(
                name="time",
                values=[np.datetime64("2025-01-13")],
                units="seconds",
                standard_name="time",
                long_name="time",
                axis="T",
            ),
            crs=crs,
            resolution_meters=10,
        )

        r10m = Sentinel2ResolutionDataset(
            resolution=ResolutionLevel.R10M,
            coordinates=coords,
            bands={
                "b02": Sentinel2DataArray(
                    name="b02", shape=(1, 1, 1), dtype="uint16", attributes=DataArrayAttributes(long_name="Blue")
                )
            },
        )

        group = Sentinel2ReflectanceGroup(r10m=r10m)
        all_bands = group.get_all_bands()
        assert "b02" in all_bands


class TestSentinel2QualityGroup:
    """Test Sentinel2QualityGroup model."""

    def test_create_quality_group(self) -> None:
        """Test creating quality group."""
        atm = Sentinel2AtmosphereData(
            aot=Sentinel2DataArray(
                name="aot",
                shape=(1, 275, 275),
                dtype="uint16",
                attributes=DataArrayAttributes(long_name="AOT"),
            )
        )
        quality = Sentinel2QualityGroup(atmosphere=atm)
        assert quality.atmosphere is not None
        assert quality.atmosphere.aot.name == "aot"


class TestSentinel2DataTree:
    """Test complete Sentinel2DataTree model."""

    @pytest.fixture
    def minimal_datatree(self) -> None:
        """Create minimal valid Sentinel2DataTree."""
        crs = ProjAttrs(code="EPSG:32632", spatial_dimensions=("x", "y"))
        coords = Sentinel2Coordinates(
            x=CoordinateArray(
                name="x", values=[600000.0], units="m", standard_name="projection_x_coordinate", long_name="x", axis="X"
            ),
            y=CoordinateArray(
                name="y", values=[5095490.0], units="m", standard_name="projection_y_coordinate", long_name="y", axis="Y"
            ),
            time=CoordinateArray(
                name="time",
                values=[np.datetime64("2025-01-13")],
                units="seconds",
                standard_name="time",
                long_name="time",
                axis="T",
            ),
            crs=crs,
            resolution_meters=10,
        )

        r10m = Sentinel2ResolutionDataset(
            resolution=ResolutionLevel.R10M,
            coordinates=coords,
            bands={
                "b02": Sentinel2DataArray(
                    name="b02",
                    shape=(1, 1, 1),
                    dtype="uint16",
                    attributes=DataArrayAttributes(
                        long_name="Blue",
                        standard_name="toa_bidirectional_reflectance",
                        grid_mapping="crs",
                        array_dimensions=["time", "y", "x"],
                    ),
                )
            },
        )

        measurements = Sentinel2ReflectanceGroup(r10m=r10m)

        stac = STACDiscoveryProperties(
            mission="sentinel-2a",
            datetime="2025-01-13T10:33:09",
            processing_level="L1C",
        )

        attrs = Sentinel2RootAttributes(
            stac_discovery={"properties": stac.model_dump()},
            title="Sentinel-2 L1C",
        )

        return Sentinel2DataTree(
            attributes=attrs,
            measurements=measurements,
        )

    def test_create_minimal_datatree(self, minimal_datatree) -> None:
        """Test creating minimal valid DataTree."""
        assert minimal_datatree.zarr_format == 3
        assert minimal_datatree.measurements is not None

    def test_list_available_bands(self, minimal_datatree):
        """Test listing available bands."""
        bands = minimal_datatree.list_available_bands()
        assert "b02" in bands

    def test_get_band_info(self, minimal_datatree):
        """Test getting band information."""
        band_info = minimal_datatree.get_band_info("b02")
        assert band_info is not None
        assert band_info.native_resolution == 10

    def test_get_native_resolution(self, minimal_datatree):
        """Test getting native resolution for band."""
        res = minimal_datatree.get_native_resolution("b02")
        assert res == 10

    def test_validate_geozarr_compliance(self, minimal_datatree):
        """Test GeoZarr compliance validation."""
        results = minimal_datatree.validate_geozarr_compliance()
        assert results["has_crs_info"] is True
        assert results["has_array_dimensions"] is True
        assert results["has_grid_mapping"] is True

    def test_no_bands_fails(self):
        """Test that DataTree without bands fails validation."""
        attrs = Sentinel2RootAttributes(
            stac_discovery={
                "properties": {
                    "mission": "sentinel-2a",
                    "datetime": "2025-01-13",
                }
            }
        )
        crs = ProjAttrs(code="EPSG:32632", spatial_dimensions=("x", "y"))
        coords = Sentinel2Coordinates(
            x=CoordinateArray(
                name="x", values=[1.0], units="m", standard_name="projection_x_coordinate", long_name="x", axis="X"
            ),
            y=CoordinateArray(
                name="y", values=[1.0], units="m", standard_name="projection_y_coordinate", long_name="y", axis="Y"
            ),
            time=CoordinateArray(
                name="time", values=[np.datetime64("2025-01-13")], units="s", standard_name="time", long_name="t", axis="T"
            ),
            crs=crs,
            resolution_meters=10,
        )
        r10m = Sentinel2ResolutionDataset(
            resolution=ResolutionLevel.R10M,
            coordinates=coords,
            bands={},  # No bands
        )

        with pytest.raises(ValidationError):
            Sentinel2DataTree(
                attributes=attrs,
                measurements=Sentinel2ReflectanceGroup(r10m=r10m),
            )

    def test_invalid_mission_fails(self) -> None:
        """Test that non-Sentinel-2 mission fails validation."""
        with pytest.raises(ValidationError):
            STACDiscoveryProperties(
                mission="sentinel-1a",  # Wrong mission
                datetime="2025-01-13",
            )

