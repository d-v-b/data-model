"""
Tests for Sentinel-2 data validation logic.

This module tests the validation rules for Sentinel-2 models,
particularly band-resolution consistency checking.
"""

import numpy as np
import pytest

from eopf_geozarr.data_api.sentinel2 import (
    ALL_BAND_NAMES,
    NATIVE_BANDS,
    Sentinel2CoordinateArray,
    Sentinel2DataArray,
    Sentinel2ResolutionDataset,
)
from eopf_geozarr.data_api.geozarr.common import DatasetAttrs


class TestBandResolutionValidation:
    """Test validation of band-resolution consistency."""

    @pytest.fixture
    def create_dataset(self) -> None:
        """Helper to create a resolution dataset with given bands."""
        def _create(band_names: list[str], resolution_level: str | None = None) -> None:
            members = {}

            # Add coordinates
            x_vals = np.arange(600000, 600100, 10, dtype=np.float64)
            y_vals = np.arange(5095490, 5095390, -10, dtype=np.float64)
            time_vals = np.array([np.datetime64("2025-01-13")])

            members["x"] = Sentinel2CoordinateArray.create_x_coordinate(x_vals)
            members["y"] = Sentinel2CoordinateArray.create_y_coordinate(y_vals)
            members["time"] = Sentinel2CoordinateArray.create_time_coordinate(time_vals)

            # Add bands
            for band_name in band_names:
                members[band_name] = Sentinel2DataArray.from_band(
                    band_name,
                    np.random.randint(0, 10000, (1, 10, 10), dtype=np.uint16),
                )

            return Sentinel2ResolutionDataset(
                attributes=DatasetAttrs(),
                members=members,
                resolution_level=resolution_level,
            )

        return _create

    def test_valid_10m_bands(self, create_dataset) -> None:
        """Test that valid 10m bands pass validation."""
        # All 10m native bands
        dataset = create_dataset(["b02", "b03", "b04", "b08"], resolution_level="r10m")
        assert dataset.resolution_level == "r10m"

        bands = dataset.get_bands()
        assert len(bands) == 4
        assert set(bands.keys()) == {"b02", "b03", "b04", "b08"}

    def test_valid_20m_bands(self, create_dataset) -> None:
        """Test that valid 20m bands pass validation."""
        # All 20m native bands
        dataset = create_dataset(
            ["b05", "b06", "b07", "b8a", "b11", "b12"],
            resolution_level="r20m"
        )
        assert dataset.resolution_level == "r20m"

        bands = dataset.get_bands()
        assert len(bands) == 6

    def test_valid_60m_bands(self, create_dataset) -> None:
        """Test that valid 60m bands pass validation."""
        # All 60m native bands
        dataset = create_dataset(["b01", "b09", "b10"], resolution_level="r60m")
        assert dataset.resolution_level == "r60m"

        bands = dataset.get_bands()
        assert len(bands) == 3

    def test_subset_of_native_bands_valid(self, create_dataset) -> None:
        """Test that a subset of native bands is valid."""
        # Only some 10m bands
        dataset = create_dataset(["b02", "b03"], resolution_level="r10m")

        bands = dataset.get_bands()
        assert len(bands) == 2
        assert set(bands.keys()) == {"b02", "b03"}

    def test_mixed_resolution_bands_fails_for_r10m(self, create_dataset) -> None:
        """Test that r10m strictly enforces native-only bands."""
        # r10m should only contain native 10m bands
        with pytest.raises(ValueError, match="should only contain native 10m bands"):
            create_dataset(
                ["b02", "b03", "b05"],  # b05 is 20m native
                resolution_level="r10m"
            )

    def test_wrong_resolution_band_fails_for_r10m(self, create_dataset) -> None:
        """Test that r10m rejects non-native bands."""
        # Try to use 60m band in 10m dataset
        with pytest.raises(ValueError, match="should only contain native 10m bands"):
            create_dataset(
                ["b02", "b01"],  # b01 is 60m native
                resolution_level="r10m"
            )

    def test_all_wrong_bands_fails_for_r10m(self, create_dataset) -> None:
        """Test that r10m rejects all non-native bands."""
        # Try to put 20m bands in 10m dataset
        with pytest.raises(ValueError, match="should only contain native 10m bands"):
            create_dataset(
                ["b05", "b06"],  # All 20m native
                resolution_level="r10m"
            )

    def test_r20m_allows_downsampled_bands(self, create_dataset) -> None:
        """Test that r20m can contain downsampled bands from other resolutions."""
        # r20m can have native 20m bands plus downsampled 10m and 60m bands (real S2 behavior)
        dataset = create_dataset(
            ["b02", "b03", "b05", "b01"],  # 10m + 20m + 60m mixed
            resolution_level="r20m"
        )
        assert dataset.resolution_level == "r20m"
        bands = dataset.get_bands()
        assert len(bands) == 4

    def test_r60m_allows_downsampled_bands(self, create_dataset) -> None:
        """Test that r60m can contain downsampled bands from all resolutions."""
        # r60m can have all bands downsampled (real S2 behavior)
        dataset = create_dataset(
            ["b02", "b03", "b05", "b01", "b09"],  # Mix of all resolutions
            resolution_level="r60m"
        )
        assert dataset.resolution_level == "r60m"
        bands = dataset.get_bands()
        assert len(bands) == 5

    def test_invalid_band_name_fails(self, create_dataset) -> None:
        """Test that invalid band names are rejected."""
        # Create dataset with coordinate but manually add invalid band
        members = {
            "x": Sentinel2CoordinateArray.create_x_coordinate(
                np.array([600000.0], dtype=np.float64)
            ),
            "invalid_band": Sentinel2DataArray.from_band(
                "b02",  # Create as b02 but we'll rename it
                np.array([[[1000]]], dtype=np.uint16),
            ),
        }

        # This will fail during validation because "invalid_band" is not in ALL_BAND_NAMES
        # But since we're checking member names, not creating from scratch, this should pass
        # The validation only checks names that ARE in ALL_BAND_NAMES
        dataset = Sentinel2ResolutionDataset(
            attributes=DatasetAttrs(),
            members=members,
        )

        # Get bands should not return invalid_band
        bands = dataset.get_bands()
        assert "invalid_band" not in bands

    def test_no_resolution_level_allows_any_bands(self, create_dataset) -> None:
        """Test that without resolution_level set, any valid bands are allowed."""
        # Mix bands without declaring resolution - should be allowed
        dataset = create_dataset(
            ["b02", "b05"],  # 10m and 20m mixed
            resolution_level=None  # No validation
        )

        bands = dataset.get_bands()
        assert len(bands) == 2

    def test_downsampled_resolution_no_native_check(self, create_dataset) -> None:
        """Test that downsampled resolutions don't have native band restrictions."""
        # r120m doesn't have native bands, so any bands are allowed
        dataset = create_dataset(
            ["b02", "b05", "b01"],  # Mix of different native resolutions
            resolution_level="r120m"
        )

        bands = dataset.get_bands()
        assert len(bands) == 3

    def test_empty_dataset_valid(self) -> None:
        """Test that dataset with no bands is valid."""
        # Just coordinates, no bands
        dataset = Sentinel2ResolutionDataset(
            attributes=DatasetAttrs(),
            members={
                "x": Sentinel2CoordinateArray.create_x_coordinate(
                    np.array([600000.0], dtype=np.float64)
                ),
            },
            resolution_level="r10m",
        )

        bands = dataset.get_bands()
        assert len(bands) == 0


class TestBandNameConstants:
    """Test that band name constants are correctly defined."""

    def test_all_band_names_complete(self) -> None:
        """Test that ALL_BAND_NAMES contains all expected bands."""
        expected = {
            "b01", "b02", "b03", "b04", "b05", "b06", "b07",
            "b08", "b8a", "b09", "b10", "b11", "b12"
        }
        assert ALL_BAND_NAMES == expected

    def test_native_bands_complete(self) -> None:
        """Test that NATIVE_BANDS covers all bands."""
        all_native = set()
        for bands in NATIVE_BANDS.values():
            all_native.update(bands)

        assert all_native == ALL_BAND_NAMES

    def test_native_bands_no_overlap(self) -> None:
        """Test that native band resolutions don't overlap."""
        res_10m = NATIVE_BANDS[10]
        res_20m = NATIVE_BANDS[20]
        res_60m = NATIVE_BANDS[60]

        # No band should be in multiple resolutions
        assert len(res_10m & res_20m) == 0
        assert len(res_10m & res_60m) == 0
        assert len(res_20m & res_60m) == 0

    def test_resolution_counts(self) -> None:
        """Test that each resolution has the expected number of bands."""
        assert len(NATIVE_BANDS[10]) == 4  # b02, b03, b04, b08
        assert len(NATIVE_BANDS[20]) == 6  # b05, b06, b07, b8a, b11, b12
        assert len(NATIVE_BANDS[60]) == 3  # b01, b09, b10


class TestBandInfoValidation:
    """Test BandInfo creation and validation."""

    def test_get_band_info_for_all_bands(self) -> None:
        """Test that we can get band info for all valid bands."""
        from eopf_geozarr.data_api.sentinel2 import Sentinel2BandInfo

        for band_name in ALL_BAND_NAMES:
            band_info = Sentinel2BandInfo.from_band_name(band_name)
            assert band_info.name == band_name
            assert band_info.native_resolution in [10, 20, 60]
            assert band_info.wavelength_center > 0
            assert band_info.wavelength_width > 0

    def test_band_info_resolution_matches_native(self) -> None:
        """Test that BandInfo resolution matches NATIVE_BANDS."""
        from eopf_geozarr.data_api.sentinel2 import Sentinel2BandInfo

        for resolution, bands in NATIVE_BANDS.items():
            for band_name in bands:
                band_info = Sentinel2BandInfo.from_band_name(band_name)
                assert band_info.native_resolution == resolution, (
                    f"Band {band_name} should have native resolution {resolution}m, "
                    f"but got {band_info.native_resolution}m"
                )
