"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_converter import (
    convert_s2_optimized,
    initialize_crs_from_dataset,
    simple_root_consolidation,
)


@pytest.fixture
def mock_s2_dataset():
    """Create a mock S2 dataset for testing."""
    # Create test data arrays
    coords = {
        "x": (["x"], np.linspace(0, 1000, 100)),
        "y": (["y"], np.linspace(0, 1000, 100)),
        "time": (["time"], [np.datetime64("2023-01-01")]),
    }

    # Create test variables
    data_vars = {
        "b02": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b03": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b04": (["time", "y", "x"], np.random.rand(1, 100, 100)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add rioxarray CRS
    ds = ds.rio.write_crs("EPSG:32632")

    # Create datatree
    dt = xr.DataTree(ds)
    dt.attrs = {"stac_discovery": {"properties": {"mission": "sentinel-2"}}}

    return dt


class TestS2FunctionalAPI:
    """Test the S2 functional API."""

    def test_is_sentinel2_dataset_placeholder(self):
        """Placeholder test for is_sentinel2_dataset.

        The actual is_sentinel2_dataset function uses complex pydantic validation
        that requires a fully structured zarr group matching Sentinel1Root or
        Sentinel2Root models. Testing this would require creating a complete
        mock sentinel dataset, which is better done in integration tests.
        """
        # This test is kept as a placeholder to maintain test structure
        assert True


class TestCRSInitialization:
    """Test CRS initialization functionality."""

    def test_initialize_crs_from_cpm_260_metadata(self):
        """Test CRS initialization from CPM >= 2.6.0 metadata with integer EPSG."""
        # Create a DataTree with CPM 2.6.0+ style metadata (integer format)
        dt = xr.DataTree()
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": 32632  # EPSG:32632 (WGS 84 / UTM zone 32N)
            }
        }

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32632

    def test_initialize_crs_from_cpm_260_metadata_string(self):
        """Test CRS initialization from CPM >= 2.6.0 metadata with string EPSG."""
        # Create a DataTree with CPM 2.6.0+ style metadata (string format)
        dt = xr.DataTree()
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": "EPSG:32632"  # String format
            }
        }

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32632

    def test_initialize_crs_from_cpm_260_metadata_invalid_epsg(self):
        """Test CRS initialization with invalid EPSG code in CPM 2.6.0 metadata."""
        # Create a DataTree with invalid EPSG code
        dt = xr.DataTree()
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": 999999  # Invalid EPSG code
            }
        }

        # Should fall through to other methods or return None
        crs = initialize_crs_from_dataset(dt)

        # CRS should be None since there's no other CRS information
        assert crs is None

    def test_initialize_crs_from_rio_accessor(self):
        """Test CRS initialization from rioxarray accessor."""
        # Create a dataset with rioxarray CRS
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.rio.write_crs("EPSG:32633")

        # Create DataTree without CPM 2.6.0 metadata
        dt = xr.DataTree(ds)

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32633

    def test_initialize_crs_from_proj_epsg_attribute(self):
        """Test CRS initialization from proj:epsg attribute."""
        # Create a dataset with proj:epsg attribute
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        ds["b02"].attrs["proj:epsg"] = 32634

        # Create DataTree
        dt = xr.DataTree(ds)

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32634

    def test_initialize_crs_no_crs_information(self):
        """Test CRS initialization when no CRS information is available."""
        # Create a dataset without any CRS information
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)

        # Create DataTree without any CRS metadata
        dt = xr.DataTree(ds)

        crs = initialize_crs_from_dataset(dt)

        assert crs is None

    def test_initialize_crs_priority_cpm_260_over_rio(self):
        """Test that CPM 2.6.0 metadata takes priority over rio accessor."""
        # Create a dataset with both CPM 2.6.0 metadata and rio CRS
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.rio.write_crs("EPSG:32633")  # Different EPSG

        # Create DataTree with CPM 2.6.0 metadata
        dt = xr.DataTree(ds)
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": 32632  # This should take priority
            }
        }

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        # CPM 2.6.0 metadata should take priority
        assert crs.to_epsg() == 32632


def test_simple_root_consolidation_success(tmp_path: Path):
    """
    Test that simple_root_consolidation produces consolidated metadata at the root, and for the
    measurements/reflectance group, but not for other groups.
    """

    datasets = {
        "/measurements/reflectance/r10m": xr.Dataset(),
        "/quality/atmosphere": xr.Dataset(),
    }

    [
        v.to_zarr(
            str(tmp_path / f"test.zarr{k}/"),
            mode="a",
            zarr_format=3,
            consolidated=False,
        )
        for k, v in datasets.items()
    ]

    simple_root_consolidation(str(tmp_path / "test.zarr"), datasets=datasets)

    root_z_meta = json.loads((tmp_path / "test.zarr/zarr.json").read_text())
    reflectance_zmeta = json.loads(
        (tmp_path / "test.zarr/measurements/reflectance/zarr.json").read_text()
    )
    atmos_zmeta = json.loads((tmp_path / "test.zarr/quality/zarr.json").read_text())

    assert "consolidated_metadata" in root_z_meta and isinstance(
        root_z_meta["consolidated_metadata"], dict
    )
    assert "consolidated_metadata" in reflectance_zmeta and isinstance(
        reflectance_zmeta["consolidated_metadata"], dict
    )
    if "consolidated_metadata" in atmos_zmeta:
        assert atmos_zmeta["consolidated_metadata"] is None


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_convert_s2_optimized_convenience_function(
        self,
        s2_group_example: Path,
        tmp_path: Path,
    ):
        """Test the convenience function with real S2 data."""
        # Open the S2 example as a DataTree
        dt_input = xr.open_datatree(s2_group_example, engine="zarr")
        output_path = str(tmp_path / "test_output.zarr")

        # Run the conversion
        result = convert_s2_optimized(
            dt_input,
            output_path=output_path,
            enable_sharding=False,
            spatial_chunk=512,
            compression_level=2,
            max_retries=5,
            validate_output=False,
        )

        # Verify the output was created
        assert Path(output_path).exists()

        # Verify the result is a DataTree
        assert isinstance(result, xr.DataTree)

        # Verify basic structure - output should have multiscale groups
        output_dt = xr.open_datatree(output_path, engine="zarr")
        assert len(output_dt.groups) > 0


if __name__ == "__main__":
    pytest.main([__file__])
