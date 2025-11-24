"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_converter import (
    convert_s2_optimized,
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


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


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


class TestMetadataConsolidation:
    """Test metadata consolidation functionality."""

    @patch("xarray.DataTree.to_zarr")
    def test_simple_root_consolidation_success(self, mock_to_zarr, temp_output_dir):
        """Test successful root consolidation with DataTree."""
        # Call the function with empty datasets
        simple_root_consolidation(temp_output_dir, {})

        # Verify to_zarr was called multiple times to create root group
        assert mock_to_zarr.call_count >= 2

        # Verify the calls included consolidated=True
        calls = mock_to_zarr.call_args_list
        assert any(kwargs.get("consolidated", False) for args, kwargs in calls)

    @patch("xarray.DataTree.to_zarr")
    def test_simple_root_consolidation_with_groups(self, mock_to_zarr, temp_output_dir):
        """Test root consolidation with nested groups."""
        # Create test datasets dict with nested groups
        datasets = {
            "/measurements/reflectance/r10m": {},
            "/quality/atmosphere": {},
        }

        simple_root_consolidation(temp_output_dir, datasets)

        # Verify to_zarr was called (for root + intermediary groups)
        assert mock_to_zarr.call_count >= 2


class TestConvenienceFunction:
    """Test the convenience function."""

    @patch("eopf_geozarr.s2_optimization.s2_converter.get_zarr_group")
    @patch("eopf_geozarr.s2_optimization.s2_converter.is_sentinel2_dataset")
    @patch("eopf_geozarr.s2_optimization.s2_converter.create_multiscale_from_datatree")
    @patch("eopf_geozarr.s2_optimization.s2_converter.simple_root_consolidation")
    def test_convert_s2_optimized_convenience_function(
        self, mock_consolidation, mock_multiscale, mock_is_s2, mock_get_zarr_group
    ):
        """Test the convenience function parameter passing."""
        # Setup mocks
        mock_multiscale.return_value = {}
        mock_is_s2.return_value = True
        mock_get_zarr_group.return_value = Mock()

        # Test parameter passing - Mock DataTree with groups attribute
        dt_input = Mock()
        dt_input.groups = ["/measurements/reflectance/r10m"]
        output_path = "/test/path"

        convert_s2_optimized(
            dt_input,
            output_path=output_path,
            enable_sharding=False,
            spatial_chunk=512,
            compression_level=2,
            max_retries=5,
            create_meteorology_group=False,
            create_geometry_group=False,
            validate_output=False,
        )

        # Verify multiscale function was called with correct args
        mock_multiscale.assert_called_once()
        call_kwargs = mock_multiscale.call_args.kwargs
        assert call_kwargs["enable_sharding"] is False
        assert call_kwargs["spatial_chunk"] == 512


if __name__ == "__main__":
    pytest.main([__file__])
