"""Utility functions for GeoZarr conversion."""

import numpy as np
import rasterio  # noqa: F401  # Import to enable .rio accessor
import xarray as xr


def downsample_2d_array(
    source_data: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """
    Downsample a 2D array using block averaging.

    Parameters
    ----------
    source_data : numpy.ndarray
        Source 2D array
    target_height : int
        Target height
    target_width : int
        Target width

    Returns
    -------
    numpy.ndarray
        Downsampled 2D array
    """
    source_height, source_width = source_data.shape

    # Calculate block sizes
    block_size_y = source_height // target_height
    block_size_x = source_width // target_width

    if block_size_y > 1 and block_size_x > 1:
        # Block averaging
        reshaped = source_data[: target_height * block_size_y, : target_width * block_size_x]
        reshaped = reshaped.reshape(target_height, block_size_y, target_width, block_size_x)
        downsampled = reshaped.mean(axis=(1, 3))
    else:
        # Simple subsampling
        y_indices = np.linspace(0, source_height - 1, target_height, dtype=int)
        x_indices = np.linspace(0, source_width - 1, target_width, dtype=int)
        downsampled = source_data[np.ix_(y_indices, x_indices)]

    return downsampled


def calculate_aligned_chunk_size(dimension_size: int, desired_chunk_size: int) -> int:
    """
    Calculate a chunk size that aligns well with the data dimension.

    This function finds the largest divisor of the dimension size that is
    less than or equal to the desired chunk size, ensuring efficient chunking.

    Parameters
    ----------
    dimension_size : int
        Size of the data dimension
    desired_chunk_size : int
        Desired chunk size

    Returns
    -------
    int
        Aligned chunk size that divides evenly into the dimension
    """
    if desired_chunk_size >= dimension_size:
        return dimension_size

    # Start from desired_chunk_size and work downwards to find a good divisor
    for chunk_size in range(desired_chunk_size, 0, -1):
        if dimension_size % chunk_size == 0:
            return chunk_size

    # If no perfect divisor found, return the desired chunk size
    return desired_chunk_size


def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool:
    """
    Check if a variable is a grid_mapping variable by looking for references to it.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check
    var_name : str
        Variable name to check

    Returns
    -------
    bool
        True if this variable is referenced as a grid_mapping
    """
    for data_var in ds.data_vars:
        if data_var != var_name and "grid_mapping" in ds[data_var].attrs:
            if ds[data_var].attrs["grid_mapping"] == var_name:
                return True
    return False


def validate_existing_band_data(
    existing_ds: xr.Dataset, band_name: str, expected_ds: xr.Dataset
) -> bool:
    """
    Validate that a specific band exists and is complete in the dataset.

    Parameters
    ----------
    existing_ds : xarray.Dataset
        Existing dataset to validate
    band_name : str
        Name of the band to validate
    expected_ds : xarray.Dataset
        Expected dataset structure for comparison

    Returns
    -------
    bool
        True if the band exists and is valid, False otherwise
    """
    try:
        # Check if the band exists
        if band_name not in existing_ds.data_vars and band_name not in existing_ds.coords:
            return False

        # Check shape matches
        if band_name in expected_ds.data_vars:
            expected_shape = expected_ds[band_name].shape
            existing_shape = existing_ds[band_name].shape

            if expected_shape != existing_shape:
                return False

        # Check required attributes for data variables
        if band_name in expected_ds.data_vars and not is_grid_mapping_variable(
            expected_ds, band_name
        ):
            required_attrs = ["_ARRAY_DIMENSIONS", "standard_name", "grid_mapping"]
            for attr in required_attrs:
                if attr not in existing_ds[band_name].attrs:
                    return False

        # Basic data integrity check for data variables
        if band_name in existing_ds.data_vars and not is_grid_mapping_variable(
            existing_ds, band_name
        ):
            try:
                # Just check if we can access the array metadata without reading data
                array_info = existing_ds[band_name]
                if array_info.size == 0:
                    return False
                # read a piece of data to ensure it's valid
                test = array_info.isel({dim: 0 for dim in array_info.dims}).values.mean()
                if np.isnan(test):
                    return False
            except Exception as e:
                print(f"Error validating band {band_name}: {e}")
                return False

        return True

    except Exception:
        return False
