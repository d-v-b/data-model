# EOPF GeoZarr

GeoZarr compliant data model for EOPF (Earth Observation Processing Framework) datasets.

## Overview

This library provides tools to convert EOPF datasets to GeoZarr-spec 0.4 compliant format while maintaining native projections and using /2 downsampling logic for multiscale support.

## Key Features

- **GeoZarr Specification Compliance**: Full compliance with GeoZarr spec 0.4
- **Native CRS Preservation**: No reprojection to TMS, maintains original coordinate reference systems
- **Multiscale Support**: COG-style /2 downsampling with overview levels as children groups
- **CF Conventions**: Proper CF standard names and grid_mapping attributes
- **Robust Processing**: Band-by-band writing with validation and retry logic

## GeoZarr Compliance Features

- `_ARRAY_DIMENSIONS` attributes on all arrays
- CF standard names for all variables
- `grid_mapping` attributes referencing CF grid_mapping variables
- `GeoTransform` attributes in grid_mapping variables
- Proper multiscales metadata structure
- Native CRS tile matrix sets

## Installation

```bash
pip install eopf-geozarr
```

For development:

```bash
git clone <repository-url>
cd eopf-geozarr
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

After installation, you can use the `eopf-geozarr` command:

```bash
# Convert EOPF dataset to GeoZarr format
eopf-geozarr convert input.zarr output.zarr

# Get information about a dataset
eopf-geozarr info input.zarr

# Validate GeoZarr compliance
eopf-geozarr validate output.zarr

# Get help
eopf-geozarr --help
```

### Python API

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load your EOPF DataTree
dt = xr.open_datatree("path/to/eopf/dataset.zarr", engine="zarr")

# Define groups to convert (e.g., resolution groups)
groups = ["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"]

# Convert to GeoZarr compliant format
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=groups,
    output_path="path/to/output/geozarr.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    tile_width=256,
    max_retries=3
)
```

## API Reference

### Main Functions

#### `create_geozarr_dataset`

Create a GeoZarr-spec 0.4 compliant dataset from EOPF data.

**Parameters:**
- `dt_input` (xr.DataTree): Input EOPF DataTree
- `groups` (List[str]): List of group names to process as Geozarr datasets
- `output_path` (str): Output path for the Zarr store
- `spatial_chunk` (int, default=4096): Spatial chunk size for encoding
- `min_dimension` (int, default=256): Minimum dimension for overview levels
- `tile_width` (int, default=256): Tile width for TMS compatibility
- `max_retries` (int, default=3): Maximum number of retries for network operations

**Returns:**
- `xr.DataTree`: DataTree containing the GeoZarr compliant data

#### `setup_datatree_metadata_geozarr_spec_compliant`

Set up GeoZarr-spec compliant CF standard names and CRS information.

**Parameters:**
- `dt` (xr.DataTree): The data tree containing the datasets to process
- `groups` (List[str]): List of group names to process as Geozarr datasets

**Returns:**
- `Dict[str, xr.Dataset]`: Dictionary of datasets with GeoZarr compliance applied

### Utility Functions

#### `downsample_2d_array`

Downsample a 2D array using block averaging.

#### `calculate_aligned_chunk_size`

Calculate a chunk size that aligns well with the data dimension.

#### `is_grid_mapping_variable`

Check if a variable is a grid_mapping variable by looking for references to it.

#### `validate_existing_band_data`

Validate that a specific band exists and is complete in the dataset.

## Architecture

The library is organized into the following modules:

- **`conversion`**: Core conversion tools for EOPF to GeoZarr transformation
  - `geozarr.py`: Main conversion functions and GeoZarr spec compliance
  - `utils.py`: Utility functions for data processing and validation
- **`data_api`**: Data access API (future development with pydantic-zarr)

## GeoZarr Specification Compliance

This library implements the GeoZarr specification 0.4 with the following key requirements:

1. **Array Dimensions**: All arrays must have `_ARRAY_DIMENSIONS` attributes
2. **CF Standard Names**: All variables must have CF-compliant `standard_name` attributes
3. **Grid Mapping**: Data variables must reference CF grid_mapping variables via `grid_mapping` attributes
4. **Multiscales Structure**: Overview levels are stored as children groups with proper tile matrix metadata
5. **Native CRS**: Coordinate reference systems are preserved without reprojection

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd eopf-geozarr

# Install in development mode with all dependencies
pip install -e ".[dev,docs,all]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
pytest
```

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for automated checks

### Building Documentation

```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality checks pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built on top of the excellent [xarray](https://xarray.pydata.org/) and [zarr](https://zarr.readthedocs.io/) libraries
- Follows the [GeoZarr specification](https://github.com/zarr-developers/geozarr-spec) for geospatial data in Zarr
- Designed for compatibility with [EOPF](https://eopf.readthedocs.io/) datasets

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/developmentseed/eopf-geozarr).
