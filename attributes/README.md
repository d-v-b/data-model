# Zarr Attributes Specifications

This directory contains specifications for encoding Geospatial metadata in Zarr hierarchies.

## Overview

Zarr attributes are metadata stored in the `attributes` object in Zarr array and group metadata documents. While the `attributes` can contain arbitrary metadata, this directory provides specifications for commonly used attribute conventions within the Zarr community. These specifications also define the structure of a Zarr hierarchy, e.g. the layout of groups, the names of the nodes in the hierarchy, and the properties of the arrays.

## Registered Attributes

### [`geo/proj`](./geo/proj/README.md) - Geo Projection Attribute Extension

Defines a JSON object that encodes datum and coordinate reference system (CRS) information for geospatial data stored under the `proj` key within the `geo` dictionary.

**Key Features**:

- Simple, standardized CRS encoding without complex nested structures
- Compatible with existing geospatial tools (GDAL, rasterio, pyproj)
- Based on the proven STAC Projection Extension model
- Group-to-array inheritance model for CRS metadata
- Support for multiple CRS representations (EPSG codes, WKT2, PROJJSON)

**Specification**: [./geo/proj/README.md](./geo/proj/README.md)

## Contributing

For changes to attribute specifications:

1. **Content Changes**: Submit PRs to this repository for specification content
2. **Registration Changes**: Submit PRs to [zarr-extensions](https://github.com/zarr-developers/zarr-extensions) for registry updates
3. **New Attributes**: Follow the [registration process](https://github.com/zarr-developers/zarr-extensions#registering-an-attribute) in zarr-extensions
4. **Implementation Requirement**: Every specification should include a reference implementation in this repository to ensure the spec can be practically implemented

## License

All attribute specifications are licensed under the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/), consistent with the zarr-extensions registry.
