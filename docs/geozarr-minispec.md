# GeoZarr

This document specifies the GeoZarr model used in this repository. It's a "mini" version of the [official GeoZarr spec](https://zarr.dev/geozarr-spec/documents/standard/template/geozarr-spec.html)

## Spec conventions

### Array and Group attributes
This document only defines rules for a finite subset of the keys in Zarr array 
and group attributes. Unless otherwise stated, any external keys in Zarr array and group attributes are consistent with this specification. This means this specification composes with the presence of, e.g., [CF metadata](https://cfconventions.org/), at different levels of the Zarr hierarchy.

## Organization

GeoZarr defines a Zarr hierarchy, i.e. a particular arrangements of Zarr arrays and groups, and 
their attributes. This document defines that hierarchy from the bottom-up, starting with arrays
and their attributes before moving to higher-level structures, like groups and their attributes.

The GeoZarr specification can be implemented in Zarr V2 and V3. The main difference between the Zarr V2 and Zarr V3 implementations is how the dimension names of an array are specified.

## DataArray

A DataArray is a Zarr array with named axes. The structure of a DataArray depends on the Zarr format.

This section contains the rules for *individual* DataArrays. Additional 
constraints on collectiosn of DataArrays are defined in the section on [Datasets](#dataset)

### Zarr V2

#### Attributes

| key | type | required | notes |
|-----| ------| ----------| ----- |
| _ARRAY_DIMENSIONS | array of strings, length matches number of axes of the array | yes |  xarray convention for naming axes in Zarr V2 |

#### Array metadata

Zarr V2 DataArrays must have at least 1 dimension, i.e. scalar Zarr V2 DataArrays are not allowed.

In tabular form: 

| attribute | constraint | notes |
|-----| ----------| ----- |
| `shape` | at least 1 element | No scalar arrays allowed

#### Example 

```json
{
    ".zarray": {
        "zarr_format": 2,
        "dtype": "|u1",
        "shape": [10,11,12],
        "chunks": [10,11,12],
        "filters": None
        "compressor": None
        "order": "C"
        "dimension_separator": "/"
        }
    ".zattrs": {
        "_ARRAY_DIMENSIONS": ["lat", "lon", "time"]
        }

}
```

### Zarr V3

#### Attributes

No particular attributes are required for Zarr V3 DataArrays.

#### Array metadata

Zarr V3 DataArrays must have at least 1 dimension, i.e. scalar Zarr V3 DataArrays are not allowed. The 
`dimension_names` attribute of a Zarr V3 DataArray must be set, the elements of `dimension_names` must 
all be strings, and they must all be unique.

In tabular form:

| attribute | constraint | notes |
|-----| ----------| ----- |
| `shape` | at least 1 element | No scalar arrays allowed
| `dimension_names` | an array of unique strings | all array axes must be uniquely named.


#### Example

```json
{
    "zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10,11,12],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10,11,12]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat", "lon", "time"],
        "storage_transformers": [],
        }
}
```

## Dataset

A GeoZarr dataset is a Zarr group that contains Zarr arrays that together describe a measured quantity, 
as well as arbitrary sub-groups.

### Attributes

There are no required attributes for Datasets.

### Members

If any member of a GeoZarr Dataset is an array, then it must comply with the [DataArray](#dataarray) definition.

If the Dataset contains a DataArray `D`, then for each dimension name `N` in the list of `D`'s named dimensions, 
the Dataset must contain a one-dimensional DataArray named `N` with a shape that matches the the length 
of `D` along the axis named by `N`. In this case, `D` is called a "data variable", and the each 
DataArrays matching a dimension names of `D` is called a "coordinate variable". 

> [!Note]
> These two definitions are not mutually exclusive, as a 1-dimensional DataArray named `D` with
dimension names `["D"]` is both a coordinate variable and a data variable.


#### Examples 

This example demonstrates the stored representation of a valid Dataset. Notice how 
the dimension names defined on the DataArray named `"data"` (i.e., `"lat"` and `"lon"`) are 
the names of one-dimensional DataArrays in the same Zarr group as `"data"`.

In this case, `"data"` is a data variable, and `"lat"` and `"lon"` are coordinate variables.

```json
{
    "zarr.json" : {
        "node_type": "group",
        "zarr_format": 3,
        },
    "data/zarr.json" : {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10,11],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10,11]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat", "lon"],
        "storage_transformers": [],
        },
    "lat/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat"],
        "storage_transformers": [],
        },
    "lon/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [11],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [11]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lon"],
        "storage_transformers": [],
        },
}
```

This example demonstrates the layout of a Dataset with just one DataArray. A single array
is only permitted if that array is one dimensional, and the name of that DataArray in the Dataset 
matches the (single) dimension name defined for that DataArray. 

In this case `lat` is both a coordinate variable and a data variable.

```json
{
    "zarr.json" : {
        "node_type": "group",
        "zarr_format": 3,
        },
    "lat/zarr.json" : {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10,11]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat"],
        "storage_transformers": [],
    },
}
```

## Multiscale Dataset 

Downsampling is a process in which a collection of localized data points is resampled on a subset of its original sampling locations. 

In the case of arrays, downsampling generally reduces an array's shape along at least one dimension. To downsample the 
contents of a Dataset `D` and generate a new Dataset `E`, all of the coordinate variable - data variable 
relationships in `D` must be preserved in `E`. If `D/data` is a data variable with dimension names (`"a"` , `"b"`), then `D/a` and `D/b` are coordinate variables with shapes aligned to the dimensions of `D/data`. If we downsample `D/data` and assign the result to `E/data`, we must also generate (e.g., by more downsampling) coordinate variables `E/a` and `E/b` so that `E` can be a valid Dataset according to the relevant [Dataset members rule](#members).

The downsampling transformation is thus well-defined for Datasets. Downsampling 
is often applied multiple times in a series, e.g. to generate multiple levels of 
detail for a data variable. 

GeoZarr defines a layout for downsampled Datasets, which this document terms Given some source Dataset `s0`, 
that dataset and all downsampled Datasets `s1`, `s2`, ... are stored in a flat layout inside a Multiscale Dataset
 `D`. The presence of downsampled Datsets in `D` is signalled by a [special key](#attributes-3) in the attributes of `D`.

### Attributes

The attributes of a Multiscale Dataset function as an entry point to a collection of downsampled Datasets. Accordingly, the attributes of a Multiscale Dataset declare the names of the downsampled datasets it contains, as well as spatial metadata for those datasets.

| key | type | required | notes |
|-----| ------| ----------| ----- |
| `"multiscales"` | [`MultiscaleMetadata`](#multiscalemetadata)  | yes | this field defines the layout of the multiscale Datasets inside this Dataset  |

#### MultiscaleMetadata

`MultiscaleMetadata` is a JSON object that declares the names of the downsampled Datasets inside a Multiscale Dataset, as well as the downsampling method used. This object has the following structure:

| key | type | required | notes |
|-----| ------| ----------| ----- |
| `"resampling_method"` | [ResamplingMethod](#resamplingmethod) | yes | This is a string that declares the resampling method used to create the downsampled datasets.
| `"tile_matrix_set"` | [TileMatrixSet](#tilematrixset) or string | yes | This object declares the names of the downsampled Datasets. If `"tile_matrix_set"` is a string, it must be the name of a well-known [`TileMatrixSet`](https://docs.ogc.org/is/17-083r4/17-083r4.html#toc48).
| `"tile_matrix_limit"` | {`string`: [TileMatrixLimit](#tilematrixlimit)} | no |  

### Members

All of the members declared in the `multiscales` attribute must comply with the [Dataset](#dataset) definition. All of these Datasets must
have the exact same set of member names.

## Appendix

### Definitions

#### TileMatrixLimit

| key | type | required | notes |
|-----| ------| ----------| ----- |
| `"tileMatrix"` | string | yes | |
| `"minTileCol"` |  int | yes | | |
| `"minTileRow"` | int | yes | |
| `"maxTileCol"` | int | yes | |
| `"maxTileRow"` | int | yes | |

#### TileMatrix

| key | type | required | notes |
|-----| ------| ----------| ----- |
| `"id"` | string | yes | |
| `"scaleDenominator"` | float | yes | |
| `"cellSize"` | float | yes | |
| `"pointOfOrigin"` | [float, float] | yes | |
| `"tileWidth"` | int | yes | |
| `"tileHeight"` | int | yes | |
| `"matrixWidth"` | int | yes | |
| `"matrixHeight"` | int | yes | |


#### TileMatrixSet

| key | type | required | notes |
|-----| ------| ----------| ----- |
| `"id"` | string | yes | |
| `"title"` | string | no | |
| `"crs"` | string | no | |
| `"supportedCRS"` | string | no | |
| `"orderedAxes"` | [str, str] | no | |
| `"tileMatrices"` | [[TileMatrix](#tilematrix), ...] | yes | May not be empty |

#### ResamplingMethod

This is a string literal defined [here](https://zarr.dev/geozarr-spec/documents/standard/template/geozarr-spec.html#_71eeacb0-5e4e-8a8e-5714-02fc0838075b).

