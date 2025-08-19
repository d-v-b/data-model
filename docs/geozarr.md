# GeoZarr

This document specifies the GeoZarr model used in this repository.

## Spec conventions

This document only defines rules for a finite subset of keys in Zarr array 
and group attributes. Unless otherwise stated, external keys in Zarr array and group attributes are 
consistent with this specification.

## Organization

GeoZarr defines a Zarr hierarchy, i.e. a particular arrangements of Zarr arrays and groups, and 
their attributes. This document defines that hierarchy from the bottom-up, starting with arrays
and their attributes before moving to groups and their attributes.

The GeoZarr specification can be implemented in Zarr V2 and V3.

## DataArray

A DataArray is a Zarr array with named axes. The structure of a DataArray depends on the Zarr format.

This section contains the rules for individual DataArrays. Additional 
constraints on collectiosn of DataArrays are defined in the section on [Datasets](#dataset)

### Zarr V2

#### Attributes

| key | type | required | notes |
|-----| ------| ----------| ----- |
| _ARRAY_DIMENSIONS | array of strings, length matches number of axes of the array | yes |  xarray convention for naming axes in Zarr V2 |

#### Array metadata

DataArrays must have at least 1 dimension, i.e. scalar DataArrays are not allowed.

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

DataArrays must have at least 1 dimension, i.e. scalar DataArrays are not allowed.

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

Downsampling is a process where data is resampled on a subset of its original sampling grid. 

Downsampling generally reduces an array's shape along at least one dimension. To downsample the 
contents of a Dataset and generate a new Dataset, all of the coordinate variable - data variable 
relationships must be preserved. This means that downsampling some data variable `D` with 
dimension names `["a", "b"]` requires generating appropriately sized coordinate variables named
 `"a"` and `"b"`.

The downsampling transformation is thus well-defined for Datasets, and the downsampling 
transformation can be applied multiple times in a series, e.g. to generate multiple levels of 
detail for a data variable. 

GeoZarr defines a layout for downsampled Datasets. Given some source Dataset `s0`, 
that dataset and all downsampled Datasets `s1`, `s2`, ... are stored in a flat layout inside a parent Dataset
 `D`. The presence of downsampled Datsets in `D` is signalled by a [special key](#attributes-3) in the attributes of `D`.

### Attributes

| key | type | required | notes |
|-----| ------| ----------| ----- |
| multiscales | string or tms object  | yes | this field defines the layout of the multiscale Datasets inside this Dataset  |
| grid_mapping | <string or ...>  | yes | |


### Members

All of the members declared in the `multiscales` attribute must comply with the [Dataset](#dataset) definition. All of these Datasets must
have the same set of member names.
