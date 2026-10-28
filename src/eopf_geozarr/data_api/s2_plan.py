"""
Spec-level planning of the Sentinel-2 GeoZarr hierarchy.

This module is the metadata plane of the Sentinel-2 conversion:
`load_sentinel2` reads a Zarr store into the validated `Sentinel2Root` input
model — keeping the model as the carrier of everything the conversion needs,
instead of validating and discarding it — and `plan_geozarr` is a pure
function from that model to the GeoZarr hierarchy we want to write, as a
Zarr V3 `GroupSpec` tree. No chunk data is read or written here: realizing
the plan (copying and coarsening chunks into the planned arrays) is a
separate, data-plane step.

Everything the plan emits is derived from the input model's own metadata:
EOPF products carry `long_name`, the `proj:*` attributes, and the CF packing
attributes on every band, so the output documents (band arrays retyped and
sharded, per-level coordinate arrays, `spatial_ref` grid-mapping arrays, the
spatial/proj convention attributes, and the multiscales layout) are all
computable from the input document alone.

The reflectance pyramid follows the converter's policy
(`eopf_geozarr.s2_optimization`): the scale levels are
``(10, 20, 60, 120, 360, 720)`` metres, levels present in the input are
reused, coarser levels are derived from the level before them, and finer
bands are injected into the coarser native levels (`b08` everywhere; for L1C
also the other 10 m and 20 m bands) so they propagate through the overview
chain. The `conditions` and `quality` subtrees are carried: their own
multiscale groups (the classification masks and the cloud probabilities) are
planned by `AUX_MULTISCALE_LEVELS`, georeferenced datasets get their spatial
attributes and `spatial_ref` grid mappings, and the quicklook groups and
scalar arrays are dropped, as the converter drops them.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Mapping
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic_zarr.v3 import ArraySpec as ArraySpecV3
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3

from eopf_geozarr.data_api.geozarr.v3 import DataArray
from eopf_geozarr.data_api.s2 import Sentinel2Root

if TYPE_CHECKING:
    import zarr
    from pydantic_zarr.v3 import AnyGroupSpec

S2Level = Literal["L1C", "L2A"]

#: Reflectance pyramid levels in metres, finest first (the converter's policy).
REFLECTANCE_SCALE_LEVELS: tuple[int, ...] = (10, 20, 60, 120, 360, 720)

#: Bands injected into coarser *native* levels, per product level, keyed by
#: target resolution in metres; generated levels inherit the previous level's
#: bands. Mirrors the converter's `inject_missing_bands` calls.
INJECTED_BANDS: dict[S2Level, dict[int, frozenset[str]]] = {
    "L2A": {20: frozenset({"b08"}), 60: frozenset({"b08"})},
    "L1C": {
        20: frozenset({"b02", "b03", "b04", "b08"}),
        60: frozenset({"b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12"}),
    },
}

#: Inner (sub-shard) chunk edge length for band arrays.
BAND_CHUNK = 1024

#: Band attributes carried from the input document, in the order they are
#: carried; everything else (xarray bookkeeping such as `_ARRAY_DIMENSIONS`
#: and `_eopf_attrs`, and the per-band `proj:wkt2`, which the per-level
#: `spatial_ref` array carries instead) is dropped.
_BAND_ATTR_CARRY = (
    "add_offset",
    "long_name",
    "proj:bbox",
    "proj:epsg",
    "proj:shape",
    "proj:transform",
    "scale_factor",
    "valid_max",
    "valid_min",
)

# base64 of a little-endian float64 NaN: the `_FillValue` spelling xarray
# stores for the float coordinate arrays
_NAN_FLOAT64_B64 = base64.b64encode(struct.pack("<d", float("nan"))).decode()

_CHUNK_KEY_ENCODING: dict[str, object] = {"name": "default", "configuration": {"separator": "/"}}

# registry entries for the zarr conventions the plan declares (zarr-conventions v0.1)
_MULTISCALES_CONVENTION = {
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}
_SPATIAL_CONVENTION = {
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
    "name": "spatial",
    "description": "Spatial coordinate information",
}
_PROJ_CONVENTION = {
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data",
}


def load_sentinel2(group: zarr.Group) -> Sentinel2Root:
    """
    Read a Zarr group into the validated `Sentinel2Root` input model.

    The model is the carrier for the conversion: it holds exactly the members
    and attributes the GeoZarr plan reads, validated (member structure by the
    closed TypedDicts, attributes by the attribute models). Raises
    `pydantic.ValidationError` when the group is not a Sentinel-2 product.
    """
    from pydantic_zarr.v2 import GroupSpec as GroupSpecV2

    return Sentinel2Root.model_validate(GroupSpecV2.from_zarr(group).model_dump())


def product_level(root: Sentinel2Root) -> S2Level:
    """
    The product level (L1C or L2A) of a Sentinel-2 product, read from the
    `stac_discovery` root attribute (the STAC id and `product:type` both
    carry it). Raises `ValueError` when neither names a known level.
    """
    stac = root.attributes.stac_discovery
    candidates: list[object] = [stac.get("id")]
    properties = stac.get("properties")
    if isinstance(properties, dict):
        candidates.append(properties.get("product:type"))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for level in ("L2A", "L1C"):
            # "MSIL2A" in product ids; "S2MSI2A" in product:type
            if level in candidate or candidate.endswith(level[1:]):
                return level
    msg = "Could not determine the Sentinel-2 product level (L1C or L2A) from stac_discovery"
    raise ValueError(msg)


def _shard_and_inner(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    The (shard, inner chunk) shapes for a band array: inner chunks of
    `BAND_CHUNK` per side (the whole dimension when smaller), shards rounded
    up to whole inner chunks.
    """
    inner = tuple(min(dim, BAND_CHUNK) for dim in shape)
    shard = tuple(
        dim if dim <= i else ceil(dim / i) * i for dim, i in zip(shape, inner, strict=True)
    )
    return shard, inner


def _plan_band(
    source_attrs: Mapping[str, object],
    shape: tuple[int, ...],
    data_type: str,
    fill_value: object,
    dimension_names: tuple[str, ...],
    typesize: int,
) -> DataArray:
    """One band array of one *native* level: input attributes mapped and carried."""
    attributes: dict[str, object] = {
        "_FillValue": source_attrs.get("fill_value", fill_value),
        "grid_mapping": "spatial_ref",
        "units": "1",
    }
    for key in _BAND_ATTR_CARRY:
        if key in source_attrs:
            attributes[key] = source_attrs[key]
    return _band_array(attributes, shape, data_type, fill_value, dimension_names, typesize)


def _band_array(
    attributes: dict[str, object],
    shape: tuple[int, ...],
    data_type: str | dict[str, object],
    fill_value: object,
    dimension_names: tuple[str, ...],
    typesize: int,
) -> DataArray:
    """One band array document: sharded uint16 with the blosc policy."""
    shard, inner = _shard_and_inner(shape)
    return DataArray.model_validate(
        {
            "attributes": attributes,
            "shape": shape,
            "data_type": data_type,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard}},
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "fill_value": fill_value,
            "codecs": (
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": inner,
                        "codecs": (
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {
                                "name": "blosc",
                                "configuration": {
                                    "typesize": typesize,
                                    "cname": "zstd",
                                    "clevel": 3,
                                    "shuffle": "shuffle",
                                    "blocksize": 0,
                                },
                            },
                        ),
                        "index_codecs": (
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"},
                        ),
                        "index_location": "end",
                    },
                },
            ),
            "dimension_names": dimension_names,
        }
    )


def _plan_coordinate(name: str, size: int) -> DataArray:
    """A 1-D coordinate array (`x` or `y`): float64, NaN-filled, one chunk."""
    return DataArray.model_validate(
        {
            "attributes": {"_FillValue": _NAN_FLOAT64_B64},
            "shape": (size,),
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (size,)}},
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "fill_value": "NaN",
            "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
            "dimension_names": (name,),
        }
    )


def _grid_mapping_attributes(wkt2: str) -> dict[str, object]:
    """
    The CF grid-mapping attributes of a CRS, rioxarray-style, derived from
    the product's own `proj:wkt2` (rather than re-deriving from the EPSG
    code, which is hostage to the installed PROJ database's naming).
    """
    from pyproj import CRS

    # round-trip through CF: the stored grid mapping is exactly what a reader
    # reconstructs from the CF attributes, with the authority bookkeeping the
    # CF form cannot carry dropped from the WKT
    crs = CRS.from_cf(CRS.from_wkt(wkt2).to_cf())
    attributes: dict[str, object] = dict(crs.to_cf())
    wkt = crs.to_wkt("WKT1_GDAL")
    attributes["crs_wkt"] = wkt
    attributes["spatial_ref"] = wkt
    return attributes


def _plan_spatial_ref(wkt2: str, *, native: bool) -> ArraySpecV3:
    """
    The dimensionless `spatial_ref` grid-mapping array of one level.

    A plain `pydantic_zarr.v3.ArraySpec` rather than a GeoZarr `DataArray`:
    the grid-mapping array has no dimensions, which `DataArray` requires.
    The levels written natively carry a zstd codec; the coarsened levels are
    written uncompressed (mirroring the two write paths of the converter).
    """
    codecs: tuple[Mapping[str, object], ...] = (
        {"name": "bytes", "configuration": {"endian": "little"}},
    )
    if native:
        codecs = (*codecs, {"name": "zstd", "configuration": {"level": 0, "checksum": False}})
    return ArraySpecV3.model_validate(
        {
            "attributes": _grid_mapping_attributes(wkt2),
            "shape": (),
            "data_type": "int64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": ()}},
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "fill_value": 0,
            "codecs": codecs,
        }
    )


def _spatial_dataset_attributes(
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, ...],
    epsg: int,
) -> dict[str, object]:
    """The spatial/proj convention attributes of one spatial dataset's group."""
    res_x, _, x0, _, res_y, y0 = transform
    rows, cols = shape
    return {
        "grid_mapping": "spatial_ref",
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": [x0, y0 + rows * res_y, x0 + cols * res_x, y0],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:shape": list(shape),
        "spatial:transform": list(transform),
        "zarr_conventions": [_SPATIAL_CONVENTION, _PROJ_CONVENTION],
    }


def _level_spatial_attributes(
    *,
    meters: int,
    shape: tuple[int, ...],
    origin: tuple[float, float],
    epsg: int,
) -> dict[str, object]:
    """The spatial/proj convention attributes of one pyramid level's group."""
    x0, y0 = origin
    resolution = float(meters)
    return _spatial_dataset_attributes((resolution, 0.0, x0, 0.0, -resolution, y0), shape, epsg)


def plan_reflectance(root: Sentinel2Root) -> AnyGroupSpec:
    """
    Plan the GeoZarr `measurements/reflectance` multiscale group of a
    Sentinel-2 product: one level group per entry of
    `REFLECTANCE_SCALE_LEVELS`, native levels carried from the input (with
    the product level's finer bands injected), coarser levels derived from
    the level before them, each holding sharded uint16 band arrays, float64
    `x`/`y` coordinates and a `spatial_ref` grid-mapping array, under a
    multiscales layout attribute.
    """
    level = product_level(root)
    reflectance = root.members["measurements"].members["reflectance"]
    input_levels: dict[int, Any] = {}
    for meters in REFLECTANCE_SCALE_LEVELS:
        member = reflectance.members.get(f"r{meters}m")
        if member is not None:
            input_levels[meters] = member

    if not input_levels:
        msg = "The product's reflectance group holds none of the known resolution datasets"
        raise ValueError(msg)

    # the georeferencing of the pyramid: origin and EPSG from any finest-level band
    finest = input_levels[min(input_levels)]
    first_band_name = next(k for k in finest.members if k.startswith("b"))
    first_band_attrs: Mapping[str, object] = finest.members[first_band_name].attributes
    transform = first_band_attrs["proj:transform"]
    if not isinstance(transform, (list, tuple)) or len(transform) < 6:
        msg = f"Band {first_band_name!r} carries no usable proj:transform"
        raise ValueError(msg)
    origin = (float(transform[2]), float(transform[5]))
    epsg = int(first_band_attrs["proj:epsg"])  # type: ignore[call-overload]
    wkt2 = str(first_band_attrs["proj:wkt2"])

    # per-level planned bands: name -> (source attrs, shape, dtype itemsize)
    planned_levels: dict[int, dict[str, DataArray]] = {}
    level_attributes: dict[int, dict[str, object]] = {}
    previous_meters: int | None = None
    layout: list[dict[str, object]] = []
    members_out: dict[str, AnyGroupSpec] = {}

    for meters in REFLECTANCE_SCALE_LEVELS:
        bands: dict[str, DataArray] = {}
        native = meters in input_levels
        if native:
            source = input_levels[meters]
            for name, member in source.members.items():
                if not name.startswith("b"):
                    continue
                dims = tuple(member.attributes.get("_ARRAY_DIMENSIONS", ("y", "x")))
                bands[name] = _plan_band(
                    member.attributes,
                    tuple(member.shape),
                    "uint16",
                    0,
                    dims,
                    typesize=2,
                )
            if not bands:
                msg = f"The r{meters}m resolution dataset holds no band arrays"
                raise ValueError(msg)
            level_shape = next(iter(bands.values())).shape
            # inject finer bands so they propagate through the overview chain
            for band_name in sorted(INJECTED_BANDS.get(level, {}).get(meters, ())):
                if band_name in bands:
                    continue
                for src_meters in sorted(input_levels):
                    if src_meters >= meters:
                        break
                    src_member = input_levels[src_meters].members.get(band_name)
                    if src_member is None:
                        continue
                    dims = tuple(src_member.attributes.get("_ARRAY_DIMENSIONS", ("y", "x")))
                    bands[band_name] = _plan_band(
                        src_member.attributes, level_shape, "uint16", 0, dims, typesize=2
                    )
                    break
        else:
            if previous_meters is None:
                msg = (
                    f"The finest pyramid level (r{meters}m) is not in the input, "
                    "so there is nothing to derive the coarser levels from"
                )
                raise ValueError(msg)
            factor = meters // previous_meters
            for name, previous in planned_levels[previous_meters].items():
                shape = tuple(dim // factor for dim in previous.shape)
                # a coarsened band keeps its attributes verbatim
                dims = previous.dimension_names or ("y", "x")
                bands[name] = _band_array(
                    previous.attributes.model_dump(),
                    shape,
                    "uint16",
                    0,
                    dims,
                    typesize=2,
                )
            level_shape = next(iter(bands.values())).shape

        planned_levels[meters] = bands
        rows, cols = level_shape
        shape2d = (rows, cols)
        level_members: dict[str, DataArray | ArraySpecV3] = dict(bands)
        level_members["x"] = _plan_coordinate("x", cols)
        level_members["y"] = _plan_coordinate("y", rows)
        level_members["spatial_ref"] = _plan_spatial_ref(wkt2, native=native)

        level_attributes[meters] = _level_spatial_attributes(
            meters=meters, shape=shape2d, origin=origin, epsg=epsg
        )
        members_out[f"r{meters}m"] = GroupSpecV3(
            attributes=level_attributes[meters],
            members=level_members,
        )

        entry: dict[str, object] = {
            "asset": f"r{meters}m",
            "spatial:shape": list(shape2d),
            "spatial:transform": [float(meters), 0.0, origin[0], 0.0, -float(meters), origin[1]],
        }
        if previous_meters is not None:
            factor = meters // previous_meters
            entry = {
                "asset": entry["asset"],
                "derived_from": f"r{previous_meters}m",
                "spatial:shape": entry["spatial:shape"],
                "spatial:transform": entry["spatial:transform"],
                "transform": {"scale": [factor, factor], "translation": [0.0, 0.0]},
            }
        layout.append(entry)
        previous_meters = meters

    finest_meters = REFLECTANCE_SCALE_LEVELS[0]
    finest_level_attrs = level_attributes[min(input_levels)]
    group_attributes: dict[str, object] = {
        "multiscales": {"layout": layout, "resampling_method": "average"},
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": finest_level_attrs["spatial:bbox"],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:transform": [
            float(finest_meters),
            0.0,
            origin[0],
            0.0,
            -float(finest_meters),
            origin[1],
        ],
        "zarr_conventions": [_MULTISCALES_CONVENTION, _SPATIAL_CONVENTION, _PROJ_CONVENTION],
    }
    return GroupSpecV3(attributes=group_attributes, members=members_out)


# ---------------------------------------------------------------------------
# conditions and quality

#: Groups that get a multiscale pyramid, with their levels in metres (the
#: converter's aux multiscale specs); levels present in the input are reused,
#: the rest are coarsened from the level before them.
AUX_MULTISCALE_LEVELS: dict[str, tuple[int, ...]] = {
    "conditions/mask/l1c_classification": (60, 120, 360, 720),
    "conditions/mask/l2a_classification": (20, 60, 120, 360, 720),
    "quality/probability": (20, 60, 120, 360, 720),
}

#: Groups the output drops: the quicklooks duplicate RGB data derivable from
#: the reflectance bands (EOPF-Explorer/data-model#81).
_DROPPED_GROUPS = frozenset({"quality/l1c_quicklook", "quality/l2a_quicklook"})

#: Attribute keys never carried into the output: xarray's dimension
#: bookkeeping, the EOPF source bag, the fill value (which lives in the array
#: document, re-added as a CF attribute only for floats), and the geo-proj
#: convention keys that belong on the enclosing group.
_DROPPED_ATTRS = frozenset(
    {"_ARRAY_DIMENSIONS", "_eopf_attrs", "_FillValue", "proj:code", "proj:wkt2", "proj:projjson"}
)

# The conditions/quality planner walks the input model's *dump* — plain Zarr
# V2 documents — because the input model types those subtrees loosely, so
# their nodes are a mix of spec models and raw mappings. A node holding
# "members" is a group; one holding "shape" is an array.
_ArrayDoc = Mapping[str, Any]
_GroupDoc = Mapping[str, Any]


def _doc_attrs(doc: Mapping[str, Any]) -> Mapping[str, object]:
    attributes = doc.get("attributes") or {}
    if not isinstance(attributes, Mapping):
        msg = f"Attributes are not a mapping: {type(attributes)!r}"
        raise TypeError(msg)
    return attributes


def _v3_data_type(dtype_v2: str) -> str | dict[str, object]:
    """The Zarr V3 data type of a Zarr V2 dtype string."""
    parsed = np.dtype(dtype_v2)
    if parsed.kind == "U":
        return {
            "name": "fixed_length_utf32",
            "configuration": {"length_bytes": parsed.itemsize},
        }
    return parsed.name


def _carried_array_attributes(
    attrs: Mapping[str, object], *, is_float: bool, grid_mapping: bool, name: str
) -> dict[str, object]:
    """The output attributes of a carried array: sanitized, with the CF float
    fill, the grid-mapping reference and the SCL flag table added."""
    out = {key: value for key, value in attrs.items() if key not in _DROPPED_ATTRS}
    if is_float:
        out["_FillValue"] = _NAN_FLOAT64_B64
    if grid_mapping:
        out["grid_mapping"] = "spatial_ref"
    if name == "scl":
        # the converter's SCL flag table and NO_DATA fill declaration
        from eopf_geozarr.s2_optimization.s2_multiscale import SCL_ATTRS, SCL_NODATA

        out.update(SCL_ATTRS)
        out["_FillValue"] = SCL_NODATA
    return out


def _plan_carried_array(name: str, doc: _ArrayDoc, *, spatial: bool) -> DataArray:
    """
    One carried (non-reflectance) array, translated to its output document:
    `x`/`y` become the float64 coordinate arrays, other coordinate arrays
    (an array whose only dimension is its own name) keep their data type
    uncompressed, and data arrays are sharded with the blosc policy.
    """
    attrs = _doc_attrs(doc)
    dims_raw = attrs.get("_ARRAY_DIMENSIONS")
    if not isinstance(dims_raw, (list, tuple)):
        msg = f"Array {name!r} carries no _ARRAY_DIMENSIONS attribute"
        raise TypeError(msg)
    dims = tuple(str(dim) for dim in dims_raw)
    shape = tuple(doc["shape"])
    if name in ("x", "y") and dims == (name,):
        return _plan_coordinate(name, shape[0])
    parsed = np.dtype(str(doc["dtype"]))
    data_type = _v3_data_type(str(doc["dtype"]))
    fill_value: object = "NaN" if parsed.kind == "f" else "" if parsed.kind == "U" else 0
    attributes = _carried_array_attributes(
        attrs,
        is_float=parsed.kind == "f",
        grid_mapping=spatial and "proj:epsg" in attrs,
        name=name,
    )
    if dims == (name,):
        # a coordinate array: one chunk, uncompressed
        return DataArray.model_validate(
            {
                "attributes": attributes,
                "shape": shape,
                "data_type": data_type,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shape}},
                "chunk_key_encoding": _CHUNK_KEY_ENCODING,
                "fill_value": fill_value,
                "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                "dimension_names": dims,
            }
        )
    return _band_array(attributes, shape, data_type, fill_value, dims, typesize=parsed.itemsize)


def _carried_arrays(group: _GroupDoc) -> dict[str, _ArrayDoc]:
    """The group's non-scalar arrays (scalar arrays are dropped)."""
    members = group.get("members") or {}
    return {
        name: member
        for name, member in members.items()
        if isinstance(member, Mapping) and "shape" in member and len(member["shape"]) > 0
    }


def _spatial_anchor(arrays: Mapping[str, _ArrayDoc]) -> tuple[str, _ArrayDoc] | None:
    """The first array carrying proj attributes, with its name, if any."""
    for name, member in arrays.items():
        if "proj:epsg" in _doc_attrs(member):
            return name, member
    return None


_Transform6 = tuple[float, float, float, float, float, float]


def _proj_of(name: str, doc: _ArrayDoc) -> tuple[_Transform6, int, str]:
    """The (transform, EPSG, WKT2) a proj-bearing input array carries."""
    attrs = _doc_attrs(doc)
    transform = attrs.get("proj:transform")
    if not isinstance(transform, (list, tuple)) or len(transform) < 6:
        msg = f"Array {name!r} carries no usable proj:transform"
        raise ValueError(msg)
    a, b, c, d, e, f = (float(part) for part in transform[:6])
    return (
        (a, b, c, d, e, f),
        int(attrs["proj:epsg"]),  # type: ignore[call-overload]
        str(attrs["proj:wkt2"]),
    )


def _plan_dataset(group: _GroupDoc, path: str) -> AnyGroupSpec | None:
    """
    Plan one carried leaf dataset: every non-scalar array translated, and —
    when the dataset is georeferenced (an array carries proj attributes) —
    the group's spatial attributes and a `spatial_ref` grid-mapping array.
    Returns `None` for a dataset with no arrays to carry.
    """
    arrays = _carried_arrays(group)
    if not arrays:
        return None
    anchor = _spatial_anchor(arrays)
    members: dict[str, DataArray | ArraySpecV3] = {
        name: _plan_carried_array(name, member, spatial=anchor is not None)
        for name, member in arrays.items()
    }
    if anchor is None:
        return GroupSpecV3(attributes=dict(_doc_attrs(group)), members=members)
    transform, epsg, wkt2 = _proj_of(*anchor)
    members["spatial_ref"] = _plan_spatial_ref(wkt2, native=True)
    return GroupSpecV3(
        attributes=_spatial_dataset_attributes(transform, tuple(anchor[1]["shape"]), epsg),
        members=members,
    )


def _plan_aux_multiscale(
    group: _GroupDoc, path: str, scale_levels: tuple[int, ...]
) -> AnyGroupSpec:
    """
    Plan an auxiliary multiscale group (a classification mask or the cloud
    probabilities): levels present in the input are carried, coarser levels
    derive from the level before them with their arrays' attributes verbatim,
    and the group gets the multiscales layout — resampled `nearest` when the
    arrays carry a flag table, `average` otherwise.
    """
    group_members = group.get("members") or {}
    input_levels: dict[int, _GroupDoc] = {}
    for meters in scale_levels:
        member = group_members.get(f"r{meters}m")
        if member is not None:
            input_levels[meters] = member
    if not input_levels:
        msg = f"The {path} group holds none of the known resolution datasets"
        raise ValueError(msg)

    finest_arrays = {
        name: member
        for name, member in _carried_arrays(input_levels[min(input_levels)]).items()
        if name not in ("x", "y")
    }
    anchor = _spatial_anchor(finest_arrays)
    if anchor is None:
        msg = f"The {path} group's finest level carries no proj attributes"
        raise ValueError(msg)
    transform, epsg, wkt2 = _proj_of(*anchor)
    origin = (transform[2], transform[5])

    planned_levels: dict[int, dict[str, DataArray]] = {}
    level_attributes: dict[int, dict[str, object]] = {}
    members_out: dict[str, AnyGroupSpec] = {}
    layout: list[dict[str, object]] = []
    previous_meters: int | None = None
    uses_flags = False

    for meters in scale_levels:
        variables: dict[str, DataArray] = {}
        native = meters in input_levels
        if native:
            for name, member in _carried_arrays(input_levels[meters]).items():
                if name in ("x", "y"):
                    continue
                variables[name] = _plan_carried_array(name, member, spatial=True)
            if not variables:
                msg = f"The {path} r{meters}m dataset holds no arrays"
                raise ValueError(msg)
        else:
            if previous_meters is None:
                msg = (
                    f"The finest {path} level (r{meters}m) is not in the input, "
                    "so there is nothing to derive the coarser levels from"
                )
                raise ValueError(msg)
            factor = meters // previous_meters
            for name, previous in planned_levels[previous_meters].items():
                shape = tuple(dim // factor for dim in previous.shape)
                data_type = previous.data_type
                if not isinstance(data_type, str):
                    msg = f"Cannot coarsen {path} variable {name!r} of data type {data_type!r}"
                    raise TypeError(msg)
                # a coarsened variable keeps its attributes verbatim
                variables[name] = _band_array(
                    previous.attributes.model_dump(),
                    shape,
                    data_type,
                    previous.fill_value,
                    previous.dimension_names or ("y", "x"),
                    typesize=np.dtype(data_type).itemsize,
                )
        uses_flags = uses_flags or any(
            not {"flag_values", "flag_meanings", "flag_masks"}.isdisjoint(
                variable.attributes.model_dump()
            )
            for variable in variables.values()
        )
        planned_levels[meters] = variables
        rows, cols = next(iter(variables.values())).shape
        level_members: dict[str, DataArray | ArraySpecV3] = dict(variables)
        level_members["x"] = _plan_coordinate("x", cols)
        level_members["y"] = _plan_coordinate("y", rows)
        level_members["spatial_ref"] = _plan_spatial_ref(wkt2, native=native)
        level_attributes[meters] = _level_spatial_attributes(
            meters=meters, shape=(rows, cols), origin=origin, epsg=epsg
        )
        members_out[f"r{meters}m"] = GroupSpecV3(
            attributes=level_attributes[meters], members=level_members
        )

        entry: dict[str, object] = {
            "asset": f"r{meters}m",
            "spatial:shape": [rows, cols],
            "spatial:transform": [float(meters), 0.0, origin[0], 0.0, -float(meters), origin[1]],
        }
        if previous_meters is not None:
            factor = meters // previous_meters
            entry = {
                "asset": entry["asset"],
                "derived_from": f"r{previous_meters}m",
                "spatial:shape": entry["spatial:shape"],
                "spatial:transform": entry["spatial:transform"],
                "transform": {"scale": [factor, factor], "translation": [0.0, 0.0]},
            }
        layout.append(entry)
        previous_meters = meters

    finest_meters = scale_levels[0]
    group_attributes: dict[str, object] = {
        "multiscales": {
            "layout": layout,
            "resampling_method": "nearest" if uses_flags else "average",
        },
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": level_attributes[min(input_levels)]["spatial:bbox"],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:transform": [
            float(finest_meters),
            0.0,
            origin[0],
            0.0,
            -float(finest_meters),
            origin[1],
        ],
        "zarr_conventions": [_MULTISCALES_CONVENTION, _SPATIAL_CONVENTION, _PROJ_CONVENTION],
    }
    return GroupSpecV3(attributes=group_attributes, members=members_out)


def _plan_subtree(group: _GroupDoc, path: str) -> AnyGroupSpec | None:
    """
    Plan one conditions/quality subtree: aux multiscale groups by their specs,
    leaf datasets carried, dropped and empty groups skipped.
    """
    if path in _DROPPED_GROUPS:
        return None
    if path in AUX_MULTISCALE_LEVELS:
        return _plan_aux_multiscale(group, path, AUX_MULTISCALE_LEVELS[path])
    members = group.get("members") or {}
    subgroups = {
        name: member
        for name, member in members.items()
        if isinstance(member, Mapping) and "members" in member
    }
    if not subgroups:
        return _plan_dataset(group, path)
    planned: dict[str, AnyGroupSpec] = {}
    for name, member in subgroups.items():
        sub = _plan_subtree(member, f"{path}/{name}")
        if sub is not None:
            planned[name] = sub
    if not planned:
        return None
    return GroupSpecV3(attributes=dict(_doc_attrs(group)), members=planned)


def plan_geozarr(root: Sentinel2Root) -> AnyGroupSpec:
    """
    Plan the GeoZarr hierarchy to write for a Sentinel-2 product.

    Pure: reads only the validated input model, returns the output tree as a
    Zarr V3 `GroupSpec` — `measurements/reflectance` as the sharded multiscale
    pyramid, and the `conditions` and `quality` subtrees carried (their own
    multiscale groups planned by `AUX_MULTISCALE_LEVELS`, the quicklooks and
    scalar arrays dropped).
    """
    members: dict[str, AnyGroupSpec] = {
        "measurements": GroupSpecV3(attributes={}, members={"reflectance": plan_reflectance(root)}),
    }
    for top in ("conditions", "quality"):
        member = root.members.get(top)
        if member is None:
            continue
        planned = _plan_subtree(member.model_dump(), top)
        if planned is not None:
            members[top] = planned
    return GroupSpecV3(attributes={}, members=members)
