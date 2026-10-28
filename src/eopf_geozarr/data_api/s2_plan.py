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
chain.

Only the `measurements/reflectance` subtree is planned so far; the
`conditions` and `quality` subtrees still go through the DataTree-based
converter (a follow-up migrates them onto this plan).
"""

from __future__ import annotations

import base64
import struct
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

from pydantic_zarr.v3 import ArraySpec as ArraySpecV3
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3

from eopf_geozarr.data_api.geozarr.v3 import DataArray
from eopf_geozarr.data_api.s2 import Sentinel2Root

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    data_type: str,
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


def _level_spatial_attributes(
    *,
    meters: int,
    shape: tuple[int, ...],
    origin: tuple[float, float],
    epsg: int,
) -> dict[str, object]:
    """The spatial/proj convention attributes of one pyramid level's group."""
    x0, y0 = origin
    rows, cols = shape
    resolution = float(meters)
    return {
        "grid_mapping": "spatial_ref",
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": [x0, y0 - rows * resolution, x0 + cols * resolution, y0],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:shape": list(shape),
        "spatial:transform": [resolution, 0.0, x0, 0.0, -resolution, y0],
        "zarr_conventions": [_SPATIAL_CONVENTION, _PROJ_CONVENTION],
    }


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
    assert isinstance(transform, (list, tuple))
    origin = (float(transform[2]), float(transform[5]))
    epsg = int(first_band_attrs["proj:epsg"])  # type: ignore[call-overload]
    wkt2 = str(first_band_attrs["proj:wkt2"])

    # per-level planned bands: name -> (source attrs, shape, dtype itemsize)
    planned_levels: dict[int, dict[str, DataArray]] = {}
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
            assert previous_meters is not None
            factor = meters // previous_meters
            for name, previous in planned_levels[previous_meters].items():
                shape = tuple(dim // factor for dim in previous.shape)
                # a coarsened band keeps its attributes verbatim
                dims = previous.dimension_names
                assert dims is not None
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

        members_out[f"r{meters}m"] = GroupSpecV3(
            attributes=_level_spatial_attributes(
                meters=meters, shape=shape2d, origin=origin, epsg=epsg
            ),
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
    finest_attrs = members_out[f"r{min(input_levels)}m"].attributes
    assert isinstance(finest_attrs, dict)
    group_attributes: dict[str, object] = {
        "multiscales": {"layout": layout, "resampling_method": "average"},
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": finest_attrs["spatial:bbox"],
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


def plan_geozarr(root: Sentinel2Root) -> AnyGroupSpec:
    """
    Plan the GeoZarr hierarchy to write for a Sentinel-2 product.

    Pure: reads only the validated input model, returns the output tree as a
    Zarr V3 `GroupSpec`. Only `measurements/reflectance` is planned so far;
    the `conditions` and `quality` subtrees still come from the
    DataTree-based converter.
    """
    return GroupSpecV3(
        attributes={},
        members={
            "measurements": GroupSpecV3(
                attributes={}, members={"reflectance": plan_reflectance(root)}
            ),
        },
    )
