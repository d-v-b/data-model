"""
This module contains zarr-specific IO routines
"""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import numcodecs
import structlog
import zarr
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage._common import make_store_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from zarr.core.metadata.v2 import ArrayV2Metadata


class ChunkEncodingSpec(TypedDict):
    write_chunks: tuple[int, ...]
    read_chunks: NotRequired[tuple[int, ...]]


def _normalize_node_path(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    # reencode_group member names are relative (no leading slash)
    value = value.lstrip("/")
    while "//" in value:
        value = value.replace("//", "/")
    return value.rstrip("/")


def _normalize_omit_nodes(values: set[str] | None) -> set[str]:
    if not values:
        return set()
    normalized = {_normalize_node_path(v) for v in values}
    normalized.discard("")
    return normalized


def _is_omitted(name: str, omit_nodes: set[str]) -> bool:
    # Omit either the exact node, or any descendant below it.
    return any(name == v or name.startswith(v + "/") for v in omit_nodes)


def _iter_chunk_regions(
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> Iterator[tuple[slice, ...]] | None:
    if len(shape) != len(chunk_shape):
        return None
    if any(dim_size <= 0 for dim_size in shape):
        return None

    starts_per_dim = [
        range(0, dim_size, max(1, chunk_size))
        for dim_size, chunk_size in zip(shape, chunk_shape, strict=True)
    ]

    def _gen() -> Iterator[tuple[slice, ...]]:
        for starts in product(*starts_per_dim):
            yield tuple(
                slice(start, min(start + max(1, chunk), dim))
                for start, chunk, dim in zip(starts, chunk_shape, shape, strict=True)
            )

    return _gen()


def _dtype_itemsize(dtype: object) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    return 1


def _estimate_nbytes(shape: tuple[int, ...], dtype: object) -> int:
    try:
        n = int(math.prod(shape))
    except Exception:
        return 0
    return n * _dtype_itemsize(dtype)


def convert_compression(
    compressor: numcodecs.abc.Codec | dict[str, object], *, compression_level: int | None = None
) -> tuple[dict[str, Any], ...]:
    """
    Convert the compression parameters for a Zarr V2 array into a tuple of Zarr V3 codecs.

    Parameters
    ----------
    compressor : numcodecs.abc.Codec | dict[str, object]
        The compressor of the zarr v3 array
    compression_level: int | None = None
        The new compression level to use.
        Only applies to the specific case of blosc -> blosc compression.
    """

    if compressor is None:
        return ()

    SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
    if isinstance(compressor, numcodecs.abc.Codec):
        old_config = compressor.get_config()
    else:
        old_config = compressor
    if old_config["id"] == "blosc":
        new_level = old_config["clevel"] if compression_level is None else compression_level
        new_codec = {
            "name": "blosc",
            "configuration": {
                "cname": old_config["cname"],
                "clevel": new_level,
                "blocksize": old_config["blocksize"],
                "shuffle": SHUFFLE[old_config["shuffle"]],
            },
        }
        return (new_codec,)

    raise ValueError(f"Only blosc -> blosc or None -> () is supported. Got {compressor=}")


def default_array_reencoder(
    key: str,
    metadata: ArrayV2Metadata,
) -> ArrayV3Metadata:
    """
    Re-encode a zarr array into a new zarr v3 array.
    """

    new_codecs = convert_compression(metadata.compressor)
    if metadata.fill_value is None:
        fill_value = metadata.dtype.default_scalar()
    else:
        fill_value = metadata.fill_value
    attributes = metadata.attributes.copy()
    dimension_names = attributes.pop("_ARRAY_DIMENSIONS", None)
    chunk_grid_shape = metadata.chunks
    codecs = ({"name": "bytes"}, *new_codecs)

    return ArrayV3Metadata(
        shape=metadata.shape,
        data_type=metadata.dtype,
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunk_grid_shape},
        },
        fill_value=fill_value,
        dimension_names=dimension_names,
        codecs=codecs,
        attributes=attributes,
    )


def reencode_group(
    group: zarr.Group,
    store: zarr.storage.StoreLike,
    path: str,
    *,
    overwrite: bool = False,
    use_consolidated_for_children: bool = False,
    omit_nodes: set[str] | None = None,
    array_reencoder: Callable[[str, ArrayV2Metadata], ArrayV3Metadata] = default_array_reencoder,
) -> zarr.Group:
    """
    Re-encode a Zarr group, applying a re-encoding to all sub-groups and sub-arrays.

    Parameters
    ----------
    group : zarr.Group
        The Zarr group to re-encode
    store : zarr.storage.StoreLike
        The store to write into
    path : str
        The path in the new store to use
    overwrite : bool, default = False
        Whether to overwrite contents of the new store
    omit_nodes : set[str] | None
        Relative group/array paths to omit (e.g., "measurements/reflectance").
        Exact matches omit that node; prefix matches omit the whole subtree.
    array_reencoder : Callable[[str, ArrayV2Metadata], ArrayV3Metadata]
        Maps a v2 array metadata document to v3 metadata for the destination array.

    """
    omit_nodes = _normalize_omit_nodes(omit_nodes)

    log = structlog.get_logger()

    # Convert store-like to a proper Store object
    store_path = sync(make_store_path(store))
    store = store_path.store

    members = dict(
        group.members(max_depth=None, use_consolidated_for_children=use_consolidated_for_children)
    )

    log.info("Begin re-encoding Zarr group %s", group)
    root_attrs = group.attrs.asdict()

    new_members: dict[str, ArrayV3Metadata | GroupMetadata] = {
        path: GroupMetadata(zarr_format=3, attributes=root_attrs)
    }
    arrays_to_copy: list[str] = []
    for name in omit_nodes:
        if not any(k == name or k.startswith(name + "/") for k in members):
            log.warning(
                "The name %s was provided in omit_nodes but no such array or group exists.", name
            )
    for name, member in members.items():
        if _is_omitted(name, omit_nodes):
            log.info(
                "Skipping node %s because it is contained in a subtree declared in omit_nodes",
                name,
            )
            continue
        log.info("Re-encoding member %s", name)
        new_path = f"{path}/{name}"
        if isinstance(member, zarr.Array):
            new_meta = array_reencoder(member.path, member.metadata)
            new_members[new_path] = new_meta
            arrays_to_copy.append(name)
        else:
            new_members[new_path] = GroupMetadata(
                zarr_format=3,
                attributes=member.attrs.asdict(),
            )
    log.info("Creating new Zarr hierarchy structure at %s", f"{store}/{path}")
    tree = dict(zarr.create_hierarchy(store=store, nodes=new_members, overwrite=overwrite))
    new_group: zarr.Group = tree[path]
    for name in arrays_to_copy:
        log.info("Copying array data %s", name)
        old_array = group[name]
        new_array = new_group[name]

        if new_array.ndim == 0:
            new_array[...] = old_array[...]
            continue

        old_chunk_shape = tuple(getattr(old_array.metadata, "chunks", ()))
        new_chunk_shape = tuple(new_array.metadata.chunk_grid.chunk_shape)
        # If chunking differs, writing by destination chunk regions can re-read source chunks.
        # Prefer eager copy for smaller arrays to minimize IO; fall back to chunk-by-chunk when
        # the array is too large to comfortably materialize.
        eager_copy_max_bytes = 256 * 1024 * 1024
        if old_chunk_shape != new_chunk_shape:
            estimated = _estimate_nbytes(tuple(new_array.shape), getattr(old_array, "dtype", None))
            if estimated and estimated <= eager_copy_max_bytes:
                new_array[...] = old_array[...]
                continue

        # Iterate using the destination chunk grid to bound peak memory.
        regions_iter = _iter_chunk_regions(tuple(new_array.shape), new_chunk_shape)
        if regions_iter is None:
            new_array[...] = old_array[...]
        else:
            for region in regions_iter:
                new_array[region] = old_array[region]

    # return the root group
    return tree[path]


class ArrayEncoding(TypedDict):
    dimension_names: NotRequired[None | tuple[str | None, ...]]
    attributes: NotRequired[Mapping[str, object] | None]
