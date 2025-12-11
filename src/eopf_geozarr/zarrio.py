"""
This module contains zarr-specific IO routines
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NotRequired, TypedDict

import numcodecs
import structlog
import zarr
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata


async def set_store(store: zarr.abc.store.Store, key: str, value: Any) -> None:
    """
    Call store.set(key, value) after awaiting value, if value awaits to None then do nothing.
    """
    value_actual = await value
    if value_actual is not None:
        await store.set(key, value_actual)
    return None


class ChunkEncodingSpec(TypedDict):
    write_chunks: tuple[int, ...]
    read_chunks: NotRequired[tuple[int, ...]]


def convert_compression(
    compressor: Any,
) -> tuple[dict[str, Any], ...]:
    """
    Convert the compression parameters for a Zarr V2 array into a tuple of Zarr V3 codecs.
    """

    if compressor is None:
        return ()

    SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
    if isinstance(compressor, numcodecs.Blosc):
        old_config = compressor.get_config()
        new_codec = {
            "name": "blosc",
            "configuration": {
                "cname": old_config["cname"],
                "clevel": old_config["clevel"],
                "shuffle": SHUFFLE[old_config["shuffle"]],
            },
        }
        return (new_codec,)

    raise ValueError(
        f"Only blosc -> blosc or None -> () is supported. Got {compressor=}"
    )


def reencode_array(
    array: zarr.Array,
    *,
    dimension_names: None | tuple[str | None, ...] = None,
    attributes: Mapping[str, object] | None = None,
    chunking: ChunkEncodingSpec | None = None,
) -> zarr.core.metadata.v3.ArrayV3Metadata:
    """
    Re-encode a zarr array into a new zarr v3 array.
    """

    new_codecs = convert_compression(array.metadata.compressor)
    if array.fill_value is None:
        fill_value = array.metadata.dtype.default_scalar()
    else:
        fill_value = array.fill_value
    if attributes is None:
        attributes = array.attrs.asdict()
    else:
        attributes = attributes
    if chunking is not None:
        chunk_grid_shape = chunking["write_chunks"]
    else:
        chunk_grid_shape = array.chunks

    if chunking is not None and "read_chunks" in chunking:
        codecs = (
            {
                "name": "sharding-indexed",
                "configuration": {
                    "chunk_shape": chunking["read_chunks"],
                    "index_codecs": ({"name": "bytes"}, {"name": "crc32c"}),
                    "index_location": "end",
                    "codecs": ({"name": "bytes"}, *new_codecs),
                },
            },
        )
    else:
        codecs = ({"name": "bytes"}, *new_codecs)  # type: ignore[assignment]

    new_meta = zarr.core.metadata.v3.ArrayV3Metadata(
        shape=array.shape,
        data_type=array.metadata.dtype,
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
    return new_meta


def reencode_group(
    group: zarr.Group,
    store: Any,
    path: str,
    *,
    overwrite: bool = False,
    use_consolidated_for_children: bool = False,
    chunk_reencoder: Callable[[zarr.Array[Any]], ChunkEncodingSpec] | None = None,
) -> zarr.group:
    """
    Re-encode a Zarr group, applying a re-encoding to all sub-groups and sub-arrays.
    """
    log = structlog.get_logger()

    all_members = dict(
        group.members(
            max_depth=None, use_consolidated_for_children=use_consolidated_for_children
        )
    )

    log = structlog.get_logger()
    log.info(f"Begin re-encoding Zarr group {group}")
    new_members: dict[str, ArrayV3Metadata | GroupMetadata] = {
        path: GroupMetadata(zarr_format=3, attributes=group.attrs.asdict())
    }
    chunks_to_encode: list[tuple[str, str]] = []
    for name, member in all_members.items():
        log.info(f"re-encoding member {name}")
        new_path = f"{path}/{name}"
        member_attrs = member.attrs.asdict()
        if isinstance(member, zarr.Array):
            if "_ARRAY_DIMENSIONS" in member.attrs:
                dimension_names = member_attrs.pop("_ARRAY_DIMENSIONS")
            else:
                dimension_names = None
            if chunk_reencoder is None:
                chunking = None
            else:
                chunking = chunk_reencoder(member)
            new_members[new_path] = reencode_array(
                member,
                dimension_names=dimension_names,
                attributes=member_attrs,
                chunking=chunking,
            )
            chunks_to_encode.append((name, new_path))
        else:
            new_members[new_path] = GroupMetadata(
                zarr_format=3,
                attributes=member.attrs.asdict(),
            )
    log.info(f"Creating new Zarr hierarchy structure at {store}/{path}")
    tree = dict(
        zarr.create_hierarchy(store=store, nodes=new_members, overwrite=overwrite)
    )
    new_group = tree[path]

    for name, new_path in chunks_to_encode:
        log.info(f"Re-encoding chunks for array {name}")
        old_array = group[name]
        new_array = new_group[name]

        new_array[...] = old_array[...]

    # return the root group
    return tree[path]


class ArrayEncoding(TypedDict):
    dimension_names: NotRequired[None | tuple[str | None, ...]]
    attributes: NotRequired[Mapping[str, object] | None]
