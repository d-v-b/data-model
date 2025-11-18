from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import structlog
import zarr
from pydantic import TypeAdapter

from eopf_geozarr.data_api.s1 import Sentinel1Root
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.pyz.v2 import GroupSpec

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
log = structlog.get_logger()

AnyProduct = Sentinel1Root | Sentinel2Root


class ChunkSpec(TypedDict):
    shard_shape: int | None
    chunk_shape: int


class ChunkingScheme(TypedDict):
    time: ChunkSpec
    space: ChunkSpec


def resolve_product(group_spec: GroupSpec[Any, Any]) -> AnyProduct:
    type_adapter = TypeAdapter(AnyProduct)
    return type_adapter.validate_python(group_spec.model_dump())


def convert(input_path: str | Path) -> None:
    # resolve the input path to a specific product
    zg = zarr.open_group(input_path, mode="r", use_consolidated=True)
    group_spec: GroupSpec[Any, Any] = GroupSpec.from_zarr(zg)
    product_model = resolve_product(group_spec)
    if isinstance(product_model, Sentinel1Root):
        log.info("Found a Sentinel1 product!")
    else:
        log.info("Found a Sentinel2 product!")


def create_multiscale(data: ZDataArray, chunking_scheme: ChunkingScheme) -> None: ...
