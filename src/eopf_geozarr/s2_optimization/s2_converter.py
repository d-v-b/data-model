"""
Main S2 optimization converter.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

import structlog
import xarray as xr
import zarr
from pydantic import TypeAdapter
from pyproj import CRS

from eopf_geozarr.conversion.fs_utils import get_storage_options
from eopf_geozarr.conversion.geozarr import get_zarr_group
from eopf_geozarr.data_api.s1 import Sentinel1Root
from eopf_geozarr.data_api.s2 import Sentinel2Root

from .s2_multiscale import create_multiscale_from_datatree

log = structlog.get_logger()


def initialize_crs_from_dataset(dt_input: xr.DataTree) -> CRS | None:
    """
    Initialize CRS from dataset by checking data variables.

    Args:
        dt_input: Input DataTree

    Returns:
        CRS object if found, None otherwise
    """
    # For CPM >= 2.6.0, the EPSG code is stored in root attributes
    epsg_cpm_260 = dt_input.attrs.get("other_metadata", {}).get(
        "horizontal_CRS_code", None
    )
    if epsg_cpm_260 is not None:
        try:
            # Handle both integer (32632) and string ("EPSG:32632" or "32632") formats
            if isinstance(epsg_cpm_260, str):
                # Extract numeric part from string like "EPSG:32632" or "32632"
                epsg_code = int(epsg_cpm_260.split(":")[-1])
            else:
                # Already an integer
                epsg_code = int(epsg_cpm_260)
            crs = CRS.from_epsg(epsg_code)
            log.info("Initialized CRS from CPM 2.6.0+ metadata", epsg=epsg_code)
            return crs
        except Exception as e:
            log.warning(
                "Failed to initialize CRS from CPM 2.6.0+ metadata",
                epsg=epsg_cpm_260,
                error=str(e),
            )

    for group_path in dt_input.groups:
        if group_path == ".":
            continue
        group_node = dt_input[group_path]
        if not hasattr(group_node, "ds") or group_node.ds is None:
            continue
        dataset = group_node.ds

        # Check if dataset has rio accessor with CRS
        if hasattr(dataset, "rio"):
            try:
                crs = dataset.rio.crs
                if crs is not None:
                    log.info("Initialized CRS from dataset", crs=str(crs))
                    return crs
            except Exception:
                pass

        # Check data variables for CRS information
        for var in dataset.data_vars.values():
            if hasattr(var, "rio"):
                try:
                    crs = var.rio.crs
                    if crs is not None:
                        log.info("Initialized CRS from variable", crs=str(crs))
                        return crs
                except Exception:
                    pass

            # Check for proj:epsg attribute
            if "proj:epsg" in var.attrs:
                try:
                    epsg = var.attrs["proj:epsg"]
                    crs = CRS.from_epsg(epsg)
                    log.info("Initialized CRS from EPSG code", epsg=epsg)
                    return crs
                except Exception:
                    pass

    log.warning("Could not initialize CRS from dataset")
    return None


def convert_s2(
    dt_input: xr.DataTree,
    output_path: str,
    validate_output: bool,
    enable_sharding: bool,
    spatial_chunk: int,
) -> xr.DataTree:
    """
    Convert S2 dataset to optimized structure.

        Args:
            dt_input: Input Sentinel-2 DataTree
            output_path: Output path for optimized dataset
            validate_output: Whether to validate the output
            verbose: Enable verbose logging

        Returns:
            Optimized DataTree
    """
    start_time = time.time()

    log.info(
        "Starting S2 optimized conversion",
        num_groups=len(dt_input.groups),
        output_path=output_path,
    )

    # Validate input is S2
    if not is_sentinel2_dataset(get_zarr_group(dt_input)):
        raise ValueError("Input dataset is not a Sentinel-2 product")

    # Step 1: Process data while preserving original structure
    log.info("Step 1: Processing data with original structure preserved")

    # Step 2: Create multiscale pyramids for each group in the original structure
    log.info("Step 2: Creating multiscale pyramids (preserving original hierarchy)")
    datasets = create_multiscale_from_datatree(
        dt_input,
        output_path,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
    )

    log.info("Created multiscale pyramids", num_groups=len(datasets))

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    simple_root_consolidation(output_path, datasets)

    # Step 4: Validation
    if validate_output:
        log.info("Step 4: Validating optimized dataset")
        validation_results = validate_optimized_dataset(output_path)
        if not validation_results["is_valid"]:
            log.warning("Validation issues found", issues=validation_results["issues"])

    # Create result DataTree
    result_dt = create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    optimization_summary(dt_input, result_dt, output_path)

    return result_dt


class ConvertS2Params(TypedDict):
    enable_sharding: bool
    spatial_chunk: int
    compression_level: int
    max_retries: int


class ArrayEncoding(TypedDict):
    dimension_names: NotRequired[None | tuple[str | None, ...]]
    attributes: NotRequired[Mapping[str, object] | None]


def convert_compression(
    compressor: Any, filters: Any, dtype: Any
) -> tuple[dict[str, Any], ...]:
    """
    Convert the compression parameters for a Zarr V2 array into Zarr V3.
    """
    import numcodecs

    """The shuffle values permitted for the blosc codec"""

    if compressor is None and filters is None:
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
        f"Only blosc -> blosc is supported. Got {compressor=} and {filters=}"
    )


def reencode_array(
    array: zarr.Array,
    *,
    zarr_format: Literal[2, 3] | None = None,
    dimension_names: None | tuple[str | None, ...] = None,
    attributes: Mapping[str, object] | None = None,
) -> zarr.core.metadata.v3.ArrayV3Metadata:
    """
    Re-encode a zarr array into a new array.
    """

    if array.metadata.zarr_format == 2 and zarr_format == 3:
        new_codecs = convert_compression(
            array.metadata.compressor, array.metadata.filters, array.dtype
        )
        if array.fill_value is None:
            fill_value = 0
        else:
            fill_value = array.fill_value
        if attributes is None:
            attributes = array.attrs.asdict()
        else:
            attributes = attributes
        new_array = zarr.core.metadata.v3.ArrayV3Metadata(
            shape=array.shape,
            data_type=array.metadata.dtype,
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
            chunk_grid={
                "name": "regular",
                "configuration": {"chunk_shape": array.chunks},
            },
            fill_value=fill_value,
            dimension_names=dimension_names,
            codecs=(
                {"name": "bytes", "configuration": {"endian": "little"}},
                *new_codecs,
            ),
            attributes=attributes,
        )
        return new_array
    raise ValueError(
        f"Re-encoding from {array.zarr_format} to {zarr_format} is not supported"
    )


async def set_coro(store, key, value) -> None:
    """
    Call store.set(key, value) after awaiting value, and returning early if value is none
    """
    value_actual = await value
    if value_actual is not None:
        return await store.set(key, value_actual)
    return None


async def move_chunks(array_a, array_b, *, decompress: bool = True) -> None:
    """
    Copy chunks from one array to another. If decompress is false, the raw chunk bytes will be copied.
    If decompress is True, an error will be raised until this developer gets around to implementing
    decompression + recompression.
    """
    if not decompress:
        coros = [
            set_coro(
                array_b.store,
                f"{array_b.path}/{new_key}",
                array_a.store.get(
                    f"{array_a.path}/{old_key}",
                    prototype=zarr.core.buffer.default_buffer_prototype(),
                ),
            )
            for old_key, new_key in zip(
                array_a._iter_shard_keys(), array_b._iter_shard_keys(), strict=True
            )
        ]
    else:
        raise ValueError("Decompression not supported yet")
    await asyncio.gather(*coros)
    return None


def reencode_group(
    group: zarr.Group,
    store: Any,
    path: str,
    *,
    overwrite: bool = False,
    zarr_format: Literal[2, 3] | None = None,
    use_consolidated_for_children: bool = False,
) -> zarr.group:
    # collect all the members of this group

    all_members = dict(
        group.members(
            max_depth=None, use_consolidated_for_children=use_consolidated_for_children
        )
    )
    from zarr.core.group import GroupMetadata
    from zarr.core.metadata.v3 import ArrayV3Metadata

    log = structlog.get_logger()
    log.info(f"Begin re-encoding Zarr group {group}")
    new_members: dict[str, ArrayV3Metadata | GroupMetadata] = {
        path: GroupMetadata(zarr_format=zarr_format, attributes=group.attrs.asdict())
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
            new_members[new_path] = reencode_array(
                member, zarr_format=zarr_format, dimension_names=dimension_names
            )
            chunks_to_encode.append((name, new_path))
        else:
            new_members[new_path] = GroupMetadata(
                zarr_format=zarr_format,
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

        result = asyncio.run(move_chunks(old_array, new_array, decompress=False))

    return tree[""]


def convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    compression_level: int,
    validate_output: bool,
    max_retries: int = 3,
) -> xr.DataTree:
    """
    Convenience function for S2 optimization.

    Args:
        dt_input: Input Sentinel-2 DataTree
        output_path: Output path
        enable_sharding: Enable Zarr v3 sharding
        spatial_chunk: Spatial chunk size
        compression_level: Compression level 1-9
        validate_output: Whether to validate the output
        max_retries: Maximum number of retries for network operations

    Returns:
        Optimized DataTree
    """

    start_time = time.time()

    log.info(
        "Starting S2 optimized conversion",
        num_groups=len(dt_input.groups),
        output_path=output_path,
    )
    # Validate input is S2
    if not is_sentinel2_dataset(get_zarr_group(dt_input)):
        raise ValueError("Input dataset is not a Sentinel-2 product")
    from zarr.storage._common import make_store

    out_store = zarr.core.sync.sync(make_store(output_path))

    # re-encode the group
    reencode_group(
        get_zarr_group(dt_input), out_store, "", overwrite=True, zarr_format=3
    )

    # Initialize CRS from dataset
    crs = initialize_crs_from_dataset(dt_input)

    # Step 1: Process data while preserving original structure
    log.info("Step 1: Processing data with original structure preserved")

    # Step 2: Create multiscale pyramids for each group in the original structure
    log.info("Step 2: Creating multiscale pyramids (preserving original hierarchy)")
    new_dt_input = xr.open_datatree(out_store, engine="zarr", chunks="auto")
    datasets = create_multiscale_from_datatree(
        new_dt_input,
        output_path,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
        crs=crs,
    )

    log.info("Created multiscale pyramids", num_groups=len(datasets))

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    simple_root_consolidation(output_path, datasets)

    # Step 4: Validation
    if validate_output:
        log.info("Step 4: Validating optimized dataset")
        validation_results = validate_optimized_dataset(output_path)
        if not validation_results["is_valid"]:
            log.warning("Validation issues found", issues=validation_results["issues"])

    # Create result DataTree
    result_dt = create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    optimization_summary(dt_input, result_dt, output_path)

    return result_dt


def simple_root_consolidation(output_path: str, datasets: dict[str, dict]) -> None:
    """Simple root-level metadata consolidation with proper zarr group creation."""
    # create missing intermediary groups (/conditions, /quality, etc.)
    # using the keys of the datasets dict
    missing_groups = set()
    for group_path in datasets.keys():
        # extract all the parent paths
        parts = group_path.strip("/").split("/")
        for i in range(1, len(parts)):
            parent_path = "/" + "/".join(parts[:i])
            if parent_path not in datasets:
                missing_groups.add(parent_path)

    for group_path in missing_groups:
        dt_parent = xr.DataTree()
        dt_parent.to_zarr(
            output_path + group_path,
            mode="a",
            zarr_format=3,
            consolidated=False,
        )

    # Create root zarr group if it doesn't exist
    log.info("Creating root zarr group")
    dt_root = xr.DataTree()
    dt_root.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
    dt_root = xr.DataTree()
    for group_path, dataset in datasets.items():
        dt_root[group_path] = xr.DataTree()

    dt_root.to_zarr(
        output_path,
        mode="r+",
        consolidated=False,
        zarr_format=3,
    )
    log.info("Root zarr group created")

    # consolidate reflectance group metadata
    zarr.consolidate_metadata(output_path + "/measurements/reflectance", zarr_format=3)

    # consolidate root group metadata
    zarr.consolidate_metadata(output_path, zarr_format=3)


def optimization_summary(
    dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str
) -> None:
    """Print optimization summary statistics."""
    # Count groups
    input_groups = len(dt_input.groups) if hasattr(dt_input, "groups") else 0
    output_groups = len(dt_output.groups) if hasattr(dt_output, "groups") else 0

    # Estimate file count reduction
    estimated_input_files = input_groups * 10  # Rough estimate
    estimated_output_files = output_groups * 5  # Fewer files per group
    group_change_pct = (
        ((output_groups - input_groups) / input_groups * 100) if input_groups > 0 else 0
    )
    file_change_pct = (
        ((estimated_output_files - estimated_input_files) / estimated_input_files * 100)
        if estimated_input_files > 0
        else 0
    )

    log.info(
        "OPTIMIZATION SUMMARY",
        input_groups=input_groups,
        output_groups=output_groups,
        group_change_pct=f"{group_change_pct:+.1f}%",
        estimated_input_files=estimated_input_files,
        estimated_output_files=estimated_output_files,
        file_change_pct=f"{file_change_pct:+.1f}%",
        output_path=output_path,
        groups=[g for g in dt_output.groups if g != "."],
    )


def create_result_datatree(output_path: str) -> xr.DataTree:
    """Create result DataTree from written output."""
    try:
        storage_options = get_storage_options(output_path)
        return xr.open_datatree(
            output_path,
            engine="zarr",
            chunks="auto",
            storage_options=storage_options,
        )
    except Exception as e:
        log.warning("Could not open result DataTree", error=str(e))
        return xr.DataTree()


def is_sentinel2_dataset(group: zarr.Group) -> bool:
    from eopf_geozarr.pyz.v2 import GroupSpec

    adapter = TypeAdapter(Sentinel1Root | Sentinel2Root)  # type: ignore[var-annotated]
    try:
        model = adapter.validate_python(GroupSpec.from_zarr(group).model_dump())
    except ValueError as e:
        log.warning("Could not validate Sentinel-2 dataset", error=str(e))
        return False

    return isinstance(model, Sentinel2Root)


def validate_optimized_dataset(dataset_path: str) -> dict[str, Any]:
    """
    Validate an optimized Sentinel-2 dataset.

    Args:
        dataset_path: Path to the optimized dataset

    Returns:
        Validation results dictionary
    """
    results = {"is_valid": True, "issues": [], "warnings": [], "summary": {}}

    # Placeholder for validation logic
    return results
