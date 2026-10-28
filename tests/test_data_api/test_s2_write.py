"""
Parity tests for the plan-driven writer (`eopf_geozarr.data_api.s2_write`):
the store it writes is compared file by file against the store the DataTree
writer (`create_multiscale_from_datatree`) produces from the same input.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr
from pyproj import CRS
from structlog.testing import capture_logs

if TYPE_CHECKING:
    import pathlib

from eopf_geozarr.data_api.s2_plan import load_sentinel2
from eopf_geozarr.data_api.s2_write import write_geozarr
from eopf_geozarr.s2_optimization.s2_multiscale import create_multiscale_from_datatree

from ..conftest import create_zarrv2_group_from_json, get_stem, s2_example_json_paths

# Fallback footprint for coordinate arrays in groups that carry no proj
# attributes (geometry, meteorology), where both writers only copy the values.
_TILE_X_MIN = 300000.0
_TILE_Y_MAX = 5000040.0
_TILE_EXTENT = 109800.0

#: Arrays at most this many elements get deterministic test data, so the
#: resampling arithmetic is exercised on values, not only on nodata fill.
_PATTERN_MAX_ELEMENTS = 1830 * 1830

#: Larger arrays seeded anyway: the scene classifications, so the
#: window-centre subsampling is exercised on real values.
_PATTERNED_PATHS = (
    "conditions/mask/l2a_classification/r20m/scl",
    "conditions/mask/l1c_classification/r60m/b00",
)


def _walk_groups(group: zarr.Group, prefix: str = "") -> list[tuple[str, zarr.Group]]:
    found: list[tuple[str, zarr.Group]] = [(prefix, group)]
    for name, subgroup in group.groups():
        found.extend(_walk_groups(subgroup, f"{prefix}{name}/"))
    return found


def _group_transform(group: zarr.Group) -> tuple[float, ...] | None:
    """The proj:transform any of the group's arrays carries, if any."""
    for _, array in group.arrays():
        transform = array.attrs.get("proj:transform")
        if not isinstance(transform, (list, tuple)) or len(transform) < 6:
            continue
        values = [value for value in transform[:6] if isinstance(value, (int, float))]
        if len(values) == 6:
            return tuple(float(value) for value in values)
    return None


def _seed_input_store(group: zarr.Group) -> None:
    """Give the metadata-only fixture store coordinates and array values.

    Coordinates are the pixel centres of each group's own `proj:transform`
    (the georeferencing the plan derives from attributes, and the DataTree
    writer re-derives from these values), and real products store them as
    float64, so the fixtures' int64 declarations are recreated in that form.
    """
    patterned = set(_PATTERNED_PATHS)
    for group_path, subgroup in _walk_groups(group):
        transform = _group_transform(subgroup)
        for name, array in subgroup.arrays():
            path = f"{group_path}{name}"
            if name in ("x", "y") and array.ndim == 1:
                centers = np.arange(array.shape[0]) + 0.5
                if transform is not None:
                    res_x, _, x0, _, res_y, y0 = transform
                    values = x0 + centers * res_x if name == "x" else y0 + centers * res_y
                else:
                    res = _TILE_EXTENT / array.shape[0]
                    values = (
                        _TILE_X_MIN + centers * res if name == "x" else _TILE_Y_MAX - centers * res
                    )
                replacement = zarr.create_array(
                    group.store,
                    name=path,
                    shape=array.shape,
                    dtype="float64",
                    chunks=array.chunks,
                    zarr_format=2,
                    fill_value=float("nan"),
                    attributes=dict(array.attrs),
                    overwrite=True,
                )
                replacement[:] = values
            elif array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
                if array.size <= _PATTERN_MAX_ELEMENTS or path in patterned:
                    modulus = min(999, int(np.iinfo(array.dtype).max))
                    array[...] = (np.arange(array.size) % modulus).reshape(array.shape)


def _store_files(root: pathlib.Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
@pytest.mark.parametrize("source_path", s2_example_json_paths, ids=get_stem)
def test_write_geozarr_matches_datatree_writer(
    source_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """
    `write_geozarr` reproduces the DataTree writer's store from the same
    input: the same files (so the same chunk existence), the same metadata
    documents, and the same bytes in every chunk — across the copied native
    levels, the injected bands, the generated pyramid levels (mean and
    window-centre resampling exercised on seeded values), the coordinates,
    and the carried conditions/quality datasets.
    """
    input_path = create_zarrv2_group_from_json(source_path, tmp_path)
    _seed_input_store(zarr.open_group(input_path, mode="r+", zarr_format=2))
    input_group = zarr.open_group(input_path, mode="r", zarr_format=2)

    band = input_group["measurements/reflectance/r10m/b02"]
    assert isinstance(band, zarr.Array)
    crs = CRS.from_epsg(int(band.attrs["proj:epsg"]))  # type: ignore[arg-type]

    datatree_path = tmp_path / "datatree.zarr"
    dt_input = xr.open_datatree(
        input_group.store,  # pyright: ignore[reportArgumentType]
        engine="zarr",
        chunks={},
        mask_and_scale=False,
        decode_coords="all",
    )
    with capture_logs():
        create_multiscale_from_datatree(
            dt_input,
            output_group=zarr.create_group(str(datatree_path)),
            enable_sharding=True,
            spatial_chunk=1024,
            crs=crs,
        )

    planned_path = tmp_path / "planned.zarr"
    write_geozarr(load_sentinel2(input_group), input_group, str(planned_path))

    datatree_files = _store_files(datatree_path)
    planned_files = _store_files(planned_path)
    assert sorted(planned_files) == sorted(datatree_files)

    mismatched = []
    for key in sorted(datatree_files):
        expected = (datatree_path / key).read_bytes()
        observed = (planned_path / key).read_bytes()
        if key.rsplit("/", 1)[-1] == "zarr.json":
            if json.loads(observed) != json.loads(expected):
                mismatched.append(key)
        elif observed != expected:
            mismatched.append(key)
    assert mismatched == []


def test_write_geozarr_refuses_a_source_missing_the_planned_arrays(
    tmp_path: pathlib.Path,
) -> None:
    """A source store that lacks a planned array's data is refused."""
    input_path = create_zarrv2_group_from_json(s2_example_json_paths[0], tmp_path)
    input_group = zarr.open_group(input_path, mode="r", zarr_format=2)
    root = load_sentinel2(input_group)
    empty_source = zarr.create_group(tmp_path / "empty.zarr", zarr_format=2)
    with pytest.raises(ValueError, match="input store"):
        write_geozarr(root, empty_source, str(tmp_path / "out.zarr"))
