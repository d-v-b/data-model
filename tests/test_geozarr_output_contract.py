"""Cross-product output contract.

Reprojection to a regular grid with a declared CRS is a hard requirement for
every converted product (see
docs/superpowers/specs/2026-07-27-s3-olci-reprojection-design.md). One
parametrized test walks each converter's output and asserts the contract at
every multiscale level. Products whose converter has not yet migrated off the
legacy nested layout carry an xfail on the whole-store DataTree check so the
requirement stays on record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from pyproj import CRS as ProjCRS

from eopf_geozarr.conversion import create_geozarr_dataset
from eopf_geozarr.s3_olci_optimization.olci_converter import convert_olci_optimized

from .test_integration_sentinel1 import MockSentinel1L1GRDBuilder
from .test_integration_sentinel2 import build_sample_sentinel2_datatree
from .test_olci_integration import build_synthetic_olci

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


def _convert_olci(tmp: pathlib.Path) -> pathlib.Path:
    out = tmp / "olci.zarr"
    convert_olci_optimized(
        build_synthetic_olci(rows=256, cols=256), output_path=str(out), min_dimension=64
    )
    return out


def _convert_s1(tmp: pathlib.Path) -> pathlib.Path:
    out = tmp / "s1.zarr"
    dt = MockSentinel1L1GRDBuilder("20170508T164830_0025_A094_8604_01B54C").build()
    with patch("eopf_geozarr.conversion.geozarr.print"):
        create_geozarr_dataset(
            dt,
            groups=["measurements"],
            output_path=str(out),
            gcp_group="conditions/gcp",
        )
    return out


def _convert_s2(tmp: pathlib.Path) -> pathlib.Path:
    out = tmp / "s2.zarr"
    with patch("eopf_geozarr.conversion.geozarr.print"):
        create_geozarr_dataset(
            build_sample_sentinel2_datatree(),
            groups=["/measurements/reflectance/r10m"],
            output_path=str(out),
        )
    return out


PRODUCT_CONVERTERS: dict[str, Callable[[pathlib.Path], pathlib.Path]] = {
    "olci": _convert_olci,
    "s1": _convert_s1,
    "s2": _convert_s2,
}

#: Converters that still write the legacy layout (native at group root with
#: asset "."), which xr.open_datatree rejects. The xfail keeps the hard
#: requirement on record until the generic converter migrates (cf. PR #212).
DATATREE_XFAIL: dict[str, str] = {
    "s1": "generic converter still writes asset='.' nested overview layout",
    "s2": "generic converter still writes asset='.' nested overview layout",
}

#: Converters whose overview (non-base) pyramid levels don't carry a
#: rioxarray-detectable CRS. The generic converter (geozarr.py) only stamps
#: the zarr-cm proj:/spatial: convention attrs on the base-resolution group;
#: overview sub-groups (r2, r4, ...) get a legacy CF-style `grid_mapping`
#: attribute plus a `spatial_ref` variable that is never promoted back to a
#: coordinate on plain `xr.open_dataset` reads, so rioxarray's CF and Zarr
#: convention readers both fail to auto-detect the CRS there. OLCI does not
#: have this gap: it stamps proj:/spatial: convention attrs on every level
#: (cf. PR #212, commit c21e793).
REGULAR_GRID_XFAIL: dict[str, str] = {
    "s1": "generic converter only stamps proj:/spatial: convention attrs on the "
    "base group; overview levels (r2, ...) have no rioxarray-detectable CRS",
    "s2": "generic converter only stamps proj:/spatial: convention attrs on the "
    "base group; overview levels (r2, ...) have no rioxarray-detectable CRS",
}


@pytest.fixture(scope="module", params=sorted(PRODUCT_CONVERTERS), ids=str)
def converted_store(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[pathlib.Path, str]:
    name = str(request.param)
    store = PRODUCT_CONVERTERS[name](tmp_path_factory.mktemp(name))
    return store, name


def _multiscale_level_paths(store: pathlib.Path) -> list[str]:
    """Every pyramid-level group path, discovered via multiscales layout attrs."""
    root = zarr.open_group(str(store), mode="r")
    found: list[str] = []

    def walk(group: zarr.Group, path: str) -> None:
        attrs = dict(group.attrs)
        multiscales = attrs.get("multiscales")
        if isinstance(multiscales, dict):
            layout = multiscales.get("layout")
            assert isinstance(layout, list)
            for entry in layout:
                assert isinstance(entry, dict)
                asset = entry["asset"]
                assert isinstance(asset, str)
                found.append(path if asset == "." else f"{path}/{asset}" if path else asset)
        for key in group.group_keys():
            child = group[key]
            assert isinstance(child, zarr.Group)
            walk(child, f"{path}/{key}" if path else key)

    walk(root, "")
    return found


def test_every_level_is_regular_grid_with_crs(
    converted_store: tuple[pathlib.Path, str],
) -> None:
    """Hard requirement: each pyramid level is a regular grid with a real CRS."""
    store, name = converted_store
    if name in REGULAR_GRID_XFAIL:
        pytest.xfail(REGULAR_GRID_XFAIL[name])
    levels = _multiscale_level_paths(store)
    assert levels, f"{name}: no multiscale groups found in {store}"
    for level in levels:
        ds = xr.open_dataset(str(store), group=level, engine="zarr", consolidated=False)
        crs = ds.rio.crs
        assert crs is not None, f"{name}:{level}: no CRS declared"
        # CRS round-trips through pyproj
        ProjCRS.from_user_input(crs.to_wkt())
        for dim in ("y", "x"):
            assert dim in ds.sizes, f"{name}:{level}: missing spatial dim {dim}"
            coord = ds[dim]
            assert coord.ndim == 1, f"{name}:{level}: {dim} coordinate is not 1-D"
            steps = np.diff(coord.values)
            assert np.allclose(steps, steps[0]), (
                f"{name}:{level}: {dim} coordinate spacing is not regular"
            )
        ds.close()


def test_store_opens_as_datatree(converted_store: tuple[pathlib.Path, str]) -> None:
    """Hard requirement: the whole store opens with xr.open_datatree."""
    store, name = converted_store
    if name in DATATREE_XFAIL:
        pytest.xfail(DATATREE_XFAIL[name])
    xr.open_datatree(str(store), engine="zarr", consolidated=False, chunks={})
