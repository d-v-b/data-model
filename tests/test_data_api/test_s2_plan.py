"""Tests for the spec-level Sentinel-2 GeoZarr plan (`eopf_geozarr.data_api.s2_plan`)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
import zarr
from pydantic import ValidationError
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3

from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.data_api.s2_plan import (
    REFLECTANCE_SCALE_LEVELS,
    load_sentinel2,
    plan_geozarr,
    plan_reflectance,
    product_level,
)

_INPUT_DIR = pathlib.Path("tests/_test_data/s2_examples")
_SNAPSHOT_DIR = pathlib.Path("tests/_test_data/optimized_geozarr_examples")
_INPUT_PATHS = sorted(_INPUT_DIR.glob("*.json"))

# The snapshot harness (test_s2_multiscale) feeds one fixed, synthetic tile
# CRS and coordinate set to every product, because the metadata-only input
# fixtures carry no coordinate data; the plan instead derives georeferencing
# from each product's own band attributes. These are the snapshot attribute
# keys that carry the harness's tile rather than the product's, masked in the
# cross-product comparison and pinned by the self-consistency test instead.
_HARNESS_GEOREF_KEYS = ("proj:code", "spatial:bbox", "spatial:transform")

# The one product whose real tile matches the harness's synthetic tile, so its
# snapshot is byte-for-byte reproducible from metadata alone.
_EXACT_PRODUCT = "S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458"


def _read(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def _planned_reflectance(path: pathlib.Path) -> dict[str, Any]:
    root = Sentinel2Root.model_validate(_read(path))
    plan = plan_geozarr(root).model_dump(mode="json")
    return plan["members"]["measurements"]["members"]["reflectance"]


def _snapshot_reflectance(path: pathlib.Path) -> dict[str, Any]:
    snapshot = GroupSpecV3.model_validate(_read(_SNAPSHOT_DIR / path.name))
    dumped = snapshot.model_dump(mode="json")
    return dumped["members"]["measurements"]["members"]["reflectance"]


def _mask_harness_georef(reflectance: dict[str, Any]) -> None:
    """Blank the attribute values the snapshot harness injects (in place)."""

    def mask_attrs(attributes: dict[str, Any]) -> None:
        for key in _HARNESS_GEOREF_KEYS:
            attributes.pop(key, None)
        multiscales = attributes.get("multiscales")
        if isinstance(multiscales, dict):
            for entry in multiscales.get("layout", ()):
                entry.pop("spatial:transform", None)

    mask_attrs(reflectance["attributes"])
    for level in reflectance["members"].values():
        mask_attrs(level["attributes"])
        # the whole grid-mapping array carries the harness CRS
        level["members"].pop("spatial_ref", None)


@pytest.mark.parametrize("path", _INPUT_PATHS, ids=lambda p: p.stem[:26])
def test_plan_matches_snapshot(path: pathlib.Path) -> None:
    """
    The planned reflectance tree reproduces the stored converter snapshot:
    the member tree, every array document (shapes, dtypes, shards, codecs,
    dimension names) and every attribute, except the georeferencing values
    the snapshot harness injects from its synthetic tile, which the
    self-consistency test pins to the product's own metadata instead.
    """
    planned = _planned_reflectance(path)
    expected = _snapshot_reflectance(path)
    _mask_harness_georef(planned)
    _mask_harness_georef(expected)
    assert planned == expected


def test_plan_matches_snapshot_exactly() -> None:
    """
    For the product whose tile the snapshot harness's synthetic coordinates
    model, the plan reproduces the snapshot byte for byte, georeferencing
    included.
    """
    path = _INPUT_DIR / f"{_EXACT_PRODUCT}.json"
    assert _planned_reflectance(path) == _snapshot_reflectance(path)


@pytest.mark.parametrize("path", _INPUT_PATHS, ids=lambda p: p.stem[:26])
def test_plan_georeferencing_is_self_consistent(path: pathlib.Path) -> None:
    """
    Every planned level's spatial attributes derive from the product's own
    band attributes: the transform is the level resolution over the finest
    band's origin, the bbox is the transform applied to the level shape, and
    the grid-mapping array holds the band's own CRS.
    """
    from pyproj import CRS

    planned = _planned_reflectance(path)
    finest = planned["members"][f"r{REFLECTANCE_SCALE_LEVELS[0]}m"]
    band = next(v for k, v in finest["members"].items() if k.startswith("b"))
    transform = band["attributes"]["proj:transform"]
    x0, y0 = transform[2], transform[5]
    epsg = band["attributes"]["proj:epsg"]

    for name, level in planned["members"].items():
        meters = float(name[1:-1])
        attributes = level["attributes"]
        assert attributes["proj:code"] == f"EPSG:{epsg}"
        assert attributes["spatial:transform"] == [meters, 0.0, x0, 0.0, -meters, y0]
        rows, cols = level["members"]["x"]["shape"][0], level["members"]["y"]["shape"][0]
        rows, cols = cols, rows  # y is rows, x is cols
        assert attributes["spatial:bbox"] == [
            x0,
            y0 - rows * meters,
            x0 + cols * meters,
            y0,
        ]
        crs = CRS.from_wkt(level["members"]["spatial_ref"]["attributes"]["crs_wkt"])
        assert crs.to_epsg() == epsg


def test_load_sentinel2_keeps_the_model(tmp_path: pathlib.Path) -> None:
    """
    `load_sentinel2` reads a Zarr store into the validated input model, and
    the plan built from it equals the plan built from the source document.
    """
    path = _INPUT_DIR / f"{_EXACT_PRODUCT}.json"
    document = _read(path)
    store_path = tmp_path / "s2.zarr"
    GroupSpecV2.model_validate(document).to_zarr(store_path, path="")  # type: ignore[arg-type]

    root = load_sentinel2(zarr.open_group(store_path, mode="r", zarr_format=2))
    assert isinstance(root, Sentinel2Root)
    assert plan_geozarr(root) == plan_geozarr(Sentinel2Root.model_validate(document))


def test_load_sentinel2_refuses_non_s2(tmp_path: pathlib.Path) -> None:
    """A store that is not a Sentinel-2 product is refused with a ValidationError."""
    s1_path = next(pathlib.Path("tests/_test_data/s1_examples").glob("*.json"))
    store_path = tmp_path / "s1.zarr"
    GroupSpecV2.model_validate(_read(s1_path)).to_zarr(store_path, path="")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        load_sentinel2(zarr.open_group(store_path, mode="r", zarr_format=2))


def test_product_level_requires_stac_discovery_evidence() -> None:
    """A product whose stac_discovery names no L1C/L2A level is refused."""
    path = _INPUT_PATHS[0]
    document = _read(path)
    document["attributes"]["stac_discovery"] = {"id": "mystery", "properties": {}}
    root = Sentinel2Root.model_validate(document)
    with pytest.raises(ValueError, match="product level"):
        product_level(root)


def test_plan_reflectance_requires_a_resolution_dataset() -> None:
    """A reflectance group with none of the known resolution datasets is refused."""
    path = _INPUT_PATHS[0]
    document = _read(path)
    document["members"]["measurements"]["members"]["reflectance"]["members"] = {}
    root = Sentinel2Root.model_validate(document)
    with pytest.raises(ValueError, match="resolution datasets"):
        plan_reflectance(root)
