"""Integration tests: converter output must satisfy the GeoZarr minispec validator."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import rioxarray  # noqa: F401  # enable .rio accessor
import xarray as xr

from eopf_geozarr.conversion import create_geozarr_dataset
from eopf_geozarr.data_api.geozarr.validation import validate_store


@pytest.fixture
def synthetic_s2_tree() -> xr.DataTree:
    """Small S2-like tree; r60m is below the default min_dimension on purpose."""
    epsg = 32632
    x_min, x_max = 600000, 605490
    y_min, y_max = 5090000, 5095490
    sizes = {"r10m": 549, "r20m": 275, "r60m": 92}
    bands = {"r10m": ["b02", "b03"], "r20m": ["b05"], "r60m": ["b01"]}
    rng = np.random.default_rng(42)
    dt = xr.DataTree()
    dt["measurements"] = xr.DataTree()
    dt["measurements/reflectance"] = xr.DataTree()
    for res, n in sizes.items():
        coords = {
            "x": np.linspace(x_min, x_max, n, endpoint=False),
            "y": np.linspace(y_max, y_min, n, endpoint=False),
            "time": [np.datetime64("2025-01-13T10:33:09")],
        }
        data = {
            b: (
                ["time", "y", "x"],
                rng.integers(500, 3000, (1, n, n), dtype=np.uint16),
                {"proj:epsg": epsg},
            )
            for b in bands[res]
        }
        ds = xr.Dataset(data, coords=coords).rio.write_crs(f"EPSG:{epsg}")
        dt[f"measurements/reflectance/{res}"] = ds
    dt.attrs = {"title": "synthetic S2", "product_type": "S2MSI1C"}
    return dt


def test_create_geozarr_dataset_output_is_minispec_compliant(
    synthetic_s2_tree: xr.DataTree, tmp_path: pathlib.Path
) -> None:
    output = str(tmp_path / "geozarr.zarr")
    create_geozarr_dataset(
        dt_input=synthetic_s2_tree,
        groups=[
            "/measurements/reflectance/r10m",
            "/measurements/reflectance/r20m",
            "/measurements/reflectance/r60m",
        ],
        output_path=output,
        spatial_chunk=4096,
        min_dimension=256,
        max_retries=3,
    )
    report = validate_store(output)
    assert report.compliant, "\n".join(str(i) for i in report.issues)
