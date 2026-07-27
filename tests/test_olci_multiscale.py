"""Tests for olci_multiscale: decimate_swath, reduce_swath, grid_spatial_attrs."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio.transform
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    decimate_swath,
    grid_spatial_attrs,
    reduce_swath,
)


def _swath(rows: int = 8, cols: int = 6) -> xr.Dataset:
    """Minimal synthetic swath dataset with one radiance band and two coords."""
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={
            "scale_factor": 0.5,
            "units": "mW.m-2.sr-1.nm-1",
            "_FillValue": 65535,
        },
    )
    lat = xr.DataArray(
        np.linspace(0, 1, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.linspace(10, 11, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude"},
    )
    return xr.Dataset(
        {"oa01_radiance": rad},
        coords={"latitude": lat, "longitude": lon},
    )


# ---------------------------------------------------------------------------
# decimate_swath tests
# ---------------------------------------------------------------------------


def test_decimate_halves_each_axis() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_decimate_takes_every_other_pixel() -> None:
    ds = _swath(8, 6)
    out = decimate_swath(ds, factor=2)
    # top-left pixel is preserved exactly (no averaging)
    assert int(out["oa01_radiance"].values[0, 0]) == 0
    assert float(out["latitude"].values[0, 0]) == 0.0
    # interior pixel: stride-2 decimation means out[1, 1] comes from original [2, 2]
    assert int(out["oa01_radiance"].values[1, 1]) == int(ds["oa01_radiance"].values[2, 2])


def test_decimate_preserves_attrs() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"


def test_decimate_factor_1_returns_unchanged() -> None:
    ds = _swath(8, 6)
    out = decimate_swath(ds, factor=1)
    assert out["oa01_radiance"].shape == (8, 6)


def test_decimate_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="factor must be >= 1"):
        decimate_swath(_swath(), factor=0)


# ---------------------------------------------------------------------------
# reduce_swath tests
# ---------------------------------------------------------------------------


def test_reduce_swath_halves_each_axis() -> None:
    """reduce_swath must produce output with halved spatial dims."""
    out = reduce_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_reduce_swath_radiance_is_averaged_not_decimated() -> None:
    """Radiance must be block-averaged; top-left output != top-left input (unless accident)."""
    rng = np.random.default_rng(42)
    rad_data = rng.integers(100, 200, (8, 6)).astype("uint16")
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # block [0:2, 0:2] averages to a value; verify it's a rounded mean
    expected_block = int(np.round(rad_data[0:2, 0:2].astype("float64").mean()))
    assert int(out["oa01_radiance"].values[0, 0]) == expected_block


def test_reduce_swath_coordinates_decimated() -> None:
    """Coordinate arrays must be decimated (stride), not averaged."""
    ds = _swath(8, 6)
    out = reduce_swath(ds, factor=2)
    # lat[0,0] in output == lat[0,0] in input
    assert float(out["latitude"].values[0, 0]) == float(ds["latitude"].values[0, 0])
    # lat[1,1] in output == lat[2,2] in input (stride-2)
    assert float(out["latitude"].values[1, 1]) == float(ds["latitude"].values[2, 2])


def test_reduce_swath_fill_value_preserved_in_all_fill_block() -> None:
    """A block where all pixels are fill must produce fill output, not 65535.0 average."""
    fill = 65535
    rad_data = np.ones((4, 4), dtype="uint16") * fill
    # put some non-fill values only in lower-right block
    rad_data[2:4, 2:4] = 100
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": fill},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # top-left block: all fill -> output must be fill
    assert int(out["oa01_radiance"].values[0, 0]) == fill
    # bottom-right block: all 100 -> output must be 100
    assert int(out["oa01_radiance"].values[1, 1]) == 100


def test_reduce_swath_preserves_attrs() -> None:
    """reduce_swath must carry over variable attributes."""
    out = reduce_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"


def test_reduce_swath_factor_1_returns_unchanged() -> None:
    ds = _swath(8, 6)
    out = reduce_swath(ds, factor=1)
    assert out["oa01_radiance"].shape == (8, 6)


def test_reduce_swath_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="factor must be >= 1"):
        reduce_swath(_swath(), factor=0)


def test_reduce_swath_non_swath_var_passthrough() -> None:
    """Variables that don't span (rows, columns) must pass through unchanged."""
    ds = _swath(8, 6)
    scalar = xr.DataArray(42.0, attrs={"info": "scalar"})
    ds = ds.assign({"extra": scalar})
    out = reduce_swath(ds, factor=2)
    assert "extra" in out
    assert float(out["extra"].values) == 42.0


# ---------------------------------------------------------------------------
# odd-dimension regression tests (real OLCI: 4865 columns is odd)
# ---------------------------------------------------------------------------


def _swath_odd(rows: int = 7, cols: int = 5) -> xr.Dataset:
    """Synthetic swath with ODD spatial dimensions.

    Matches the real-world scenario where OLCI products have 4865 columns
    (odd).  Before the fix, reduce_swath on an odd-sized dimension produced
    coordinate arrays one element longer than the corresponding radiance data
    (ceil vs floor of N/factor), causing xr.open_dataset to raise a
    conflicting-sizes error.
    """
    fill = 65535
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={
            "scale_factor": 0.5,
            "units": "mW.m-2.sr-1.nm-1",
            "_FillValue": fill,
        },
    )
    lat = xr.DataArray(
        np.linspace(0, 1, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.linspace(10, 11, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude"},
    )
    return xr.Dataset(
        {"oa01_radiance": rad},
        coords={"latitude": lat, "longitude": lon},
    )


def test_reduce_swath_odd_dims_consistent_shape() -> None:
    """reduce_swath must produce identical shapes for radiance AND coordinates on odd dims.

    Regression test for the off-by-one bug where coordinate decimation via
    [::factor] yields ceil(N/factor) but coarsen(boundary="trim") yields
    floor(N/factor).  For rows=7, cols=5, factor=2 the expected output shape
    is (floor(7/2), floor(5/2)) = (3, 2).
    """
    ds = _swath_odd(rows=7, cols=5)
    out = reduce_swath(ds, factor=2)

    expected_rows = 7 // 2  # 3
    expected_cols = 5 // 2  # 2

    assert out["oa01_radiance"].shape == (expected_rows, expected_cols), (
        f"radiance shape {out['oa01_radiance'].shape} != ({expected_rows}, {expected_cols})"
    )
    assert out["latitude"].shape == (expected_rows, expected_cols), (
        f"latitude shape {out['latitude'].shape} != ({expected_rows}, {expected_cols})"
    )
    assert out["longitude"].shape == (expected_rows, expected_cols), (
        f"longitude shape {out['longitude'].shape} != ({expected_rows}, {expected_cols})"
    )


def test_reduce_swath_odd_dims_radiance_is_block_averaged() -> None:
    """Radiance values must be block-averaged (not decimated) on odd-dim inputs."""
    rng = np.random.default_rng(7)
    rows, cols = 7, 5
    rad_data = rng.integers(100, 200, (rows, cols)).astype("uint16")
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # The top-left output pixel must be the rounded mean of the 2x2 input block.
    expected = int(np.round(rad_data[0:2, 0:2].astype("float64").mean()))
    assert int(out["oa01_radiance"].values[0, 0]) == expected


def test_reduce_swath_odd_dims_coords_decimated() -> None:
    """Coordinate arrays must use stride decimation on odd-dim inputs."""
    ds = _swath_odd(rows=7, cols=5)
    out = reduce_swath(ds, factor=2)
    # Output[0,0] must equal input[0,0] (stride starts at 0).
    assert float(out["latitude"].values[0, 0]) == float(ds["latitude"].values[0, 0])
    # Output[1,1] must equal input[2,2] (stride=2 -> second step at index 2).
    assert float(out["latitude"].values[1, 1]) == float(ds["latitude"].values[2, 2])


def test_reduce_swath_odd_simulates_real_olci_columns() -> None:
    """Simulate the real-world OLCI case: 4090x4865 (odd cols) -> both 2432 cols.

    Uses smaller proxy dimensions that are proportionally odd to avoid
    heavy memory use: rows=10, cols=9 with factor=2 must yield (5, 4) for
    both radiance and coordinates.  This specifically guards floor vs ceil
    on the cols dimension (9 // 2 = 4, not 5).
    """
    ds = _swath_odd(rows=10, cols=9)
    out = reduce_swath(ds, factor=2)
    expected = (10 // 2, 9 // 2)  # (5, 4)
    assert out["oa01_radiance"].shape == expected, (
        f"radiance shape {out['oa01_radiance'].shape} != {expected}"
    )
    assert out["latitude"].shape == expected, (
        f"latitude shape {out['latitude'].shape} != {expected}"
    )
    assert out["longitude"].shape == expected, (
        f"longitude shape {out['longitude'].shape} != {expected}"
    )


# ---------------------------------------------------------------------------
# grid dims tests (for Task 4: reprojected pyramid)
# ---------------------------------------------------------------------------


def test_reduce_swath_on_grid_dims() -> None:
    """reduce_swath(dims=("y","x")) block-averages bands and strides 1-D coords.

    After reprojection the pyramid dims are (y, x): bands average fill-aware,
    the 1-D dimension coordinates decimate by trimmed stride, and non-spatial
    variables (spatial_ref) pass through unchanged.
    """
    ny, nx = 6, 5  # odd x exercises the coarsen-trim alignment
    band = np.arange(ny * nx, dtype="uint16").reshape(ny, nx)
    ds = xr.Dataset(
        {"oa01_radiance": (("y", "x"), band, {"_FillValue": 65535})},
        coords={
            "y": ("y", np.linspace(46.0, 45.0, ny)),
            "x": ("x", np.linspace(10.0, 11.0, nx)),
            "spatial_ref": ((), 0, {"crs_wkt": "stub"}),
        },
    )
    out = reduce_swath(ds, factor=2, dims=("y", "x"))
    assert dict(out.sizes) == {"y": 3, "x": 2}
    # block mean of the top-left 2x2 block, rounded
    expected00 = round((band[0, 0] + band[0, 1] + band[1, 0] + band[1, 1]) / 4)
    assert int(out["oa01_radiance"].values[0, 0]) == expected00
    # 1-D coords: trimmed stride, lengths match the data
    assert out["y"].size == 3
    assert out["x"].size == 2
    np.testing.assert_allclose(out["x"].values, ds["x"].values[0:4:2])
    # scalar passthrough survives
    assert "spatial_ref" in out.coords
    assert out["spatial_ref"].attrs["crs_wkt"] == "stub"


# ---------------------------------------------------------------------------
# grid_spatial_attrs tests
# ---------------------------------------------------------------------------


def test_grid_spatial_attrs() -> None:
    """grid_spatial_attrs derives dimensions, bbox, and 6-element transform."""
    transform = rasterio.transform.from_origin(10.0, 46.0, 0.01, 0.01)
    attrs = grid_spatial_attrs(transform, (100, 200))
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:registration"] == "pixel"  # type: ignore[index]
    assert attrs["spatial:transform"] == [  # type: ignore[index]
        0.01,
        0.0,
        10.0,
        0.0,
        -0.01,
        46.0,
    ]
    # bbox is [xmin, ymin, xmax, ymax] from array_bounds
    assert attrs["spatial:bbox"] == [10.0, 45.0, 12.0, 46.0]  # type: ignore[index]
