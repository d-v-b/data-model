"""Tests for OLCI swath -> regular grid reprojection."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_reproject import reproject_olci

FILL = 65535


def build_rotated_swath(rows: int = 64, cols: int = 60, angle_deg: float = 30.0) -> xr.Dataset:
    """Synthetic OLCI-like swath: regular grid rotated by *angle_deg* in lon/lat.

    Rotation matters: it makes the dst bounding-box corners fall outside the
    swath, exercising real geolocation warping and off-swath fill.
    """
    rr, cc = np.meshgrid(
        np.arange(rows, dtype="float64"), np.arange(cols, dtype="float64"), indexing="ij"
    )
    theta = np.deg2rad(angle_deg)
    lon = 10.0 + 0.01 * (cc * np.cos(theta) - rr * np.sin(theta))
    lat = 45.0 + 0.01 * (cc * np.sin(theta) + rr * np.cos(theta))

    band = (100.0 * rr + cc).astype("uint16")
    band[:4, :4] = FILL  # a filled block inside the swath
    da = xr.DataArray(
        band,
        dims=("rows", "columns"),
        attrs={"scale_factor": 0.0139, "add_offset": 0.0, "_FillValue": FILL},
    )
    time_stamp = xr.DataArray(np.arange(rows).astype("datetime64[ns]"), dims=("rows",))
    return xr.Dataset(
        {"oa01_radiance": da},
        coords={
            "latitude": (("rows", "columns"), lat),
            "longitude": (("rows", "columns"), lon),
            "altitude": (("rows", "columns"), np.full((rows, cols), 7, dtype="int16")),
            "time_stamp": time_stamp,
        },
    )


def test_reproject_olci_produces_regular_grid_with_crs() -> None:
    """One behavior test: grid shape, CRS, dtype, attrs, fill, and dropped vars."""
    ds = build_rotated_swath()
    out = reproject_olci(ds)

    # Regular 1-D coordinate grid over (y, x)
    assert set(out.sizes) == {"y", "x"}
    for dim in ("y", "x"):
        coord = out[dim]
        assert coord.ndim == 1
        steps = np.diff(coord.values)
        assert np.allclose(steps, steps[0])
    # y descends (north-up grid)
    assert out["y"].values[0] > out["y"].values[-1]

    # CRS declared in both idioms
    assert out.rio.crs is not None
    assert out.rio.crs.to_epsg() == 4326
    assert "spatial_ref" in out.coords or "spatial_ref" in out.variables
    assert out["oa01_radiance"].attrs["grid_mapping"] == "spatial_ref"

    # dtype and CF scaling preserved; fill recorded
    band = out["oa01_radiance"]
    assert band.dtype == np.dtype("uint16")
    assert band.attrs["scale_factor"] == 0.0139
    assert int(band.attrs["_FillValue"]) == FILL

    # Off-swath cells (bbox corners of a rotated swath) are fill
    vals = band.values
    assert vals[0, 0] == FILL
    assert vals[-1, -1] == FILL
    # …and real data survived the warp
    valid = vals[vals != FILL]
    assert valid.size > 0
    assert valid.max() <= (100.0 * 64 + 60)

    # altitude warped onto the grid; per-scan-line time_stamp dropped
    assert "altitude" in out.data_vars
    assert out["altitude"].dims == ("y", "x")
    assert "time_stamp" not in out.variables


def test_reproject_olci_missing_geolocation_raises() -> None:
    ds = build_rotated_swath().drop_vars("latitude")
    with pytest.raises(ValueError, match="latitude"):
        reproject_olci(ds)


def test_reproject_olci_degenerate_extent_raises() -> None:
    ds = build_rotated_swath()
    ds = ds.assign_coords(
        latitude=(("rows", "columns"), np.full((64, 60), 45.0)),
        longitude=(("rows", "columns"), np.full((64, 60), 10.0)),
    )
    with pytest.raises(ValueError, match="degenerate"):
        reproject_olci(ds)
