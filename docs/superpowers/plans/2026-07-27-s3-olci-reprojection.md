# Sentinel-3 OLCI Reprojection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warp OLCI swath measurements to a regular EPSG:4326 grid with declared CRS metadata, and encode "reprojection is a hard requirement for all products" in a cross-product contract test.

**Architecture:** A new `olci_reproject` module warps the native swath once via rasterio's dense geolocation-array support; the existing r0/r2/… pyramid then builds on the grid with 1-D `y`/`x` coordinates. CRS is declared per level in both idioms (rioxarray `spatial_ref`/`grid_mapping` + zarr-cm `geo-proj` convention). A parametrized contract test asserts the grid+CRS requirement for S1, S2, and OLCI outputs.

**Tech Stack:** rasterio 1.5 (`calculate_default_transform`/`reproject` with `src_geoloc_array`), rioxarray, pyproj, zarr-cm via `eopf_geozarr.conversion.utils.build_convention_attrs`.

**Spec:** `docs/superpowers/specs/2026-07-27-s3-olci-reprojection-design.md`

## Global Constraints

- Never use `typing.Any` — use `object` or precise types and narrow (user rule).
- Conventional commits; every commit ends with trailer `Assisted-by: ClaudeCode:claude-fable-5`. No Co-Authored-By line.
- TDD: write the failing test, watch it fail, implement, watch it pass, commit.
- All commands run via `uv run …`. `uv run pyright` must stay at `0 errors`.
- Test structure rule: one behavior test covering reasonable input combinations + one test function per error case.
- Target CRS parameter default is exactly `"EPSG:4326"`; CLI flag is `--target-crs`.
- Snapshot regeneration uses the uncomment/run/re-comment block inside `test_olci_conversion_matches_snapshot` (never leave it uncommented in a commit).

---

### Task 1: `reproject_olci` warp helper

**Files:**
- Create: `src/eopf_geozarr/s3_olci_optimization/olci_reproject.py`
- Create: `tests/test_olci_reproject.py`

**Interfaces:**
- Consumes: nothing new (rasterio/rioxarray already installed).
- Produces: `reproject_olci(ds: xr.Dataset, *, target_crs: str = "EPSG:4326", resampling: Resampling = Resampling.bilinear) -> xr.Dataset` and module constant `GRID_DIMS = ("y", "x")`. Output dataset: dims `("y", "x")`; 1-D `y`/`x` dimension coords; `spatial_ref` coordinate; every data var carries `grid_mapping: "spatial_ref"` and `_FillValue`; `ds.rio.crs` set. Raises `ValueError` on missing lat/lon coords and on degenerate (zero-area) geolocation extent.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_olci_reproject.py`:

```python
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
    time_stamp = xr.DataArray(
        np.arange(rows).astype("datetime64[ns]"), dims=("rows",)
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_olci_reproject.py -v`
Expected: FAIL (all three) with `ModuleNotFoundError: No module named 'eopf_geozarr.s3_olci_optimization.olci_reproject'`

- [ ] **Step 3: Write the implementation**

Create `src/eopf_geozarr/s3_olci_optimization/olci_reproject.py`:

```python
"""Reprojection of OLCI swath measurements to a regular grid.

OLCI L1 EFR is a curvilinear swath geolocated by dense per-pixel 2-D
latitude/longitude arrays.  Generic GeoZarr readers (titiler, rioxarray)
require a regular grid with an affine transform and a declared CRS, so the
converter warps the swath once at native resolution using rasterio's
geolocation-array support (``src_geoloc_array``, rasterio >= 1.4).
"""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  # enables the .rio accessor
import structlog
import xarray as xr
from pyproj import CRS as ProjCRS
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

log = structlog.get_logger()

#: CRS of the OLCI geolocation arrays (per-pixel lat/lon in degrees).
GEOLOC_CRS = "EPSG:4326"

#: Dimension names of the reprojected regular grid.
GRID_DIMS: tuple[str, str] = ("y", "x")

_SWATH_DIMS = ("rows", "columns")


def _nodata_for(var: xr.DataArray) -> float:
    """Warp nodata for *var*: its ``_FillValue`` if present, else a dtype default.

    Integer variables without a fill value get the dtype maximum (matching the
    OLCI convention of 65535 for uint16 radiances); floats get NaN.
    """
    fill = var.attrs.get("_FillValue")
    if fill is None:
        fill = var.encoding.get("_FillValue")
    if fill is not None:
        return float(fill)
    if np.issubdtype(var.dtype, np.integer):
        return float(np.iinfo(var.dtype).max)
    return float("nan")


def _grid_coord_attrs(target_crs: str) -> tuple[dict[str, str], dict[str, str]]:
    """CF attrs for the 1-D (y, x) dimension coordinates in *target_crs*."""
    if ProjCRS.from_user_input(target_crs).is_geographic:
        y_attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
        x_attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    else:
        y_attrs = {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}
        x_attrs = {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}
    return y_attrs, x_attrs


def reproject_olci(
    ds: xr.Dataset,
    *,
    target_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
) -> xr.Dataset:
    """Warp an OLCI swath dataset onto a regular *target_crs* grid.

    Every numeric variable spanning exactly ``(rows, columns)`` — radiance
    bands and the 2-D ``altitude`` coordinate alike — is warped onto a common
    grid sized by :func:`rasterio.warp.calculate_default_transform` from the
    dense per-pixel geolocation (~native resolution preserved).  The
    ``latitude``/``longitude`` geolocation arrays are consumed by the warp and
    replaced by 1-D ``y``/``x`` dimension coordinates.  Variables that carry a
    swath dim but cannot live on the grid (e.g. per-scan-line ``time_stamp``)
    are dropped; variables without swath dims pass through unchanged.

    Off-swath cells are set to each variable's ``_FillValue`` (dtype max for
    integer variables without one), and ``_FillValue`` is recorded in the
    output attrs so downstream fill-aware averaging keeps working.

    Raises
    ------
    ValueError
        If the 2-D ``latitude``/``longitude`` coordinates are missing, or if
        the geolocation spans a zero-area (degenerate) extent.
    """
    for required in ("latitude", "longitude"):
        if required not in ds.coords:
            raise ValueError(
                f"cannot reproject OLCI swath: missing 2-D coordinate {required!r}"
            )
    lat = ds.coords["latitude"]
    lon = ds.coords["longitude"]
    if tuple(str(d) for d in lat.dims) != _SWATH_DIMS or lat.dims != lon.dims:
        raise ValueError("latitude/longitude must be 2-D over (rows, columns)")
    lat_vals = np.asarray(lat.values, dtype="float64")
    lon_vals = np.asarray(lon.values, dtype="float64")
    if lat_vals.max() == lat_vals.min() or lon_vals.max() == lon_vals.min():
        raise ValueError(
            "degenerate geolocation extent: latitude/longitude span zero area"
        )

    src_height, src_width = lat_vals.shape
    geoloc = np.stack([lon_vals, lat_vals])  # (2, H, W): x first, then y
    transform, width, height = calculate_default_transform(
        src_crs=CRS.from_string(GEOLOC_CRS),
        dst_crs=CRS.from_string(target_crs),
        width=src_width,
        height=src_height,
        src_geoloc_array=geoloc,
    )
    assert width is not None
    assert height is not None
    log.info(
        "Reprojecting OLCI swath",
        target_crs=target_crs,
        src_shape=(src_height, src_width),
        dst_shape=(height, width),
    )

    result_vars: dict[str, xr.DataArray] = {}
    passthrough_coords: dict[str, xr.DataArray] = {}
    all_names = [str(k) for k in ds.data_vars] + [
        str(k) for k in ds.coords if str(k) not in ("latitude", "longitude")
    ]
    for name in all_names:
        var = ds[name] if name in ds.data_vars else ds.coords[name]
        var_dims = tuple(str(d) for d in var.dims)
        if var_dims == _SWATH_DIMS and np.issubdtype(var.dtype, np.number):
            nodata = _nodata_for(var)
            src_nodata = (
                nodata
                if (var.attrs.get("_FillValue") is not None
                    or var.encoding.get("_FillValue") is not None)
                else None
            )
            dest = np.full((height, width), nodata, dtype=var.dtype)
            reproject(
                source=np.ascontiguousarray(var.values),
                destination=dest,
                src_crs=CRS.from_string(GEOLOC_CRS),
                src_geoloc_array=geoloc,
                src_nodata=src_nodata,
                dst_crs=CRS.from_string(target_crs),
                dst_transform=transform,
                dst_nodata=nodata,
                resampling=resampling,
            )
            out_attrs = dict(var.attrs)
            if np.issubdtype(var.dtype, np.integer):
                out_attrs["_FillValue"] = int(nodata)
            elif not np.isnan(nodata):
                out_attrs["_FillValue"] = float(nodata)
            out_attrs.pop("coordinates", None)  # swath geolocation is gone
            result_vars[name] = xr.DataArray(dest, dims=GRID_DIMS, attrs=out_attrs)
        elif any(d in var_dims for d in _SWATH_DIMS):
            log.info("Dropping swath-bound variable with no grid home", variable=name)
        elif name in ds.coords:
            passthrough_coords[name] = var
        else:
            result_vars[name] = var

    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    y_attrs, x_attrs = _grid_coord_attrs(target_crs)
    out = xr.Dataset(
        result_vars,
        coords={
            "y": ("y", ys, y_attrs),
            "x": ("x", xs, x_attrs),
            **passthrough_coords,
        },
        attrs=dict(ds.attrs),
    )
    out = out.rio.write_crs(target_crs)
    if not isinstance(out, xr.Dataset):
        raise TypeError(
            f"expected an xarray.Dataset after write_crs, got {type(out).__name__}"
        )
    # rioxarray records grid_mapping in encoding; pin it in attrs so it is
    # guaranteed to reach the zarr store.
    for name in out.data_vars:
        out[name].attrs["grid_mapping"] = "spatial_ref"
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_olci_reproject.py -v`
Expected: 3 passed. If `test_reproject_olci_produces_regular_grid_with_crs` fails on the corner-fill assertions, print `band.values[0, 0]` — a rotated swath must leave bbox corners at fill; investigate the warp call rather than weakening the test.

- [ ] **Step 5: Run pyright and commit**

Run: `uv run pyright` — expected `0 errors`.

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_reproject.py tests/test_olci_reproject.py
git commit -m "feat(s3-olci): add swath-to-grid reprojection via rasterio geolocation arrays

Assisted-by: ClaudeCode:claude-fable-5"
```

---

### Task 2: Generalize `reduce_swath`/`decimate_swath` to grid dims

**Files:**
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py`
- Test: `tests/test_olci_multiscale.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `decimate_swath(ds, factor=2, *, dims: tuple[str, str] = SWATH_DIMS)` and `reduce_swath(ds, factor=2, *, dims: tuple[str, str] = SWATH_DIMS)`. Existing call sites without `dims` behave identically; Task 4 calls `reduce_swath(current, factor=2, dims=("y", "x"))`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_olci_multiscale.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_multiscale.py::test_reduce_swath_on_grid_dims -v`
Expected: FAIL with `TypeError: reduce_swath() got an unexpected keyword argument 'dims'`

- [ ] **Step 3: Implement**

In `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py`, thread `dims` through both functions. Signatures:

```python
def decimate_swath(
    ds: xr.Dataset, factor: int = 2, *, dims: tuple[str, str] = SWATH_DIMS
) -> xr.Dataset:
```

```python
def reduce_swath(
    ds: xr.Dataset, factor: int = 2, *, dims: tuple[str, str] = SWATH_DIMS
) -> xr.Dataset:
```

Inside both function bodies, replace every use of the module constant `SWATH_DIMS` with the `dims` parameter (five sites: the `indexers` comprehension in `decimate_swath`; and in `reduce_swath` the `dim_trim` comprehension, the `is_swath_2d` comparison, the coarsen dict — change `{"rows": factor, "columns": factor}` to `{dims[0]: factor, dims[1]: factor}` — and the two stride-indexer comprehensions). Update both docstrings' first lines to say "spanning the *dims* spatial dimensions (default ``(rows, columns)``)". Do not change `SWATH_DIMS` itself or `OLCI_BANDS` handling.

- [ ] **Step 4: Run tests to verify pass (new + existing)**

Run: `uv run pytest tests/test_olci_multiscale.py -v`
Expected: all pass (existing swath-dims tests confirm default behavior unchanged).

- [ ] **Step 5: Run pyright and commit**

Run: `uv run pyright` — expected `0 errors`.

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py tests/test_olci_multiscale.py
git commit -m "refactor(s3-olci): parameterize reduce/decimate over spatial dims for gridded pyramids

Assisted-by: ClaudeCode:claude-fable-5"
```

---

### Task 3: `grid_spatial_attrs` helper

**Files:**
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py`
- Test: `tests/test_olci_multiscale.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `grid_spatial_attrs(transform: Affine, shape: tuple[int, int]) -> SpatialAttrs` (shape is `(height, width)`). Task 4 calls it per pyramid level. `swath_spatial_attrs` stays for now — Task 4 deletes it with its call site.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_olci_multiscale.py`:

```python
def test_grid_spatial_attrs() -> None:
    """grid_spatial_attrs derives dimensions, bbox, and 6-element transform."""
    transform = rasterio.transform.from_origin(10.0, 46.0, 0.01, 0.01)
    attrs = grid_spatial_attrs(transform, (100, 200))
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:registration"] == "pixel"
    assert attrs["spatial:transform"] == [0.01, 0.0, 10.0, 0.0, -0.01, 46.0]
    # bbox is [xmin, ymin, xmax, ymax] from array_bounds
    assert attrs["spatial:bbox"] == [10.0, 45.0, 12.0, 46.0]
```

Add the imports at the top of the test file: `import rasterio.transform` and extend the existing `olci_multiscale` import with `grid_spatial_attrs`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_olci_multiscale.py::test_grid_spatial_attrs -v`
Expected: FAIL with `ImportError: cannot import name 'grid_spatial_attrs'`

- [ ] **Step 3: Implement**

In `olci_multiscale.py` add (with `import rasterio.transform` at module top and `from affine import Affine` under `TYPE_CHECKING`):

```python
def grid_spatial_attrs(transform: Affine, shape: tuple[int, int]) -> SpatialAttrs:
    """Spatial-convention data for a regular grid with an affine *transform*.

    *shape* is ``(height, width)``.  Emits ``spatial:dimensions`` ``["y","x"]``,
    pixel registration, the bounding box, and the 6-element row-major affine
    transform — the gridded counterpart of :func:`swath_spatial_attrs`.
    """
    height, width = shape
    left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
    return {
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:bbox": [float(left), float(bottom), float(right), float(top)],
        "spatial:transform": [
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_olci_multiscale.py -v`
Expected: all pass.

- [ ] **Step 5: Run pyright and commit**

Run: `uv run pyright` — expected `0 errors`.

```bash
git add src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py tests/test_olci_multiscale.py
git commit -m "feat(s3-olci): add gridded spatial-convention attrs with affine transform

Assisted-by: ClaudeCode:claude-fable-5"
```

---

### Task 4: Integrate reprojection into `convert_olci_optimized`

**Files:**
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_converter.py`
- Modify: `tests/test_olci_integration.py`
- Modify: `src/eopf_geozarr/s3_olci_optimization/olci_multiscale.py` (delete `swath_spatial_attrs`)
- Modify: `tests/test_olci_multiscale.py` (delete `swath_spatial_attrs` tests)

**Interfaces:**
- Consumes: `reproject_olci`, `GRID_DIMS` (Task 1); `reduce_swath(..., dims=...)` (Task 2); `grid_spatial_attrs` (Task 3).
- Produces: `convert_olci_optimized(..., target_crs: str = "EPSG:4326")`. Output store: `measurements/r0..rN` on a regular grid, per-level `spatial_ref` + `grid_mapping` + zarr-cm spatial/geo-proj attrs; parent `measurements` carries multiscales + spatial + geo-proj. Task 5 (CLI) and Task 6 (contract test) rely on this signature and layout.

- [ ] **Step 1: Fix the degenerate synthetic fixture and update the acceptance test (RED)**

In `tests/test_olci_integration.py`, replace the lat/lon construction in `build_synthetic_olci` (currently a flattened `linspace` reshape — collinear in lon/lat and therefore unwarpable):

```python
def build_synthetic_olci(rows: int = 512, cols: int = 480) -> xr.DataTree:
    """Minimal synthetic OLCI L1 EFR datatree (measurements only).

    Geolocation is an axis-aligned ~300 m grid (0.003 deg spacing) so the
    dataset is genuinely warpable to a regular lat/lon grid.
    """
    rng = np.random.default_rng(0)
    lat_1d = np.linspace(45.0, 45.0 + 0.003 * (rows - 1), rows)
    lon_1d = np.linspace(10.0, 10.0 + 0.003 * (cols - 1), cols)
    lat = np.repeat(lat_1d[:, None], cols, axis=1)
    lon = np.repeat(lon_1d[None, :], rows, axis=0)
    alt = np.zeros((rows, cols), dtype="int16")
```

(keep the band-building loop and Dataset assembly unchanged below this point).

Apply the same replacement to the inline lat/lon in `test_convert_olci_conditions_quality_passthrough` (the `128, 128` case):

```python
    lat_1d = np.linspace(45.0, 45.0 + 0.003 * (rows - 1), rows)
    lon_1d = np.linspace(10.0, 10.0 + 0.003 * (cols - 1), cols)
    lat = np.repeat(lat_1d[:, None], cols, axis=1)
    lon = np.repeat(lon_1d[None, :], rows, axis=0)
```

Rewrite `test_convert_olci_output_opens_as_datatree` for the gridded contract (warped sizes are ~input sizes but not exact, so assert structure relative to observed r0):

```python
def test_convert_olci_output_opens_as_datatree(tmp_path: object) -> None:
    """Acceptance: the exported store is a regular grid with CRS at every level.

    xr.open_datatree must open the whole store; measurements/r0 holds the
    warped native-resolution grid with 1-D y/x coordinates and a declared
    CRS, overview siblings halve it, and the multiscales layout references
    the base level by name.
    """
    dt = build_synthetic_olci(rows=1024, cols=1024)
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    result = convert_olci_optimized(dt, output_path=out, min_dimension=256)

    opened = xr.open_datatree(out, engine="zarr", consolidated=False, chunks={})

    r0 = opened["/measurements/r0"].to_dataset()
    assert set(r0.sizes) == {"y", "x"}
    for i in range(1, 22):
        assert f"oa{i:02d}_radiance" in r0
    # CRS declared at every level, 1-D regular coordinates
    for level in ("r0", "r2", "r4"):
        ds = opened[f"/measurements/{level}"].to_dataset()
        assert ds.rio.crs is not None, f"{level}: no CRS"
        assert ds.rio.crs.to_epsg() == 4326
        for dim in ("y", "x"):
            coord = ds[dim]
            assert coord.ndim == 1
            steps = np.diff(coord.values)
            assert np.allclose(steps, steps[0])
        band = ds["oa01_radiance"]
        assert band.attrs["grid_mapping"] == "spatial_ref"
    # halving structure relative to observed r0
    r2 = opened["/measurements/r2"].to_dataset()
    assert r2.sizes["y"] == r0.sizes["y"] // 2
    assert r2.sizes["x"] == r0.sizes["x"] // 2

    # measurements itself holds only convention metadata
    meas = opened["/measurements"].to_dataset()
    assert len(meas.data_vars) == 0

    meas_attrs = dict(zarr.open_group(out, mode="r")["measurements"].attrs)
    multiscales = meas_attrs["multiscales"]
    assert isinstance(multiscales, dict)
    layout = multiscales["layout"]
    assert isinstance(layout, list)
    assert layout[0] == {"asset": "r0"}
    # per-level geo-proj convention present
    r0_attrs = dict(zarr.open_group(out, mode="r")["measurements"]["r0"].attrs)
    assert "proj:code" in r0_attrs or any(k.startswith("proj:") for k in r0_attrs)

    assert "/measurements/r0" in result.groups
    assert "/measurements/r2" in result.groups
```

Add `import rioxarray  # noqa: F401` to the test file's imports.

Note: before asserting on the exact `proj:` key, check what `proj_attrs_for_crs("EPSG:4326")` emits (`uv run python -c "from eopf_geozarr.conversion.utils import proj_attrs_for_crs; print(proj_attrs_for_crs('EPSG:4326'))"`) and pin the real key instead of the `any(...)` fallback.

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/test_olci_integration.py::test_convert_olci_output_opens_as_datatree -v`
Expected: FAIL — r0 has dims `rows`/`columns` (no warp yet), `set(r0.sizes) == {"y", "x"}` assertion fails.

- [ ] **Step 3: Implement converter changes**

In `olci_converter.py`:

1. Imports: replace the `swath_spatial_attrs` import with `grid_spatial_attrs`; add `from eopf_geozarr.s3_olci_optimization.olci_reproject import GRID_DIMS, reproject_olci` and `import rioxarray  # noqa: F401`.
2. Signature: add `target_crs: str = "EPSG:4326"` after `keep_scale_offset`, and document it in the docstring ("Target CRS for the output grid; the swath is warped once at native resolution before the pyramid builds.").
3. After `measurements = _sanitize_data_vars(measurements)` insert:

```python
    measurements = reproject_olci(measurements, target_crs=target_crs)
```

4. Replace the two native-size reads: `rows = measurements.sizes["y"]` and `cols = measurements.sizes["x"]`.
5. Overview loop: `current = reduce_swath(current, factor=2, dims=GRID_DIMS)`; collect levels while writing — before the loop add `level_datasets: dict[str, xr.Dataset] = {"r0": measurements}` and inside the loop, after computing `group_name`, add `level_datasets[group_name] = current`.
6. Replace the convention-attrs block (the `if n_levels > 0:` section) with per-level + parent writes:

```python
    root_rw = zarr.open_group(output_path, mode="a")
    base_transform = measurements.rio.transform(recalc=True)
    base_spatial = grid_spatial_attrs(
        base_transform, (measurements.sizes["y"], measurements.sizes["x"])
    )
    for group_name, level_ds in level_datasets.items():
        level_transform = level_ds.rio.transform(recalc=True)
        level_conv = build_convention_attrs(
            spatial=grid_spatial_attrs(
                level_transform, (level_ds.sizes["y"], level_ds.sizes["x"])
            ),
            crs=target_crs,
        )
        root_rw[f"measurements/{group_name}"].attrs.update(
            cast("dict[str, JSON]", level_conv)
        )

    if n_levels > 0:
        ms: MultiscalesAttrs = {"layout": layout, "resampling_method": "average"}
        conv = build_convention_attrs(
            multiscales=ms, spatial=base_spatial, crs=target_crs
        )
    else:
        conv = build_convention_attrs(spatial=base_spatial, crs=target_crs)

    root_rw["measurements"].attrs.update(cast("dict[str, JSON]", conv))
```

(The existing `zarr.open_group(output_path, mode="a")["measurements"].attrs.update(...)` line is replaced by this block; keep the `layout` construction above it unchanged.)

7. Docstring: update the flow description to "warps the swath once to a regular *target_crs* grid, writes it to ``measurements/r0``, then /2-reduced overview siblings"; note that `time_stamp` is dropped (survives in the source product) and that every level carries `spatial_ref`/`grid_mapping` plus zarr-cm spatial/geo-proj attrs.
8. In `olci_multiscale.py` delete `swath_spatial_attrs` entirely; in `tests/test_olci_multiscale.py` delete its tests (search `swath_spatial_attrs`).

- [ ] **Step 4: Run OLCI tests, fix the remaining old-layout assertions**

Run: `uv run pytest tests/test_olci_integration.py tests/test_olci_multiscale.py -v`

Expected initially: the acceptance test passes; these tests fail on swath-era assumptions and must be updated in place:

- `test_convert_olci_creates_overviews`: replace exact-size expectations with structural ones — read the observed r0 sizes and assert (a) every level's dims `>= 256` for case 2, (b) each level halves the previous, and (c) the deepest level would violate `min_dimension` if halved again:

```python
    r0_2 = meas2["r0"]
    assert isinstance(r0_2, zarr.Group)
    band0 = r0_2["oa01_radiance"]
    assert isinstance(band0, zarr.Array)
    prev_shape = band0.shape
    for sg_name in [k for k in subgroups2 if k != "r0"]:
        sg = meas2[sg_name]
        assert isinstance(sg, zarr.Group)
        band = sg["oa01_radiance"]
        assert isinstance(band, zarr.Array)
        assert band.shape[0] == prev_shape[0] // 2
        assert band.shape[1] == prev_shape[1] // 2
        assert band.shape[0] >= 256 and band.shape[1] >= 256
        prev_shape = band.shape
    assert prev_shape[0] // 2 < 256 or prev_shape[1] // 2 < 256
```

  For case 1 keep `subgroups1 == ["r0"]` only if the warped 512x480 grid still yields zero levels at `min_dimension=256` (halving 480-ish < 256 — it does).
- `test_convert_olci_odd_dims_overview_no_conflicting_sizes`: the warped grid's dims won't be exactly 10x9; keep the intent (coord/data length agreement at every level) by asserting, for each written level, `ds["oa01_radiance"].shape == (ds["y"].size, ds["x"].size)` and that each level floor-halves the previous; drop the exact `(5, 4)` assertion.
- Any test asserting `latitude`/`longitude` arrays exist in the output: update to `y`/`x` (`test_convert_olci_writes_measurements` band loop is unaffected).

Then: `uv run pytest tests/test_olci_integration.py tests/test_olci_multiscale.py tests/test_olci_reproject.py -v` — all pass except `test_olci_conversion_matches_snapshot` (fails on degenerate zero geolocation — fixed in Task 5).

- [ ] **Step 5: Commit**

```bash
git add src/eopf_geozarr/s3_olci_optimization/ tests/test_olci_integration.py tests/test_olci_multiscale.py
git commit -m "feat(s3-olci): reproject measurements to regular EPSG:4326 grid with per-level CRS

Assisted-by: ClaudeCode:claude-fable-5"
```

---

### Task 5: Snapshot, CLI flag, docs, notebook

**Files:**
- Modify: `tests/test_olci_integration.py` (snapshot test seeding)
- Regenerate: `tests/_test_data/optimized_olci_examples/S3A_OL_1_EFR____20251101T073957_20251101T074257_20251102T084255_0179_132_149_2160_PS1_O_NT_004.json`
- Modify: `src/eopf_geozarr/cli.py`
- Modify: `README.md`, `docs/converter.md`
- Modify + re-execute: `docs/notebooks/sentinel3_olci_geozarr.ipynb`

**Interfaces:**
- Consumes: `convert_olci_optimized(..., target_crs=...)` from Task 4.
- Produces: `--target-crs` CLI flag; regenerated golden snapshot; docs describing the gridded layout.

- [ ] **Step 1: Seed real geolocation in the snapshot test**

The golden fixture materializes arrays with no data (all zeros), which is now a degenerate geolocation. In `test_olci_conversion_matches_snapshot`, before the `xr.open_datatree(...)` call, add:

```python
    # The JSON fixture materializes arrays as zeros; zero lat/lon is a
    # degenerate geolocation the warp rejects. Seed a plausible ~300 m grid.
    fixture_group = zarr.open_group(str(s3_olci_group_example), mode="a")
    fixture_meas = fixture_group["measurements"]
    assert isinstance(fixture_meas, zarr.Group)
    lat_arr = fixture_meas["latitude"]
    assert isinstance(lat_arr, zarr.Array)
    ny, nx = lat_arr.shape
    lat_1d = np.linspace(45.0, 45.0 + 0.003 * (ny - 1), ny)
    lon_1d = np.linspace(10.0, 10.0 + 0.003 * (nx - 1), nx)
    lat_arr[:] = np.repeat(lat_1d[:, None], nx, axis=1)
    lon_arr = fixture_meas["longitude"]
    assert isinstance(lon_arr, zarr.Array)
    lon_arr[:] = np.repeat(lon_1d[None, :], ny, axis=0)
```

Also update the `_assert_radiance_dtype_and_attrs` call sites if level group names changed (they did not — still `r0`/`r2`).

- [ ] **Step 2: Regenerate the snapshot**

Uncomment the regeneration block in `test_olci_conversion_matches_snapshot`, run:

`uv run pytest "tests/test_olci_integration.py::test_olci_conversion_matches_snapshot" -v`

then RE-COMMENT the block and re-run to confirm it passes against the committed file. Inspect the snapshot diff: expect `rows`/`columns` dims replaced by `y`/`x`, new `spatial_ref` arrays, `grid_mapping` attrs, per-level `proj:*`/`spatial:*` attrs, no `time_stamp`/`latitude`/`longitude` arrays under `measurements/r*`.

- [ ] **Step 3: CLI flag**

In `src/eopf_geozarr/cli.py`, in `add_s3_olci_optimization_commands` (after the `--keep-scale-offset` argument):

```python
    p.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS for the reprojected output grid (default: EPSG:4326)",
    )
```

and in `convert_s3_olci_optimized_command`, pass `target_crs=args.target_crs` in the `convert_olci_optimized(...)` call. The auto-detect path in `convert` keeps the default (no change).

Run: `uv run pytest tests/test_olci_integration.py::test_cli_convert_s3_olci_optimized -v` — expected PASS.

- [ ] **Step 4: Docs**

`README.md` "What is converted" OLCI section — replace the three bullets with:

```markdown
- **`/measurements/r0`**: all 21 OLCI radiance bands warped once from the
  native swath onto a regular grid (default `EPSG:4326`, ~300 m preserved),
  with 1-D `y`/`x` coordinates, a `spatial_ref` variable, and `grid_mapping`
  on every band; the parent `measurements/` group carries the GeoZarr
  `multiscales`, `spatial:`, and `proj:` convention metadata.
- **Overview subgroups** (`r2`, `r4`, …): /2 fill-aware block-averaged copies
  stored as sibling Zarr groups next to `r0`, each with its own CRS metadata.
- **`/conditions` and `/quality`**: copied through unmodified.
```

Add below the note: per-scan-line `time_stamp` is not representable on a regular grid and is dropped from the converted measurements (it remains in the source product).

`docs/converter.md` OLCI section — update the "native swath geometry" paragraph to say measurements are reprojected to a regular grid (default EPSG:4326) with the swath geolocation consumed by the warp; update the output-layout tree (`r0/` line becomes "Warped native-resolution OLCI bands … with 1-D y/x coordinates and spatial_ref") and add a `--target-crs` row to the flag table:

```markdown
| `--target-crs` | EPSG:4326 | Target CRS for the reprojected output grid |
```

Run: `uv run pytest tests/test_docs.py -v` — expected PASS (no executable snippets changed).

- [ ] **Step 5: Notebook**

Source edits in `docs/notebooks/sentinel3_olci_geozarr.ipynb` (edit with `uv run --with nbformat python` and `nbformat.read`/`write`, asserting each old string exists before replacing, as in the r0-layout change):

- Conversion markdown cell: describe the warp ("`convert_olci_optimized` first warps the swath onto a regular EPSG:4326 grid using the per-pixel geolocation, then writes the `r0` base level plus /2 block-averaged overview siblings").
- Add to the pyramid-levels code cell (after the `levels` print): `print("CRS:", xr.open_dataset(output_path, engine="zarr", group="measurements/r0").rio.crs)` and `import rioxarray  # noqa: F401` in the imports cell.
- The offline-fallback path in cell 3 uses the zero-filled fixture, which the warp now rejects; extend the fallback to seed lat/lon exactly as in Task 5 Step 1 (same meshgrid code against the materialized fixture store) before opening it.

Re-execute end-to-end (remote product has real geolocation):

`uv run --with jupyter,nbconvert,ipykernel,matplotlib jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/sentinel3_olci_geozarr.ipynb`

Then verify with nbformat: zero error outputs, pyramid levels print `r0…`, the CRS print shows `EPSG:4326`, quadrant figure rendered.

- [ ] **Step 6: Full verification and commit**

Run: `uv run pytest tests/ -q -p no:warnings --deselect tests/test_cli_e2e.py` — expected: 0 failures.
Run: `uv run pyright` — expected `0 errors`.

```bash
git add tests/test_olci_integration.py tests/_test_data/optimized_olci_examples/ src/eopf_geozarr/cli.py README.md docs/converter.md docs/notebooks/sentinel3_olci_geozarr.ipynb
git commit -m "feat(s3-olci): --target-crs flag, regenerated snapshot, gridded-layout docs and notebook

Assisted-by: ClaudeCode:claude-fable-5"
```

---

### Task 6: Cross-product output contract test

**Files:**
- Create: `tests/test_geozarr_output_contract.py`
- Modify: `tests/test_integration_sentinel2.py` (extract fixture body into an importable builder)

**Interfaces:**
- Consumes: `convert_olci_optimized` (Task 4), `build_synthetic_olci` (Task 4 version), `MockSentinel1L1GRDBuilder` (exists in `tests/test_integration_sentinel1.py`), `create_geozarr_dataset` (exists), and the new `build_sample_sentinel2_datatree()`.
- Produces: the executable statement of "reprojection to a regular grid with CRS is a hard requirement for all products".

- [ ] **Step 1: Extract the S2 fixture builder**

In `tests/test_integration_sentinel2.py`, rename the body of the `sample_sentinel2_datatree` fixture into a module-level function and have the fixture delegate:

```python
def build_sample_sentinel2_datatree() -> xr.DataTree:
    """Build the sample Sentinel-2 EOPF DataTree (importable, non-fixture form)."""
    # <existing fixture body, unchanged, ending in `return dt`>


@pytest.fixture
def sample_sentinel2_datatree() -> xr.DataTree:
    """Create a sample Sentinel-2 EOPF DataTree structure for testing."""
    return build_sample_sentinel2_datatree()
```

Run: `uv run pytest tests/test_integration_sentinel2.py -q` — expected: unchanged results (refactor only).

- [ ] **Step 2: Write the contract test (RED where it should be)**

Create `tests/test_geozarr_output_contract.py`:

```python
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

import pathlib
from collections.abc import Callable
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
    levels = _multiscale_level_paths(store)
    assert levels, f"{name}: no multiscale groups found in {store}"
    for level in levels:
        ds = xr.open_dataset(
            str(store), group=level, engine="zarr", consolidated=False
        )
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
```

- [ ] **Step 3: Run and reconcile**

Run: `uv run pytest tests/test_geozarr_output_contract.py -v`

Expected: OLCI params pass; S1/S2 `test_store_opens_as_datatree` xfail. If `test_every_level_is_regular_grid_with_crs` fails for S1 or S2, investigate before touching the test: open the failing level by hand and determine whether the converter genuinely omits CRS/regular coords at that level (a real product gap — report it and add a targeted xfail with a reason naming the gap) or the walker mis-resolved a group path (fix the walker). Do not weaken the OLCI assertions — OLCI must pass everything.

- [ ] **Step 4: Full verification**

Run: `uv run pytest tests/ -q -p no:warnings --deselect tests/test_cli_e2e.py` — expected 0 failures.
Run: `uv run pyright` — expected `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_geozarr_output_contract.py tests/test_integration_sentinel2.py
git commit -m "test: cross-product contract - every converted product is a regular grid with CRS

Assisted-by: ClaudeCode:claude-fable-5"
```
