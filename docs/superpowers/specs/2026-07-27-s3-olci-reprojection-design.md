# Sentinel-3 OLCI reprojection to a regular grid — design

Date: 2026-07-27
Status: approved
Context: PR #212 review feedback (titiler cannot tile unprojected swath data);
supersedes the "native swath geometry, no reprojection" choice in
[2026-06-21-sentinel3-olci-export-design.md](2026-06-21-sentinel3-olci-export-design.md).

## Problem

The OLCI converter writes measurements in instrument (swath) geometry with
per-pixel 2-D `latitude`/`longitude` arrays and no declared CRS. Generic
GeoZarr readers cannot serve xyz tiles from this: without a projected grid
there is no geotransform (confirmed by titiler maintainers on PR #212), and
without `grid_mapping`/`spatial_ref`, `ds.rio.crs` is `None` — the same gap
Sentinel-1 had (#176/#201).

Decision (with maintainer sign-off on the PR thread): **reprojection to a
regular grid is a hard requirement for all converted products.** The OLCI
output contains only the reprojected grid; the swath representation is not
copied into the output (it remains in the source product).

## Approach

Warp with rasterio's dense geolocation-array support (rasterio >= 1.4;
repo has 1.5.0). No new dependencies; no GCP subsampling or polynomial
fitting — the full per-pixel geolocation is used directly.

Rejected alternatives: reusing the Sentinel-1 GCP path (discards geolocation
density; polynomial transformer distorts the curved swath), and pyresample
(new heavyweight dependency for something rasterio already does here).

## Converter flow

In `convert_olci_optimized`, after the existing encoding-strip and
attr-sanitize steps:

1. **Warp once at native resolution.** Pass the 2-D `latitude`/`longitude`
   coordinate arrays as `src_geoloc_array` to
   `rasterio.warp.calculate_default_transform` (grid sizing, ~native 300 m
   resolution preserved) and `rasterio.warp.reproject` (per-band warp).
   Target CRS defaults to `EPSG:4326`; exposed as `target_crs` on the entry
   point and `--target-crs` on the CLI subcommand.
2. **Band handling.** Radiances stay raw `uint16` with CF
   `scale_factor`/`add_offset` attrs. `_FillValue` (65535) doubles as warp
   nodata, so off-swath cells are fill. Resampling: bilinear (linear scale
   means averaging digital counts is exact in radiance space).
3. **Coordinates.** Output grid has 1-D `x`/`y` dimension coordinates derived
   from the affine transform. The 2-D `altitude` coordinate is warped onto
   the grid as a variable. Per-scan-line `time_stamp` has no home on a
   regular grid and is dropped (documented; it survives in the source
   product).
4. **Pyramid.** Overviews build as today — fill-aware /2 block averaging —
   on the grid, in `measurements/r0`, `r2`, … sibling groups. 1-D
   coordinates decimate by simple striding consistent with the coarsen trim.

## CRS metadata (both idioms, per level)

- `rio.write_crs`: a `spatial_ref` variable plus `grid_mapping` attr on every
  band — what titiler/rioxarray require (closes the #176/#201-style gap).
- zarr-cm `geo-proj` convention attrs via the existing
  `build_convention_attrs(crs=...)` path.
- `swath_spatial_attrs()` is replaced by gridded spatial attrs carrying the
  real affine transform. The multiscales layout keeps `r0`-relative entries
  with per-level scale transforms.

## Tests

**Cross-product contract** — new `tests/test_geozarr_output_contract.py`,
one parametrized acceptance test over each converter's output (S1, S2, OLCI;
reusing their existing synthetic fixtures):

- store opens via `xr.open_datatree`;
- every measurement level reports `ds.rio.crs is not None`;
- spatial dims have 1-D regular coordinate arrays;
- the declared CRS round-trips through pyproj.

**OLCI unit tests** (per repo test-structure convention): one behavior test
for the warp helper (correct output grid, fill propagation outside the
swath, dtype preservation) and one test per error case (missing geolocation
coordinates; degenerate/empty extent).

**Regeneration:** OLCI golden snapshot regenerates; the demonstration
notebook re-executes against the real EODC product (the quadrant demo works
unchanged on the gridded pyramid).

## Out of scope

- `conditions`/`quality` groups stay passthrough in tie-point/swath form.
- No swath copy of measurements in the output.
- Encoding wiring for `--enable-sharding` / `--spatial-chunk` /
  `--compression-level` / `--keep-scale-offset` (pre-existing follow-up).
