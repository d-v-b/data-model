"""Multiscale (overview) generation for OLCI data.

Both reduction strategies operate over a configurable pair of spatial
dimensions (default the raw-swath ``(rows, columns)``; the converter's
reprojected pyramid uses ``(y, x)``):

* :func:`decimate_swath` — pure stride-based decimation; every 2-D spatial
  variable is subsampled ``[::factor, ::factor]``.  Intended for cases where
  preserving pixel identity matters.

* :func:`reduce_swath` — radiance bands and altitude are fill-aware
  block-averaged (mean of ``factor x factor`` blocks); other spatial
  variables (including 1-D dimension coordinates) are stride-decimated;
  non-spatial variables pass through unchanged.  Intended for producing
  GeoZarr multiscale overview groups.

:func:`grid_spatial_attrs` emits the zarr-cm spatial-convention attrs for a
regular grid level with an affine transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio.transform
import structlog
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_band_mapping import OLCI_BANDS

if TYPE_CHECKING:
    from affine import Affine
    from zarr_cm import SpatialAttrs

log = structlog.get_logger()

SWATH_DIMS = ("rows", "columns")


def decimate_swath(
    ds: xr.Dataset, factor: int = 2, *, dims: tuple[str, str] = SWATH_DIMS
) -> xr.Dataset:
    """Return *ds* with every array spanning the *dims* spatial dimensions (default ``(rows, columns)``) subsampled by *factor*.

    Both data variables and coordinate variables that span exactly the swath
    dims are decimated ``[::factor, ::factor]``; everything else is passed
    through unchanged. Attributes and encoding are preserved by xarray's isel.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return ds
    indexers = {dim: slice(None, None, factor) for dim in dims if dim in ds.sizes}
    if not indexers:
        return ds
    return ds.isel(indexers)


def _fill_value_of(var: xr.DataArray) -> int | float | None:
    """Declared ``_FillValue`` from attrs or encoding, or None."""
    fill = var.attrs.get("_FillValue")
    if fill is None:
        fill = var.encoding.get("_FillValue")
    return fill


def reduce_swath(
    ds: xr.Dataset, factor: int = 2, *, dims: tuple[str, str] = SWATH_DIMS
) -> xr.Dataset:
    """Return *ds* with radiance and altitude block-averaged, other spatial variables decimated.

    Overviews are an unweighted index-block mean that ASSUMES locally-uniform
    pixel spacing; intended for visualization, not quantitative analysis at
    reduced resolution.

    Parameters spanning the *dims* spatial dimensions (default ``(rows, columns)``)
    are processed; other variables pass through unchanged.

    Radiance variables (those named in :data:`OLCI_BANDS`) are averaged over
    ``factor x factor`` pixel blocks with fill-value awareness: fill pixels
    are masked before averaging so a single fill pixel does not contaminate
    the entire block.  Blocks where ALL pixels are fill are set back to the
    fill value in the output.

    Altitude (identified by ``standard_name`` falling back to variable name)
    is fill-aware block-averaged exactly like radiance, so the elevation of
    an overview cell describes the same pixel block its radiance was averaged
    over; being linear, its mean commutes with CF packing and runs on raw
    values.  Remaining 2-D spatial variables and 1-D dimension coordinates
    are decimated by trimmed stride; variables that do not span the *dims*
    spatial dimensions are passed through unchanged.

    Parameters
    ----------
    ds:
        Input dataset.  May contain any mix of radiance bands, 2-D coordinate
        arrays, and other variables.
    factor:
        Spatial reduction factor.  Must be >= 1.  Factor 1 returns *ds*
        unchanged.

    Returns
    -------
    xr.Dataset
        Reduced dataset with the same variables but smaller (rows, columns)
        dimensions.

    Raises
    ------
    ValueError
        If *factor* < 1.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return ds

    olci_band_set: frozenset[str] = frozenset(OLCI_BANDS)
    result_vars: dict[str, xr.DataArray] = {}
    result_coords: dict[str, xr.DataArray] = {}

    coord_names: frozenset[str] = frozenset(str(k) for k in ds.coords)

    # Pre-compute the trimmed length for each swath dimension so that both the
    # coarsen path (radiance) and the isel stride path (coordinates) produce
    # exactly floor(N / factor) output elements.  coarsen(boundary="trim")
    # already truncates to a multiple of factor; we match it by stopping the
    # stride at the same trimmed limit:  slice(0, n_trim, factor).
    #
    # Without this, an odd-length dimension N yields:
    #   coarsen → floor(N / factor)   e.g. 4865 → 2432
    #   isel[::factor] → ceil(N / factor)  e.g. 4865 → 2433
    # producing a store where coordinate arrays are longer than the data they
    # describe, which makes xr.open_dataset raise a conflicting-sizes error.
    dim_trim: dict[str, int] = {
        dim: (ds.sizes[dim] // factor) * factor for dim in dims if dim in ds.sizes
    }

    all_names: list[str] = [str(k) for k in ds.data_vars] + [str(k) for k in ds.coords]
    for name in all_names:
        var: xr.DataArray = ds[name] if name in ds.data_vars else ds.coords[name]
        # Order-insensitive: a variant product storing a band transposed as
        # (columns, rows) must still get fill-aware averaging, not silently
        # fall through to stride decimation. Normalize to *dims* order below.
        var_dims = tuple(str(d) for d in var.dims)
        is_swath_2d: bool = len(var_dims) == 2 and set(var_dims) == set(dims)
        # Altitude gets the same fill-aware block mean as radiance so the
        # elevation of an overview cell describes the same pixel block its
        # radiance was averaged over (a stride sample would sit ~half a block
        # away). It is a linear quantity, and a mean commutes with affine CF
        # packing, so averaging raw packed values is exact.
        is_altitude: bool = str(var.attrs.get("standard_name") or name) == "altitude"

        if (name in olci_band_set or is_altitude) and is_swath_2d:
            if var_dims != dims:
                log.info("Transposing band to canonical spatial dim order", band=name)
                var = var.transpose(*dims)
            # Fill-aware block averaging for radiance bands.
            fill_value: int | float | None = _fill_value_of(var)
            orig_dtype = var.dtype

            float_var = var.astype("float64")
            if fill_value is not None:
                if np.isnan(float(fill_value)):
                    # NaN sentinel (float sources): equality comparison would
                    # be a no-op (NaN != NaN is always True); mask explicitly.
                    float_var = float_var.where(float_var.notnull())
                else:
                    float_var = float_var.where(float_var != float(fill_value))

            # coarsen().mean() is available at runtime; pyright stubs don't expose .mean()
            # on DataArrayCoarsen, so we suppress the type-check on the reduction call.
            coarsened = float_var.coarsen({dims[0]: factor, dims[1]: factor}, boundary="trim")
            # skipna=True explicitly: the fill-aware promise (one fill pixel
            # must not contaminate its block) depends on it, so don't rely on
            # the version-dependent default.
            averaged: xr.DataArray = coarsened.mean(skipna=True)  # type: ignore[attr-defined,assignment]
            # Materialize once: with dask-backed input, the isnull()/where()/
            # .values accesses below would otherwise each re-evaluate the
            # coarsen+mask graph — the heaviest compute in the converter.
            averaged = averaged.compute()

            if fill_value is not None:
                valid_mask = ~averaged.isnull().values
                fill_da = xr.where(averaged.isnull(), float(fill_value), averaged)
                result_arr = fill_da.values
            else:
                valid_mask = None
                result_arr = averaged.values
            # Round only when packing back into an integer dtype; float
            # radiance (e.g. a CF-decoded source) must not be quantized.
            if np.issubdtype(orig_dtype, np.integer):
                unrounded = result_arr
                result_arr = np.round(result_arr)
                if fill_value is not None and valid_mask is not None:
                    # A valid block whose mean rounds to the fill sentinel
                    # would be recoded as fill in the overview; nudge it one
                    # step back into the valid domain, on the side of the
                    # sentinel the unrounded mean came from (fill may sit at
                    # either end of the dtype range).
                    collision = valid_mask & (result_arr == float(fill_value))
                    if collision.any():
                        # Nudge candidates, clamped to the dtype range: when
                        # the sentinel sits at a dtype bound (e.g. 0 for an
                        # unsigned dtype), both sides resolve to the one
                        # in-range neighbour instead of wrapping around.
                        info = np.iinfo(orig_dtype)
                        down = float(fill_value) - 1
                        up = float(fill_value) + 1
                        if down < float(info.min):
                            down = up
                        if up > float(info.max):
                            up = down
                        nudged = np.where(unrounded <= float(fill_value), down, up)
                        result_arr = np.where(collision, nudged, result_arr)
            result_val = result_arr.astype(orig_dtype)

            out_var = xr.DataArray(result_val, dims=averaged.dims, attrs=var.attrs)

            if name in ds.data_vars:
                result_vars[name] = out_var
            else:
                result_coords[name] = out_var

        elif is_swath_2d:
            # Coordinate or non-radiance 2-D swath variable: decimate.
            # Use slice(0, n_trim, factor) rather than slice(None, None, factor)
            # so that an odd-length dimension N yields floor(N / factor) elements,
            # matching the output length of coarsen(boundary="trim").mean().
            indexers: dict[str, slice] = {
                dim: slice(0, dim_trim[dim], factor) for dim in dims if dim in dim_trim
            }
            decimated = var.isel(indexers)
            if name in coord_names:
                result_coords[name] = decimated
            else:
                result_vars[name] = decimated

        elif any(dim in (str(d) for d in var.dims) for dim in dims):
            # 1-D (or higher) variable sharing a swath dim but not 2-D swath:
            # decimate along whichever swath dims it carries.
            var_dims = {str(d) for d in var.dims}
            idx: dict[str, slice] = {
                dim: slice(0, dim_trim[dim], factor)
                for dim in dims
                if dim in var_dims and dim in dim_trim
            }
            decimated = var.isel(idx)
            if name in coord_names:
                result_coords[name] = decimated
            else:
                result_vars[name] = decimated

        else:
            # Non-swath variable: pass through unchanged.
            if name in coord_names:
                result_coords[name] = var
            else:
                result_vars[name] = var

    return xr.Dataset(result_vars, coords=result_coords)


def grid_spatial_attrs(transform: Affine, shape: tuple[int, int]) -> SpatialAttrs:
    """Spatial-convention data for a regular grid with an affine *transform*.

    *shape* is ``(height, width)``.  Emits ``spatial:dimensions`` ``["y","x"]``,
    pixel registration, the bounding box, and the 6-element row-major affine
    transform.
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
