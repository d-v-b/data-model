"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

import json
import pathlib
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec
from structlog.testing import capture_logs
from zarr.codecs import BloscCodec, CastValue, ScaleOffset
from zarr.core.dtype import Int16

from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized
from eopf_geozarr.s2_optimization.s2_multiscale import (
    _coarsen_variable,
    calculate_aligned_chunk_size,
    calculate_simple_shard_dimensions,
    create_downsampled_resolution_group,
    create_measurements_encoding,
    inject_missing_bands,
)

try:
    from pyproj import CRS as ProjCRS
except ImportError:
    ProjCRS = None


def add_spatial_ref_to_group(group: zarr.Group, epsg_code: int | None = None) -> None:
    """
    Add spatial_ref coordinate to all resolution levels in a group.

    This manually creates the spatial_ref coordinate that rioxarray would create,
    avoiding the need for rasterio dependency in the core re-encoding logic.

    Parameters
    ----------
    group : zarr.Group
        The group containing resolution level subgroups (e.g., r10m, r20m)
    epsg_code : int, optional
        EPSG code to use. If None, will try to detect from data variable attributes.
    """
    if ProjCRS is None:
        pytest.skip("pyproj not available")
        return

    # Iterate over all resolution levels
    for _, member in group.members():
        if not isinstance(member, zarr.Group):
            continue

        # Check if we can determine EPSG from array attributes in the zarr group
        detected_epsg = epsg_code
        if detected_epsg is None:
            for _, array in member.arrays():
                array_attrs = dict(array.attrs)
                if "proj:epsg" in array_attrs:
                    detected_epsg = int(array_attrs["proj:epsg"])
                    break

        if detected_epsg is None:
            continue

        # Create CRS from EPSG
        crs = ProjCRS.from_epsg(detected_epsg)
        cf_attrs = crs.to_cf()

        # Create spatial_ref array directly in zarr (scalar int64 array)
        # This avoids issues with xarray's to_zarr mode="a" handling
        spatial_ref_array = member.create_array(
            "spatial_ref",
            shape=(),
            dtype="int64",
            fill_value=0,
            overwrite=True,
        )
        spatial_ref_array.attrs.update(cf_attrs)
        spatial_ref_array[()] = 0  # Set the scalar value


@pytest.mark.filterwarnings("ignore:.*:zarr.errors.ZarrUserWarning")
def test_add_spatial_ref_matches_rasterio(tmp_path: pathlib.Path) -> None:
    """Test that our manual spatial_ref creation matches rasterio's output."""
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841

    if ProjCRS is None:
        pytest.skip("pyproj not available")

    # Create a test dataset with spatial dimensions
    x = np.arange(0, 100, 10, dtype=np.float64)
    y = np.arange(0, 100, 10, dtype=np.float64)
    data = np.random.rand(len(y), len(x)).astype(np.float32)

    test_epsg = 32632  # UTM zone 32N

    ds = xr.Dataset(
        {
            "test_var": (["y", "x"], data, {"proj:epsg": test_epsg}),
        },
        coords={"x": x, "y": y},
    )

    # Create two stores: one for rasterio, one for manual
    rasterio_store = tmp_path / "rasterio.zarr"
    manual_store = tmp_path / "manual.zarr"

    # Create group structure with a resolution level for manual approach
    manual_root = zarr.create_group(str(manual_store))
    _ = manual_root.create_group("r10m")  # Create the subgroup structure

    # Write dataset using rioxarray (which creates spatial_ref automatically)
    ds_with_crs = ds.rio.write_crs(test_epsg)
    ds_with_crs.to_zarr(str(rasterio_store), group="r10m", mode="w")

    # Write dataset without spatial_ref, then add it manually
    ds.to_zarr(str(manual_store), group="r10m", mode="a")
    add_spatial_ref_to_group(manual_root, epsg_code=test_epsg)

    # Reopen groups to access arrays (without consolidated metadata since we added spatial_ref manually)
    rasterio_res_group_reopened = zarr.open_group(
        str(rasterio_store), mode="r", use_consolidated=False
    )["r10m"]
    manual_res_group_reopened = zarr.open_group(
        str(manual_store), mode="r", use_consolidated=False
    )["r10m"]

    # Get spatial_ref arrays
    rasterio_spatial_ref_member = rasterio_res_group_reopened["spatial_ref"]
    manual_spatial_ref_member = manual_res_group_reopened["spatial_ref"]

    # Ensure they are arrays
    assert isinstance(rasterio_spatial_ref_member, zarr.Array)
    assert isinstance(manual_spatial_ref_member, zarr.Array)

    # Check that both exist and have the same shape (scalar)
    assert rasterio_spatial_ref_member.shape == ()
    assert manual_spatial_ref_member.shape == ()

    # Check that both have the same dtype
    assert rasterio_spatial_ref_member.dtype == manual_spatial_ref_member.dtype

    # Check that both have the same value
    rasterio_value = rasterio_spatial_ref_member[()]
    manual_value = manual_spatial_ref_member[()]
    assert rasterio_value == manual_value

    # Check that CF attributes match
    rasterio_attrs = dict(rasterio_spatial_ref_member.attrs)
    manual_attrs = dict(manual_spatial_ref_member.attrs)

    # Both should have grid_mapping_name
    assert "grid_mapping_name" in rasterio_attrs
    assert "grid_mapping_name" in manual_attrs
    assert rasterio_attrs["grid_mapping_name"] == manual_attrs["grid_mapping_name"]

    # Both should have crs_wkt, but the format may differ (WKT1 vs WKT2)
    # Verify both are valid WKT and represent the same CRS
    assert "crs_wkt" in rasterio_attrs
    assert "crs_wkt" in manual_attrs

    # Parse both WKT strings and verify they represent the same EPSG code
    rasterio_crs = ProjCRS.from_wkt(str(rasterio_attrs["crs_wkt"]))
    manual_crs = ProjCRS.from_wkt(str(manual_attrs["crs_wkt"]))

    # Both should have the same EPSG code
    assert rasterio_crs.to_epsg() == test_epsg
    assert manual_crs.to_epsg() == test_epsg

    # Verify the manual implementation has the expected CF attribute keys
    # (pyproj's to_cf() generates these)
    expected_cf_keys = {"grid_mapping_name", "crs_wkt"}
    assert expected_cf_keys.issubset(set(manual_attrs.keys()))

    # The manual implementation should have all standard CF-compliant attributes
    # that pyproj generates (there may be additional ones)
    assert len(manual_attrs) >= len(expected_cf_keys)


@pytest.fixture
def sample_dataset(s2_group_example: pathlib.Path) -> xr.Dataset:
    """Create a sample xarray dataset for testing."""
    with pytest.warns((RuntimeWarning, FutureWarning)):
        return xr.open_datatree(s2_group_example, engine="zarr")[
            "measurements/reflectance/r10m"
        ].to_dataset()


def test_create_downsampled_resolution_group_quality_mask() -> None:
    """Quality-mask downsampling should not crash and should preserve dtype."""
    x = np.arange(8)
    y = np.arange(6)
    quality = xr.DataArray(
        np.random.randint(0, 2, (6, 8), dtype=np.uint8),
        dims=["y", "x"],
        coords={"y": y, "x": x},
        name="quality_clouds",
    )
    ds = xr.Dataset({"quality_clouds": quality})

    out = create_downsampled_resolution_group(ds, factor=2)

    assert "quality_clouds" in out.data_vars
    assert out["quality_clouds"].dtype == np.uint8
    assert out["quality_clouds"].shape == (3, 4)


def test_calculate_simple_shard_dimensions() -> None:
    """Test simplified shard dimensions calculation."""
    # Test 3D data (time, y, x) - shards are multiples of chunks
    data_shape = (5, 1024, 1024)
    chunks = (1, 256, 256)

    shard_dims = calculate_simple_shard_dimensions(data_shape, chunks)

    assert len(shard_dims) == 3
    assert shard_dims[0] == 1  # Time dimension should be 1
    assert shard_dims[1] == 1024  # Y dimension matches exactly (divisible by 256)
    assert shard_dims[2] == 1024  # X dimension matches exactly (divisible by 256)

    # Test 2D data (y, x) with non-divisible dimensions
    data_shape = (1000, 1000)
    chunks = (256, 256)

    shard_dims = calculate_simple_shard_dimensions(data_shape, chunks)

    assert len(shard_dims) == 2
    # Should use largest multiple of chunk_size that fits
    assert shard_dims[0] == 768  # 3 * 256 = 768 (largest multiple that fits in 1000)
    assert shard_dims[1] == 768  # 3 * 256 = 768


def test_create_measurements_encoding_experimental_scale_offset_codec() -> None:
    """Test that experimental_scale_offset_codec adds ScaleOffset + CastValue filters."""
    # Create a dataset with CF-style scale-offset encoding, as xarray would
    # produce when reading a CF-encoded zarr/netCDF variable.
    data = xr.DataArray(
        np.arange(0, 100, dtype="float64").reshape(10, 10),
        dims=["y", "x"],
    )
    data.encoding = {
        "scale_factor": 0.01,
        "add_offset": 273.15,
        "dtype": np.dtype("int16"),
    }
    ds = xr.Dataset({"temperature": data})

    encoding = create_measurements_encoding(
        ds,
        enable_sharding=True,
        spatial_chunk=256,
        keep_scale_offset=False,
        experimental_scale_offset_codec=True,
    )

    int16_min = int(np.iinfo(np.int16).min)
    assert encoding == {
        "temperature": {
            "chunks": (10, 10),
            "compressors": (BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0),),
            "shards": (10, 10),
            "filters": (
                ScaleOffset(offset=273.15, scale=100.0),
                CastValue(
                    data_type=Int16(endianness="little"),
                    rounding="nearest-even",
                    out_of_range=None,
                    scalar_map={
                        "encode": [("NaN", int16_min)],
                        "decode": [(int16_min, "NaN")],
                    },
                ),
            ),
            "fill_value": "NaN",
        }
    }


@pytest.mark.filterwarnings("ignore:.*:zarr.errors.ZarrUserWarning")
@pytest.mark.parametrize("keep_scale_offset", [True, False])
def test_create_measurements_encoding(keep_scale_offset: bool, sample_dataset: xr.Dataset) -> None:
    """Test measurements encoding creation with xy-aligned sharding."""
    encoding = create_measurements_encoding(
        sample_dataset,
        enable_sharding=True,
        spatial_chunk=1024,
        keep_scale_offset=keep_scale_offset,
    )

    # Check that encoding is created for all variables
    for var_name in sample_dataset.data_vars:
        assert var_name in encoding
        var_encoding = encoding[var_name]

        # Check basic encoding structure
        assert "chunks" in var_encoding
        # Zarr v3 uses 'compressors' (plural)
        assert "compressors" in var_encoding or "compressor" in var_encoding

        # Check sharding is included when enabled
        assert "shards" in var_encoding

    # Check coordinate encoding
    for coord_name in sample_dataset.coords:
        if coord_name in encoding:
            # Coordinates may have either compressor or compressors set to None
            assert (
                encoding[coord_name].get("compressor") is None
                or encoding[coord_name].get("compressors") is None
            )
    # Store data and check that we are conditionally applying the scale-offset transformation
    # based on the request passed to the encoding
    stored = sample_dataset.to_zarr({}, encoding=encoding)
    zg = stored.zarr_group
    for var_name in sample_dataset.data_vars:
        if "add_offset" in sample_dataset[var_name].encoding:
            if keep_scale_offset:
                assert zg[var_name].dtype != sample_dataset[var_name].dtype
            else:
                assert zg[var_name].dtype == sample_dataset[var_name].dtype


def test_create_measurements_encoding_time_chunking(sample_dataset: xr.Dataset) -> None:
    """Test that time dimension is chunked to 1 for single file per time."""
    encoding = create_measurements_encoding(
        sample_dataset, enable_sharding=True, spatial_chunk=1024
    )

    for var_name in sample_dataset.data_vars:
        if sample_dataset[var_name].ndim == 3:  # 3D variable with time
            chunks = encoding[var_name]["chunks"]
            assert chunks[0] == 1  # Time dimension should be chunked to 1


def test_calculate_aligned_chunk_size() -> None:
    """Test aligned chunk size calculation."""
    # Test with spatial_chunk that divides evenly
    chunk_size = calculate_aligned_chunk_size(1024, 256)
    assert chunk_size == 256

    # Test with spatial_chunk that doesn't divide evenly
    chunk_size = calculate_aligned_chunk_size(1000, 256)
    # Should return a value that divides evenly into 1000
    assert 1000 % chunk_size == 0


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_convert_s2_optimized(
    s2_group_example: Path,
    tmp_path: pathlib.Path,
) -> None:
    """Snapshot test: convert an S2 example and compare the output structure
    against a stored fixture."""
    input_group = zarr.open_group(s2_group_example)

    output_path = tmp_path / "output.zarr"

    dt_input = xr.open_datatree(input_group.store, engine="zarr", chunks="auto")
    with capture_logs():
        convert_s2_optimized(
            dt_input,
            output_path=str(output_path),
            enable_sharding=True,
            spatial_chunk=256,
            compression_level=1,
            validate_output=False,
            keep_scale_offset=False,
            experimental_scale_offset_codec=False,
        )
    observed_group = zarr.open_group(output_path)
    observed_structure = GroupSpec.from_zarr(observed_group)

    # Comparing JSON objects is sensitive to the difference between tuples and lists, but we
    # don't care about that here, so we convert all lists to tuples before creating the GroupSpec
    observed_structure_json = observed_structure.model_dump()
    observed_structure = GroupSpec(**tuplify_json(observed_structure_json))
    observed_structure_flat = observed_structure.to_flat()
    expected_structure_path = Path("tests/_test_data/optimized_geozarr_examples/") / (
        s2_group_example.stem + ".json"
    )

    # Write out the observed structure for analysis
    observed_output_path = Path("/tmp/observed_structure.json")
    observed_output_path.write_text(json.dumps(observed_structure_json, indent=2, sort_keys=True))

    expected_structure_json = tuplify_json(json.loads(expected_structure_path.read_text()))
    expected_structure = GroupSpec(**expected_structure_json)
    expected_structure_flat = expected_structure.to_flat()

    # Uncomment this section to write out the expected structure from the observed structure
    # This is useful when the expected structure needs to be updated
    # expected_structure_path.write_text(
    #    json.dumps(observed_structure_json, indent=2, sort_keys=True)
    # )

    # check that all multiscale levels have the same data type
    # this check is redundant with the later check, but it's expedient to check this here.
    # eventually this check should be spun out into its own test
    _, res_groups = zip(*observed_group["measurements/reflectance"].groups(), strict=False)

    dtype_mismatch: set[object] = set()
    for group_a, group_b in pairwise(res_groups):
        ds_a = xr.open_dataset(group_a.store, engine="zarr", group=group_a.path)
        ds_b = xr.open_dataset(group_b.store, engine="zarr", group=group_b.path)

        for name in ds_a.data_vars:
            dtype_a = ds_a[name].dtype
            if name in ds_b.data_vars:
                dtype_b = ds_b[name].dtype
                if dtype_a != dtype_b:
                    dtype_mismatch.add(
                        (f"{group_a.path}/{name}::{dtype_a}", f"{group_b.path}/{name}::{dtype_b}")
                    )
    assert dtype_mismatch == set()

    o_keys = set(observed_structure_flat.keys())
    e_keys = set(expected_structure_flat.keys())

    # Write out the expected structure to the temporary test directory
    # This is useful when the expected structure needs to be updated

    observed_tree_path = tmp_path / "observed_tree.json"
    observed_flat_path = tmp_path / "observed_flat.json"
    expected_flat_path = tmp_path / "expected_flat.json"

    observed_tree_path.write_text(json.dumps(observed_structure_json, indent=2, sort_keys=True))

    observed_flat_path.write_text(
        json.dumps({k: v.model_dump() for k, v in observed_structure_flat.items()})
    )
    expected_flat_path.write_text(
        json.dumps({k: v.model_dump() for k, v in expected_structure_flat.items()})
    )

    # Check that all of the keys are the same
    if o_keys != e_keys:
        (tmp_path / "extra_observed_keys.json").write_text(
            json.dumps(
                {k: observed_structure_flat[k].model_dump() for k in o_keys - e_keys},
                indent=2,
            )
        )
        (tmp_path / "extra_expected_keys.json").write_text(
            json.dumps(
                {k: expected_structure_flat[k].model_dump() for k in e_keys - o_keys},
                indent=2,
            )
        )

        raise AssertionError(
            f"Key mismatch.\n"
            f"Extra in observed: {sorted(o_keys - e_keys)[:20]}\n"
            f"Missing in observed: {sorted(e_keys - o_keys)[:20]}\n"
            f"Check {observed_tree_path!s} to see the observed JSON structure."
        )

    # Check that all values are the same for common keys
    common_keys = o_keys & e_keys
    mismatch_values = [
        k for k in common_keys if expected_structure_flat.get(k) != observed_structure_flat.get(k)
    ]

    if mismatch_values != []:
        mismatch_values_path = tmp_path / "mismatch_values.json"
        mismatch_values_path.write_text(
            json.dumps(
                {
                    k: {
                        "expected": expected_structure_flat[k].model_dump(),
                        "observed": observed_structure_flat[k].model_dump(),
                    }
                    for k in mismatch_values
                },
                indent=2,
            )
        )
        raise AssertionError(
            f"Value mismatches: {mismatch_values[:20]}\n"
            f"Check {mismatch_values_path!s} to see the observed JSON structure."
        )


# ---------------------------------------------------------------------------
# _coarsen_variable
# ---------------------------------------------------------------------------


def test_coarsen_variable_classification() -> None:
    """Classification variables should be downsampled via subsample."""
    data = np.arange(16, dtype="uint8").reshape(4, 4)
    var = xr.DataArray(data, dims=["y", "x"], coords={"y": np.arange(4.0), "x": np.arange(4.0)})
    result = _coarsen_variable("scl", var, factor=2)
    assert result.shape == (2, 2)
    assert result.dtype == np.uint8
    # subsample picks top-left of each 2x2 block
    np.testing.assert_array_equal(result.values, data[::2, ::2])


def test_coarsen_variable_quality_mask() -> None:
    """Quality mask variables should be downsampled via max."""
    data = np.array([[0, 1], [2, 3]], dtype="uint8")
    var = xr.DataArray(data, dims=["y", "x"], coords={"y": np.arange(2.0), "x": np.arange(2.0)})
    result = _coarsen_variable("quality_cirrus", var, factor=2)
    assert result.shape == (1, 1)
    assert result.values.item() == 3


# ---------------------------------------------------------------------------
# inject_missing_bands
# ---------------------------------------------------------------------------


def _make_reflectance_datatree() -> xr.DataTree:
    """Build a minimal DataTree with /measurements/reflectance/r10m and r20m."""
    size_10m = 120  # must be divisible by 2 (→60) and 6 (→20)
    x10 = np.arange(size_10m, dtype="float64")
    y10 = np.arange(size_10m, dtype="float64")

    r10m_ds = xr.Dataset(
        {
            "b02": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b03": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b04": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b08": (["y", "x"], np.full((size_10m, size_10m), 42, dtype="uint16")),
        },
        coords={"x": x10, "y": y10},
    )

    size_20m = size_10m // 2
    x20 = np.arange(size_20m, dtype="float64")
    y20 = np.arange(size_20m, dtype="float64")
    r20m_ds = xr.Dataset(
        {
            "b05": (["y", "x"], np.ones((size_20m, size_20m), dtype="uint16")),
        },
        coords={"x": x20, "y": y20},
    )

    dt = xr.DataTree()
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m_ds)
    dt["measurements/reflectance/r20m"] = xr.DataTree(r20m_ds)
    return dt


def test_inject_missing_bands_respects_bands_filter() -> None:
    """With bands={"b08"}, only b08 should be injected even when others are eligible."""
    dt = _make_reflectance_datatree()
    r20m_ds = dt["measurements/reflectance/r20m"].to_dataset()

    result = inject_missing_bands(r20m_ds, dt, target_resolution=20, bands={"b08"})

    assert "b08" in result.data_vars
    assert result["b08"].shape == (60, 60)
    assert result["b08"].dtype == np.uint16
    # b02/b03/b04 are also eligible (10m native, missing from r20m) but excluded
    for excluded in ("b02", "b03", "b04"):
        assert excluded not in result.data_vars


def test_inject_missing_bands_skips_existing() -> None:
    """Bands already present in the dataset should not be overwritten."""
    dt = _make_reflectance_datatree()
    r20m_ds = dt["measurements/reflectance/r20m"].to_dataset()

    # Pre-populate b08 with a sentinel value so we can verify it is NOT replaced.
    sentinel = np.full((60, 60), 999, dtype="uint16")
    r20m_ds["b08"] = (["y", "x"], sentinel)

    result = inject_missing_bands(r20m_ds, dt, target_resolution=20, bands={"b08"})

    # b08 was already present — inject_missing_bands must leave it untouched.
    np.testing.assert_array_equal(result["b08"].values, sentinel)


def test_inject_missing_bands_noop_when_no_source() -> None:
    """If the source group is missing from the DataTree, return dataset unchanged."""
    dt = xr.DataTree()
    ds = xr.Dataset({"b05": (["y", "x"], np.ones((60, 60)))})

    result = inject_missing_bands(ds, dt, target_resolution=20)

    assert "b08" not in result.data_vars


def test_inject_missing_bands_default_injects_all() -> None:
    """With bands=None (default), all eligible finer bands should be injected."""
    size_10m = 120
    x10 = np.arange(size_10m, dtype="float64")
    y10 = np.arange(size_10m, dtype="float64")

    r10m_ds = xr.Dataset(
        {
            "b02": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b03": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b04": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b08": (["y", "x"], np.full((size_10m, size_10m), 42, dtype="uint16")),
        },
        coords={"x": x10, "y": y10},
    )

    size_20m = 60
    x20 = np.arange(size_20m, dtype="float64")
    y20 = np.arange(size_20m, dtype="float64")
    r20m_ds = xr.Dataset(
        {
            "b05": (["y", "x"], np.ones((size_20m, size_20m), dtype="uint16")),
            "b06": (["y", "x"], np.ones((size_20m, size_20m), dtype="uint16")),
        },
        coords={"x": x20, "y": y20},
    )

    size_60m = 20
    x60 = np.arange(size_60m, dtype="float64")
    y60 = np.arange(size_60m, dtype="float64")
    r60m_ds = xr.Dataset(
        {
            "b01": (["y", "x"], np.ones((size_60m, size_60m), dtype="uint16")),
        },
        coords={"x": x60, "y": y60},
    )

    dt = xr.DataTree()
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m_ds)
    dt["measurements/reflectance/r20m"] = xr.DataTree(r20m_ds)
    dt["measurements/reflectance/r60m"] = xr.DataTree(r60m_ds)

    result = inject_missing_bands(r60m_ds, dt, target_resolution=60)

    # All 10m bands (b02, b03, b04, b08) and 20m bands (b05, b06) should be injected
    for band in ("b02", "b03", "b04", "b08", "b05", "b06"):
        assert band in result.data_vars, f"{band} missing from r60m"
        assert result[band].shape == (size_60m, size_60m)
