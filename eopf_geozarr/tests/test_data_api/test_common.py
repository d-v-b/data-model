from __future__ import annotations
import re

import zarr
from eopf_geozarr.data_api.geozarr.v2 import CoordArrayAttrs
import pytest
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec
from zarr.core.buffer import default_buffer_prototype
from eopf_geozarr.data_api.geozarr.common import CF_STANDARD_NAME_URL, check_standard_name, get_cf_standard_names
from eopf_geozarr.tests.test_data_api.conftest import example_zarr_json


def test_get_cf_standard_names() -> None:
    """
    Test the get_cf_standard_names function to ensure it retrieves the CF standard names correctly.
    """
    standard_names = get_cf_standard_names(CF_STANDARD_NAME_URL)
    assert isinstance(standard_names, tuple)
    assert len(standard_names) > 0
    assert all(isinstance(name, str) for name in standard_names)


@pytest.mark.parametrize(
    "name", ["air_temperature", "sea_surface_temperature", "precipitation_flux"]
)
def test_check_standard_name_valid(name: str) -> None:
    """
    Test the check_standard_name function with valid standard names.
    """
    assert check_standard_name(name) == name


def test_check_standard_name_invalid() -> None:
    """
    Test the check_standard_name function with an invalid standard name.
    """
    with pytest.raises(ValueError):
        check_standard_name("invalid_standard_name")


def test_coord_array_attrs_dimensions_length() -> None:
    """
    Test that the array_dimensions attribute must have length 1.
    """
    msg = (
        "1 validation error for CoordArrayAttrs\n_ARRAY_DIMENSIONS\n "
        " Tuple should have at most 1 item after validation, not 2"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CoordArrayAttrs(
            _ARRAY_DIMENSIONS=("time", "lat"),
            standard_name="air_temperature",
            units="mm",
            axis="Y",
        )


def test_multiscales_round_trip() -> None:


    from eopf_geozarr.data_api.geozarr.common import Multiscales

    source_store = {"zarr.json": default_buffer_prototype().buffer.from_bytes(example_zarr_json)}
    source_untyped = GroupSpec.from_zarr(zarr.open_group(source_store, mode="r"))
    flat = source_untyped.to_flat()
    meta = flat["/measurements/reflectance/r60m"].attributes["multiscales"]
    assert Multiscales(**meta).model_dump() == tuplify_json(meta)