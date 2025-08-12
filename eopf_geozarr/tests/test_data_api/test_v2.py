from __future__ import annotations
from eopf_geozarr.data_api.geozarr.common import BaseDataArrayAttrs
import pytest
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from typing import Any
from eopf_geozarr.data_api.geozarr.v2 import CoordArrayAttrs, DataArrayAttrs


import re

from eopf_geozarr.data_api.geozarr.v2 import CoordArray, DataArray, check_valid_coordinates


def test_coord_array_dimensionality() -> None:
    """
    Test that only 1-dimensional arrays are allowed.
    """
    msg = (
        "1 validation error for CoordArray\nshape\n  "
        "Tuple should have at most 1 item after validation, not 2"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CoordArray(
            shape=(10, 11),
            dtype="|u1",
            chunks=(10, 11),
            attributes=CoordArrayAttrs(
                _ARRAY_DIMENSIONS=("time",),
                standard_name="air_temperature",
                units="s",
                axis="Y",
            ),
        )


class TestCheckValidCoordinates:
    @pytest.mark.parametrize(
        "example",
        [
            {
                "": GroupSpec(attributes={}, members=None),
                "/data_var": DataArray(
                    shape=(10, 11),
                    dtype="|u1",
                    chunks=(10, 11),
                    attributes=DataArrayAttrs(
                        array_dimensions=["time", "lat"],
                        standard_name="air_temperature",
                        grid_mapping=None,
                        grid_mapping_name="latitude_longitude",
                    ),
                ),
                "/time": CoordArray(
                    shape=(10,),
                    dtype="|u1",
                    chunks=(10,),
                    attributes=CoordArrayAttrs(
                        array_dimensions=["time"], standard_name="time", units="s", axis="T"
                    ),
                ),
                "/lat": CoordArray(
                    shape=(11,),
                    dtype="|u1",
                    chunks=(11,),
                    attributes=CoordArrayAttrs(
                        array_dimensions=["lat"], standard_name="latitude", units="m", axis="Y"
                    ),
                ),
            },
        ],
    )
    @staticmethod
    def test_valid(example: dict[str, ArraySpec[Any] | GroupSpec[Any, Any]]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.
        """
        group = GroupSpec.from_flat(example)
        assert check_valid_coordinates(group) == group

    @pytest.mark.parametrize(
        "example",
        [
            {
                "": GroupSpec(attributes={}, members=None),
                "/data_var": DataArray(
                    shape=(9, 10),
                    dtype="|u1",
                    chunks=(10, 11),
                    attributes=DataArrayAttrs(
                        array_dimensions=["time", "lat"],
                        standard_name="air_temperature",
                        grid_mapping=None,
                        grid_mapping_name="latitude_longitude",
                    ),
                ),
                "/time": CoordArray(
                    shape=(10,),
                    dtype="|u1",
                    chunks=(10,),
                    attributes=CoordArrayAttrs(
                        _ARRAY_DIMENSIONS=["time"], standard_name="time", units="s", axis="T"
                    ),
                ),
                "/lat": CoordArray(
                    shape=(11,),
                    dtype="|u1",
                    chunks=(11,),
                    attributes=CoordArrayAttrs(
                        _ARRAY_DIMENSIONS=["lat"], standard_name="latitude", units="m", axis="Y"
                    ),
                ),
            },
        ],
    )
    @staticmethod
    def test_invalid_coordinates(example: dict[str, ArraySpec[Any] | GroupSpec[Any, Any]]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.

        This test checks that the function raises a ValueError when the dimensions of the data variable
        do not match the dimensions of the coordinate arrays.
        """
        group = GroupSpec[Any, DataArray | CoordArray].from_flat(example)
        with pytest.raises(ValueError):
            check_valid_coordinates(group)