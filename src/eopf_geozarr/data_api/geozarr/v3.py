"""Zarr V3 Models for the GeoZarr Zarr Hierarchy."""

from __future__ import annotations

from typing import Any, Self

from pydantic import model_validator
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    DatasetAttrs,
    check_grid_mapping,
    check_valid_coordinates,
)
from eopf_geozarr.data_api.geozarr.multiscales import MultiscaleGroupAttrs


class DataArray(ArraySpec[BaseDataArrayAttrs]):
    """
    A Zarr array that represents as GeoZarr DataArray variable.

    The attributes of this array are defined in `BaseDataArrayAttrs`.

    This array has an additional constraint: the dimension_names field must be a tuple of strings.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """

    # The dimension names must be a tuple of strings
    dimension_names: tuple[str, ...]

    @property
    def array_dimensions(self) -> tuple[str, ...]:
        return self.dimension_names


class Dataset(GroupSpec[DatasetAttrs, GroupSpec[Any, Any] | DataArray]):
    """
    A GeoZarr Dataset.
    """

    @model_validator(mode="after")
    def check_valid_coordinates(self) -> Self:
        """
        Validate the coordinates of the GeoZarr DataSet.

        This method checks that all DataArrays in the dataset have valid coordinates
        according to the GeoZarr specification.

        Returns
        -------
        GroupSpec[Any, Any]
            The validated GeoZarr DataSet.
        """
        return check_valid_coordinates(self)

    @model_validator(mode="after")
    def validate_grid_mapping(self) -> Self:
        return check_grid_mapping(self)


class MultiscaleGroup(GroupSpec[MultiscaleGroupAttrs, DataArray | GroupSpec[Any, Any]]):
    """
    A GeoZarr Multiscale Group.
    """
