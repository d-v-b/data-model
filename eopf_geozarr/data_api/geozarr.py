from __future__ import annotations
from typing import Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from pydantic_zarr.v2 import ArraySpec, GroupSpec

class GeoZarrDataArrayAttrs(BaseModel):
    """
    Attributes for a GeoZarr DataArray.

    Attributes
    ----------
    array_dimensions : tuple[str, ...]
        Alias for the _ARRAY_DIMENSIONS attribute, which lists the dimension names for this array.
    standard_name : str
        The CF standard name of the variable.
    grid_mapping : object
        The grid mapping of the variable, which is a reference to a grid mapping variable that
        describes the spatial reference of the variable.
    grid_mapping_name : str
        The name of the grid mapping, which is a string that describes the type of grid mapping
        used for the variable.
    """

    # todo: validate that this names listed here are the names of zarr arrays
    # unless the variable is an auxiliary variable
    # see https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-coordinates
    array_dimensions: tuple[str, ...] = Field(alias="_ARRAY_DIMENSIONS")
    standard_name: str
    grid_mapping: object
    grid_mapping_name: str


class GeoZarrDataArray(ArraySpec[GeoZarrDataArrayAttrs]):
    """
    A GeoZarr DataArray variable.
    

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """


def check_valid_coordinates(model: GroupSpec[Any, Any]) -> GroupSpec[Any, Any]:
    """
    Check if the coordinates of a GeoZarr DataArray are valid.

    Parameters
    ----------
    model : GroupSpec[Any, Any]
        The GeoZarr DataArray model to check.

    Returns
    -------
    GroupSpec[Any, Any]
        The validated GeoZarr DataArray model.
    """
    if model.members is None:
        raise ValueError("Model members cannot be None")

    arrays: dict[str, GeoZarrDataArray] = {k: v for k, v in model.members.items() if isinstance(v, GeoZarrDataArray)}
    for key, array in arrays.items():
        for idx, dim in enumerate(array.attributes.array_dimensions):
            if dim not in model.members:
                raise ValueError(f"Dimension '{dim}' for array '{key}' is not defined in the model members.")
            member = model.members[dim]
            if isinstance(member, GroupSpec):
                raise ValueError(f"Dimension '{dim}' for array '{key}' should be a group. Found an array instead.")
            if member.shape[0] != array.shape[idx]:
                raise ValueError(f"Dimension '{dim}' for array '{key}' has a shape mismatch: "
                                 f"{member.shape[0]} != {array.shape[idx]}.")
    return model


class GeoZarrDataset(GroupSpec[Any, GroupSpec[Any, Any] | GeoZarrDataArray]):
    ...

