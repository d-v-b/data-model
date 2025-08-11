"""Common utilities for GeoZarr data API."""
import io
import urllib
import urllib.request
from typing import Literal, TypeAlias, TypeVar

from cf_xarray.utils import parse_cf_standard_name_table
from pydantic import BaseModel


def get_cf_standard_names(url: str) -> tuple[str, ...]:
    """Retrieve the set of CF standard names and return them as a tuple."""

    headers = {"User-Agent": "eopf_geozarr"}

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()  # Read the entire response body into memory
            content_fobj = io.BytesIO(content)
    except urllib.error.URLError as e:
        raise e

    _info, table, _aliases = parse_cf_standard_name_table(source=content_fobj)
    return tuple(table.keys())


# This is a URL to the CF standard names table.
CF_STANDARD_NAME_URL = (
    "https://raw.githubusercontent.com/cf-convention/cf-convention.github.io/"
    "master/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
)


# this does IO against github. consider locally storing this data instead if fetching every time
# is problematic.
CF_STANDARD_NAMES = get_cf_standard_names(url=CF_STANDARD_NAME_URL)


def check_standard_name(name: str) -> str:
    """
    Check if the standard name is valid according to the CF conventions.

    Parameters
    ----------
    name : str
        The standard name to check.

    Returns
    -------
    str
        The validated standard name.

    Raises
    ------
    ValueError
        If the standard name is not valid.
    """

    if name in CF_STANDARD_NAMES:
        return name
    raise ValueError(
        f"Invalid standard name: {name}. This name was not found in the list of CF standard names."
    )


# todo: narrow to literal type
ResamplingMethod = Literal[
    "nearest",
    "average",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "mode",
    "max",
    "min",
    "med",
    "sum",
    "q1",
    "q3",
    "rms",
    "gauss",
]
"""A string literal indicating a resampling method"""

TileMatrixSet: TypeAlias = str | dict[str, object]
"""Identifier, URI, or inline JSON object compliant with OGC TileMatrixSet v2"""


class TileMatrixSetLimits(BaseModel):
    min_tile_col: int
    min_tile_row: int
    max_tile_col: int
    max_tile_row: int


class MultiscaleAttrs(BaseModel):
    """
    Attributes for a GeoZarr multiscale dataset.

    Attributes
    ----------
    tile_matrix_set : str
        The tile matrix set identifier for the multiscale dataset.
    resampling_method : ResamplingMethod
        The name of the resampling method for the multiscale dataset.
    tile_matrix_set_limits : dict[str, TileMatrixSetLimits] | None, optional
        The tile matrix set limits for the multiscale dataset.
    """

    tile_matrix_set: TileMatrixSet
    resampling_method: ResamplingMethod
    tile_matrix_set_limits: dict[str, TileMatrixSetLimits] | None = None
