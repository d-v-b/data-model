"""Types and constants for the GeoZarr data API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, NotRequired, TypedDict


class TileMatrixLimitJSON(TypedDict):
    tileMatrix: str
    minTileCol: int
    minTileRow: int
    maxTileCol: int
    maxTileRow: int


class XarrayEncodingJSON(TypedDict):
    chunks: NotRequired[tuple[int, ...]]
    compressors: Any


class StandardXCoordAttrsJSON(TypedDict):
    units: Literal["m"]
    long_name: Literal["x coordinate of projection"]
    standard_name: Literal["projection_x_coordinate"]
    _ARRAY_DIMENSIONS: list[Literal["x"]]


class StandardYCoordAttrsJSON(TypedDict):
    units: Literal["m"]
    long_name: Literal["y coordinate of projection"]
    standard_name: Literal["projection_y_coordinate"]
    _ARRAY_DIMENSIONS: list[Literal["y"]]


class OverviewLevelJSON(TypedDict):
    level: int
    zoom: int
    width: int
    height: int
    scale_factor: int


class TileMatrixJSON(TypedDict):
    id: str
    scaleDenominator: float
    cellSize: float
    pointOfOrigin: tuple[float, float] | list[float]
    tileWidth: int
    tileHeight: int
    matrixWidth: int
    matrixHeight: int


class TileMatrixSetJSON(TypedDict):
    id: str
    title: str | None
    crs: str | None
    supportedCRS: str | None
    orderedAxes: tuple[str, str] | None | list[str]
    tileMatrices: tuple[TileMatrixJSON, ...] | list[TileMatrixJSON]


class TMSMultiscalesJSON(TypedDict):
    """
    Typeddict model of the `multiscales` attribute of Zarr groups that implement the
    OGC TileMatrixSet multiscales structure
    """

    tile_matrix_set: TileMatrixSetJSON
    resampling_method: ResamplingMethod
    tile_matrix_limits: Mapping[str, TileMatrixLimitJSON]


class TMSMultiscalesAttrsJSON(TypedDict):
    multiscales: TMSMultiscalesJSON


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
XARRAY_DIMS_KEY: Final = "_ARRAY_DIMENSIONS"
