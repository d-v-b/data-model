"""Types and constants for the GeoZarr data API."""

from typing import Any, Final, Literal, TypedDict

from typing_extensions import NotRequired


class TileMatrixLimitJSON(TypedDict):
    tileMatrix: str
    minTileCol: int
    minTileRow: int
    maxTileCol: int
    maxTileRow: int


class XarrayEncodingJSON(TypedDict):
    chunks: NotRequired[tuple[int, ...]]
    compressors: Any
    shards: NotRequired[Any]


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


class StandardLonCoordAttrsJSON(TypedDict):
    units: Literal["degrees_east"]
    long_name: Literal["longitude"]
    standard_name: Literal["longitude"]
    _ARRAY_DIMENSIONS: list[Literal["x"]]


class StandardLatCoordAttrsJSON(TypedDict):
    units: Literal["degrees_north"]
    long_name: Literal["latitude"]
    standard_name: Literal["latitude"]
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


# Why is endpoint URL specified twice?
class S3ClientOptions(TypedDict):
    """
    S3 client options
    """

    region_name: NotRequired[str]
    endpoint_url: NotRequired[str]


class S3FsOptions(TypedDict):
    """
    S3FS options
    """

    anon: NotRequired[bool]
    use_ssl: NotRequired[bool]
    client_kwargs: NotRequired[S3ClientOptions]
    endpoint_url: NotRequired[str]
    asynchronous: NotRequired[bool]


class S3Credentials(TypedDict):
    """
    S3 credentials
    """

    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_session_token: str | None
    aws_default_region: str
    aws_profile: str | None
    AWS_ENDPOINT_URL: str | None
