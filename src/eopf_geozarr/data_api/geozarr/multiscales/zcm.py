from __future__ import annotations

from typing import Final, Literal, NotRequired

from pydantic import BaseModel, field_validator
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import (
    ZarrConventionMetadata,
    ZarrConventionMetadataJSON,
)

ConventionID = Literal["d35379db-88df-4056-af3a-620245f8e347"]
CONVENTION_ID: Final[ConventionID] = "d35379db-88df-4056-af3a-620245f8e347"

ConventionSchemaURL = Literal[
    "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1.0/schema.json"
]
CONVENTION_SCHEMA_URL: Final[ConventionSchemaURL] = (
    "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1.0/schema.json"
)

ConventionSpecURL = Literal[
    "https://github.com/zarr-conventions/multiscales/blob/v0.1.0/README.md"
]
CONVENTION_SPEC_URL: Final[ConventionSpecURL] = (
    "https://github.com/zarr-conventions/multiscales/blob/v0.1.0/README.md"
)


class MultiscaleConventionMetadata(ZarrConventionMetadata):
    uuid: ConventionID = CONVENTION_ID
    schema_url: ConventionSchemaURL = CONVENTION_SCHEMA_URL
    name: Literal["multiscales"] = "multiscales"
    description: Literal["Multiscale layout of zarr datasets"] = (
        "Multiscale layout of zarr datasets"
    )
    spec_url: ConventionSpecURL = CONVENTION_SPEC_URL


class MultiscaleConventionMetadataJSON(TypedDict):
    uuid: NotRequired[Literal["d35379db-88df-4056-af3a-620245f8e347"]]
    schema_url: NotRequired[
        Literal[
            "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/heads/main/schema.json"
        ]
    ]
    name: NotRequired[Literal["multiscales"]]
    description: NotRequired[Literal["Multiscale layout of zarr datasets"]]
    spec_url: NotRequired[
        Literal[
            "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/heads/main/README.md"
        ]
    ]


class ZarrConventionAttrs(BaseModel):
    zarr_conventions: tuple[ZarrConventionMetadata, ...]

    model_config = {"extra": "allow"}


class Transform(BaseModel):
    scale: tuple[float, ...] | MISSING = MISSING
    translation: tuple[float, ...] | MISSING = MISSING


class TransformJSON(TypedDict):
    scale: NotRequired[tuple[float, ...]]
    translation: NotRequired[tuple[float, ...]]


class ScaleLevel(BaseModel):
    asset: str
    derived_from: str | MISSING = MISSING
    transform: Transform | MISSING = MISSING
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class ScaleLevelJSON(TypedDict):
    asset: str
    derived_from: NotRequired[str]
    transform: TransformJSON
    resampling_method: NotRequired[str]


class Multiscales(BaseModel):
    layout: tuple[ScaleLevel, ...]
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class MultiscalesJSON(TypedDict):
    layout: tuple[ScaleLevelJSON, ...]
    resampling_method: NotRequired[str]


class MultiscalesAttrs(ZarrConventionAttrs):
    multiscales: Multiscales
    model_config = {"extra": "allow"}

    @field_validator("zarr_conventions", mode="after")
    @classmethod
    def ensure_multiscales_convention(
        cls, value: tuple[ZarrConventionMetadata, ...]
    ) -> tuple[ZarrConventionMetadata, ...]:
        """
        Iterate over the elements of zarr_conventions and check that at least one of them is
        multiscales
        """
        success: bool = False
        errors: dict[int, ValueError] = {}
        for idx, convention_meta in enumerate(value):
            try:
                MultiscaleConventionMetadata(**convention_meta.model_dump())
                success = True
            except ValueError as e:
                errors[idx] = e
                pass
        if not success:
            raise ValueError("Multiscales convention not found. Errors: " + str(errors))
        return value


class MultiscalesAttrsJSON(TypedDict):
    zarr_conventions: tuple[ZarrConventionMetadataJSON, ...]
    multiscales: MultiscalesJSON
