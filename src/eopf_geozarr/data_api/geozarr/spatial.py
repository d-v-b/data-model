"""
Models for the Spatial Zarr Convention
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.common import is_none

SPATIAL_UUID = spatial_cm.UUID

# Re-export the zarr-cm TypedDict for the convention metadata object
SpatialConvention = spatial_cm.SpatialAttrs


class Spatial(BaseModel):
    dimensions: list[str] = Field(alias="spatial:dimensions")  # Required field
    bbox: list[float] | None = Field(None, alias="spatial:bbox", exclude_if=is_none)
    transform_type: str = Field("affine", alias="spatial:transform_type")
    transform: list[float] | None = Field(None, alias="spatial:transform", exclude_if=is_none)
    shape: list[int] | None = Field(None, alias="spatial:shape", exclude_if=is_none)
    registration: str = Field("pixel", alias="spatial:registration")

    model_config = {"extra": "allow", "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_dimensions_not_empty(self) -> Spatial:
        """Validate that dimensions list is not empty."""
        if not self.dimensions:
            raise ValueError("spatial:dimensions must contain at least one dimension")
        return self
