"""
Pydantic-zarr integrated models for Sentinel-1 EOPF Zarr data structure.
"""
from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    CFStandardName,
    DatasetAttrs,
)


class Sentinel1DataArrayAttrs(BaseDataArrayAttrs):
    """Extended attributes for Sentinel-1 data arrays."""

    long_name: str
    standard_name: CFStandardName | str | None = None
    units: str = "1"
    model_config = {"extra": "allow"}


class Sentinel1RootAttrs(BaseModel):
    """Root-level attributes for Sentinel-1 DataTree."""

    Conventions: str | None = Field(default=None)
    title: str | None = Field(default=None)
    institution: str | None = Field(default=None)
    source: str | None = Field(default=None)
    history: str | None = Field(default=None)
    model_config = {"extra": "allow"}


class Sentinel1DataArray(ArraySpec[Sentinel1DataArrayAttrs]):
    """Sentinel-1 data array integrated with pydantic-zarr."""
    pass


# Conditions groups
class Sentinel1AntennaPatternGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Antenna pattern group containing antenna characteristics."""
    pass


class Sentinel1AttitudeGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Attitude group containing spacecraft attitude data."""
    pass


class Sentinel1AzimuthFmRateGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Azimuth FM rate group."""
    pass


class Sentinel1CoordinateConversionGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Coordinate conversion group."""
    pass


class Sentinel1DopplerCentroidGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Doppler centroid group."""
    pass


class Sentinel1GcpGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Ground Control Points (GCP) group."""
    pass


class Sentinel1OrbitGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Orbit group containing spacecraft position and velocity."""
    pass


class Sentinel1ReferenceReplicaGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Reference replica group."""
    pass


class Sentinel1ReplicaGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Replica group containing pulse replica data."""
    pass


class Sentinel1TerrainHeightGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Terrain height group."""
    pass


class Sentinel1ConditionsGroup(GroupSpec[DatasetAttrs, GroupSpec[Any, Any]]):
    """Conditions group containing acquisition and processing metadata."""

    def get_antenna_pattern(self) -> GroupSpec[Any, Any] | None:
        """Get antenna pattern subgroup."""
        if self.members is None:
            return None
        return self.members.get("antenna_pattern")

    def get_attitude(self) -> GroupSpec[Any, Any] | None:
        """Get spacecraft attitude subgroup."""
        if self.members is None:
            return None
        return self.members.get("attitude")

    def get_azimuth_fm_rate(self) -> GroupSpec[Any, Any] | None:
        """Get azimuth FM rate subgroup."""
        if self.members is None:
            return None
        return self.members.get("azimuth_fm_rate")

    def get_coordinate_conversion(self) -> GroupSpec[Any, Any] | None:
        """Get coordinate conversion subgroup."""
        if self.members is None:
            return None
        return self.members.get("coordinate_conversion")

    def get_doppler_centroid(self) -> GroupSpec[Any, Any] | None:
        """Get Doppler centroid subgroup."""
        if self.members is None:
            return None
        return self.members.get("doppler_centroid")

    def get_gcp(self) -> GroupSpec[Any, Any] | None:
        """Get Ground Control Points subgroup."""
        if self.members is None:
            return None
        return self.members.get("gcp")

    def get_orbit(self) -> GroupSpec[Any, Any] | None:
        """Get orbit subgroup."""
        if self.members is None:
            return None
        return self.members.get("orbit")

    def get_reference_replica(self) -> GroupSpec[Any, Any] | None:
        """Get reference replica subgroup."""
        if self.members is None:
            return None
        return self.members.get("reference_replica")

    def get_replica(self) -> GroupSpec[Any, Any] | None:
        """Get replica subgroup."""
        if self.members is None:
            return None
        return self.members.get("replica")

    def get_terrain_height(self) -> GroupSpec[Any, Any] | None:
        """Get terrain height subgroup."""
        if self.members is None:
            return None
        return self.members.get("terrain_height")


# Quality groups
class Sentinel1CalibrationGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Calibration group containing radiometric calibration data."""
    pass


class Sentinel1NoiseGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Noise group containing noise estimation data."""
    pass


class Sentinel1NoiseAzimuthGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Noise azimuth group containing azimuth noise vectors."""
    pass


class Sentinel1NoiseRangeGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Noise range group containing range noise vectors."""
    pass


class Sentinel1QualityGroup(GroupSpec[DatasetAttrs, GroupSpec[Any, Any]]):
    """Quality group containing quality assurance and calibration data."""

    def get_calibration(self) -> GroupSpec[Any, Any] | None:
        """Get calibration subgroup."""
        if self.members is None:
            return None
        return self.members.get("calibration")

    def get_noise(self) -> GroupSpec[Any, Any] | None:
        """Get noise subgroup."""
        if self.members is None:
            return None
        return self.members.get("noise")

    def get_noise_azimuth(self) -> GroupSpec[Any, Any] | None:
        """Get noise azimuth subgroup."""
        if self.members is None:
            return None
        return self.members.get("noise_azimuth")

    def get_noise_range(self) -> GroupSpec[Any, Any] | None:
        """Get noise range subgroup."""
        if self.members is None:
            return None
        return self.members.get("noise_range")


# Measurements
class Sentinel1MeasurementsGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Measurements group containing SAR imagery data."""
    pass


# Polarization group
class Sentinel1PolarizationGroup(GroupSpec[DatasetAttrs, GroupSpec[Any, Any]]):
    """Polarization-specific group containing all data for one polarization."""

    @property
    def conditions(self) -> GroupSpec[Any, Any] | None:
        """Get the conditions group."""
        if self.members is None:
            return None
        return self.members.get("conditions")

    @property
    def measurements(self) -> GroupSpec[Any, Any] | None:
        """Get the measurements group."""
        if self.members is None:
            return None
        return self.members.get("measurements")

    @property
    def quality(self) -> GroupSpec[Any, Any] | None:
        """Get the quality group."""
        if self.members is None:
            return None
        return self.members.get("quality")


# Root model
class Sentinel1Root(
    GroupSpec[
        Sentinel1RootAttrs,
        Sentinel1PolarizationGroup | GroupSpec[Any, Any],
    ]
):
    """Complete Sentinel-1 EOPF Zarr hierarchy integrated with pydantic-zarr.

    The hierarchy follows EOPF organization with separate groups for each
    polarization (VH and VV):

    Root
    ├── S01SIWGRD_[timestamp]_..._VH/ (VH Polarization)
    │   ├── conditions/
    │   ├── measurements/ (GRD imagery)
    │   └── quality/
    └── S01SIWGRD_[timestamp]_..._VV/ (VV Polarization)
        ├── conditions/
        ├── measurements/
        └── quality/
    """

    @model_validator(mode="after")
    def validate_sentinel1_structure(self) -> Self:
        """Validate overall Sentinel-1 dataset structure."""
        if self.members is None:
            raise ValueError("Sentinel-1 root must have members")

        # Check for at least one polarization group (VH or VV)
        pol_groups = [k for k in self.members.keys() if 'VH' in k or 'VV' in k]
        if not pol_groups:
            raise ValueError("Sentinel-1 dataset must contain at least one polarization group (VH or VV)")

        return self

    def get_polarization_groups(self) -> dict[str, GroupSpec[Any, Any]]:
        """Get all polarization groups (VH, VV, etc.)."""
        if self.members is None:
            return {}

        pol_groups = {}
        for name, member in self.members.items():
            if isinstance(member, GroupSpec):
                pol_groups[name] = member
        return pol_groups

    def get_vh_group(self) -> GroupSpec[Any, Any] | None:
        """Get the VH polarization group."""
        if self.members is None:
            return None
        for name, member in self.members.items():
            if 'VH' in name:
                return member
        return None

    def get_vv_group(self) -> GroupSpec[Any, Any] | None:
        """Get the VV polarization group."""
        if self.members is None:
            return None
        for name, member in self.members.items():
            if 'VV' in name:
                return member
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary representation."""
        return self.model_dump(exclude_none=True)
