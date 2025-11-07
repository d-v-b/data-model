"""
Pydantic-zarr integrated models for Sentinel-2 EOPF Zarr data structure.

Uses the new pyz.GroupSpec with TypedDict members to enforce strict structure validation.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, Self, Union
from typing_extensions import TypedDict

import numpy as np
from pydantic import BaseModel, Field, model_validator

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    CFStandardName,
    DatasetAttrs,
)
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec
from eopf_geozarr.s2_optimization.s2_band_mapping import BAND_INFO


# ============================================================================
# Sentinel-2 Array Attributes Models
# ============================================================================


class BandMetadata(BaseModel):
    """Metadata for a single Sentinel-2 spectral band.

    Contains physical and spectral characterization of a band.
    """

    bandwidth: Annotated[float | str, Field(description="Bandwidth in nm")]
    central_wavelength: Annotated[float, Field(description="Central wavelength in nm")]
    onboard_compression_rate: Annotated[str | float, Field(description="Compression rate")]
    onboard_integration_time: Annotated[str | float, Field(description="Integration time")]
    physical_gain: Annotated[str | float, Field(description="Physical gain factor")]
    spectral_response_step: Annotated[str, Field(description="Spectral response step")]
    spectral_response_values: Annotated[str, Field(description="Spectral response curve values")]
    units: Annotated[str, Field(description="Unit of measurement")]
    wavelength_max: Annotated[float, Field(description="Maximum wavelength in nm")]
    wavelength_min: Annotated[float, Field(description="Minimum wavelength in nm")]


class BandDescription(BaseModel):
    """Collection of band metadata for all Sentinel-2 bands.

    Maps band identifiers (e.g., '01', '02', '8A', 'b02') to their metadata.
    """

    # Allow arbitrary band IDs as keys with BandMetadata values
    model_config = {"extra": "allow"}

    def __getitem__(self, key: str) -> BandMetadata:
        """Get band metadata by band ID."""
        value = getattr(self, key, None)
        if value is None:
            raise KeyError(f"Band {key} not found")
        return value

    def __iter__(self):
        """Iterate over band IDs."""
        for key in self.model_fields:
            yield key

    def items(self):
        """Get (band_id, metadata) pairs."""
        for key in self.model_fields:
            yield key, getattr(self, key)


class OtherMetadata(BaseModel):
    """Sentinel-2 product metadata container.

    Stores various metadata about the product including:
    - Quality information (L0/L2A, processing details)
    - Band descriptions (spectral characteristics)
    - Atmospheric corrections (AOT, water vapor)
    - Geolocation and timing information
    """

    # Core metadata fields
    AOT_retrieval_model: Annotated[str, Field(description="Aerosol Optical Thickness retrieval model")]
    L0_ancillary_data_quality: Annotated[str, Field(description="L0 ancillary data quality indicator")]
    L0_ephemeris_data_quality: Annotated[str, Field(description="L0 ephemeris data quality")]
    NUC_table_ID: Annotated[int | str, Field(description="Non-Uniformity Correction table ID")]
    SWIR_rearrangement_flag: Annotated[str | None, Field(description="SWIR band rearrangement flag")]
    UTM_zone_identification: Annotated[str, Field(description="UTM zone identifier")]
    absolute_location_assessment_from_AOCS: Annotated[str, Field(description="Location assessment")]

    # Band information
    band_description: Annotated[dict[str, BandMetadata], Field(description="Spectral band metadata")]

    # Accuracy declarations
    declared_accuracy_of_AOT_model: Annotated[float | None, Field(description="AOT model accuracy")]
    declared_accuracy_of_radiative_transfer_model: Annotated[float | None, Field(description="Radiative transfer accuracy")]
    declared_accuracy_of_water_vapour_model: Annotated[float | None, Field(description="Water vapor model accuracy")]

    # Correction flags
    electronic_crosstalk_correction_flag: Annotated[str | bool, Field(description="Electronic crosstalk correction")]
    optical_crosstalk_correction_flag: Annotated[str | bool, Field(description="Optical crosstalk correction")]
    onboard_compression_flag: Annotated[str | bool, Field(description="Onboard compression applied")]
    onboard_equalization_flag: Annotated[str | bool, Field(description="Onboard equalization applied")]

    # Product and geometry information
    eopf_category: Annotated[str, Field(description="EOPF product category")]
    geometric_refinement: Annotated[dict[str, Any] | str | None, Field(description="Geometric refinement information")]
    history: Annotated[list[dict[str, Any]] | str | None, Field(description="Processing history")]
    horizontal_CRS_code: Annotated[str, Field(description="Coordinate Reference System code")]
    horizontal_CRS_name: Annotated[str, Field(description="CRS name")]
    mean_sensing_time: Annotated[str | None, Field(description="Mean acquisition time")]

    # Sun/sensor geometry
    mean_sun_azimuth_angle_in_deg_for_all_bands_all_detectors: Annotated[float, Field(description="Mean sun azimuth in degrees")]
    mean_sun_zenith_angle_in_deg_for_all_bands_all_detectors: Annotated[float, Field(description="Mean sun zenith in degrees")]

    # Atmospheric parameters
    mean_value_of_aerosol_optical_thickness: Annotated[float | None, Field(description="Mean AOT value")]
    mean_value_of_total_water_vapour_content: Annotated[float | None, Field(description="Mean water vapor content")]

    # Meteo information (flexible structure)
    meteo: Annotated[dict[str, Any] | None, Field(description="Meteorological parameters")]

    # Quality assessments
    multispectral_registration_assessment: Annotated[str | None, Field(description="Registration quality")]
    product_quality_status: Annotated[str, Field(description="Product quality status")]
    planimetric_stability_assessment_from_AOCS: Annotated[str | None, Field(description="Planimetric stability")]

    # Data degradation
    percentage_of_degraded_MSI_data: Annotated[float | None, Field(description="Percentage of degraded data")]

    # Ozone information
    ozone_source: Annotated[str | None, Field(description="Ozone data source")]
    ozone_value: Annotated[float | str | None, Field(description="Ozone value")]

    # Reference band
    spectral_band_of_reference: Annotated[str, Field(description="Reference spectral band")]

    # Allow extra fields for extensibility
    model_config = {"extra": "allow"}


class Sentinel2ArrayAttributes(BaseModel):
    """Array attributes for Sentinel-2 measurement data.

    Contains structured metadata from Sentinel-2 products including:
    - other_metadata: Detailed processing and band information
    - stac_discovery: STAC-compliant discovery metadata
    """

    other_metadata: Annotated[OtherMetadata, Field(description="Product metadata")]
    stac_discovery: Annotated[dict[str, Any] | None, Field(description="STAC discovery metadata")]

    # Allow extra fields for extensibility
    model_config = {"extra": "allow"}


# Resolution levels
ResolutionLevel = Literal["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]

# Member type for groups with any nested structures (groups or arrays)
# Used for groups with dynamic or variable nested structures
AnyMembers = Mapping[str, Union[GroupSpec[Any, Any], ArraySpec[Any]]]

# Sentinel-2 spectral band identifiers
BandName = Literal[
    # 10m bands
    "b02",  # Blue
    "b03",  # Green
    "b04",  # Red
    "b08",  # NIR
    # 20m bands
    "b05",  # Red Edge 1
    "b06",  # Red Edge 2
    "b07",  # Red Edge 3
    "b8a",  # NIR Narrow
    "b11",  # SWIR 1
    "b12",  # SWIR 2
    # 60m bands
    "b01",  # Coastal aerosol
    "b09",  # Water Vapor
    "b10",  # Cirrus
]

# All valid band names
ALL_BAND_NAMES = {
    "b01",
    "b02",
    "b03",
    "b04",
    "b05",
    "b06",
    "b07",
    "b08",
    "b8a",
    "b09",
    "b10",
    "b11",
    "b12",
}

# Native band mappings for validation
NATIVE_BANDS = {
    10: {"b02", "b03", "b04", "b08"},
    20: {"b05", "b06", "b07", "b8a", "b11", "b12"},
    60: {"b01", "b09", "b10"},
}

# Resolution name to meters mapping
RESOLUTION_TO_METERS = {
    "r10m": 10,
    "r20m": 20,
    "r60m": 60,
    "r120m": 120,
    "r360m": 360,
    "r720m": 720,
}

# Quality data names
QualityDataName = Literal[
    "scl",  # Scene Classification Layer
    "aot",  # Aerosol Optical Thickness
    "wvp",  # Water Vapor
    "cld",  # Cloud probability
    "snw",  # Snow probability
]

# Variable type for determining resampling strategy
VariableType = Literal[
    "reflectance",     # Block averaging
    "classification",  # Nearest neighbor
    "quality_mask",    # Logical OR
    "probability",     # Averaged clamping
]


class Sentinel2BandInfo(BaseModel):
    """Complete information about a Sentinel-2 spectral band."""

    name: BandName
    native_resolution: Annotated[int, Field(ge=10, le=60)]
    data_type: str = "uint16"
    wavelength_center: Annotated[float, Field(gt=0, description="nm")]
    wavelength_width: Annotated[float, Field(gt=0, description="nm")]
    long_name: str
    standard_name: str = "toa_bidirectional_reflectance"
    units: str = "1"

    @classmethod
    def from_band_name(cls, band_name: BandName) -> Sentinel2BandInfo:
        """Create BandInfo from band name."""
        if band_name not in BAND_INFO:
            raise ValueError(f"Unknown band name: {band_name}")
        info = BAND_INFO[band_name]
        return cls(
            name=band_name,
            native_resolution=info.native_resolution,
            wavelength_center=info.wavelength_center,
            wavelength_width=info.wavelength_width,
            long_name=f"Band {band_name.upper()}",
        )


class Sentinel2DataArrayAttrs(BaseDataArrayAttrs):
    """Extended attributes for Sentinel-2 data arrays."""

    long_name: str
    standard_name: CFStandardName | str | None = None
    units: str = "1"
    model_config = {"extra": "allow"}


class Sentinel2RootAttrs(BaseModel):
    """Root-level attributes for Sentinel-2 DataTree."""

    Conventions: str | None = Field(default=None)
    title: str | None = Field(default=None)
    institution: str | None = Field(default=None)
    source: str | None = Field(default=None)
    history: str | None = Field(default=None)
    model_config = {"extra": "allow"}


class Sentinel2DataArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Sentinel-2 data array integrated with pydantic-zarr."""

    @classmethod
    def from_band(
        cls,
        band_name: BandName,
        data: np.ndarray,
        crs_name: str = "crs",
    ) -> Sentinel2DataArray:
        """Create a Sentinel-2 band array from numpy data."""
        band_info = Sentinel2BandInfo.from_band_name(band_name)

        attrs = Sentinel2DataArrayAttrs(
            long_name=band_info.long_name,
            standard_name=band_info.standard_name,
            units=band_info.units,
            grid_mapping=crs_name,
        )

        return cls.from_array(data, attributes=attrs)

    @classmethod
    def from_zarr(cls, zarr_array: Any) -> Self:
        """Load from zarr array and convert dict attributes to Pydantic model."""
        # Call parent from_zarr to load the array
        result = super().from_zarr(zarr_array)

        # Convert dict attributes to Sentinel2DataArrayAttrs if needed
        if isinstance(result.attributes, dict):
            attrs = Sentinel2DataArrayAttrs(**result.attributes)
            result = result.model_copy(update={"attributes": attrs})

        return result


class Sentinel2CoordinateArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Coordinate array for Sentinel-2 data."""

    @classmethod
    def create_x_coordinate(
        cls,
        values: np.ndarray,
        units: str = "m",
    ) -> Sentinel2CoordinateArray:
        """Create X coordinate array."""
        attrs = Sentinel2DataArrayAttrs(
            long_name="x coordinate of projection",
            standard_name="projection_x_coordinate",
            units=units,
        )
        return cls.from_array(values, attributes=attrs)

    @classmethod
    def create_y_coordinate(
        cls,
        values: np.ndarray,
        units: str = "m",
    ) -> Sentinel2CoordinateArray:
        """Create Y coordinate array."""
        attrs = Sentinel2DataArrayAttrs(
            long_name="y coordinate of projection",
            standard_name="projection_y_coordinate",
            units=units,
        )
        return cls.from_array(values, attributes=attrs)

    @classmethod
    def create_time_coordinate(
        cls,
        values: np.ndarray,
    ) -> Sentinel2CoordinateArray:
        """Create time coordinate array."""
        attrs = Sentinel2DataArrayAttrs(
            long_name="time",
            standard_name="time",
            units="seconds since 1970-01-01",
        )
        return cls.from_array(values, attributes=attrs)

    @classmethod
    def from_zarr(cls, zarr_array: Any) -> Self:
        """Load from zarr array and convert dict attributes to Pydantic model."""
        # Call parent from_zarr to load the array
        result = super().from_zarr(zarr_array)

        # Convert dict attributes to Sentinel2DataArrayAttrs if needed
        if isinstance(result.attributes, dict):
            attrs = Sentinel2DataArrayAttrs(**result.attributes)
            result = result.model_copy(update={"attributes": attrs})

        return result


# TypedDict definitions for members structure
class Sentinel2ResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members dict for a resolution dataset (r10m, r20m, r60m).

    Closed TypedDict - no extra keys are allowed beyond those explicitly defined.
    All fields are optional since not all bands are present in every resolution.
    """

    x: ArraySpec[Any]
    y: ArraySpec[Any]
    time: ArraySpec[Any]
    b01: ArraySpec[Any]
    b02: ArraySpec[Any]
    b03: ArraySpec[Any]
    b04: ArraySpec[Any]
    b05: ArraySpec[Any]
    b06: ArraySpec[Any]
    b07: ArraySpec[Any]
    b08: ArraySpec[Any]
    b8a: ArraySpec[Any]
    b09: ArraySpec[Any]
    b10: ArraySpec[Any]
    b11: ArraySpec[Any]
    b12: ArraySpec[Any]


class Sentinel2ResolutionDataset(GroupSpec[DatasetAttrs, Sentinel2ResolutionMembers]):  # type: ignore[type-var]
    """A single resolution dataset within reflectance (r10m, r20m, r60m)."""

    resolution_level: ResolutionLevel = Field(default="r10m")

    def get_coordinates(self) -> dict[str, ArraySpec[Any]] | None:
        """Get coordinate arrays (x, y, time)."""
        coords = {}
        for name in ["x", "y", "time"]:
            if name in self.members:
                coords[name] = self.members[name]  # type: ignore[literal-required]
        return coords if coords else None

    def get_bands(self) -> dict[str, ArraySpec[Any]]:
        """Get all band arrays."""
        bands = {}
        for band in ALL_BAND_NAMES:
            if band in self.members:
                bands[band] = self.members[band]  # type: ignore[literal-required]
        return bands


class Sentinel2ReflectanceMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for reflectance group.

    Closed TypedDict - only r10m, r20m, r60m keys are allowed.
    All fields are optional since not all resolutions are always present.
    """

    r10m: Sentinel2ResolutionDataset
    r20m: Sentinel2ResolutionDataset
    r60m: Sentinel2ResolutionDataset


class Sentinel2ReflectanceGroup(GroupSpec[DatasetAttrs, Sentinel2ReflectanceMembers]):  # type: ignore[type-var]
    """Reflectance data organized by resolution."""

    def get_resolution_dataset(
        self, resolution: ResolutionLevel
    ) -> Sentinel2ResolutionDataset | None:
        """Get a resolution dataset by name."""
        return self.members.get(resolution)  # type: ignore[return-value]

    def list_resolutions(self) -> list[str]:
        """List available resolutions."""
        return [k for k in self.members.keys() if k.startswith("r")]


class Sentinel2MeasurementsMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for measurements group.

    Closed TypedDict - only 'reflectance' key is allowed.
    """

    reflectance: Sentinel2ReflectanceGroup


class Sentinel2MeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel2MeasurementsMembers]):  # type: ignore[type-var]
    """Measurements group containing reflectance data."""

    @property
    def reflectance(self) -> Sentinel2ReflectanceGroup:
        """Get reflectance subgroup."""
        return self.members["reflectance"]


# Quality data groups - need resolution-level typed groups

class Sentinel2AtmosphereResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for atmosphere data at a specific resolution.

    Closed TypedDict - may contain aot and/or wvp arrays depending on available data.
    All fields are optional since not all data products are always present.
    """
    aot: ArraySpec[Any]
    wvp: ArraySpec[Any]
    x: ArraySpec[Any]  # Coordinate arrays
    y: ArraySpec[Any]


class Sentinel2AtmosphereResolutionDataset(GroupSpec[DatasetAttrs, Sentinel2AtmosphereResolutionMembers]):  # type: ignore[type-var]
    """Atmosphere data at a single resolution."""
    pass


class Sentinel2AtmosphereMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for atmosphere group containing resolution datasets."""
    r10m: Sentinel2AtmosphereResolutionDataset
    r20m: Sentinel2AtmosphereResolutionDataset
    r60m: Sentinel2AtmosphereResolutionDataset


class Sentinel2AtmosphereDataset(GroupSpec[DatasetAttrs, Sentinel2AtmosphereMembers]):  # type: ignore[type-var]
    """Atmosphere quality data (AOT, WVP) at multiple resolutions."""

    pass


class Sentinel2ProbabilityDataset(GroupSpec[DatasetAttrs, AnyMembers]):
    """Probability data (cloud, snow) at multiple resolutions."""

    pass


class Sentinel2QuicklookDataset(GroupSpec[DatasetAttrs, AnyMembers]):
    """True Color Image (TCI) quicklook data at multiple resolutions."""

    pass


class Sentinel2MaskDataset(GroupSpec[DatasetAttrs, AnyMembers]):
    """Mask data containing classification and detector footprints."""

    pass


class Sentinel2QualityMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for quality group.

    Closed TypedDict - only atmosphere, probability, l2a_quicklook, mask keys are allowed.
    """

    atmosphere: Sentinel2AtmosphereDataset
    probability: Sentinel2ProbabilityDataset
    l2a_quicklook: Sentinel2QuicklookDataset
    mask: Sentinel2MaskDataset


class Sentinel2QualityGroup(GroupSpec[DatasetAttrs, Sentinel2QualityMembers]):  # type: ignore[type-var]
    """Quality group containing atmosphere, probability, classification, and quicklook data."""

    def get_atmosphere_data(self) -> Sentinel2AtmosphereDataset | None:
        """Get atmosphere subgroup."""
        return self.members.get("atmosphere")

    def get_probability_data(self) -> Sentinel2ProbabilityDataset | None:
        """Get probability subgroup."""
        return self.members.get("probability")

    def get_quicklook_data(self) -> Sentinel2QuicklookDataset | None:
        """Get L2A quicklook subgroup."""
        return self.members.get("l2a_quicklook")

    def get_mask_data(self) -> Sentinel2MaskDataset | None:
        """Get mask subgroup."""
        return self.members.get("mask")


# Conditions groups
class Sentinel2GeometryGroup(GroupSpec[DatasetAttrs, AnyMembers]):
    """Geometry group containing sun and viewing angles."""

    pass


class Sentinel2MeteorologyGroup(GroupSpec[DatasetAttrs, AnyMembers]):
    """Meteorology group containing CAMS and ECMWF atmospheric data."""

    pass


class Sentinel2ConditionsMaskGroup(GroupSpec[DatasetAttrs, AnyMembers]):
    """Mask subgroup in conditions."""

    pass


class Sentinel2ConditionsMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for conditions group.

    Closed TypedDict - only geometry, mask, meteorology keys are allowed.
    """

    geometry: Sentinel2GeometryGroup
    mask: Sentinel2ConditionsMaskGroup
    meteorology: Sentinel2MeteorologyGroup


class Sentinel2ConditionsGroup(GroupSpec[DatasetAttrs, Sentinel2ConditionsMembers]):  # type: ignore[type-var]
    """Conditions group containing geometry and meteorology data."""

    def get_geometry_data(self) -> Sentinel2GeometryGroup | None:
        """Get geometry subgroup."""
        return self.members.get("geometry")

    def get_mask_data(self) -> Sentinel2ConditionsMaskGroup | None:
        """Get mask subgroup."""
        return self.members.get("mask")

    def get_meteorology_data(self) -> Sentinel2MeteorologyGroup | None:
        """Get meteorology subgroup."""
        return self.members.get("meteorology")


# Root model
class Sentinel2RootMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for Sentinel-2 root group.

    Closed TypedDict - only measurements, quality, conditions keys are allowed.
    """

    measurements: Sentinel2MeasurementsGroup
    quality: Sentinel2QualityGroup
    conditions: Sentinel2ConditionsGroup


class Sentinel2Root(GroupSpec[Sentinel2RootAttrs, Sentinel2RootMembers]):  # type: ignore[type-var]
    """Complete Sentinel-2 EOPF Zarr hierarchy.

    The hierarchy follows EOPF organization:
    Root
    ├── measurements/
    │   └── reflectance/
    │       ├── r10m/
    │       ├── r20m/
    │       └── r60m/
    ├── quality/
    │   ├── atmosphere/
    │   ├── l2a_quicklook/
    │   ├── mask/
    │   └── probability/
    └── conditions/
        ├── geometry/
        ├── mask/
        └── meteorology/
    """

    @model_validator(mode="after")
    def validate_sentinel2_structure(self) -> Self:
        """Validate overall Sentinel-2 dataset structure."""
        # Ensure required groups exist
        required = {"measurements", "quality", "conditions"}
        existing = set(self.members.keys())
        missing = required - existing
        if missing:
            raise ValueError(
                f"Sentinel-2 dataset must contain groups: {missing}"
            )

        return self

    @property
    def measurements(self) -> Sentinel2MeasurementsGroup:
        """Get measurements group."""
        return self.members["measurements"]

    @property
    def quality(self) -> Sentinel2QualityGroup:
        """Get quality group."""
        return self.members["quality"]

    @property
    def conditions(self) -> Sentinel2ConditionsGroup:
        """Get conditions group."""
        return self.members["conditions"]

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary representation."""
        return self.model_dump(exclude_none=True)
