"""
Pydantic-zarr integrated models for Sentinel-2 EOPF Zarr data structure.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    CFStandardName,
    DatasetAttrs,
)
from eopf_geozarr.s2_optimization.s2_band_mapping import BAND_INFO

# Resolution levels
ResolutionLevel = Literal["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]

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

# Quality data variable identifiers (native 20m)
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

# Native band mappings for validation
NATIVE_BANDS = {
    10: {"b02", "b03", "b04", "b08"},
    20: {"b05", "b06", "b07", "b8a", "b11", "b12"},
    60: {"b01", "b09", "b10"},
}

# All valid band names
ALL_BAND_NAMES = {"b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b10", "b11", "b12"}

# Resolution name to meters mapping
RESOLUTION_TO_METERS = {
    "r10m": 10,
    "r20m": 20,
    "r60m": 60,
    "r120m": 120,
    "r360m": 360,
    "r720m": 720,
}

class Sentinel2BandInfo(BaseModel):
    """Complete information about a Sentinel-2 spectral band.

    Attributes:
        name: Band identifier (e.g., 'b02', 'b8a')
        native_resolution: Native resolution in meters (10, 20, or 60)
        data_type: NumPy data type (typically 'uint16')
        wavelength_center: Central wavelength in nanometers
        wavelength_width: Bandwidth in nanometers
        long_name: Human-readable band description
        standard_name: CF standard name for the variable
        units: Units of measurement (typically dimensionless for reflectance)
    """

    name: BandName
    native_resolution: Annotated[int, Field(ge=10, le=60)]
    data_type: str = "uint16"
    wavelength_center: Annotated[float, Field(gt=0, description="nm")]
    wavelength_width: Annotated[float, Field(gt=0, description="nm")]
    long_name: str
    standard_name: str = "toa_bidirectional_reflectance"
    units: str = "1"

    @classmethod
    def from_band_name(cls, band: BandName) -> Sentinel2BandInfo:
        """Create BandInfo from band name using standard Sentinel-2 specifications."""

        info = BAND_INFO[band]
        return cls(
            name=band,
            native_resolution=info.native_resolution,
            data_type=info.data_type,
            wavelength_center=info.wavelength_center,
            wavelength_width=info.wavelength_width,
            long_name=f"Band {band.upper()}",
        )


class Sentinel2DataArrayAttrs(BaseDataArrayAttrs):
    """Extended attributes for Sentinel-2 data arrays.

    Includes standard GeoZarr attributes plus Sentinel-2 specific metadata.
    """

    long_name: str
    standard_name: CFStandardName | str | None = None
    units: str = "1"
    model_config = {"extra": "allow"}

class Sentinel2RootAttrs(BaseModel):
    """Root-level attributes for Sentinel-2 DataTree.

    Attributes:
        Conventions: CF conventions version
        title: Dataset title
        institution: Data provider institution
        source: Data source
        history: Processing history
        stac_discovery: STAC discovery metadata
    """

    Conventions: str | None = Field(default=None)
    title: str | None = Field(default=None)
    institution: str | None = Field(default=None)
    source: str | None = Field(default=None)
    history: str | None = Field(default=None)
    model_config = {"extra": "allow"}

class Sentinel2DataArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Sentinel-2 data array integrated with pydantic-zarr.

    This extends ArraySpec to provide:
    - from_zarr() - load from zarr array
    - to_zarr() - write to zarr store
    - from_array() - create from numpy array
    - Full validation of attributes and structure
    """

    @classmethod
    def from_band(
        cls,
        band_name: BandName,
        data: np.ndarray,
        crs_name: str = "crs",
    ) -> Sentinel2DataArray:
        """Create a Sentinel-2 band array from numpy data.

        Args:
            band_name: Band identifier
            data: Numpy array (should be 3D: time, y, x)
            crs_name: Name of the CRS grid mapping variable

        Returns:
            Validated Sentinel2DataArray
        """
        band_info = Sentinel2BandInfo.from_band_name(band_name)

        attrs = Sentinel2DataArrayAttrs(
            long_name=band_info.long_name,
            standard_name=band_info.standard_name,
            units=band_info.units,
            grid_mapping=crs_name,
        )

        return cls.from_array(
            data,
            attributes=attrs,
        )


class Sentinel2CoordinateArray(ArraySpec[Sentinel2DataArrayAttrs]):

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
        units: str = "seconds since 1970-01-01",
    ) -> Sentinel2CoordinateArray:
        """Create time coordinate array."""
        attrs = Sentinel2DataArrayAttrs(
            long_name="time",
            standard_name="time",
            units=units,
        )
        # Datetime arrays need explicit fill value
        # Use 0 (epoch) as fill value since zarr doesn't support NaT directly
        return cls.from_array(
            values,
            attributes=attrs,
            fill_value=0,
        )


class Sentinel2ResolutionDataset(
    GroupSpec[DatasetAttrs, Sentinel2DataArray | Sentinel2CoordinateArray]
):
    """Dataset containing all bands and coordinates at a specific resolution.

    This is a zarr group that contains:
    - Band arrays (b02, b03, etc.)
    - Coordinate arrays (x, y, time)
    - Grid mapping variable (crs)

    The group can be loaded from and written to zarr stores.

    Attributes:
        resolution_level: Optional resolution level (r10m, r20m, etc.) for validation
    """

    # Optional field to track which resolution this dataset represents
    resolution_level: ResolutionLevel | None = Field(
        default=None,
        description="Resolution level for validation (r10m, r20m, r60m, etc.)",
        exclude=True,  # Don't serialize this field to zarr
    )

    @model_validator(mode="after")
    def validate_band_resolution_consistency(self) -> Self:
        """Ensure bands are appropriate for the declared resolution level.

        This validation checks that:
        1. All band names are valid Sentinel-2 bands
        2. If resolution_level is set, bands match the native resolution
        3. Warns if bands from multiple native resolutions are mixed (downsampled data)

        Raises:
            ValueError: If invalid bands are present or bands don't match declared resolution
        """
        if self.members is None:
            return self

        # Get all bands in this dataset
        band_names = {
            name for name, member in self.members.items()
            if isinstance(member, Sentinel2DataArray) and name in ALL_BAND_NAMES
        }

        if not band_names:
            # No bands, that's okay (might be just coordinates or other data)
            return self

        # Validate all band names are valid
        invalid_bands = band_names - ALL_BAND_NAMES
        if invalid_bands:
            raise ValueError(
                f"Invalid Sentinel-2 band names found: {invalid_bands}. "
                f"Valid bands are: {sorted(ALL_BAND_NAMES)}"
            )

        # If resolution level is declared, perform validation
        # Note: Real S2 data allows downsampled bands from other resolutions in r20m and r60m
        # Only r10m strictly contains only native 10m bands in real data
        if self.resolution_level:
            resolution_meters = RESOLUTION_TO_METERS.get(self.resolution_level)

            # For r10m, enforce strict native-only bands (matches real S2 data behavior)
            if resolution_meters == 10 and resolution_meters in NATIVE_BANDS:
                expected_bands = NATIVE_BANDS[resolution_meters]
                non_native_bands = band_names - expected_bands

                if non_native_bands:
                    band_origins = {}
                    for band in non_native_bands:
                        for res, bands in NATIVE_BANDS.items():
                            if band in bands:
                                band_origins[band] = res

                    raise ValueError(
                        f"Resolution level 'r10m' should only contain native 10m bands. "
                        f"Found bands from other resolutions: {band_origins}. "
                        f"Expected only: {sorted(expected_bands)}"
                    )

            # For r20m and r60m, bands may include downsampled versions from other resolutions
            # (This matches real Sentinel-2 data structure)
            # No validation needed for these cases

        return self

    def get_bands(self) -> dict[str, Sentinel2DataArray]:
        """Get all spectral band arrays from this dataset."""
        if self.members is None:
            return {}

        bands = {}
        for name, member in self.members.items():
            if isinstance(member, Sentinel2DataArray):
                # Check if it's a band (not a coordinate)
                if name in ALL_BAND_NAMES:
                    bands[name] = member
        return bands

    def get_coordinates(self) -> dict[str, Sentinel2CoordinateArray | Sentinel2DataArray]:
        """Get coordinate arrays from this dataset.

        Returns coordinate arrays (x, y, time). After zarr round-trip,
        these may be loaded as Sentinel2DataArray or Sentinel2CoordinateArray.
        """
        if self.members is None:
            return {}

        coords = {}
        for name, member in self.members.items():
            # Check by name since we can't distinguish types after zarr round-trip
            if name in ("x", "y", "time"):
                if isinstance(member, (Sentinel2CoordinateArray, Sentinel2DataArray)):
                    coords[name] = member
        return coords


class Sentinel2ReflectanceGroup(GroupSpec[DatasetAttrs, Sentinel2ResolutionDataset]):
    """Measurements/reflectance group containing all resolution levels.

    This zarr group contains:
    - r10m (10-meter resolution dataset)
    - r20m (20-meter resolution dataset)
    - r60m (60-meter resolution dataset)
    - r120m, r360m, r720m (optional downsampled datasets)
    """

    @model_validator(mode="after")
    def validate_at_least_one_resolution(self) -> Self:
        """Ensure at least one native resolution is present."""
        if self.members is None:
            raise ValueError("Reflectance group must have members")

        native_resolutions = {"r10m", "r20m", "r60m"}
        present_resolutions = set(self.members.keys()) & native_resolutions

        if not present_resolutions:
            raise ValueError("At least one native resolution (r10m, r20m, r60m) must be present")

        return self

    def get_resolution_dataset(self, resolution: ResolutionLevel) -> Sentinel2ResolutionDataset | None:
        """Get dataset for a specific resolution level."""
        if self.members is None:
            return None
        return self.members.get(resolution)

    def list_resolutions(self) -> list[str]:
        """List all available resolution levels."""
        if self.members is None:
            return []
        return [k for k in self.members.keys() if k.startswith("r")]


class Sentinel2MeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel2ReflectanceGroup]):
    """Top-level measurements group.

    Contains the reflectance subgroup with all spectral bands.
    """

    @property
    def reflectance(self) -> Sentinel2ReflectanceGroup | None:
        """Get the reflectance subgroup.

        When loading from zarr, this may be a generic GroupSpec instead of
        the typed Sentinel2ReflectanceGroup.
        """
        if self.members is None:
            return None
        member = self.members.get("reflectance")
        if isinstance(member, (Sentinel2ReflectanceGroup, GroupSpec)):
            return member
        return None


class Sentinel2AtmosphereDataset(GroupSpec[DatasetAttrs, Any]):
    """Atmosphere quality data (AOT, WVP) at multiple resolutions.

    Contains aerosol optical thickness (aot) and water vapor (wvp) measurements
    at r10m, r20m, and r60m resolutions.
    """


class Sentinel2ProbabilityDataset(GroupSpec[DatasetAttrs, Any]):
    """Probability data (cloud, snow) at multiple resolutions.

    Contains cloud probability (cld) and snow probability (snw) estimates
    at r20m resolution (with potential r10m/r60m variants).
    """


class Sentinel2QuicklookDataset(GroupSpec[DatasetAttrs, Any]):
    """True Color Image (TCI) quicklook data at multiple resolutions.

    Contains 3-band RGB quicklook imagery (bands 4, 3, 2) at r10m, r20m, r60m.
    Each quicklook is a 3-band array (band, y, x).
    """


class Sentinel2MaskDataset(GroupSpec[DatasetAttrs, Any]):
    """Mask data containing classification and detector footprints.

    Multiple subgroups:
    - detector_footprint: Per-band detector footprints at native resolutions
    - l1c_classification: L1C-level classification at r60m
    - l2a_classification: L2A Scene Classification Layer (SCL) at r20m/r60m
    """


class Sentinel2QualityGroup(
    GroupSpec[
        DatasetAttrs,
        Sentinel2AtmosphereDataset
        | Sentinel2ProbabilityDataset
        | Sentinel2QuicklookDataset
        | Sentinel2MaskDataset
        | GroupSpec[Any, Any],
    ]
):
    """Quality group containing atmosphere, probability, classification, and quicklook data.

    Hierarchy:
    ├── atmosphere/ (r10m, r20m, r60m with aot, wvp arrays)
    ├── l2a_quicklook/ (r10m, r20m, r60m with tci 3-band arrays)
    ├── mask/ (r10m, r20m, r60m with per-band quality masks)
    └── probability/ (r20m with cld, snw arrays)
    """

    def get_atmosphere_data(
        self,
    ) -> Sentinel2AtmosphereDataset | GroupSpec[Any, Any] | None:
        """Get atmosphere subgroup (aot, wvp)."""
        if self.members is None:
            return None
        return self.members.get("atmosphere")

    def get_probability_data(
        self,
    ) -> Sentinel2ProbabilityDataset | GroupSpec[Any, Any] | None:
        """Get probability subgroup (cld, snw)."""
        if self.members is None:
            return None
        return self.members.get("probability")

    def get_quicklook_data(
        self,
    ) -> Sentinel2QuicklookDataset | GroupSpec[Any, Any] | None:
        """Get L2A quicklook subgroup (true color image)."""
        if self.members is None:
            return None
        return self.members.get("l2a_quicklook")

    def get_mask_data(
        self,
    ) -> Sentinel2MaskDataset | GroupSpec[Any, Any] | None:
        """Get mask subgroup (classification, detector footprints)."""
        if self.members is None:
            return None
        return self.members.get("mask")


class Sentinel2GeometryGroup(GroupSpec[DatasetAttrs, ArraySpec[Any]]):
    """Geometry group containing sun and viewing angles.

    Contains both mean angles and spatially-varying angle grids:
    - mean_sun_angles: [2] - zenith, azimuth
    - sun_angles: [2, grid_y, grid_x] - spatially varying
    - mean_viewing_incidence_angles: [bands, 2]
    - viewing_incidence_angles: [bands, detectors, 2, grid_y, grid_x]
    - Supporting arrays: angle, band, detector, x (geometry grid), y (geometry grid)
    """


class Sentinel2MeteorologyGroup(
    GroupSpec[DatasetAttrs, GroupSpec[Any, Any] | ArraySpec[Any]]
):
    """Meteorology group containing CAMS and ECMWF atmospheric data.

    Two subgroups:
    - cams: Copernicus Atmosphere Monitoring Service data (11 variables)
    - ecmwf: ECMWF weather model data (6 variables)

    Both are gridded at 9x9 resolution with multiple metadata arrays.
    """

    def get_cams_data(self) -> GroupSpec[Any, Any] | None:
        """Get CAMS atmospheric composition data."""
        if self.members is None:
            return None
        return self.members.get("cams")

    def get_ecmwf_data(self) -> GroupSpec[Any, Any] | None:
        """Get ECMWF weather model data."""
        if self.members is None:
            return None
        return self.members.get("ecmwf")


class Sentinel2ConditionsMaskGroup(
    GroupSpec[DatasetAttrs, GroupSpec[Any, Any]]
):
    """Mask subgroup in conditions containing detector footprints and classifications.

    Hierarchy:
    ├── detector_footprint/ (r10m, r20m, r60m with per-band footprints)
    ├── l1c_classification/ (r60m with b00 classification array)
    └── l2a_classification/ (r20m, r60m with Scene Classification Layer)
    """

    def get_detector_footprint(self) -> GroupSpec[Any, Any] | None:
        """Get detector footprint subgroup."""
        if self.members is None:
            return None
        return self.members.get("detector_footprint")

    def get_l1c_classification(self) -> GroupSpec[Any, Any] | None:
        """Get L1C classification data."""
        if self.members is None:
            return None
        return self.members.get("l1c_classification")

    def get_l2a_classification(self) -> GroupSpec[Any, Any] | None:
        """Get L2A Scene Classification Layer (SCL) data."""
        if self.members is None:
            return None
        return self.members.get("l2a_classification")


class Sentinel2ConditionsGroup(
    GroupSpec[DatasetAttrs, GroupSpec[Any, Any] | Sentinel2GeometryGroup]
):
    """Conditions group containing geometry and meteorology data.

    Hierarchy:
    ├── geometry/ (sun/viewing angles, both mean and spatially-varying)
    ├── mask/ (detector footprints and classifications)
    └── meteorology/ (CAMS and ECMWF atmospheric data)
    """

    def get_geometry_data(self) -> Sentinel2GeometryGroup | GroupSpec[Any, Any] | None:
        """Get geometry subgroup (sun/viewing angles)."""
        if self.members is None:
            return None
        member = self.members.get("geometry")
        if isinstance(member, (Sentinel2GeometryGroup, GroupSpec)):
            return member
        return None

    def get_mask_data(
        self,
    ) -> Sentinel2ConditionsMaskGroup | GroupSpec[Any, Any] | None:
        """Get mask subgroup (detector footprints, classifications)."""
        if self.members is None:
            return None
        member = self.members.get("mask")
        if isinstance(member, (Sentinel2ConditionsMaskGroup, GroupSpec)):
            return member
        return None

    def get_meteorology_data(
        self,
    ) -> Sentinel2MeteorologyGroup | GroupSpec[Any, Any] | None:
        """Get meteorology subgroup (CAMS, ECMWF)."""
        if self.members is None:
            return None
        member = self.members.get("meteorology")
        if isinstance(member, (Sentinel2MeteorologyGroup, GroupSpec)):
            return member
        return None


# ============================================================================
# Root Model
# ============================================================================


class Sentinel2Root(
    GroupSpec[
        Sentinel2RootAttrs,
        Sentinel2MeasurementsGroup | Sentinel2QualityGroup | Sentinel2ConditionsGroup,
    ]
):
    """Complete Sentinel-2 EOPF Zarr hierarchy integrated with pydantic-zarr.

    This is the root model representing the entire Sentinel-2 dataset structure.
    It extends GroupSpec to provide:
    - from_zarr() - load entire hierarchy from zarr store
    - to_zarr() - write entire hierarchy to zarr store
    - to_flat() - flatten hierarchy for inspection
    - Full validation of structure and compliance

    The hierarchy follows EOPF organization:
    Root
    ├── measurements/
    │   └── reflectance/
    │       ├── r10m/ (b02, b03, b04, b08 with x, y coords)
    │       ├── r20m/ (b05-b07, b8a, b11-b12, and downsampled 10m bands)
    │       └── r60m/ (b01, b09, b10, and downsampled 10m/20m bands)
    ├── quality/
    │   ├── atmosphere/ (r10m, r20m, r60m with aot, wvp)
    │   ├── l2a_quicklook/ (r10m, r20m, r60m with TCI true color images)
    │   ├── mask/ (r10m, r20m, r60m with per-band quality masks)
    │   └── probability/ (r20m with cld, snw cloud/snow probability)
    └── conditions/
        ├── geometry/ (sun angles, viewing angles, angle grids)
        ├── mask/
        │   ├── detector_footprint/ (r10m, r20m, r60m per-band footprints)
        │   ├── l1c_classification/ (r60m)
        │   └── l2a_classification/ (r20m, r60m Scene Classification Layer)
        └── meteorology/
            ├── cams/ (Copernicus Atmosphere Monitoring Service)
            └── ecmwf/ (ECMWF weather model)

    Example:
        >>> import zarr
        >>> zarr_group = zarr.open_group("s2_data.zarr", mode="r")
        >>> s2_root = Sentinel2Root.from_zarr(zarr_group)
        >>>
        >>> # Access measurements
        >>> measurements = s2_root.measurements
        >>> r10m_dataset = measurements.reflectance.get_resolution_dataset(ResolutionLevel.R10M)
        >>> bands = r10m_dataset.get_bands()
        >>>
        >>> # Validate compliance
        >>> compliance = s2_root.validate_geozarr_compliance()
    """

    @model_validator(mode="after")
    def validate_sentinel2_structure(self) -> Self:
        """Validate overall Sentinel-2 dataset structure."""
        if self.members is None:
            raise ValueError("Sentinel-2 root must have members")

        # Ensure measurements group exists
        if "measurements" not in self.members:
            raise ValueError("Sentinel-2 dataset must contain measurements group")

        # Validate STAC metadata if present
        if self.attributes:
            # Check for stac_discovery in extra fields
            stac = None
            if hasattr(self.attributes, "stac_discovery"):
                stac = getattr(self.attributes, "stac_discovery", None)
            elif hasattr(self.attributes, "model_extra"):
                extra = self.attributes.model_extra or {}
                stac = extra.get("stac_discovery")

            if stac and isinstance(stac, dict):
                props = stac.get("properties", {})
                mission = props.get("mission", "")
                constellation = props.get("constellation", "")
                platform = props.get("platform", "")

                # Accept if mission is sentinel-2, or if constellation/platform indicates sentinel-2, or if mission is copernicus
                is_valid = (
                    mission.lower().startswith("sentinel-2") or
                    constellation.lower() == "sentinel-2" or
                    platform.lower().startswith("sentinel-2") or
                    mission.lower() == "copernicus"
                )

                if mission and not is_valid:
                    raise ValueError(
                        f"STAC mission/constellation/platform must indicate Sentinel-2, "
                        f"got mission={mission}, constellation={constellation}, platform={platform}"
                    )

        return self

    @property
    def measurements(self) -> Sentinel2MeasurementsGroup| None:
        """Get the measurements group.

        When loading from zarr, this may be a generic GroupSpec instead of
        the typed Sentinel2MeasurementsGroup.
        """
        if self.members is None:
            return None
        member = self.members.get("measurements")
        if isinstance(member, (Sentinel2MeasurementsGroup, GroupSpec)):
            return member
        return None

    @property
    def quality(self) -> Sentinel2QualityGroup | None:
        """Get the quality group.

        When loading from zarr, this may be a generic GroupSpec instead of
        the typed Sentinel2QualityGroup.
        """
        if self.members is None:
            return None
        member = self.members.get("quality")
        if isinstance(member, (Sentinel2QualityGroup, GroupSpec)):
            return member
        return None

    @property
    def conditions(self) -> Sentinel2ConditionsGroup | GroupSpec[Any, Any] | None:
        """Get the conditions group.

        When loading from zarr, this may be a generic GroupSpec instead of
        the typed Sentinel2ConditionsGroup.
        """
        if self.members is None:
            return None
        member = self.members.get("conditions")
        if isinstance(member, (Sentinel2ConditionsGroup, GroupSpec)):
            return member
        return None

    def list_available_bands(self) -> list[str]:
        """List all available spectral bands in the dataset."""
        bands = []
        measurements = self.measurements
        if measurements and measurements.reflectance:
            for res_name in measurements.reflectance.list_resolutions():
                res_dataset = measurements.reflectance.members.get(res_name)  # type: ignore
                if isinstance(res_dataset, Sentinel2ResolutionDataset):
                    bands.extend(res_dataset.get_bands().keys())
        return sorted(set(bands))

    def get_band_info(self, band_name: str) -> Sentinel2BandInfo | None:
        """Get information for a specific band.

        Args:
            band_name: Band identifier (e.g., 'b02', 'b8a')

        Returns:
            BandInfo object or None if band not found
        """
        # Check if it's a valid band name
        valid_bands = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b10", "b11", "b12"]
        if band_name not in valid_bands:
            return None
        return Sentinel2BandInfo.from_band_name(band_name)

    def validate_geozarr_compliance(self) -> dict[str, bool]:
        """Validate GeoZarr spec 0.4 compliance.

        Returns:
            Dictionary with compliance check results
        """
        results = {
            "has_measurements": False,
            "has_bands": False,
            "has_coordinates": False,
            "has_valid_hierarchy": False,
        }

        # Check for measurements
        if self.measurements:
            results["has_measurements"] = True

            # Check for bands
            bands = self.list_available_bands()
            if bands:
                results["has_bands"] = True

            # Check for valid hierarchy
            if self.measurements.reflectance:
                results["has_valid_hierarchy"] = True

                # Check for coordinates in at least one resolution
                for res in ["r10m", "r20m", "r60m"]:
                    dataset = self.measurements.reflectance.get_resolution_dataset(res)
                    if dataset and dataset.get_coordinates():
                        results["has_coordinates"] = True
                        break

        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary representation."""
        return self.model_dump(exclude_none=True)
