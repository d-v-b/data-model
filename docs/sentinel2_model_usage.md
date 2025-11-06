# Sentinel-2 Pydantic Data Model

This document describes the declarative Pydantic data model for Sentinel-2 earth observation data stored in Zarr format.

## Overview

The Sentinel-2 data model provides a type-safe, validated representation of the EOPF (Earth Observation Processing Framework) hierarchy for Sentinel-2 datasets. It ensures GeoZarr spec 0.4 compliance and enables programmatic access to all data components.

## Key Components

### 1. Root Model: `Sentinel2DataTree`

The top-level model representing the entire dataset:

```python
from eopf_geozarr.data_api import Sentinel2DataTree

# Load and validate a Sentinel-2 dataset
s2_data = Sentinel2DataTree.from_datatree(xr.open_datatree("s2.zarr"))

# Access metadata
print(s2_data.attributes.title)
print(s2_data.zarr_format)  # 2 or 3

# List available bands
bands = s2_data.list_available_bands()
print(f"Available bands: {bands}")

# Get band information
band_info = s2_data.get_band_info("b02")
print(f"B02 native resolution: {band_info.native_resolution}m")
print(f"B02 wavelength: {band_info.wavelength_center}nm")

# Validate GeoZarr compliance
compliance = s2_data.validate_geozarr_compliance()
print(f"GeoZarr compliant: {all(compliance.values())}")
```

### 2. Band Information: `Sentinel2BandInfo`

Complete spectral band metadata:

```python
from eopf_geozarr.data_api import Sentinel2BandInfo, BandName

# Get band info from band name
band_info = Sentinel2BandInfo.from_band_name(BandName.B02)

print(f"Band: {band_info.name}")
print(f"Resolution: {band_info.native_resolution}m")
print(f"Type: {band_info.data_type}")
print(f"Wavelength: {band_info.wavelength_center}nm ± {band_info.wavelength_width/2}nm")
print(f"CF name: {band_info.standard_name}")
```

### 3. Resolution Levels: `ResolutionLevel`

Sentinel-2 supports multiple resolution levels:

```python
from eopf_geozarr.data_api import ResolutionLevel

# Native resolutions
ResolutionLevel.R10M   # 10-meter (B02, B03, B04, B08)
ResolutionLevel.R20M   # 20-meter (B05-B07, B8A, B11-B12)
ResolutionLevel.R60M   # 60-meter (B01, B09, B10)

# Downsampled resolutions (multiscale)
ResolutionLevel.R120M  # 120-meter
ResolutionLevel.R360M  # 360-meter
ResolutionLevel.R720M  # 720-meter
```

### 4. Data Organization

#### Measurements Group

Contains all spectral bands organized by resolution:

```python
# Access measurements
measurements = s2_data.measurements

# Access 10m bands
if measurements.r10m:
    print("10m bands:", measurements.r10m.bands.keys())
    b02 = measurements.r10m.bands["b02"]
    print(f"B02 shape: {b02.shape}")
    print(f"B02 dtype: {b02.dtype}")

# Access 20m bands
if measurements.r20m:
    print("20m bands:", measurements.r20m.bands.keys())

# Get all bands across resolutions
all_bands = measurements.get_all_bands()
```

#### Quality Group

Atmosphere, probability, classification, and quicklook data:

```python
quality = s2_data.quality

# Atmosphere data (AOT, WVP) - native 20m
if quality and quality.atmosphere:
    if quality.atmosphere.aot:
        print(f"AOT shape: {quality.atmosphere.aot.shape}")
    if quality.atmosphere.wvp:
        print(f"WVP shape: {quality.atmosphere.wvp.shape}")

# Cloud and snow probability - native 20m
if quality and quality.probability:
    if quality.probability.cld:
        print(f"Cloud probability: {quality.probability.cld.shape}")
    if quality.probability.snw:
        print(f"Snow probability: {quality.probability.snw.shape}")

# Scene classification - native 20m
if quality and quality.classification:
    if quality.classification.scl:
        print(f"SCL shape: {quality.classification.scl.shape}")

# RGB quicklook - 10m
if quality and quality.quicklook:
    print(f"Quicklook RGB: {quality.quicklook.red.shape}")
```

#### Conditions Group

Geometry and meteorology data:

```python
conditions = s2_data.conditions

# Geometry (sun/view angles)
if conditions and conditions.geometry:
    if conditions.geometry.sun_zenith:
        print(f"Solar zenith: {conditions.geometry.sun_zenith.shape}")

# Meteorology
if conditions and conditions.meteorology:
    # CAMS data
    if conditions.meteorology.cams:
        print("CAMS data available")

    # ECMWF data
    if conditions.meteorology.ecmwf:
        print("ECMWF data available")
```

## Band Specifications

### 10-meter Bands
- **B02** (Blue): 490nm ± 33nm
- **B03** (Green): 560nm ± 18nm
- **B04** (Red): 665nm ± 15.5nm
- **B08** (NIR): 842nm ± 53nm

### 20-meter Bands
- **B05** (Red Edge 1): 705nm ± 7.5nm
- **B06** (Red Edge 2): 740nm ± 7.5nm
- **B07** (Red Edge 3): 783nm ± 10nm
- **B8A** (NIR Narrow): 865nm ± 10.5nm
- **B11** (SWIR 1): 1614nm ± 45.5nm
- **B12** (SWIR 2): 2202nm ± 87.5nm

### 60-meter Bands
- **B01** (Coastal Aerosol): 443nm ± 10.5nm
- **B09** (Water Vapor): 945nm ± 10nm
- **B10** (Cirrus): 1375nm ± 15nm

## Quality Data (Native 20m)

- **SCL**: Scene Classification Layer
- **AOT**: Aerosol Optical Thickness
- **WVP**: Water Vapor content
- **CLD**: Cloud probability (0-100)
- **SNW**: Snow probability (0-100)

## Validation

The model includes comprehensive validation:

```python
from pydantic import ValidationError

try:
    # Create a data tree
    s2_data = Sentinel2DataTree(
        attributes=attrs,
        measurements=measurements,
        quality=quality,
        conditions=conditions,
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Validation Rules

1. **Band Resolution Consistency**: Bands must be at their native resolution
   - 10m bands: B02, B03, B04, B08
   - 20m bands: B05-B07, B8A, B11-B12
   - 60m bands: B01, B09, B10

2. **Coordinate Consistency**: Coordinate spacing must match resolution

3. **Shape Consistency**: All arrays must have valid 3D shapes (time, y, x)

4. **Chunk Alignment**: Chunk sizes cannot exceed dimension sizes

5. **STAC Metadata**: Mission must be Sentinel-2

6. **GeoZarr Compliance**:
   - Must have CRS information
   - Must have `_ARRAY_DIMENSIONS` attributes (specified via `array_dimensions` field)
   - Must have CF standard names
   - Must have grid_mapping references

## Resampling Types

Different variable types use different resampling strategies:

```python
from eopf_geozarr.data_api import VariableType

# Reflectance bands use block averaging
VariableType.REFLECTANCE

# Classification uses nearest neighbor
VariableType.CLASSIFICATION

# Quality masks use logical OR
VariableType.QUALITY_MASK

# Probabilities use averaged clamping
VariableType.PROBABILITY
```

## Example: Complete Data Tree Construction

```python
from eopf_geozarr.data_api import (
    Sentinel2DataTree,
    Sentinel2RootAttributes,
    Sentinel2ReflectanceGroup,
    Sentinel2ResolutionDataset,
    Sentinel2Coordinates,
    Sentinel2DataArray,
    DataArrayAttributes,
    CoordinateArray,
    STACDiscoveryProperties,
    ResolutionLevel,
    BandName,
)
from eopf_geozarr.data_api.geozarr.common import ProjAttrs
import numpy as np

# 1. Create CRS
crs = ProjAttrs(
    code="EPSG:32632",
    bbox=(600000.0, 5090000.0, 605490.0, 5095490.0),
    transform=(10.0, 0.0, 600000.0, 0.0, -10.0, 5095490.0),
    spatial_dimensions=("x", "y"),
)

# 2. Create coordinates
coords = Sentinel2Coordinates(
    x=CoordinateArray(
        name="x",
        values=list(np.linspace(600000, 605490, 549)),
        units="m",
        standard_name="projection_x_coordinate",
        long_name="x coordinate of projection",
        axis="X",
    ),
    y=CoordinateArray(
        name="y",
        values=list(np.linspace(5095490, 5090000, 549)),
        units="m",
        standard_name="projection_y_coordinate",
        long_name="y coordinate of projection",
        axis="Y",
    ),
    time=CoordinateArray(
        name="time",
        values=[np.datetime64("2025-01-13T10:33:09")],
        units="seconds since 1970-01-01",
        standard_name="time",
        long_name="time",
        axis="T",
    ),
    crs=crs,
    resolution_meters=10,
)

# 3. Create data arrays
b02 = Sentinel2DataArray(
    name="b02",
    shape=(1, 549, 549),
    dtype="uint16",
    chunks=(1, 256, 256),
    attributes=DataArrayAttributes(
        long_name="Blue band (B02)",
        standard_name="toa_bidirectional_reflectance",
        units="1",
        grid_mapping="crs",
        array_dimensions=["time", "y", "x"],
    ),
)

# 4. Create resolution dataset
r10m = Sentinel2ResolutionDataset(
    resolution=ResolutionLevel.R10M,
    coordinates=coords,
    bands={"b02": b02},
)

# 5. Create measurements group
measurements = Sentinel2ReflectanceGroup(r10m=r10m)

# 6. Create root attributes
stac = STACDiscoveryProperties(
    mission="sentinel-2a",
    platform="sentinel-2a",
    instruments=["msi"],
    datetime="2025-01-13T10:33:09",
    processing_level="L1C",
)

attrs = Sentinel2RootAttributes(
    stac_discovery={"properties": stac.model_dump()},
    Conventions="CF-1.7",
    title="Sentinel-2A MSI L1C",
    institution="ESA",
    source="Copernicus Sentinel-2A satellite",
)

# 7. Create complete data tree
s2_data = Sentinel2DataTree(
    attributes=attrs,
    measurements=measurements,
    zarr_format=3,
)

# 8. Validate
print(f"Valid: {s2_data}")
print(f"Bands: {s2_data.list_available_bands()}")
print(f"Compliance: {s2_data.validate_geozarr_compliance()}")
```

## Benefits of the Declarative Model

1. **Type Safety**: All fields are typed and validated
2. **Automatic Validation**: Pydantic ensures data integrity
3. **IDE Support**: Full autocomplete and type hints
4. **Documentation**: Self-documenting structure
5. **Serialization**: Easy conversion to/from JSON/dict
6. **Maintainability**: Declarative structure is easier to understand and modify
7. **GeoZarr Compliance**: Built-in validation ensures spec compliance

## Integration with Existing Code

The model can be integrated with the existing procedural conversion code:

```python
# Before: Procedural approach
def process_sentinel2(dt_input):
    # Manual data extraction
    for group in dt_input.groups:
        if "reflectance" in group:
            # Process bands...
            pass

# After: Declarative approach
def process_sentinel2(dt_input):
    # Validate and structure data
    s2_model = Sentinel2DataTree.from_datatree(dt_input)

    # Type-safe access
    for band_name, band in s2_model.measurements.get_all_bands().items():
        print(f"Processing {band_name}: {band.shape}")
        # Guaranteed to have proper structure
```

## Future Enhancements

1. **Complete `from_datatree()` implementation**: Parse xarray DataTree to Pydantic model
2. **`to_datatree()` method**: Convert Pydantic model back to xarray DataTree
3. **Zarr store integration**: Direct reading/writing from Zarr stores
4. **Validation helpers**: Additional compliance checking utilities
5. **Metadata enrichment**: Automatic attribute generation
6. **Schema export**: Generate JSON Schema for API documentation
