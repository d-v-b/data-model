# EOPF GeoZarr Data API

This package provides declarative Pydantic models for working with Earth Observation data in GeoZarr format.

## Modules

### `sentinel2.py`

Complete Pydantic data model for Sentinel-2 datasets following the EOPF hierarchy.

**Key Models:**
- `Sentinel2DataTree` - Root model for entire dataset
- `Sentinel2BandInfo` - Spectral band metadata
- `Sentinel2ReflectanceGroup` - Measurements/reflectance hierarchy
- `Sentinel2QualityGroup` - Quality data (atmosphere, probability, classification)
- `Sentinel2ConditionsGroup` - Geometry and meteorology data
- `Sentinel2Coordinates` - Spatial and temporal coordinates

**Usage:**
```python
from eopf_geozarr.data_api import Sentinel2DataTree, BandName

# Create or load a validated Sentinel-2 dataset
s2_data = Sentinel2DataTree(attributes=..., measurements=...)

# Type-safe access to data
bands = s2_data.measurements.get_all_bands()
band_info = s2_data.get_band_info("b02")
```

### `geozarr/`

GeoZarr specification models (v2 and v3).

**Key Models:**
- `ProjAttrs` - CRS encoding (EPSG codes, WKT2, PROJJSON)
- `Multiscales` - Multiscale metadata
- `TileMatrixSet` - Tile matrix definitions
- `CFStandardName` - Validated CF standard names

## Benefits

1. **Type Safety**: Full type hints and IDE autocomplete
2. **Validation**: Automatic validation of data structure
3. **GeoZarr Compliance**: Built-in spec compliance checking
4. **Documentation**: Self-documenting structure
5. **Maintainability**: Declarative over procedural

## Documentation

See `docs/sentinel2_model_usage.md` for comprehensive usage examples.

## Testing

Run tests with:
```bash
uv run pytest tests/test_sentinel2_model.py -v
```
