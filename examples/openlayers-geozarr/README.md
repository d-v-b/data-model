# GeoZarr Viewer

A minimal OpenLayers-based viewer for browsing GeoZarr multiscale imagery with band selection and contrast stretch controls.

Requires the [OpenLayers GeoZarr source](https://github.com/openlayers/openlayers/pull/17355), available in `ol@dev` builds (not yet in a stable release).

## Quick start

```bash
# Install dependencies (once)
npm install

# View a local zarr store
./serve.sh /path/to/store.zarr

# View a remote zarr store
./serve.sh https://example.com/store.zarr
```

The script starts a [vite](https://vite.dev/) dev server and opens the viewer in your browser. For local paths, it also starts a CORS-enabled file server on port 8000.

## Usage

```
./serve.sh <zarr-path-or-url> [multiscales-group]
```

- **zarr-path-or-url**: Local directory or remote URL of a zarr store.
- **multiscales-group**: Group path within the store containing multiscale metadata. Defaults to `measurements/reflectance`.

### Examples

```bash
# Local S2 converted output (default group)
./serve.sh /tmp/s2_converted.zarr

# Explicit group path
./serve.sh /tmp/s2_converted.zarr measurements/reflectance

# Remote store
./serve.sh https://s3.example.com/sentinel-2/product.zarr measurements/reflectance
```

### Manual usage

If you prefer to manage the servers yourself:

1. Serve the zarr store over HTTP with CORS headers (for local data):
   ```bash
   cd /path/to/store.zarr
   python3 -m http.server 8000
   ```

2. Start the vite dev server:
   ```bash
   npx vite
   ```

3. Open in browser with the `?data=` query parameter:
   ```
   http://localhost:5173/?data=http://localhost:8000/measurements/reflectance
   ```

## Controls

- **Band selectors**: Map visible red, green, blue, or short-wave infrared bands to each RGB display channel.
- **Max sliders**: Adjust contrast stretch per channel (range 0.2–0.5).

## Requirements

The zarr store must have:
- Zarr V3 format
- [Multiscales convention](https://github.com/zarr-conventions/multiscales) metadata (`zarr_conventions` with multiscales UUID)
- [Spatial convention](https://github.com/zarr-conventions/spatial) metadata (`spatial:bbox`, `spatial:transform`)
- [Proj convention](https://github.com/zarr-experimental/geo-proj) metadata (`proj:code`)
- Consolidated metadata in the multiscales group `zarr.json`
