"""
Integration tests verifying that S2 converter output can be loaded by OpenLayers' GeoZarr source.

Runs the OL GeoZarr source configuration in Node.js (headless, no browser) against
a local HTTP server serving the converted zarr store. This exercises the same code path
that the browser would: fetch zarr.json, parse zarr conventions, build the WMTS tile grid.

Requires:
    - /tmp/s2_source.zarr to exist (raw S2 product) — or a pre-converted store
    - Node.js >= 18 installed
    - npm dependencies installed in examples/openlayers-geozarr/
"""

from __future__ import annotations

import http.server
import json
import pathlib
import shutil
import socket
import subprocess
import tempfile
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
import xarray as xr

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VIEWER_DIR = REPO_ROOT / "examples" / "openlayers-geozarr"
TEST_SCRIPT = VIEWER_DIR / "test_geozarr_ol.mjs"

s2_source_path = pathlib.Path("/tmp/s2_source.zarr")
s2_converted_path = pathlib.Path("/tmp/s2_converted_test.zarr")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not s2_source_path.exists() and not s2_converted_path.exists(),
        reason="No S2 data available (need /tmp/s2_source.zarr or /tmp/s2_converted_test.zarr)",
    ),
    pytest.mark.skipif(
        shutil.which("node") is None,
        reason="Node.js not installed",
    ),
    pytest.mark.skipif(
        not TEST_SCRIPT.exists(),
        reason=f"OL test script not found at {TEST_SCRIPT}",
    ),
]


def _npm_install_if_needed() -> None:
    """Install npm dependencies if node_modules is missing."""
    node_modules = VIEWER_DIR / "node_modules"
    if not node_modules.exists():
        subprocess.run(
            ["npm", "install"],
            cwd=str(VIEWER_DIR),
            check=True,
            capture_output=True,
            timeout=120,
        )


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress request logs during tests


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def converted_s2_path() -> pathlib.Path:
    """Return path to converted S2 data.

    Uses pre-converted data at /tmp/s2_converted_test.zarr if available,
    otherwise runs the converter on /tmp/s2_source.zarr.
    """
    if s2_converted_path.exists():
        return s2_converted_path

    if not s2_source_path.exists():
        pytest.skip("No S2 data available")

    from eopf_geozarr.s2_optimization.s2_converter import convert_s2

    tmpdir = tempfile.mkdtemp(prefix="ol_test_")
    output_path = pathlib.Path(tmpdir) / "s2_optimized.zarr"

    dt_input = xr.open_datatree(
        str(s2_source_path),
        engine="zarr",
        chunks="auto",
    )

    convert_s2(
        dt_input,
        output_path=str(output_path),
        validate_output=False,
        enable_sharding=False,
        spatial_chunk=512,
    )

    return output_path


@pytest.fixture(scope="module")
def zarr_server(converted_s2_path: pathlib.Path) -> Generator[tuple[str, http.server.HTTPServer]]:
    """Start a CORS-enabled HTTP server serving the converted zarr store."""
    _npm_install_if_needed()

    port = _find_free_port()
    handler = lambda *args, **kwargs: _CORSHandler(  # noqa: E731
        *args, directory=str(converted_s2_path), **kwargs
    )
    server = http.server.HTTPServer(("", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}", server
    server.shutdown()


def _run_ol_test(
    server_url: str, group: str = "measurements/reflectance", bands: str = "b04,b03,b02"
) -> dict[str, Any]:
    """Run the Node.js OL test script and return parsed JSON output."""
    result = subprocess.run(
        ["node", str(TEST_SCRIPT), f"{server_url}/{group}", bands],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(VIEWER_DIR),
    )
    if result.returncode != 0:
        pytest.fail(
            f"OL GeoZarr test failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return json.loads(result.stdout)


class TestOpenLayersGeoZarr:
    """Test that OpenLayers can configure a GeoZarr source from our converted data."""

    def test_source_reaches_ready_state(
        self, zarr_server: tuple[str, http.server.HTTPServer]
    ) -> None:
        """The GeoZarr source must reach 'ready' state without errors."""
        url, _ = zarr_server
        checks = _run_ol_test(url)
        assert checks["state"] == "ready"

    def test_resolutions_descending(self, zarr_server: tuple[str, http.server.HTTPServer]) -> None:
        """Resolutions must be sorted in descending order (OL requirement)."""
        url, _ = zarr_server
        checks = _run_ol_test(url)
        resolutions = checks["resolutions"]
        assert resolutions == sorted(resolutions, reverse=True), (
            f"Resolutions not descending: {resolutions}"
        )

    def test_all_resolutions_unique(self, zarr_server: tuple[str, http.server.HTTPServer]) -> None:
        """Each multiscale level must have a distinct resolution."""
        url, _ = zarr_server
        checks = _run_ol_test(url)
        resolutions = checks["resolutions"]
        assert len(resolutions) == len(set(resolutions)), f"Duplicate resolutions: {resolutions}"

    def test_expected_resolution_levels(
        self, zarr_server: tuple[str, http.server.HTTPServer]
    ) -> None:
        """Should have 6 levels: 10, 20, 60, 120, 360, 720."""
        url, _ = zarr_server
        checks = _run_ol_test(url)
        assert checks["numLevels"] == 6
        assert set(checks["resolutions"]) == {10, 20, 60, 120, 360, 720}

    def test_valid_extent(self, zarr_server: tuple[str, http.server.HTTPServer]) -> None:
        """Extent must be a valid bounding box."""
        url, _ = zarr_server
        checks = _run_ol_test(url)
        assert checks["extentValid"]
        extent = checks["extent"]
        assert extent[2] - extent[0] > 0, "Zero-width extent"
        assert extent[3] - extent[1] > 0, "Zero-height extent"

    def test_projection_detected(self, zarr_server: tuple[str, http.server.HTTPServer]) -> None:
        """OL must detect the projection from zarr conventions."""
        url, _ = zarr_server
        checks = _run_ol_test(url)
        assert checks["projection"].startswith("EPSG:")
