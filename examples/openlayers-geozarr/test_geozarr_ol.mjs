/**
 * Headless test that verifies a GeoZarr store can be loaded by OpenLayers.
 *
 * Exercises the same code path as the browser: constructs an ol/source/GeoZarr,
 * waits for it to configure (fetch zarr.json, parse conventions, build tile grid),
 * and asserts it reaches the 'ready' state.
 *
 * Usage:
 *   node test_geozarr_ol.mjs <multiscales-group-url>
 *
 * Example:
 *   node test_geozarr_ol.mjs http://localhost:8000/measurements/reflectance
 *
 * Exit codes:
 *   0 — success
 *   1 — GeoZarr source failed to configure
 */

// OL expects a global `document` for some modules — provide a minimal stub.
import {JSDOM} from 'jsdom';
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="map"></div></body></html>');
globalThis.document = dom.window.document;
globalThis.window = dom.window;
globalThis.HTMLCanvasElement = dom.window.HTMLCanvasElement;
globalThis.HTMLImageElement = dom.window.HTMLImageElement;
globalThis.Image = dom.window.Image;
Object.defineProperty(globalThis, 'navigator', {
  value: dom.window.navigator,
  writable: true,
  configurable: true,
});
globalThis.URL = dom.window.URL;

import GeoZarr from 'ol/source/GeoZarr.js';

const url = process.argv[2];
if (!url) {
  console.error('Usage: node test_geozarr_ol.mjs <multiscales-group-url>');
  process.exit(1);
}

const bands = process.argv[3]
  ? process.argv[3].split(',')
  : ['b04', 'b03', 'b02'];

const TIMEOUT_MS = 30_000;

const source = new GeoZarr({url, bands});

const result = await Promise.race([
  new Promise((resolve, reject) => {
    source.on('change', () => {
      const state = source.getState();
      if (state === 'ready') {
        resolve({state, tileGrid: source.tileGrid, projection: source.projection});
      } else if (state === 'error') {
        reject(source.error_ || new Error('GeoZarr source entered error state'));
      }
    });
  }),
  new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Timed out after ${TIMEOUT_MS}ms`)), TIMEOUT_MS),
  ),
]);

// Validate the tile grid
const resolutions = result.tileGrid.getResolutions();
const extent = result.tileGrid.getExtent();

const checks = {
  state: result.state,
  projection: result.projection?.getCode?.() || String(result.projection),
  resolutions,
  extent,
  numLevels: resolutions.length,
  resolutionsDescending:
    resolutions.every((r, i) => i === 0 || r <= resolutions[i - 1]),
  extentValid:
    extent[0] < extent[2] && extent[1] < extent[3],
};

console.log(JSON.stringify(checks, null, 2));

if (!checks.resolutionsDescending) {
  console.error('FAIL: resolutions are not in descending order');
  process.exit(1);
}
if (!checks.extentValid) {
  console.error('FAIL: extent is invalid');
  process.exit(1);
}
console.error('PASS: GeoZarr source configured successfully');
process.exit(0);
