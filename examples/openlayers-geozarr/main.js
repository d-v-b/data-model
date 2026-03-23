import Map from 'ol/Map.js';
import {getView, withExtentCenter, withHigherResolutions} from 'ol/View.js';
import TileLayer from 'ol/layer/WebGLTile.js';
import GeoZarr from 'ol/source/GeoZarr.js';
import OSM from 'ol/source/OSM.js';

const params = new URLSearchParams(window.location.search);
const zarrUrl = params.get('data');

if (!zarrUrl) {
  document.getElementById('map').innerHTML =
    '<p style="padding:2em;color:#666">No data URL provided. Add <code>?data=...</code> to the URL, pointing to a GeoZarr multiscales group.</p>';
  throw new Error('No ?data= parameter provided');
}

const channels = ['red', 'green', 'blue'];
for (const channel of channels) {
  const selector = document.getElementById(channel);
  selector.addEventListener('change', update);

  const input = document.getElementById(`${channel}Max`);
  input.addEventListener('input', update);
}

function getVariables() {
  const variables = {};
  for (const channel of channels) {
    const selector = document.getElementById(channel);
    variables[channel] = parseFloat(selector.value);

    const inputId = `${channel}Max`;
    const input = document.getElementById(inputId);
    variables[inputId] = parseFloat(input.value);
  }
  return variables;
}

const source = new GeoZarr({
  url: zarrUrl,
  bands: ['b04', 'b03', 'b02', 'b11'],
});

const layer = new TileLayer({
  style: {
    variables: getVariables(),
    gamma: 1.5,
    color: [
      'array',
      ['/', ['band', ['var', 'red']], ['var', 'redMax']],
      ['/', ['band', ['var', 'green']], ['var', 'greenMax']],
      ['/', ['band', ['var', 'blue']], ['var', 'blueMax']],
      1,
    ],
  },
  source,
});

function update() {
  layer.updateStyleVariables(getVariables());
}

const map = new Map({
  target: 'map',
  layers: [new TileLayer({source: new OSM()}), layer],
  view: getView(source, withHigherResolutions(2), withExtentCenter()),
});
