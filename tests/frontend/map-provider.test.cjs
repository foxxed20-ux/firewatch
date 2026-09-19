const test = require('node:test');
const assert = require('node:assert/strict');
const provider = require('../../web/map-provider.js');

test('Earth Engine tile URLs are restricted to approved HTTPS hosts', () => {
  assert.equal(provider.allowedImageryURL('https://earthengine.googleapis.com/v1/projects/p/maps/abc/tiles/{z}/{x}/{y}'), true);
  assert.equal(provider.allowedImageryURL('https://earthengine-highvolume.googleapis.com/v1/projects/p/maps/abc/tiles/{z}/{x}/{y}'), true);
  assert.equal(provider.allowedImageryURL('http://earthengine.googleapis.com/tile'), false);
  assert.equal(provider.allowedImageryURL('https://example.test/tile'), false);
});

test('fallback client config never contains a browser secret', () => {
  const config = provider.fallbackConfig();
  assert.deepEqual(config.maps, { provider: 'osm', api_key: null, configured: false });
});

test('Google load failure preserves Earth Engine configuration for the fallback', () => {
  const config = { maps: { provider: 'google', api_key: 'restricted-key', configured: true }, imagery: { provider: 'earth_engine', configured: true, endpoint: 'https://earthengine.googleapis.com/' } };
  const fallback = provider.googleFailureConfig(config, new Error('referrer rejected'));
  assert.equal(fallback.maps.provider, 'osm');
  assert.equal(fallback.maps.api_key, null);
  assert.equal(fallback.imagery.configured, true);
});

test('geometry and feature styles are retained without mutating input', () => {
  const geometry = { type: 'Polygon', coordinates: [[[37, 55], [37.1, 55], [37, 55.1], [37, 55]]] };
  const snapshot = JSON.stringify(geometry);
  const layer = provider.geoJSON(geometry, { style: { color: '#ff0000', interactive: false } });
  assert.deepEqual(layer.entries[0].feature.geometry, geometry);
  assert.equal(layer.getBounds().getWest(), 37);
  assert.equal(layer.getBounds().getNorth(), 55.1);
  assert.equal(JSON.stringify(geometry), snapshot);
});
