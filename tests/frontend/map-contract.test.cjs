const test = require('node:test');
const assert = require('node:assert/strict');

function sdk() {
  delete global.gm_authFailure;
  class Emitter {
    constructor() { this.handlers = new Map(); }
    addListener(name, fn) { (this.handlers.get(name) || this.handlers.set(name, []).get(name)).push(fn); return { remove() {} }; }
    fire(name, value) { (this.handlers.get(name) || []).forEach((fn) => fn(value)); }
  }
  class GMap extends Emitter { constructor(host, opts) { super(); this.host = host; this.center = opts.center; this.zoom = opts.zoom; this.overlayMapTypes = { items: [], push: (x) => this.overlayMapTypes.items.push(x), getArray: () => this.overlayMapTypes.items, removeAt: (i) => this.overlayMapTypes.items.splice(i, 1) }; } setCenter(v) { this.center = v; } setZoom(v) { this.zoom = v; } getCenter() { return { lat: () => this.center.lat, lng: () => this.center.lng }; } getZoom() { return this.zoom; } fitBounds(v) { this.fitted = v; } setMapTypeId(v) { this.type = v; } }
  class Data extends Emitter { addGeoJson(v) { this.geojson = v; } setStyle(v) { this.style = v; } setMap(v) { this.map = v; } }
  global.google = { maps: { Map: GMap, Data, InfoWindow: class { setContent() {} setPosition() {} open() {} close() {} }, Size: class { constructor(w, h) { this.width = w; this.height = h; } }, SymbolPath: { CIRCLE: 'circle' }, event: { addListenerOnce(_m, _n, fn) { fn(); return { remove() {} }; }, trigger() {}, clearInstanceListeners() {} } } };
  const host = { replaceChildren() {}, appendChild() {}, style: {} };
  global.document = { getElementById: () => host, createElement: () => ({ style: {}, setAttribute() {}, remove() {}, appendChild() {} }), head: { appendChild() {} } };
  return { host, Data };
}

test('native facade accepts raw Polygon, MultiPolygon and maps Leaflet styles', async () => {
  const { Data } = sdk();
  const provider = require('../../web/map-provider.js');
  const config = { maps: { provider: 'google', configured: true, api_key: 'x' }, imagery: { provider: 'earth_engine', configured: true } };
  const context = await provider.create('map', { center: [55, 37], zoom: 6, minZoom: 3, maxZoom: 18 }, config);
  const polygon = context.L.geoJSON({ type: 'Polygon', coordinates: [[[37, 55], [38, 55], [38, 56], [37, 55]]] }, { style: { color: '#123456', weight: 5, interactive: false } }).addTo(context.map);
  assert.equal(polygon.getBounds().getEast(), 38);
  assert.ok(polygon.native instanceof Data);
  const style = polygon.native.style({ getId: () => 0 });
  assert.equal(style.strokeColor, '#123456');
  assert.equal(style.strokeWeight, 5);
  assert.equal(style.clickable, false);
  const multi = context.L.geoJSON({ type: 'MultiPolygon', coordinates: [[[[37, 55], [38, 55], [38, 56], [37, 55]]], [[[40, 57], [41, 57], [41, 58], [40, 57]]]] }).addTo(context.map);
  assert.equal(multi.getBounds().getWest(), 37);
  assert.equal(multi.getBounds().getNorth(), 58);
});

test('pointToLayer receives feature and latlng; bounds fit native map', async () => {
  sdk();
  const provider = require('../../web/map-provider.js');
  const context = await provider.create('map', { center: [55, 37], zoom: 6 }, { maps: { provider: 'google', configured: true, api_key: 'x' }, imagery: { provider: 'earth_engine', configured: true } });
  let args;
  const af = context.L.geoJSON({ type: 'FeatureCollection', features: [{ type: 'Feature', properties: { kind: 'af' }, geometry: { type: 'Point', coordinates: [37.5, 55.5] } }] }, { pointToLayer(feature, point) { args = [feature, point]; return context.L.circleMarker(point, { radius: 4, color: '#ff783e' }); } }).addTo(context.map);
  assert.equal(args[0].properties.kind, 'af');
  assert.deepEqual([args[1].lat, args[1].lng], [55.5, 37.5]);
  context.map.fitBounds(af.getBounds().pad(.1));
  assert.deepEqual(context.map.native.fitted, { west: 37.5, south: 55.5, east: 37.5, north: 55.5 });
});

test('OSM fallback exposes full imagery and fit contract', async () => {
  const { host } = sdk();
  let added = 0;
  const native = { setView() { return this; }, on() { return this; }, createPane() { return { style: {} }; }, fitBounds(value) { this.fitted = value; }, invalidateSize() {} };
  global.L = { map: () => native, tileLayer: () => ({ addTo() { added++; return this; }, on() { return this; }, remove() { added--; } }), control: { scale: () => ({ addTo() {} }) } };
  const provider = require('../../web/map-provider.js');
  const context = await provider.create(host, { center: [55, 37], zoom: 6 }, { maps: { provider: 'osm', configured: false }, imagery: { provider: 'earth_engine', configured: true } });
  context.setImageryTileLayer('https://earthengine.googleapis.com/v1/projects/p/maps/m/tiles/{z}/{x}/{y}', 'Earth Engine');
  assert.equal(added, 2);
  context.fitBounds([37, 55, 38, 56]);
  assert.deepEqual(native.fitted, [[55, 37], [56, 38]]);
  context.clearImageryTileLayer();
  assert.equal(added, 1);
});

test('late Google auth failure migrates live layers and imagery to the same OSM facade', async () => {
  const { host } = sdk();
  const rendered = [];
  const leafMap = { setView() { return this; }, on() { return this; }, createPane() { return { style: {} }; }, fitBounds() {}, invalidateSize() {} };
  global.L = {
    map: () => leafMap,
    tileLayer: () => ({ addTo() { return this; }, on() { return this; }, remove() {} }),
    control: { scale: () => ({ addTo() {} }) },
    circleMarker: (_point, style) => ({ entries: [{ style }] }),
    geoJSON(data, options) {
      rendered.push({ data, options });
      return { addTo() { return this; }, remove() {} };
    },
  };
  const provider = require('../../web/map-provider.js');
  const context = await provider.create(host, { center: [55, 37], zoom: 6 }, { maps: { provider: 'google', configured: true, api_key: 'x' }, imagery: { provider: 'earth_engine', configured: true } });
  const aoi = context.L.geoJSON({ type: 'Polygon', coordinates: [[[37, 55], [38, 55], [38, 56], [37, 55]]] }, { style: { color: '#ff965c', interactive: false } }).addTo(context.map);
  const af = context.L.geoJSON({ type: 'Point', coordinates: [37.2, 55.2] }, { pointToLayer: (_feature, point) => context.L.circleMarker(point, { radius: 4, color: '#ff783e' }) }).addTo(context.map).bindPopup('<b>AF</b>');
  const bs = context.L.geoJSON({ type: 'MultiPolygon', coordinates: [[[[37, 55], [37.1, 55], [37.1, 55.1], [37, 55]]]] }, { style: { color: '#ee6455', fillOpacity: .48 } }).addTo(context.map);
  const removed = context.L.geoJSON({ type: 'Point', coordinates: [40, 60] }).addTo(context.map);
  removed.remove();
  context.setImageryTileLayer('https://earthengine.googleapis.com/v1/projects/p/maps/m/tiles/{z}/{x}/{y}', 'EE test');
  const before = { aoi: aoi.getBounds().literal(), popup: af.entries[0].popup, bsStyle: { ...bs.entries[0].style }, imagery: { ...context.imagerySpec } };
  global.gm_authFailure();
  assert.equal(context.provider, 'osm');
  assert.equal(context.map.context, context);
  assert.equal(context.map.layers.has(removed), false);
  assert.equal(context.map.layers.has(aoi) && context.map.layers.has(af) && context.map.layers.has(bs), true);
  assert.deepEqual(aoi.getBounds().literal(), before.aoi);
  assert.equal(af.entries[0].popup, before.popup);
  assert.deepEqual(bs.entries[0].style, before.bsStyle);
  assert.deepEqual(context.imagerySpec, before.imagery);
  assert.equal(rendered.length, 3);
});

test('AF points stay above BS polygons in both Google and Leaflet renderers', async () => {
  const { host } = sdk();
  const provider = require('../../web/map-provider.js');
  assert.equal(provider.googleStyle({ color: '#ff783e' }, true).zIndex, 5);
  assert.equal(provider.googleStyle({ color: '#ee6455' }, false).zIndex, 3);
  const front = [];
  const native = { setView() { return this; }, on() { return this; }, createPane() { return { style: {} }; }, fitBounds() {}, invalidateSize() {} };
  global.L = {
    map: () => native,
    tileLayer: () => ({ addTo() { return this; }, on() { return this; } }),
    control: { scale: () => ({ addTo() {} }) },
    circleMarker: (_point, style) => ({ entries: [{ style }] }),
    geoJSON(data) {
      const kind = data.features[0].geometry.type;
      return { addTo() { return this; }, remove() {}, bringToFront() { front.push(kind); } };
    },
  };
  const context = await provider.create(host, { center: [55, 37], zoom: 6 }, { maps: { provider: 'osm', configured: false }, imagery: { configured: false } });
  const af = context.L.geoJSON({ type: 'Point', coordinates: [37, 55] }, { pointToLayer: (_f, p) => context.L.circleMarker(p, { radius: 4 }) }).addTo(context.map);
  context.L.geoJSON({ type: 'Polygon', coordinates: [[[37, 55], [38, 55], [38, 56], [37, 55]]] }).addTo(context.map);
  assert.equal(front.filter((kind) => kind === 'Point').length >= 2, true);
  assert.equal(front.includes('Polygon'), false);
  assert.equal(context.map.layers.has(af), true);
});
