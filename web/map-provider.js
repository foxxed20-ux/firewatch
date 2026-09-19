/* One active map renderer. Application layers survive a Google auth failure. */
(function (root, factory) {
  "use strict";
  const api = factory(root);
  root.FireWatchMapProvider = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function (root) {
  "use strict";
  const OSM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
  const OSM_CREDIT = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>';
  const fallbackConfig = () => ({ maps: { provider: "osm", api_key: null, configured: false }, imagery: { provider: "earth_engine", configured: false } });
  const googleFailureConfig = (config, error) => ({ ...config, maps: { ...config.maps, provider: "osm", api_key: null, configured: false, error: error.message || String(error) } });
  function allowedImageryURL(value) {
    if (typeof value !== "string") return false;
    try {
      const url = new URL(value);
      return url.protocol === "https:" && !url.username && !url.password && !url.port &&
        ["earthengine.googleapis.com", "earthengine-highvolume.googleapis.com"].includes(url.hostname) &&
        /^\/v1(?:alpha)?\/projects\/[^/]+\/maps\/[^/]+\/tiles\//.test(url.pathname) &&
        ["{x}", "{y}", "{z}"].every((part) => value.includes(part)) &&
        !url.searchParams.has("access_token");
    } catch { return false; }
  }
  function event(name, detail) {
    if (root.dispatchEvent && root.CustomEvent) root.dispatchEvent(new root.CustomEvent(name, { detail }));
  }
  function latLng(value, lng) {
    const lat = Number(Array.isArray(value) ? value[0] : typeof value === "object" ? typeof value.lat === "function" ? value.lat() : value.lat : value);
    const lon = Number(Array.isArray(value) ? value[1] : typeof value === "object" ? typeof value.lng === "function" ? value.lng() : value.lng : lng);
    return { lat, lng: lon, wrap() { return latLng(lat, ((lon + 180) % 360 + 360) % 360 - 180); } };
  }
  class Bounds {
    constructor(points = []) {
      this.west = Infinity; this.south = Infinity; this.east = -Infinity; this.north = -Infinity;
      points.forEach((p) => this.extend(p));
    }
    extend(value) {
      const p = latLng(value);
      if (Number.isFinite(p.lat) && Number.isFinite(p.lng)) {
        this.west = Math.min(this.west, p.lng); this.east = Math.max(this.east, p.lng);
        this.south = Math.min(this.south, p.lat); this.north = Math.max(this.north, p.lat);
      }
      return this;
    }
    isValid() { return [this.west, this.south, this.east, this.north].every(Number.isFinite); }
    getWest() { return this.west; } getEast() { return this.east; }
    getSouth() { return this.south; } getNorth() { return this.north; }
    pad(ratio) {
      if (!this.isValid()) return new Bounds();
      const dx = (this.east - this.west) * ratio, dy = (this.north - this.south) * ratio;
      return new Bounds([[Math.max(-85, this.south - dy), this.west - dx], [Math.min(85, this.north + dy), this.east + dx]]);
    }
    literal() { return { west: this.west, south: this.south, east: this.east, north: this.north }; }
  }
  function boundsOf(geometry, result = new Bounds()) {
    if (!geometry) return result;
    if (geometry.type === "GeometryCollection") geometry.geometries.forEach((g) => boundsOf(g, result));
    else {
      const visit = (coordinates) => {
        if (!Array.isArray(coordinates)) return;
        if (typeof coordinates[0] === "number") result.extend([coordinates[1], coordinates[0]]);
        else coordinates.forEach(visit);
      };
      visit(geometry.coordinates);
    }
    return result;
  }
  function featuresOf(data) {
    if (!data) return [];
    if (data.type === "FeatureCollection") return data.features.flatMap(featuresOf);
    if (data.type === "Feature") return data.geometry ? [data] : [];
    return [{ type: "Feature", properties: {}, geometry: data }];
  }
  function googleStyle(style = {}, point = false) {
    const result = {
      strokeColor: style.color || "#ff965c", strokeWeight: style.weight ?? 2,
      strokeOpacity: style.opacity ?? 1, fillColor: style.fillColor || style.color || "#ff965c",
      fillOpacity: style.fillOpacity ?? 0.2, clickable: style.interactive !== false,
      zIndex: style.interactive === false ? 2 : point ? 5 : 3,
    };
    if (point) result.icon = {
      path: root.google?.maps?.SymbolPath?.CIRCLE || 0, scale: style.radius ?? 4,
      fillColor: result.fillColor, fillOpacity: result.fillOpacity,
      strokeColor: result.strokeColor, strokeWeight: result.strokeWeight, strokeOpacity: result.strokeOpacity,
    };
    return result;
  }
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  class Layer {
    constructor(entries = []) { this.entries = entries; this.owner = null; this.native = null; this.cleanup = []; }
    addTo(map) {
      if (this.owner === map) return this;
      this.remove(); this.owner = map; map.layers.add(this); this.render(); return this;
    }
    detach() {
      this.cleanup.splice(0).forEach((cleanup) => cleanup());
      if (this.native) {
        if (this.native.setMap) this.native.setMap(null);
        else this.native.remove();
      }
      this.native = null;
    }
    remove() { this.detach(); this.owner?.layers.delete(this); this.owner = null; return this; }
    getBounds() { const bounds = new Bounds(); this.entries.forEach((entry) => boundsOf(entry.feature.geometry, bounds)); return bounds; }
    bindPopup(html) { this.entries.forEach((entry) => { entry.popup = html; }); return this; }
    setStyle(style) {
      this.entries.forEach((entry) => { entry.style = { ...entry.style, ...(typeof style === "function" ? style(entry.feature) : style) }; });
      if (this.native) this.native.setStyle(this.styleFunction());
      return this;
    }
    styleFunction() {
      return (feature) => {
        const index = this.owner.context.provider === "google" ? Number(feature.getId()) : Number(feature.id);
        const entry = this.entries[index];
        return this.owner.context.provider === "google" ? googleStyle(entry.style, entry.feature.geometry.type === "Point") : entry.style;
      };
    }
    render() {
      this.detach();
      if (!this.owner) return;
      const owner = this.owner;
      // Renderer-only IDs bind styling/popups without changing downloaded GeoJSON.
      const data = { type: "FeatureCollection", features: this.entries.map((entry, id) => ({ ...entry.feature, id })) };
      if (owner.context.provider === "google") {
        const maps = root.google.maps;
        this.native = new maps.Data();
        this.native.addGeoJson(data);
        this.native.setStyle(this.styleFunction());
        this.native.setMap(owner.native);
        const handler = this.native.addListener("click", (e) => {
          const entry = this.entries[Number(e.feature.getId())];
          if (entry?.popup) {
            owner.popup.setContent(entry.popup); owner.popup.setPosition(e.latLng);
            owner.popup.open({ map: owner.native });
          } else owner.fire("click", { latlng: latLng(e.latLng), originalEvent: e.domEvent });
        });
        this.cleanup.push(() => handler.remove());
      } else {
        this.native = root.L.geoJSON(data, {
          style: this.styleFunction(),
          pointToLayer: (f, point) => root.L.circleMarker(point, this.entries[Number(f.id)].style),
          onEachFeature: (f, child) => {
            const entry = this.entries[Number(f.id)];
            if (entry.popup) child.bindPopup(entry.popup);
          },
        }).addTo(owner.native);
        // Hotspots must stay clickable above burn polygons, regardless of load order.
        owner.layers.forEach((layer) => {
          if (layer.entries.some((entry) => entry.feature.geometry.type === "Point")) layer.native?.bringToFront?.();
        });
      }
    }
  }
  function geoJSON(data, options = {}) {
    const entries = featuresOf(data).map((feature) => {
      const style = { ...(typeof options.style === "function" ? options.style(feature) : options.style) };
      if (options.interactive != null) style.interactive = options.interactive;
      let entry = { feature, style };
      if (feature.geometry.type === "Point" && options.pointToLayer) {
        const [lng, lat] = feature.geometry.coordinates;
        const marker = options.pointToLayer(feature, latLng(lat, lng));
        entry = { ...entry, style: { ...style, ...marker.entries[0].style } };
      }
      if (options.onEachFeature) options.onEachFeature(feature, new Layer([entry]));
      return entry;
    });
    return new Layer(entries);
  }
  const layerAPI = {
    geoJSON, latLng, latLngBounds: (a, b) => new Bounds(b ? [a, b] : a),
    circleMarker: (point, style) => { const p = latLng(point); return geoJSON({ type: "Point", coordinates: [p.lng, p.lat] }, { style }); },
    polyline: (points, style) => points.length < 2 ? new Layer() : geoJSON({ type: "LineString", coordinates: points.map((v) => { const p = latLng(v); return [p.lng, p.lat]; }) }, { style }),
    rectangle: (bounds, style) => geoJSON({ type: "Polygon", coordinates: [[[bounds.west, bounds.south], [bounds.east, bounds.south], [bounds.east, bounds.north], [bounds.west, bounds.north], [bounds.west, bounds.south]]] }, { style }),
    featureGroup: (children) => new Layer(children.flatMap((child) => child.entries)),
  };
  class MapFacade {
    constructor(context, host, options) {
      this.context = context; this.host = host; this.options = options;
      this.layers = new Set(); this.listeners = new Map(); this.native = null; this.popup = null;
      this.fitEpoch = 0;
    }
    on(names, fn) { names.split(/\s+/).forEach((name) => { const set = this.listeners.get(name) || new Set(); set.add(fn); this.listeners.set(name, set); }); return this; }
    fire(name, event) { this.listeners.get(name)?.forEach((fn) => fn(event)); return this; }
    removeLayer(layer) { layer?.remove(); return this; }
    setView(center, zoom) {
      this.fitEpoch++;
      if (this.context.provider === "google") {
        const p = latLng(center); this.native.setCenter({ lat: p.lat, lng: p.lng });
        if (zoom != null) this.native.setZoom(zoom);
      } else this.native.setView(center, zoom, { animate: false });
      return this;
    }
    getCenter() { return latLng(this.native.getCenter()); }
    getZoom() { return this.native.getZoom(); }
    zoomIn() { return this.setView(this.getCenter(), Math.min(this.options.maxZoom || 18, this.getZoom() + 1)); }
    zoomOut() { return this.setView(this.getCenter(), Math.max(this.options.minZoom || 3, this.getZoom() - 1)); }
    fitBounds(value, options = {}) {
      const bounds = value instanceof Bounds ? value : new Bounds(value);
      if (!bounds.isValid()) return this;
      const epoch = ++this.fitEpoch;
      if (this.context.provider === "google") {
        this.native.fitBounds(bounds.literal(), 28);
        if (options.maxZoom) root.google.maps.event.addListenerOnce(this.native, "idle", () => {
          if (epoch === this.fitEpoch && this.context.provider === "google" && this.getZoom() > options.maxZoom) this.native.setZoom(options.maxZoom);
        });
      } else this.native.fitBounds([[bounds.south, bounds.west], [bounds.north, bounds.east]], options);
      return this;
    }
    invalidateSize() {
      if (this.context.provider === "google") root.google.maps.event.trigger(this.native, "resize");
      else this.native.invalidateSize();
    }
    initialize() {
      const options = this.options;
      if (this.context.provider === "google") {
        const maps = root.google.maps;
        this.native = new maps.Map(this.host, {
          center: { lat: options.center[0], lng: options.center[1] }, zoom: options.zoom,
          minZoom: options.minZoom || 3, maxZoom: options.maxZoom || 18,
          mapTypeId: "satellite", disableDefaultUI: true, scaleControl: true,
          disableDoubleClickZoom: true, clickableIcons: false, gestureHandling: "greedy",
          tilt: 0, heading: 0, isFractionalZoomEnabled: false,
        });
        this.popup = new maps.InfoWindow();
        ["click", "mousemove"].forEach((name) => this.native.addListener(name, (e) => {
          if (e.latLng) this.fire(name, { latlng: latLng(e.latLng), originalEvent: e.domEvent });
        }));
      } else {
        this.native = root.L.map(this.host, options).setView(options.center, options.zoom);
        root.L.tileLayer(OSM, { maxZoom: 19, attribution: OSM_CREDIT, crossOrigin: true }).addTo(this.native)
          .on("tileerror", () => event("firewatch:base-tile-error", {}));
        root.L.control.scale({ imperial: false, position: "bottomleft", maxWidth: 95 }).addTo(this.native);
        ["click", "mousemove"].forEach((name) => this.native.on(name, (e) => this.fire(name, e)));
        this.native.createPane("imagery").style.zIndex = "250";
      }
    }
  }
  async function fetchConfig() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await root.fetch("/v1/client-config", { signal: controller.signal, credentials: "same-origin", headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`Config HTTP ${response.status}`);
      const config = await response.json();
      if (!config || typeof config !== "object") throw new Error("Некорректная конфигурация карты");
      return config;
    } catch { return { ...fallbackConfig(), config_error: true }; }
    finally { clearTimeout(timer); }
  }
  function loadGoogle(key, auth) {
    if (root.google?.maps?.Map) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const callback = "__firewatchGoogleMapsReady";
      const script = document.createElement("script");
      let settled = false;
      const timer = setTimeout(() => finish(new Error("Google Maps не загрузился вовремя.")), 12000);
      function finish(error) {
        if (settled) return;
        settled = true; clearTimeout(timer); auth.reject = null;
        // A late network response may still execute its callback after timeout.
        root[callback] = () => {};
        if (error) { script.remove(); reject(error); } else resolve();
      }
      root[callback] = () => finish(root.google?.maps?.Map ? null : new Error("Google Maps загрузился некорректно."));
      auth.reject = finish;
      script.async = true;
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&v=quarterly&loading=async&language=ru&callback=${callback}`;
      script.onerror = () => finish(new Error("Google Maps JavaScript API недоступен."));
      document.head.appendChild(script);
    });
  }
  async function create(hostId, options, suppliedConfig) {
    const config = suppliedConfig || await fetchConfig();
    const context = { provider: "osm", config, L: layerAPI, map: null, imagery: null, imagerySpec: null, getAOI: () => null };
    const host = typeof hostId === "string" ? document.getElementById(hostId) : hostId;
    const auth = { reject: null, error: null };
    const previousAuth = root.gm_authFailure;
    const authFailure = () => {
      const error = new Error("Google Maps отклонил ключ или настройки доступа.");
      auth.error = error;
      if (auth.reject) auth.reject(error);
      if (context.provider === "google" && context.map?.native) fallback(error);
      if (typeof previousAuth === "function") previousAuth();
    };
    if (config.maps?.provider === "google" && config.maps?.configured && config.maps?.api_key) {
      root.gm_authFailure = authFailure;
      try { await loadGoogle(config.maps.api_key, auth); if (auth.error) throw auth.error; context.provider = "google"; }
      catch (error) { context.config = googleFailureConfig(config, error); }
    }
    context.map = new MapFacade(context, host, options);
    function fallback(error) {
      if (context.provider !== "google") return;
      const map = context.map;
      const currentCenter = map.native?.getCenter?.();
      const center = currentCenter ? latLng(currentCenter) : latLng(options.center);
      const zoom = map.native?.getZoom?.() ?? options.zoom;
      const imagerySpec = context.imagerySpec;
      context.clearImageryTileLayer();
      map.layers.forEach((layer) => layer.detach());
      map.popup?.close();
      if (map.native) root.google.maps.event.clearInstanceListeners(map.native);
      host.replaceChildren();
      context.provider = "osm";
      context.config = googleFailureConfig(config, error);
      map.options = { ...options, center: [center.lat, center.lng], zoom };
      map.initialize();
      map.layers.forEach((layer) => layer.render());
      if (imagerySpec) context.setImageryTileLayer(imagerySpec.url, imagerySpec.attribution);
      event("firewatch:map-status", { provider: "osm", error: error.message });
    }
    context.setTheme = (theme) => {
      if (context.provider === "google") context.map.native.setMapTypeId(theme === "roadmap" ? "roadmap" : "satellite");
    };
    context.useFallback = () => fallback(new Error("Выбрана резервная карта."));
    context.fitBounds = (value) => {
      if (Array.isArray(value) && value.length === 4 && value.every(Number.isFinite)) context.map.fitBounds([[value[1], value[0]], [value[3], value[2]]], { maxZoom: 14, animate: false });
    };
    function credit(value = "") {
      const node = document.getElementById("imagery-attribution");
      if (node) { node.textContent = value; node.hidden = !value; }
    }
    context.clearImageryTileLayer = () => {
      if (context.imagery) {
        if (context.provider === "google") {
          const layers = context.map.native.overlayMapTypes;
          const index = layers.getArray().indexOf(context.imagery);
          if (index >= 0) layers.removeAt(index);
        } else context.imagery.remove();
      }
      context.imagery = null; context.imagerySpec = null; credit();
    };
    context.setImageryTileLayer = (url, attribution = "Google Earth Engine · Copernicus Sentinel-2") => {
      if (!allowedImageryURL(url)) throw new Error("Сервис вернул недопустимую ссылку на спутниковый снимок.");
      context.clearImageryTileLayer();
      context.imagerySpec = { url, attribution };
      let reported = false;
      const failed = () => { if (!reported) { reported = true; event("firewatch:imagery-error", {}); } };
      if (context.provider === "google") {
        const tileLayer = {
          tileSize: new root.google.maps.Size(256, 256), minZoom: 0, maxZoom: 22,
          getTile(coordinate, zoom, ownerDocument) {
            const tile = ownerDocument.createElement("img");
            tile.width = 256; tile.height = 256; tile.alt = "";
            const count = 2 ** zoom;
            if (coordinate.y < 0 || coordinate.y >= count) return tile;
            tile.onerror = failed;
            tile.src = url.replace("{z}", zoom).replace("{x}", ((coordinate.x % count) + count) % count).replace("{y}", coordinate.y);
            return tile;
          },
          releaseTile(tile) { tile.onerror = null; },
        };
        context.map.native.overlayMapTypes.push(tileLayer);
        context.imagery = tileLayer;
      } else {
        context.imagery = root.L.tileLayer(url, { pane: "imagery", maxZoom: 22, opacity: 1, attribution: escape(attribution) }).addTo(context.map.native);
        context.imagery.on("tileerror", failed);
      }
      credit(attribution);
      return context.imagery;
    };
    try { context.map.initialize(); }
    catch (error) {
      if (context.provider !== "google") throw error;
      fallback(error);
    }
    if (auth.error && context.provider === "google") fallback(auth.error);
    return context;
  }
  return { create, allowedImageryURL, fallbackConfig, googleFailureConfig, Bounds, latLng, googleStyle, geoJSON, layerAPI };
});
