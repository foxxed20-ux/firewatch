/* Dated Sentinel-2 discovery is separate from the model's prepared datasets. */
(function (root) {
  "use strict";
  const DAY = 86400000;
  function validDay(value) {
    return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value) &&
      Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
  }
  function searchPayload(aoi, start, end, cloud) {
    if (!aoi || !["Polygon", "MultiPolygon"].includes(aoi.type)) throw new Error("Выберите область на карте.");
    if (!validDay(start) || !validDay(end)) throw new Error("Укажите корректные даты начала и конца.");
    if (start < "2017-03-28") throw new Error("Архив Sentinel-2 SR доступен с 28 марта 2017 года.");
    const days = (Date.parse(end) - Date.parse(start)) / DAY + 1;
    if (days < 1) throw new Error("Дата начала должна быть не позже даты конца.");
    if (days > 366) throw new Error("Выберите период не длиннее 366 дней.");
    if (cloud === "" || !Number.isFinite(Number(cloud)) || Number(cloud) < 0 || Number(cloud) > 100) throw new Error("Облачность должна быть от 0 до 100%.");
    return { aoi, date_start: start, date_end: end, cloud_max: Number(cloud), limit: 20 };
  }
  function cloudLabel(value) {
    return typeof value === "number" && Number.isFinite(value)
      ? `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(value)}%`
      : "нет оценки";
  }
  const errors = {
    GOOGLE_NOT_CONFIGURED: "Google Earth Engine ещё не настроен на сервере.",
    EARTH_ENGINE_UNAVAILABLE: "Earth Engine сейчас недоступен. Повторите позже или проверьте подключение Google Cloud.",
    IMAGERY_RATE_LIMITED: "Достигнут лимит поиска снимков. Повторите через минуту.",
    IMAGERY_BUSY: "Сервис обрабатывает другой запрос к Earth Engine. Повторите через несколько секунд.",
    IMAGERY_AOI_TOO_LARGE: "Уменьшите область: до 10 000 км², не более 300 км по каждой стороне.",
    IMAGERY_AOI_INVALID: "Выберите область без пересечения 180-го меридиана и вне полярных широт.",
    IMAGERY_DATE_RANGE_TOO_LARGE: "Выберите период не длиннее 366 дней.",
    EARTH_ENGINE_TIMEOUT: "Earth Engine не ответил вовремя. Уменьшите область или период и повторите позже.",
    RATE_LIMITED: "Слишком много запросов. Повторите через минуту.",
    UNAUTHORIZED: "Укажите токен доступа в параметрах приложения и повторите поиск.",
  };
  function errorMessage(error) {
    const code = error?.body?.error?.code;
    if (errors[code]) return errors[code];
    if (error?.status === 429) return errors.RATE_LIMITED;
    if (error?.name === "AbortError") return "Поиск не успел завершиться. Попробуйте меньшую область или повторите позже.";
    if (error instanceof TypeError) return "Нет связи с сервисом. Проверьте подключение и повторите.";
    return error?.body?.error?.message || error?.message || "Не удалось получить снимки.";
  }
  const api = { searchPayload, validDay, cloudLabel, errorMessage };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.FireWatchImagery = api;
  if (typeof document === "undefined") return;
  const $ = (id) => document.getElementById(id);
  const dialog = $("imagery-dialog");
  if (!dialog) return;
  const state = { map: null, config: null, busy: false, epoch: 0, scenes: [], payload: null, selected: null, opened: false, controller: null };
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const dateLabel = (value) => {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? date.toLocaleString("ru-RU", { timeZone: "UTC", dateStyle: "medium", timeStyle: "short" }) : "Дата не указана";
  };
  function status(message, bad = false) {
    $("imagery-status").textContent = message;
    $("imagery-status").classList.toggle("is-error", bad);
  }
  function readyControls() {
    const area = state.map?.getAOI?.();
    $("imagery-aoi-label").textContent = area ? "Поиск внутри выбранной области на карте" : "Сначала выберите область на карте.";
    $("imagery-search").disabled = state.busy || !area || !state.config?.imagery?.configured;
    $("imagery-search").querySelector("span").textContent = state.busy ? "Загружаем…" : "Найти снимки";
    $("imagery-results").setAttribute("aria-busy", String(state.busy));
    $("imagery-setup").hidden = !state.config || !!state.config.imagery?.configured;
    document.querySelectorAll("[data-scene-preview]").forEach((button) => { button.disabled = state.busy; });
  }
  function clearOverlay() {
    state.map?.clearImageryTileLayer?.();
    state.selected = null;
    $("imagery-active").hidden = true;
    document.querySelectorAll("[data-scene-preview]").forEach((button) => {
      button.setAttribute("aria-pressed", "false");
      button.textContent = "На карту";
    });
  }
  function invalidate(message) {
    state.epoch++;
    state.controller?.abort();
    state.controller = null;
    state.busy = false;
    state.scenes = [];
    state.payload = null;
    clearOverlay();
    $("imagery-results").replaceChildren();
    status(message || "Параметры изменились. Нажмите «Найти снимки».");
    readyControls();
  }
  async function post(path, payload) {
    const controller = new AbortController();
    state.controller = controller;
    const timeout = setTimeout(() => controller.abort(), 28000);
    let token = "";
    try { token = sessionStorage.getItem("firewatch-token") || ""; } catch { /* session storage may be unavailable */ }
    const headers = state.map?.apiHeaders?.({ "Content-Type": "application/json" }) || {
      Accept: "application/json", "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };
    try {
      const response = await fetch(path, { method: "POST", headers, body: JSON.stringify(payload), signal: controller.signal, redirect: "error" });
      const body = await response.json();
      if (!response.ok) {
        if (response.status === 401) $("token-wrap").hidden = false;
        throw Object.assign(new Error(`HTTP ${response.status}`), { status: response.status, body });
      }
      return body;
    } finally {
      clearTimeout(timeout);
      if (state.controller === controller) state.controller = null;
    }
  }
  function renderScenes() {
    $("imagery-results").innerHTML = state.scenes.map((scene, index) =>
      `<article class="scene-card"><div class="scene-date"><span class="scene-orbit" aria-hidden="true"><svg><use href="#i-satellite" /></svg></span><div><h3>${escape(dateLabel(scene.acquired_at))} <small>UTC</small></h3><p>${escape(scene.sensor || "Sentinel-2")} · ${escape(scene.resolution_m || 10)} м/пиксель</p></div></div><dl><div><dt>Облачность сцены</dt><dd>${escape(cloudLabel(scene.cloud_percent))}</dd></div></dl><p class="scene-id" title="${escape(scene.id)}">${escape(scene.id)}</p><div class="scene-actions"><button type="button" class="secondary-button" data-scene-preview="${index}" aria-pressed="false">На карту</button><button type="button" class="text-button" data-scene-download="${index}">Метаданные ↓</button></div></article>`
    ).join("");
  }
  async function search(event) {
    event.preventDefault();
    let payload;
    try { payload = searchPayload(state.map?.getAOI?.(), $("imagery-start").value, $("imagery-end").value, $("imagery-cloud").value); }
    catch (error) { status(error.message, true); return; }
    if (state.busy || !state.config?.imagery?.configured) return;
    invalidate("Ищем сцены Sentinel-2 в выбранной области…");
    const epoch = state.epoch;
    state.busy = true;
    readyControls();
    try {
      const result = await post("/v1/imagery/search", payload);
      if (epoch !== state.epoch) return;
      if (!Array.isArray(result.scenes)) throw new Error("Сервис вернул некорректный список сцен.");
      state.payload = payload;
      state.scenes = result.scenes.filter((scene) => typeof scene?.id === "string").slice(0, 20);
      renderScenes();
      status(state.scenes.length
        ? `${result.truncated ? "Показаны первые" : "Найдено сцен:"} ${state.scenes.length}. ${result.truncated ? "Сузьте период для остальных сцен." : "Выберите снимок для просмотра на карте."}`
        : "Снимков по этим условиям нет. Увеличьте допустимую облачность или измените период.");
    } catch (error) { if (epoch === state.epoch) status(errorMessage(error), true); }
    finally { if (epoch === state.epoch) { state.busy = false; readyControls(); } }
  }
  async function preview(index) {
    if (state.busy || !state.payload) return;
    const scene = state.scenes[index];
    if (!scene) return;
    const epoch = state.epoch;
    state.busy = true;
    readyControls();
    status("Готовим снимок для карты…");
    try {
      const result = await post("/v1/imagery/preview", { scene_id: scene.id, aoi: state.payload.aoi });
      if (epoch !== state.epoch) return;
      state.map.setImageryTileLayer(result.tile_url, result.attribution);
      state.selected = scene;
      $("imagery-active-title").textContent = `Sentinel-2 · ${dateLabel(scene.acquired_at)} UTC`;
      $("imagery-active-detail").textContent = `RGB · облачность сцены ${cloudLabel(scene.cloud_percent)}`;
      $("imagery-active").hidden = false;
      document.querySelectorAll("[data-scene-preview]").forEach((button) => {
        const selected = Number(button.dataset.scenePreview) === index;
        button.setAttribute("aria-pressed", String(selected));
        button.textContent = selected ? "На карте" : "На карту";
      });
      status("Снимок показан на карте. Если тайлы перестали загружаться, откройте снимок повторно.");
      dialog.close();
      document.querySelector('[data-panel="map-section"]')?.click();
      // A hidden mobile map has no usable viewport until its panel is shown.
      requestAnimationFrame(() => {
        if (epoch !== state.epoch || state.selected !== scene) return;
        state.map.map?.invalidateSize();
        if (Array.isArray(result.bounds)) state.map.fitBounds(result.bounds);
      });
    } catch (error) { if (epoch === state.epoch) status(errorMessage(error), true); }
    finally { if (epoch === state.epoch) { state.busy = false; readyControls(); } }
  }
  function download(index) {
    const scene = state.scenes[index];
    if (!scene) return;
    const data = { provider: "Google Earth Engine", collection: "COPERNICUS/S2_SR_HARMONIZED", scene, search: state.payload, visualization: "RGB B4/B3/B2", model_ready: false };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `firewatch-sentinel2-${String(scene.acquired_at || "scene").slice(0, 10).replace(/[^0-9-]/g, "")}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function open() {
    if (!state.opened) {
      const today = new Date().toISOString().slice(0, 10);
      const earlier = new Date(Date.now() - 30 * DAY).toISOString().slice(0, 10);
      $("imagery-start").value = $("date-start").value || earlier;
      $("imagery-end").value = $("date-end").value || today;
      state.opened = true;
    }
    readyControls();
    dialog.showModal();
  }
  $("open-imagery").addEventListener("click", open);
  $("imagery-reopen").addEventListener("click", open);
  $("imagery-remove").addEventListener("click", clearOverlay);
  $("imagery-select-area").addEventListener("click", () => { dialog.close(); $("draw-rect").click(); });
  $("imagery-form").addEventListener("submit", search);
  ["imagery-start", "imagery-end", "imagery-cloud"].forEach((id) => $(id).addEventListener("change", () => invalidate()));
  $("imagery-results").addEventListener("click", (event) => {
    const previewButton = event.target.closest("[data-scene-preview]");
    const downloadButton = event.target.closest("[data-scene-download]");
    if (previewButton) preview(Number(previewButton.dataset.scenePreview));
    if (downloadButton) download(Number(downloadButton.dataset.sceneDownload));
  });
  root.addEventListener("firewatch:aoi-changed", () => invalidate("Территория изменилась. Выполните поиск для новой области."));
  root.addEventListener("firewatch:imagery-error", () => {
    status("Снимок не загрузился. Откройте его повторно для обновления временной ссылки.", true);
    $("imagery-active-detail").textContent = "Не удалось загрузить тайлы · откройте снимок повторно";
  });
  Promise.resolve(root.FireWatchMapReady).then((map) => {
    state.map = map;
    state.config = map?.config || { imagery: { configured: false } };
    readyControls();
  }).catch(() => {
    state.config = { imagery: { configured: false } };
    status("Не удалось подключить карту. Обновите страницу.", true);
    readyControls();
  });
  readyControls();
})(typeof globalThis !== "undefined" ? globalThis : this);
