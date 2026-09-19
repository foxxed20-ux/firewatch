/* FireWatch working surface. Data and areas always come from the service. */
(async () => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const C = globalThis.FireWatchCore;
  const icon = (name) =>
    `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const esc = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const fmt = (n, digits = 1) =>
    n == null || !Number.isFinite(n)
      ? "—"
      : new Intl.NumberFormat("ru-RU", {
          maximumFractionDigits: digits,
        }).format(n);
  const dayLabel = (value) =>
    C.dateInput(value)
      ? new Date(C.dateInput(value) + "T12:00:00Z").toLocaleDateString(
          "ru-RU",
          { timeZone: "UTC" },
        )
      : "—";
  const storage = {
    get(key, fallback = null) {
      try {
        return JSON.parse(localStorage.getItem(`firewatch-${key}`)) ?? fallback;
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        if (value == null) localStorage.removeItem(`firewatch-${key}`);
        else localStorage.setItem(`firewatch-${key}`, JSON.stringify(value));
      } catch {
        /* Storage may be disabled; the current analysis still works. */
      }
    },
  };
  let accessToken = "";
  try {
    accessToken = sessionStorage.getItem("firewatch-token") || "";
  } catch {
    /* session-only memory fallback */
  }
  const history = storage.get("history", []);
  const state = {
    catalog: [],
    models: [],
    ready: false,
    metadataLoaded: false,
    preset: null,
    aoi: null,
    drawMode: null,
    points: [],
    rectStart: null,
    drawLayer: null,
    preview: null,
    coverage: null,
    resultLayers: {},
    busy: false,
    pending: storage.get("pending"),
    history: Array.isArray(history) ? history.slice(0, 20) : [],
    timer: null,
    pollEpoch: 0,
    resultEpoch: 0,
    context: null,
    result: null,
    normalized: null,
    retry: null,
    toastTimer: null,
    fetching: false,
  };
  if (!globalThis.L || !C || !globalThis.FireWatchMapProvider) {
    $("error-box").hidden = false;
    $("error-message").textContent =
      "Не удалось загрузить компоненты интерфейса. Обновите страницу.";
    return;
  }
  const presetRow = document.createElement("div");
  presetRow.className = "preset-row";
  presetRow.hidden = true;
  presetRow.innerHTML =
    '<label for="preset">Готовая территория</label><select id="preset" aria-label="Готовая территория"></select>';
  $("dataset-note").before(presetRow);
  const leaveJob = document.createElement("button");
  leaveJob.id = "leave-job";
  leaveJob.hidden = true;
  leaveJob.textContent = "Вернуться к параметрам";
  $("error-box").append(leaveJob);
  const mapContextPromise = globalThis.FireWatchMapProvider.create("map", {
    zoomControl: false,
    doubleClickZoom: false,
    preferCanvas: true,
    minZoom: 3,
    maxZoom: 18,
    worldCopyJump: true,
    center: [49.5, 40], zoom: 6,
  });
  globalThis.FireWatchMapReady = mapContextPromise;
  const mapContext = await mapContextPromise;
  const L = mapContext.L, map = mapContext.map;
  const mapProvider = mapContext;
  // apiHeaders is declared below; defer evaluation until imagery actually sends a request.
  mapContext.apiHeaders = (extra) => apiHeaders(extra);
  mapContext.getAOI = () => state.aoi?.geometry || (state.aoi?.bbox ? C.bboxGeometry(state.aoi.bbox) : null);
  const setMapProviderStatus = (text) => {
    const node = $("map-provider-status");
    if (node) node.textContent = text;
  };
  function updateMapProvider() {
    const google = mapProvider.provider === "google";
    $("map").dataset.provider = mapProvider.provider;
    $("map-dark").textContent = google ? "Спутник" : "Тёмная";
    $("map-light").textContent = google ? "Карта" : "Светлая";
    $("map-fallback").hidden = !google;
    $("map-notice").hidden = !mapProvider.config?.maps?.error;
    if (mapProvider.config?.maps?.error) $("map-notice").textContent = "Используется резервная карта. Область и результаты сохранены.";
    setMapProviderStatus(google ? "Подложка: Google Maps" : mapProvider.config?.maps?.error ? "OpenStreetMap · резервная карта" : "OpenStreetMap · Google Maps ещё не подключён");
  }
  updateMapProvider();
  $("map-fallback").addEventListener("click", () => mapProvider.useFallback());
  globalThis.addEventListener("firewatch:map-status", updateMapProvider);
  globalThis.addEventListener("firewatch:base-tile-error", () => {
    $("map-notice").textContent = "Подложка недоступна. Контуры и выбор области продолжают работать.";
    $("map-notice").hidden = false;
  });
  const aoiStyle = {
    color: "#ff965c",
    weight: 2,
    fillColor: "#ff965c",
    fillOpacity: 0.035,
    dashArray: "7 6",
    interactive: false,
  };
  const coverageStyle = {
    color: "#96b9c1",
    weight: 1,
    fillColor: "#96b9c1",
    fillOpacity: 0.025,
    dashArray: "3 6",
    interactive: false,
  };
  const dataset = () =>
    state.catalog.find((d) => (d.dataset_id || d.id) === $("dataset").value);
  const model = () =>
    state.models.find((m) => (m.model_bundle_id || m.id) === $("model").value);
  const selectedTasks = () =>
    ["af", "bs"].filter(
      (t) => $(`task-${t}`).checked && !$(`task-${t}`).dataset.unsupported,
    );
  const datasetLabel = (d) =>
    d?.dataset_id === "official-validation-demo"
      ? "Спутниковые сцены · train"
      : d?.label || d?.dataset_id || "Набор данных";
  const contextLabel = () =>
    state.preset ? presetLabel(state.preset) : datasetLabel(dataset());
  const apiHeaders = (extra) => ({
    Accept: "application/json",
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    ...extra,
  });
  function safeURL(value) {
    if (typeof value !== "string" || !value)
      throw new Error("Сервис не передал ссылку на результат.");
    const url = new URL(value, location.origin);
    if (
      url.origin !== location.origin ||
      !/^\/(v1|health)\//.test(url.pathname)
    )
      throw new Error("Некорректная ссылка сервиса.");
    return url.href;
  }
  const messages = {
    MODEL_NOT_READY: "Модель ещё готовится. Обновите её статус через минуту.",
    UNAUTHORIZED: "Для этого сервиса нужен токен доступа.",
    UNKNOWN_DATASET: "Этот набор больше не доступен. Обновите каталог.",
    JOB_TIMEOUT:
      "Сервис остановил анализ по лимиту времени. Попробуйте меньшую область.",
    JOB_NOT_FOUND:
      "Задание не найдено. История хранится в браузере, а результаты — на сервере.",
    WORKER_LOST:
      "Анализ прервался при перезапуске сервиса. Запустите его повторно.",
    ARTIFACT_NOT_FOUND: "Файл больше не доступен на сервере.",
    INPUT_SCHEMA_MISMATCH:
      "Сервис отклонил параметры. Проверьте территорию и период.",
    IDEMPOTENCY_CONFLICT:
      "Ключ запроса уже использован для других параметров. Обновите страницу.",
  };
  function errorText(error) {
    const code = error?.body?.error?.code || error?.error?.code;
    if (messages[code]) return messages[code];
    if (error?.name === "AbortError")
      return "Сервис не ответил вовремя. Можно безопасно повторить запрос.";
    if (error instanceof TypeError)
      return "Нет связи с сервисом. Проверьте подключение и повторите.";
    return (
      error?.body?.error?.message ||
      error?.error?.message ||
      error?.message ||
      "Неизвестная ошибка сервиса."
    );
  }
  async function request(path, options = {}) {
    const controller = new AbortController(),
      timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(safeURL(path), {
        ...options,
        headers: apiHeaders(options.headers),
        signal: controller.signal,
        redirect: "error",
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        const e = new Error(`Сервис вернул HTTP ${response.status}`);
        e.status = response.status;
        e.body = data;
        const retry = response.headers.get("Retry-After");
        e.retryAt = retry
          ? Number.isFinite(Number(retry))
            ? Date.now() + Number(retry) * 1000
            : Date.parse(retry)
          : 0;
        if (response.status === 401) $("token-wrap").hidden = false;
        throw e;
      }
      return { data, response };
    } finally {
      clearTimeout(timeout);
    }
  }
  function toast(message) {
    clearTimeout(state.toastTimer);
    $("toast").textContent = message;
    $("toast").hidden = false;
    state.toastTimer = setTimeout(() => ($("toast").hidden = true), 4200);
  }
  function showError(message, retry = null) {
    $("error-message").textContent = message;
    $("error-box").hidden = false;
    state.retry = retry;
    $("retry-action").hidden = !retry;
    leaveJob.hidden = !state.pending || !!state.pending.job;
    if (!leaveJob.hidden)
      leaveJob.textContent = "Отложить запрос и изменить параметры";
  }
  function clearError() {
    $("error-box").hidden = true;
    state.retry = null;
    leaveJob.hidden = true;
  }
  function apiStatus(kind, label) {
    $("api-dot").className = `pulse ${kind}`;
    $("api-state").textContent = label;
  }
  function setPanel(id, scroll = false) {
    ["parameters", "map-section", "results-panel"].forEach((key) =>
      $(key).classList.toggle("mobile-active", key === id),
    );
    document
      .querySelectorAll("[data-panel]")
      .forEach((b) => b.classList.toggle("active", b.dataset.panel === id));
    requestAnimationFrame(() => map.invalidateSize());
    if (scroll && matchMedia("(max-width:760px)").matches)
      document
        .querySelector(".mobile-tabs")
        .scrollIntoView({ block: "start", behavior: "smooth" });
  }
  function fit(layer) {
    if (layer?.getBounds?.().isValid()) {
      map.invalidateSize();
      map.fitBounds(layer.getBounds().pad(0.16), {
        maxZoom: 13,
        animate: false,
      });
    }
  }
  function clearPreview() {
    if (state.preview) map.removeLayer(state.preview);
    state.preview = null;
  }
  function drawing(mode) {
    clearPreview();
    state.drawMode = mode;
    state.rectStart = null;
    state.points = [];
    ["rect", "poly"].forEach((t) => {
      $(`draw-${t}`).classList.toggle("active", mode === t);
      $(`draw-${t}`).setAttribute("aria-pressed", String(mode === t));
    });
    $("map").classList.toggle("draw-cursor", !!mode);
    $("draw-message").hidden = !mode;
    $("finish-polygon").hidden = mode !== "poly";
    $("finish-polygon").disabled = true;
    $("draw-instruction").textContent =
      mode === "rect"
        ? "Нажмите на два противоположных угла области."
        : "Отметьте вершины контура на карте.";
    if (mode) {
      $("empty-map").hidden = true;
      setPanel("map-section", true);
    }
    refreshControls();
  }
  function setAOI(geometry, kind = "geometry", shouldFit = false) {
    invalidateResult();
    if (state.drawLayer) map.removeLayer(state.drawLayer);
    state.drawLayer = L.geoJSON(geometry, {
      style: aoiStyle,
      interactive: false,
    }).addTo(map);
    const b = state.drawLayer.getBounds();
    if (!b.isValid()) return;
    state.aoi =
      kind === "bbox"
        ? { bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] }
        : { geometry };
    globalThis.dispatchEvent(new CustomEvent("firewatch:aoi-changed", { detail: { geometry } }));
    ["bbox-w", "bbox-s", "bbox-e", "bbox-n"].forEach(
      (id, i) =>
        ($(id).value = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()][
          i
        ].toFixed(5)),
    );
    $("aoi-note").textContent =
      kind === "bbox"
        ? "Область выбрана · можно уточнить на карте"
        : `Контур выбран · ${geometry.coordinates[0].length - 1} вершин`;
    $("aoi-note").classList.add("selected");
    $("empty-map").hidden = true;
    drawing(null);
    if (shouldFit) fit(state.drawLayer);
    refreshControls();
  }
  function clearAOI() {
    invalidateResult();
    drawing(null);
    if (state.drawLayer) map.removeLayer(state.drawLayer);
    state.drawLayer = null;
    state.aoi = null;
    globalThis.dispatchEvent(new CustomEvent("firewatch:aoi-changed", { detail: { geometry: null } }));
    ["bbox-w", "bbox-s", "bbox-e", "bbox-n"].forEach(
      (id) => ($(id).value = ""),
    );
    $("aoi-note").textContent =
      "Выберите покрытие или нарисуйте область на карте.";
    $("aoi-note").classList.remove("selected");
    refreshControls();
  }
  function applyBBox() {
    const raw = ["bbox-w", "bbox-s", "bbox-e", "bbox-n"].map((id) =>
      $(id).value.trim(),
    );
    const values = raw.map((v) => Number(v.replace(",", ".")));
    if (raw.some((v) => !v) || !C.validBBox(values)) {
      toast("Введите все координаты: запад < восток, юг < север.");
      return;
    }
    setAOI(C.bboxGeometry(values), "bbox", true);
    clearError();
  }
  function finishPolygon() {
    const message = C.validateRing(state.points);
    if (message) {
      toast(message);
      return;
    }
    const p = state.points.map((v) => [...v]);
    setAOI({ type: "Polygon", coordinates: [[...p, [...p[0]]]] }, "geometry");
    toast("Контур сохранён. Можно запускать анализ.");
  }
  map.on("click", (e) => {
    if (state.busy || !state.drawMode) return;
    const p = e.latlng.wrap();
    if (state.drawMode === "rect") {
      if (!state.rectStart) {
        state.rectStart = p;
        $("draw-instruction").textContent =
          "Теперь выберите противоположный угол.";
      } else {
        const b = L.latLngBounds(state.rectStart, p),
          values = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
        if (!C.validBBox(values)) {
          toast("Область должна иметь ненулевую площадь.");
          return;
        }
        setAOI(C.bboxGeometry(values), "bbox");
      }
    } else {
      const previous = state.points.at(-1);
      if (previous && previous[0] === p.lng && previous[1] === p.lat) return;
      if (state.points.length >= 200) {
        toast("Достигнут предел 200 вершин. Завершите контур.");
        return;
      }
      state.points.push([p.lng, p.lat]);
      clearPreview();
      state.preview = L.featureGroup([
        L.polyline(
          state.points.map((p) => [p[1], p[0]]),
          { color: "#ffab72", weight: 2, interactive: false },
        ),
        ...state.points.map((p) =>
          L.circleMarker([p[1], p[0]], {
            radius: 3,
            color: "#fff",
            weight: 1,
            fillColor: "#ed783d",
            fillOpacity: 1,
            interactive: false,
          }),
        ),
      ]).addTo(map);
      $("draw-instruction").textContent =
        `Вершин: ${state.points.length}. ${state.points.length < 3 ? "Добавьте ещё вершины." : "Завершите контур кнопкой."}`;
      $("finish-polygon").disabled = state.points.length < 3;
    }
  });
  map.on("mousemove", (e) => {
    const p = e.latlng.wrap();
    $("cursor-coords").textContent =
      `${Math.abs(p.lat).toFixed(4)}° ${p.lat < 0 ? "S" : "N"}  ${Math.abs(p.lng).toFixed(4)}° ${p.lng < 0 ? "W" : "E"}`;
    if (state.drawMode === "rect" && state.rectStart) {
      clearPreview();
      state.preview = L.rectangle(
        L.latLngBounds(state.rectStart, p),
        aoiStyle,
      ).addTo(map);
    }
  });
  function availableBounds() {
    return state.preset?.bounds || dataset()?.bounds || dataset()?.bbox;
  }
  function showCoverage() {
    if (state.coverage) map.removeLayer(state.coverage);
    state.coverage = null;
    const bounds = availableBounds();
    if (C.validBBox(bounds))
      state.coverage = L.geoJSON(C.bboxGeometry(bounds), {
        style: coverageStyle,
        interactive: false,
      }).addTo(map);
  }
  function useCoverage() {
    const b = availableBounds();
    if (C.validBBox(b)) setAOI(C.bboxGeometry(b), "bbox", true);
    else
      toast("Для этого набора не задано покрытие. Нарисуйте область вручную.");
  }
  function presetLabel(p) {
    const tasks = C.taskList(p.tasks);
    const id = p.preset_id || p.id || "";
    return `${tasks.length === 1 && tasks[0] === "af" ? "Термоточки" : tasks.length === 1 && tasks[0] === "bs" ? "Выгоревшая территория" : "Наблюдение"} · ${id.replace(/^(AF|BS)_tr_0*/, "") || p.label}`;
  }
  function applyPreset(reset = true) {
    state.preset =
      (dataset()?.presets || []).find(
        (p) => (p.preset_id || p.id) === $("preset").value,
      ) || null;
    const d = dataset(),
      range = state.preset
        ? C.presetDays(state.preset.period)
        : {
            start: C.dateInput(d?.date_range?.start),
            end: C.dateInput(d?.date_range?.end),
          };
    if (reset) {
      $("date-start").value = range.start;
      $("date-end").value = range.end;
    }
    showCoverage();
    if (reset) useCoverage();
    $("dataset-note").textContent = state.preset
      ? `${dayLabel(range.start)} — ${dayLabel(range.end)} · доступный пример`
      : "Выберите готовую территорию или задайте свою область.";
    const tasks = C.taskList(state.preset?.tasks);
    $("map-title").textContent =
      tasks.length === 1
        ? tasks[0] === "bs"
          ? "Оценка выгоревшей территории"
          : "Поиск активных пожаров"
        : "Обзор территории";
    $("use-coverage").querySelector("span").textContent = state.preset
      ? "Вся территория примера"
      : "Вся территория набора";
    $("map-subtitle").textContent = contextLabel();
    $("empty-map").hidden = !!state.coverage || !!state.aoi;
    syncTasks(reset);
    refreshControls();
  }
  function applyDataset(reset = true) {
    const d = dataset();
    $("dataset-scenes").textContent = d
      ? `${fmt(d.scene_count, 0)} сцен`
      : "Нет сцен";
    $("dataset-sources").textContent =
      C.taskList(d?.tasks)
        .map((t) => (t === "af" ? "VIIRS" : "Sentinel-2"))
        .join(" / ") || "Каталог данных";
    $("training-note").hidden = !C.isTraining(d);
    const presets = d?.presets || [],
      previous = $("preset").value;
    presetRow.hidden = !presets.length;
    $("preset").innerHTML =
      presets
        .map(
          (p) =>
            `<option value="${esc(p.preset_id || p.id)}">${esc(presetLabel(p))}</option>`,
        )
        .join("") + '<option value="">Всё покрытие набора</option>';
    if (!reset && presets.some((p) => (p.preset_id || p.id) === previous))
      $("preset").value = previous;
    else
      $("preset").value =
        presets.find((p) => C.taskList(p.tasks).includes("bs"))?.preset_id ||
        presets[0]?.preset_id ||
        "";
    if (d) applyPreset(reset);
    else {
      clearAOI();
      if (state.coverage) map.removeLayer(state.coverage);
      state.coverage = null;
      $("dataset-note").textContent =
        "Геопривязанные сцены ещё не добавлены в каталог.";
      $("empty-map").hidden = false;
    }
  }
  function syncTasks(reset = false) {
    const supportedDataset = C.taskList(
        state.preset?.tasks || dataset()?.tasks,
      ),
      supportedModel = C.taskList(model()?.tasks);
    ["af", "bs"].forEach((t) => {
      const node = $(`task-${t}`),
        allowed =
          supportedDataset.includes(t) &&
          (!model() || supportedModel.includes(t));
      node.dataset.unsupported = allowed ? "" : "1";
      if (!allowed) node.checked = false;
      else if (reset) node.checked = true;
    });
  }
  function validationMessage() {
    if (!state.metadataLoaded)
      return "Нет связи с каталогом. Нажмите «Обновить» вверху.";
    if (!dataset()) return "Ожидаем доступные спутниковые сцены.";
    if (!C.modelReady(model()) || !state.ready)
      return "Модель готовится. Обновите статус через минуту.";
    if (state.drawMode) return "Завершите выбор области на карте.";
    if (!state.aoi) return "Выберите территорию для анализа.";
    try {
      C.inclusivePeriod($("date-start").value, $("date-end").value);
    } catch (e) {
      return e.message;
    }
    if (!selectedTasks().length)
      return "Выберите хотя бы один доступный модуль.";
    return "";
  }
  function refreshControls() {
    const locked = state.busy || !!state.pending;
    [
      "dataset",
      "preset",
      "model",
      "date-start",
      "date-end",
      "draw-rect",
      "draw-poly",
      "clear-aoi",
      "apply-bbox",
      "bbox-w",
      "bbox-s",
      "bbox-e",
      "bbox-n",
      "use-coverage",
    ].forEach((id) => ($(id).disabled = locked));
    $("dataset").disabled = locked || !state.catalog.length;
    $("model").disabled = locked || !state.models.some(C.modelReady);
    $("use-coverage").disabled = locked || !C.validBBox(availableBounds());
    ["af", "bs"].forEach(
      (t) =>
        ($(`task-${t}`).disabled =
          locked || !!$(`task-${t}`).dataset.unsupported),
    );
    $("zoom-coverage").disabled = !state.coverage;
    $("zoom-aoi").disabled = !state.aoi;
    const issue = validationMessage();
    $("run-analysis").disabled = state.busy || (!state.pending && !!issue);
    $("run-label").textContent = state.busy
      ? "Анализ выполняется…"
      : state.pending?.job
        ? "Продолжить проверку"
        : state.pending
          ? "Повторить отправку"
          : "Запустить анализ";
    $("run-hint").textContent = state.busy
      ? "Можно изучать карту, пока модель работает."
      : state.pending
        ? "Предыдущий запрос сохранён — дубликат не создастся."
        : issue || "Термоточки, контуры и площади по выбранным данным";
    $("model-note").textContent = C.modelReady(model())
      ? `${C.taskList(model().tasks)
          .map((t) => t.toUpperCase())
          .join(" + ")} · модель готова к анализу`
      : "Модель ещё не готова. Каталог и карта доступны.";
    $("refresh-data").disabled = state.fetching;
  }
  async function loadMetadata(initial = false) {
    if (state.fetching) return;
    state.fetching = true;
    refreshControls();
    const oldDataset = $("dataset").value,
      oldModel = $("model").value;
    try {
      const [catalogResult, modelsResult, readyResult] =
        await Promise.allSettled([
          request("/v1/catalog"),
          request("/v1/models"),
          request("/health/ready"),
        ]);
      if (catalogResult.status === "rejected") throw catalogResult.reason;
      if (modelsResult.status === "rejected") throw modelsResult.reason;
      state.catalog = (catalogResult.value.data?.datasets || []).filter(
        (d) => !d.mode || d.mode === "geospatial_demo",
      );
      state.models = (modelsResult.value.data?.models || []).filter(
        (m) => m.model_bundle_id || m.id,
      );
      state.ready =
        readyResult.status === "fulfilled" &&
        readyResult.value.data?.status === "ready";
      state.metadataLoaded = true;
      $("dataset").innerHTML = state.catalog.length
        ? state.catalog
            .map(
              (d) =>
                `<option value="${esc(d.dataset_id || d.id)}">${esc(datasetLabel(d))}</option>`,
            )
            .join("")
        : '<option value="">Нет доступных сцен</option>';
      if (state.catalog.some((d) => (d.dataset_id || d.id) === oldDataset))
        $("dataset").value = oldDataset;
      $("model").innerHTML = state.models.length
        ? state.models
            .map(
              (m) =>
                `<option value="${esc(m.model_bundle_id || m.id)}" ${C.modelReady(m) ? "" : "disabled"}>${esc(m.model_bundle_id || m.id)}${C.modelReady(m) ? "" : " · готовится"}</option>`,
            )
            .join("")
        : '<option value="">Модель готовится…</option>';
      $("model").value =
        state.models.find(
          (m) => (m.model_bundle_id || m.id) === oldModel && C.modelReady(m),
        )?.model_bundle_id ||
        state.models.find(C.modelReady)?.model_bundle_id ||
        state.models[0]?.model_bundle_id ||
        "";
      if (state.pending && initial) restoreForm(state.pending);
      else applyDataset(initial || oldDataset !== $("dataset").value);
      apiStatus(
        state.ready ? "ready" : "warning",
        state.ready ? "Сервис готов" : "Модель готовится",
      );
      if (!state.pending) clearError();
      $("footer-status").textContent =
        `${state.catalog.length} набор · ${state.catalog.reduce((n, d) => n + (Number(d.scene_count) || 0), 0)} доступных сцен`;
      if (!initial)
        toast(
          state.ready
            ? "Каталог обновлён. Модель готова."
            : "Каталог обновлён. Модель ещё готовится.",
        );
    } catch (e) {
      state.ready = false;
      state.metadataLoaded = false;
      apiStatus("error", e.status === 401 ? "Нужен токен" : "Нет связи");
      showError(errorText(e), () => loadMetadata());
      $("footer-status").textContent = "Нет связи с каталогом";
    } finally {
      state.fetching = false;
      refreshControls();
    }
  }
  function clearLayers() {
    Object.values(state.resultLayers).forEach((l) => map.removeLayer(l));
    state.resultLayers = {};
    $("legend").hidden = true;
  }
  function invalidateResult() {
    if (!state.result || state.busy || state.pending) return;
    state.result = null;
    state.normalized = null;
    state.resultEpoch++;
    clearLayers();
    clearError();
    $("result-content").hidden = true;
    $("result-notice").hidden = true;
    $("job-progress").hidden = true;
    $("result-empty").hidden = false;
    $("result-title").textContent = "Параметры изменены";
    $("result-context").textContent =
      "Запустите анализ для выбранной территории и периода. Предыдущий результат доступен в истории.";
    jobState("neutral");
    $("map-context").textContent = "Контуры появятся после анализа";
  }
  function restoreForm(context) {
    const p = context.payload;
    if (!p) return;
    if (state.catalog.some((d) => (d.dataset_id || d.id) === p.dataset_id))
      $("dataset").value = p.dataset_id;
    applyDataset(false);
    const matching = (dataset()?.presets || []).find(
      (x) =>
        (x.preset_id || x.id) === context.preset_id ||
        presetLabel(x) === context.label,
    );
    $("preset").value = matching?.preset_id || matching?.id || "";
    applyPreset(false);
    if (
      state.models.some(
        (m) => (m.model_bundle_id || m.id) === p.model_bundle_id,
      )
    )
      $("model").value = p.model_bundle_id;
    const days = C.presetDays(p.period);
    $("date-start").value = days.start;
    $("date-end").value = days.end;
    syncTasks();
    ["af", "bs"].forEach(
      (t) => ($(`task-${t}`).checked = (p.tasks || []).includes(t)),
    );
    if (p.aoi?.bbox && C.validBBox(p.aoi.bbox))
      setAOI(C.bboxGeometry(p.aoi.bbox), "bbox", true);
    else if (p.aoi?.geometry) setAOI(p.aoi.geometry, "geometry", true);
    refreshControls();
  }
  function resetResult() {
    state.resultEpoch++;
    clearLayers();
    clearError();
    state.result = null;
    state.normalized = null;
    $("result-empty").hidden = true;
    $("result-content").hidden = true;
    $("result-notice").hidden = true;
    $("job-progress").hidden = false;
  }
  const jobLabels = {
    queued: "В ОЧЕРЕДИ",
    running: "АНАЛИЗ",
    succeeded: "ГОТОВО",
    failed: "ОШИБКА",
    paused: "ПАУЗА",
  };
  function jobState(status, label) {
    const e = $("job-status");
    e.textContent = label || jobLabels[status] || "ОЖИДАНИЕ";
    e.className = `status-pill ${status === "succeeded" ? "success" : status === "failed" ? "failed" : ["queued", "running"].includes(status) ? "running" : "neutral"}`;
  }
  function progress(job) {
    const p = Math.max(0, Math.min(1, Number(job.progress) || 0)),
      percentage = Math.round(p * 100);
    $("job-progress").hidden = false;
    $("progress-bar").style.width = `${percentage}%`;
    $("progress-percent").textContent = `${percentage}%`;
    document
      .querySelector(".progress-track")
      .setAttribute("aria-valuenow", String(percentage));
    const stageLabels = {
      queued: "Запрос принят. Ожидаем свободный обработчик.",
      running: "Модель обрабатывает спутниковые данные.",
      processing: "Модель обрабатывает спутниковые данные.",
      inference: "Модель ищет очаги и контуры.",
      succeeded: "Результаты готовы.",
      finished: "Результаты готовы.",
    };
    $("progress-text").textContent =
      stageLabels[job.stage] ||
      (job.stage && /[а-я]/i.test(job.stage)
        ? job.stage
        : stageLabels[job.status]) ||
      "Сервис обрабатывает задание…";
    $("step-queued").className = job.status === "queued" ? "active" : "done";
    $("step-running").className =
      job.status === "running"
        ? "active"
        : job.status === "succeeded"
          ? "done"
          : "";
    $("step-result").className = job.status === "succeeded" ? "active" : "";
  }
  function persistPending() {
    storage.set("pending", state.pending);
  }
  function saveHistory(job) {
    if (!job.job_id) return;
    const prev = state.history.find((h) => h.job_id === job.job_id);
    const item = {
      ...prev,
      ...state.context,
      ...job,
      created_at:
        prev?.created_at || job.created_at || new Date().toISOString(),
    };
    state.history = [
      item,
      ...state.history.filter((h) => h.job_id !== job.job_id),
    ].slice(0, 20);
    storage.set("history", state.history);
    updateHistoryCount();
  }
  function updateHistoryCount() {
    $("history-count").textContent = String(state.history.length);
    $("history-count").hidden = !state.history.length;
  }
  function setContext(context) {
    state.context = context;
    const period = context?.payload?.period,
      days = period ? C.presetDays(period) : {};
    $("result-context").textContent =
      `${context?.label || "Территория анализа"} · ${dayLabel(days.start)} — ${dayLabel(days.end)}${context?.training ? " · обучающий пример" : ""}`;
  }
  async function runAnalysis() {
    if (state.busy) return;
    if (state.pending?.retryAt > Date.now()) {
      toast(
        `Сервис просит подождать ${Math.ceil((state.pending.retryAt - Date.now()) / 1000)} с перед повтором.`,
      );
      return;
    }
    if (state.pending?.job) {
      resumeJob(state.pending);
      return;
    }
    if (!state.pending) {
      const issue = validationMessage();
      if (issue) {
        toast(issue);
        return;
      }
      const payload = {
        dataset_id: dataset().dataset_id || dataset().id,
        model_bundle_id: model().model_bundle_id || model().id,
        aoi: state.aoi,
        period: C.inclusivePeriod($("date-start").value, $("date-end").value),
        tasks: selectedTasks(),
      };
      state.pending = {
        key:
          globalThis.crypto?.randomUUID?.() ||
          `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        payload,
        preset_id: state.preset?.preset_id || state.preset?.id || null,
        label: contextLabel(),
        training: C.isTraining(dataset()),
        created_at: new Date().toISOString(),
      };
      persistPending();
    }
    state.busy = true;
    resetResult();
    setContext(state.pending);
    refreshControls();
    setPanel("results-panel", true);
    $("result-title").textContent = "Подготавливаем анализ";
    jobState("queued");
    progress({ status: "queued", progress: 0 });
    try {
      const { data: job, response } = await request("/v1/analyses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": state.pending.key,
        },
        body: JSON.stringify(state.pending.payload),
      });
      if (!job?.job_id)
        throw new Error("Сервис не передал номер задания. Повторите отправку.");
      state.pending.job = {
        ...job,
        status_url:
          job.status_url ||
          response.headers.get("Location") ||
          `/v1/jobs/${encodeURIComponent(job.job_id)}`,
        result_url:
          job.result_url || `/v1/jobs/${encodeURIComponent(job.job_id)}/result`,
      };
      persistPending();
      saveHistory(state.pending.job);
      resumeJob(state.pending);
      state.history = state.history.filter(
        (h) => h.request_key !== state.pending?.key || h.job_id,
      );
      storage.set("history", state.history);
      updateHistoryCount();
    } catch (e) {
      state.busy = false;
      $("job-progress").hidden = true;
      jobState("failed");
      $("result-title").textContent = "Запрос не завершён";
      if (
        e.status &&
        e.status >= 400 &&
        e.status < 500 &&
        ![401, 408, 429].includes(e.status)
      ) {
        state.pending = null;
        persistPending();
      }
      if (state.pending) {
        state.pending.retryAt = e.retryAt || 0;
        persistPending();
      }
      showError(
        errorText(e) +
          (state.pending
            ? " Статус отправки неизвестен; безопасный повтор сохранит исходный запрос."
            : ""),
        () => (state.pending ? runAnalysis() : loadMetadata()),
      );
      if (state.pending) {
        leaveJob.textContent = "Отложить запрос и изменить параметры";
        leaveJob.hidden = false;
      }
      refreshControls();
    }
  }
  function resumeJob(pending) {
    if (!pending?.job?.job_id) return;
    if (pending.retryAt > Date.now()) {
      toast(
        `Сервис просит подождать ${Math.ceil((pending.retryAt - Date.now()) / 1000)} с перед повтором.`,
      );
      return;
    }
    clearTimeout(state.timer);
    const epoch = ++state.pollEpoch;
    state.busy = true;
    state.pending = pending;
    persistPending();
    setContext(pending);
    clearError();
    refreshControls();
    $("result-title").textContent = "Анализ территории";
    $("result-empty").hidden = true;
    progress(pending.job);
    pollJob(epoch, Date.now(), 0);
  }
  async function pollJob(epoch, started, retries) {
    if (epoch !== state.pollEpoch || !state.pending?.job) return;
    const saved = state.pending.job;
    if (Date.now() - started > 20 * 60 * 1000) {
      pausePolling(
        "Анализ занимает больше обычного. Задание сохранено; продолжите проверку его статуса.",
      );
      return;
    }
    try {
      const { data: job, response } = await request(
        saved.status_url || `/v1/jobs/${encodeURIComponent(saved.job_id)}`,
      );
      if (epoch !== state.pollEpoch) return;
      if (
        !job ||
        !["queued", "running", "succeeded", "failed"].includes(job.status)
      )
        throw new Error("Сервис вернул неизвестное состояние задания.");
      state.pending.job = { ...saved, ...job };
      persistPending();
      saveHistory(state.pending.job);
      jobState(job.status);
      progress(job);
      if (job.status === "failed") {
        state.pending = null;
        persistPending();
        state.busy = false;
        $("job-progress").hidden = true;
        $("result-title").textContent = "Анализ прерван";
        showError(errorText(job));
        refreshControls();
        return;
      }
      if (job.status === "succeeded") {
        const { data: result } = await request(
          job.result_url || saved.result_url,
        );
        if (epoch !== state.pollEpoch) return;
        await renderResult(result);
        state.pending = null;
        persistPending();
        state.busy = false;
        refreshControls();
        return;
      }
      const delay = Math.max(
        1200,
        Math.min(
          10000,
          (Number(response.headers.get("Retry-After")) || 2) * 1000,
        ),
      );
      state.timer = setTimeout(() => pollJob(epoch, started, 0), delay);
    } catch (e) {
      if (epoch !== state.pollEpoch) return;
      state.pending.retryAt = e.retryAt || 0;
      persistPending();
      if (e.status === 404) {
        saveHistory({ ...saved, status: "failed" });
        state.pending = null;
        persistPending();
        state.busy = false;
        $("job-progress").hidden = true;
        $("result-title").textContent = "Задание недоступно";
        jobState("failed");
        showError(errorText(e));
        refreshControls();
        return;
      }
      if (retries < 3 && (!e.status || e.status >= 500 || e.status === 429)) {
        $("progress-text").textContent =
          "Связь прервалась. Восстанавливаем получение статуса…";
        state.timer = setTimeout(
          () => pollJob(epoch, started, retries + 1),
          Math.max(2000 * (retries + 1), (e.retryAt || 0) - Date.now()),
        );
        return;
      }
      pausePolling(errorText(e));
    }
  }
  function pausePolling(message) {
    state.busy = false;
    jobState("paused");
    $("progress-text").textContent = "Получение статуса приостановлено";
    showError(message, () => resumeJob(state.pending));
    leaveJob.textContent = "Вернуться к параметрам";
    leaveJob.hidden = !state.pending?.job;
    refreshControls();
  }
  function humanWarning(w) {
    if (w === "no scenes intersect AOI and period")
      return "В выбранной области и периоде нет подходящих сцен.";
    if (w === "all selected observations are invalid")
      return "Все выбранные наблюдения непригодны для оценки.";
    return w;
  }
  async function renderResult(raw) {
    state.result = raw;
    const n = C.normalizeResult(raw);
    state.normalized = n;
    const epoch = ++state.resultEpoch;
    clearLayers();
    $("job-progress").hidden = true;
    $("result-empty").hidden = true;
    $("result-content").hidden = false;
    const noData = n.status === "no_data",
      unavailable = n.status === "model_unavailable";
    $("result-title").textContent = noData
      ? "Нет пригодных наблюдений"
      : unavailable
        ? "Модель недоступна"
        : "Территория проанализирована";
    jobState(
      unavailable ? "failed" : "succeeded",
      noData ? "НЕТ ДАННЫХ" : n.status === "partial" ? "ЧАСТИЧНО" : undefined,
    );
    $("result-notice").hidden = !(
      noData ||
      unavailable ||
      n.status === "partial"
    );
    $("result-notice").textContent = noData
      ? "Для этой области и периода нет пригодных наблюдений. Это не означает отсутствие пожара. Попробуйте готовый пример или другой период."
      : unavailable
        ? "Модель не смогла выполнить анализ. Обновите её статус и повторите запуск."
        : "Данные покрывают территорию частично. Площади и обнаружения относятся только к наблюдаемой части.";
    $("hotspot-count").textContent = fmt(n.af.count, 0);
    $("burn-area").textContent = fmt(n.bs.area);
    const tasks = state.context?.payload?.tasks || raw.tasks || [];
    $("hotspot-count").closest("article").hidden = !tasks.includes("af");
    $("burn-area").closest("article").hidden = !tasks.includes("bs");
    document
      .querySelector(".metrics")
      .classList.toggle("single-metric", tasks.length === 1);
    document.querySelector(".severity-card").hidden = !tasks.includes("bs");
    ["af", "bs"].forEach(
      (t) => ($(`layer-${t}`).closest("label").hidden = !tasks.includes(t)),
    );
    document.querySelector(".severity-keys").hidden = !tasks.includes("bs");
    document.querySelector(".opacity-label").hidden = !tasks.includes("bs");
    $("layer-opacity").hidden = !tasks.includes("bs");
    $("af-metric-note").textContent = !tasks.includes("af")
      ? "модуль не выбран"
      : n.af.count === null
        ? "нет пригодных данных"
        : "обнаружений AF";
    $("bs-metric-note").textContent = !tasks.includes("bs")
      ? "модуль не выбран"
      : n.bs.area === null
        ? "нет пригодных данных"
        : "гектаров";
    const values = ["1", "2", "3"].map((k) => n.bs.severity[k]),
      total = values.reduce((a, b) => a + (b || 0), 0);
    values.forEach((v, i) => {
      $(`sev${i + 1}`).textContent = fmt(v);
      document.querySelector(`.severity-stack .s${i + 1}`).style.width =
        `${total && v != null ? (v / total) * 100 : 0}%`;
    });
    $("severity-note").textContent =
      n.bs.area === null
        ? "Нет данных для оценки повреждений."
        : "Площади рассчитаны сервисом по наблюдаемым данным.";
    $("quality-list").innerHTML =
      tasks
        .map((t) => {
          const q = n.quality[t];
          const percent =
            q.coverage === null
              ? null
              : Math.max(0, Math.min(100, q.coverage * 100));
          return `<div class="quality-row"><span>${t === "af" ? "Активные пожары" : "Выгоревшая территория"}</span><b>${percent === null ? "—" : `${fmt(percent)}%`}</b><div class="quality-line"><i style="width:${percent ?? 0}%"></i></div><small>${q.observedArea === null ? "Площадь наблюдения не указана" : `${fmt(q.observedArea)} га наблюдаемо`}${q.sceneCount === null ? "" : ` · сцен: ${fmt(q.sceneCount, 0)}`}</small></div>`;
        })
        .join("") || '<p class="field-note">Покрытие не указано сервисом.</p>';
    const warnings = [...new Set(n.warnings.map(humanWarning))];
    if (state.context?.training)
      warnings.unshift(
        "Обучающий пример: сцена из открытого train-набора. Показ не является независимой тестовой оценкой.",
      );
    $("warnings").hidden = !warnings.length;
    $("warnings-list").innerHTML = warnings
      .map((w) => `<li>${esc(w)}</li>`)
      .join("");
    renderDownloads(
      n.artifacts.filter((a) =>
        a.id === "hotspots"
          ? tasks.includes("af")
          : ["burn_polygons", "contours"].includes(a.id)
            ? tasks.includes("bs")
            : true,
      ),
    );
    $("empty-map").hidden = true;
    $("map-context").textContent = noData
      ? "Нет пригодных наблюдений"
      : `${tasks.map((t) => t.toUpperCase()).join(" + ")} · ${n.status === "partial" ? "частичное покрытие" : "анализ завершён"}`;
    const af =
        tasks.includes("af") && n.artifacts.find((a) => a.id === "hotspots"),
      bs =
        tasks.includes("bs") &&
        n.artifacts.find((a) => ["burn_polygons", "contours"].includes(a.id));
    const results = await Promise.allSettled([
      af ? loadLayer(af, "af", epoch) : null,
      bs ? loadLayer(bs, "bs", epoch) : null,
    ]);
    if (epoch !== state.resultEpoch) return;
    const failures = results.filter((r) => r.status === "rejected");
    if (failures.length)
      showError(
        "Часть слоёв карты не загрузилась. Метрики сохранены. " +
          errorText(failures[0].reason),
        () => renderResult(raw),
      );
    if (state.drawLayer) fit(state.drawLayer);
    $("legend").hidden = !Object.keys(state.resultLayers).length;
    $("layer-af-count").textContent = fmt(n.af.count, 0);
    if (!noData && !unavailable)
      toast("Анализ готов. Результаты и выгрузки доступны.");
  }
  async function loadLayer(artifact, type, epoch) {
    const { data } = await request(artifact.href);
    if (epoch !== state.resultEpoch) return;
    if (data?.type !== "FeatureCollection" || !Array.isArray(data.features))
      throw new Error("Сервис вернул некорректный GeoJSON.");
    const opacity = Number($("layer-opacity").value) / 100;
    const layer = L.geoJSON(data, {
      pointToLayer: (_, latlng) =>
        L.circleMarker(latlng, {
          radius: 4,
          color: "#ffd5b7",
          weight: 1,
          fillColor: "#ff783e",
          fillOpacity: 0.85,
          bubblingMouseEvents: false,
        }),
      style: (f) => ({
        color:
          { 1: "#eed076", 2: "#f29947", 3: "#ee6455" }[
            f.properties?.class_id
          ] || "#ed9651",
        weight: 1.2,
        fillOpacity: opacity,
        bubblingMouseEvents: false,
      }),
      onEachFeature: (f, l) => {
        const p = f.properties || {};
        const title =
          type === "af"
            ? "Термоточка"
            : {
                1: "Слабое повреждение",
                2: "Среднее повреждение",
                3: "Сильное повреждение",
              }[p.class_id] || "Выгоревшая территория";
        l.bindPopup(
          `<span class="popup-label">${type === "af" ? "АКТИВНЫЙ ПОЖАР" : "КОНТУР ГАРИ"}</span><b>${esc(title)}</b>${p.area_ha != null ? `<br>Площадь: ${fmt(Number(p.area_ha))} га` : ""}${p.observed_at ? `<br>${esc(new Date(p.observed_at).toLocaleString("ru-RU", { timeZone: "UTC" }))} UTC` : ""}${p.date_post ? `<br>Снимок после: ${dayLabel(p.date_post)}` : ""}`,
        );
      },
    });
    state.resultLayers[type] = layer;
    $(`layer-${type}`).checked = true;
    layer.addTo(map);
  }
  const exportLabels = {
    hotspots: ["Термоточки", "GeoJSON · точки"],
    burn_polygons: ["Контуры выгоревшей территории", "GeoJSON · полигоны"],
    contours: ["Контуры выгоревшей территории", "GeoJSON · полигоны"],
    summary: ["Сводка анализа", "JSON · площади и покрытие"],
    provenance: ["Источники и модель", "JSON · происхождение данных"],
    pixel_rles: ["Пиксельные маски", "JSON · RLE"],
  };
  function artifactFilename(a) {
    return (
      a.filename?.split(/[\\/]/).at(-1) ||
      `${a.id}.${a.id.startsWith("mask_") ? "tif" : ["hotspots", "burn_polygons", "contours"].includes(a.id) ? "geojson" : "json"}`
    );
  }
  function renderDownloads(artifacts) {
    const primary = artifacts.filter((a) =>
        ["hotspots", "burn_polygons", "contours", "summary"].includes(a.id),
      ),
      extra = artifacts.filter((a) => !primary.includes(a));
    const item = (a) => {
      const label = exportLabels[a.id] || [
        a.id.startsWith("mask_") ? "Растровая маска" : a.id,
        a.id.startsWith("mask_")
          ? "GeoTIFF · " + a.id.slice(5)
          : "Файл результата",
      ];
      let href;
      try {
        href = safeURL(a.href);
      } catch {
        return "";
      }
      return `<a class="download-link" href="${esc(href)}" data-artifact="${esc(a.id)}" download="${esc(artifactFilename(a))}">${icon("report")}<span>${esc(label[0])}<small>${esc(label[1])}</small></span>${icon("download")}</a>`;
    };
    $("download-list").innerHTML =
      primary.map(item).join("") +
        (extra.length
          ? `<details><summary>Дополнительные файлы · ${extra.length}</summary>${extra.map(item).join("")}</details>`
          : "") ||
      '<p class="field-note">Для этого анализа файлы не сформированы.</p>';
    $("download-list")
      .querySelectorAll("a[data-artifact]")
      .forEach((a) =>
        a.addEventListener("click", (e) => {
          e.preventDefault();
          const artifact = artifacts.find((x) => x.id === a.dataset.artifact);
          downloadArtifact(artifact);
        }),
      );
  }
  function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  async function downloadArtifact(a) {
    const controller = new AbortController(),
      timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch(safeURL(a.href), {
        headers: apiHeaders(),
        redirect: "error",
        signal: controller.signal,
      });
      if (!response.ok)
        throw new Error(`Не удалось скачать файл (HTTP ${response.status}).`);
      saveBlob(await response.blob(), artifactFilename(a));
      toast("Файл передан браузеру для скачивания.");
    } catch (e) {
      showError(errorText(e), () => downloadArtifact(a));
    } finally {
      clearTimeout(timeout);
    }
  }
  function openHistory() {
    $("history-list").innerHTML = state.history.length
      ? state.history
          .map(
            (h, i) =>
              `<button class="history-item" data-history="${i}" ${state.busy || (state.pending && !state.pending.job) ? "disabled" : ""}>${icon("map")}<span><b>${esc(h.label || h.payload?.dataset_id || h.job_id)}</b><small>${esc(new Date(h.created_at).toLocaleString("ru-RU"))} · ${(h.payload?.tasks || []).join(" + ").toUpperCase()}</small></span><span class="status-pill ${h.status === "succeeded" ? "success" : h.status === "failed" ? "failed" : "neutral"}">${h.request_key && !h.job_id ? "ОТПРАВКА" : jobLabels[h.status] || "СОХРАНЁН"}</span></button>`,
          )
          .join("")
      : '<div class="history-empty">Здесь появятся ваши анализы.<br>Начните с готовой территории в каталоге.</div>';
    $("history-list")
      .querySelectorAll("[data-history]")
      .forEach((b) =>
        b.addEventListener("click", () => {
          if (state.busy) return;
          const h = state.history[Number(b.dataset.history)];
          if (!h?.payload) return;
          $("history-dialog").close();
          resetResult();
          restoreForm(h);
          if (!h.job_id && h.request_key) {
            state.pending = { ...h, key: h.request_key };
            persistPending();
            setContext(state.pending);
            state.busy = false;
            $("job-progress").hidden = true;
            $("result-title").textContent = "Запрос ожидает отправки";
            jobState("paused");
            showError(
              "Ответ на отправку не получен. Повтор использует сохранённый ключ запроса.",
              runAnalysis,
            );
            refreshControls();
            setPanel("results-panel", true);
            return;
          }
          resumeJob({
            ...h,
            job: {
              job_id: h.job_id,
              status: h.status,
              status_url:
                h.status_url || `/v1/jobs/${encodeURIComponent(h.job_id)}`,
              result_url:
                h.result_url ||
                `/v1/jobs/${encodeURIComponent(h.job_id)}/result`,
            },
          });
          setPanel("results-panel", true);
        }),
      );
    $("history-dialog").showModal();
  }
  function reportDocument() {
    return {
      generated_at: new Date().toISOString(),
      training_example: !!state.context?.training,
      request: state.context?.payload,
      result: state.result,
    };
  }
  function openReport() {
    if (!state.result) return;
    const n = state.normalized,
      p = state.context?.payload,
      days = C.presetDays(p?.period);
    const rows = [
      ["Термоточки", fmt(n.af.count, 0)],
      ["Площадь гари, га", fmt(n.bs.area)],
      ["Слабое повреждение, га", fmt(n.bs.severity["1"])],
      ["Среднее повреждение, га", fmt(n.bs.severity["2"])],
      ["Сильное повреждение, га", fmt(n.bs.severity["3"])],
    ];
    const coords =
      p?.aoi?.bbox?.map((x) => fmt(x, 5)).join(" / ") || "Произвольный полигон";
    $("report-body").innerHTML =
      `<dl class="report-meta"><dt>Территория</dt><dd>${esc(state.context?.label)}</dd><dt>Период UTC</dt><dd>${dayLabel(days.start)} — ${dayLabel(days.end)}</dd><dt>Модель</dt><dd>${esc(state.result.model_bundle_id || p?.model_bundle_id)}</dd><dt>Координаты</dt><dd>${esc(coords)}</dd><dt>Задание</dt><dd>${esc(state.result.job_id)}</dd><dt>Покрытие AF / BS</dt><dd>${n.quality.af.coverage === null ? "—" : fmt(n.quality.af.coverage * 100) + "%"} / ${n.quality.bs.coverage === null ? "—" : fmt(n.quality.bs.coverage * 100) + "%"}</dd></dl><table class="report-table"><thead><tr><th>Показатель</th><th>Результат</th></tr></thead><tbody>${rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("")}</tbody></table><p class="report-limitations">${state.context?.training ? "Обучающий пример из открытого train-набора. Не независимая оценка качества модели. " : ""}${n.status === "no_data" ? "Пригодных наблюдений нет. Отсутствие данных не означает отсутствие пожара. " : n.status === "partial" ? "Территория покрыта наблюдениями частично. " : ""}Площади рассчитаны сервисом. «—» означает отсутствие значения. Результат требует проверки специалистом.</p>`;
    $("report-dialog").showModal();
  }
  const bind = (id, fn) => $(id).addEventListener("click", fn);
  bind("run-analysis", runAnalysis);
  bind("refresh-data", () => loadMetadata());
  bind("draw-rect", () => drawing(state.drawMode === "rect" ? null : "rect"));
  bind("draw-poly", () => drawing(state.drawMode === "poly" ? null : "poly"));
  bind("cancel-draw", () => drawing(null));
  bind("finish-polygon", finishPolygon);
  bind("clear-aoi", clearAOI);
  bind("apply-bbox", applyBBox);
  bind("use-coverage", useCoverage);
  bind("zoom-coverage", () => fit(state.coverage));
  bind("zoom-aoi", () => fit(state.drawLayer));
  bind("zoom-in", () => map.zoomIn());
  bind("zoom-out", () => map.zoomOut());
  bind("open-history", openHistory);
  bind("open-help", () => $("help-dialog").showModal());
  bind("nav-map", () => setPanel("map-section", true));
  bind("open-report", openReport);
  bind("print-report", () => window.print());
  bind("download-report", () =>
    saveBlob(
      new Blob([JSON.stringify(reportDocument(), null, 2)], {
        type: "application/json",
      }),
      `firewatch-${state.result?.job_id || "report"}.json`,
    ),
  );
  bind("retry-action", () => state.retry?.());
  bind("apply-token", async () => {
    accessToken = $("token").value.trim();
    try {
      sessionStorage.setItem("firewatch-token", accessToken);
    } catch {}
    clearError();
    await loadMetadata();
    if (state.metadataLoaded && state.result) await renderResult(state.result);
    else if (state.metadataLoaded && state.pending?.job && !state.busy)
      resumeJob(state.pending);
  });
  bind("leave-job", () => {
    clearTimeout(state.timer);
    state.pollEpoch++;
    const uncertain = state.pending && !state.pending.job;
    if (uncertain) {
      const saved = {
        ...state.pending,
        request_key: state.pending.key,
        status: "uncertain",
      };
      state.history = [
        saved,
        ...state.history.filter((h) => h.request_key !== saved.request_key),
      ].slice(0, 20);
      storage.set("history", state.history);
      updateHistoryCount();
    }
    state.pending = null;
    persistPending();
    state.busy = false;
    clearError();
    $("job-progress").hidden = true;
    $("result-title").textContent = "Запрос сохранён в истории";
    $("result-context").textContent = uncertain
      ? "Сервис мог принять запрос. Его безопасный повтор доступен в истории. Новый анализ создаст отдельное задание."
      : "Проверка в браузере остановлена. Выполнение на сервере продолжается.";
    refreshControls();
    setPanel("parameters", true);
    toast("Запрос сохранён в истории. Его можно открыть позднее.");
  });
  $("token").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("apply-token").click();
  });
  $("dataset").addEventListener("change", () => {
    invalidateResult();
    applyDataset(true);
  });
  $("preset").addEventListener("change", () => {
    invalidateResult();
    applyPreset(true);
  });
  $("model").addEventListener("change", () => {
    invalidateResult();
    syncTasks(true);
    refreshControls();
  });
  ["date-start", "date-end", "task-af", "task-bs"].forEach((id) =>
    $(id).addEventListener("change", () => {
      invalidateResult();
      refreshControls();
    }),
  );
  ["af", "bs"].forEach((t) =>
    $(`layer-${t}`).addEventListener("change", (e) => {
      const layer = state.resultLayers[t];
      if (layer) e.target.checked ? layer.addTo(map) : map.removeLayer(layer);
    }),
  );
  $("layer-opacity").addEventListener("input", (e) => {
    $("opacity-value").textContent = `${e.target.value}%`;
    state.resultLayers.bs?.setStyle({
      fillOpacity: Number(e.target.value) / 100,
    });
  });
  bind("toggle-legend", () => {
    const hidden = !$("legend-body").hidden;
    $("legend-body").hidden = hidden;
    $("toggle-legend").textContent = hidden ? "+" : "−";
    $("toggle-legend").setAttribute("aria-expanded", String(!hidden));
  });
  ["dark", "light"].forEach((theme) =>
    bind(`map-${theme}`, () => {
      mapProvider?.setTheme?.(theme === "dark" ? "satellite" : "roadmap");
      $("map").classList.toggle("dark-map", theme === "dark");
      $("map-section").classList.toggle("light-theme", theme === "light");
      ["dark", "light"].forEach((t) => {
        $(`map-${t}`).classList.toggle("active", t === theme);
        $(`map-${t}`).setAttribute("aria-pressed", String(t === theme));
      });
    }),
  );
  document
    .querySelectorAll("[data-panel]")
    .forEach((b) =>
      b.addEventListener("click", () => setPanel(b.dataset.panel)),
    );
  document
    .querySelectorAll("[data-close-dialog]")
    .forEach((b) =>
      b.addEventListener("click", () => b.closest("dialog").close()),
    );
  document.querySelectorAll("dialog").forEach((d) =>
    d.addEventListener("click", (e) => {
      if (e.target === d) {
        const r = d.getBoundingClientRect();
        if (
          e.clientX < r.left ||
          e.clientX > r.right ||
          e.clientY < r.top ||
          e.clientY > r.bottom
        )
          d.close();
      }
    }),
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.drawMode) drawing(null);
  });
  new ResizeObserver(() => map.invalidateSize()).observe($("map-section"));
  updateHistoryCount();
  $("token").value = accessToken;
  loadMetadata(true).then(() => {
    if (state.pending?.payload) {
      const a = state.pending.payload.aoi;
      if (a?.bbox && C.validBBox(a.bbox))
        setAOI(C.bboxGeometry(a.bbox), "bbox", true);
      else if (a?.geometry) setAOI(a.geometry, "geometry", true);
      if (state.pending.job) {
        resetResult();
        resumeJob(state.pending);
      } else {
        setContext(state.pending);
        showError(
          "Ответ на предыдущий запрос не получен. Нажмите «Повторить отправку»: ключ запроса сохранён.",
          runAnalysis,
        );
        refreshControls();
      }
    }
  });
})().catch((error) => {
  document.getElementById("error-box").hidden = false;
  document.getElementById("error-message").textContent = "Не удалось запустить карту. Обновите страницу.";
  console.error("FireWatch initialization failed", error);
});
