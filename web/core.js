(function (root, factory) {
  "use strict";
  var api = factory();
  root.FireWatchCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var DAY = 24 * 60 * 60 * 1000;
  var TASKS = ["af", "bs"];
  var STATUSES = ["complete", "partial", "no_data", "model_unavailable"];

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function dateInput(value) {
    if (value === null || value === undefined || value === "") return "";
    if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)) {
      var parts = value.split("-").map(Number);
      var date = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
      return date.getUTCFullYear() === parts[0] &&
        date.getUTCMonth() === parts[1] - 1 &&
        date.getUTCDate() === parts[2]
        ? value
        : "";
    }
    var parsed =
      value instanceof Date ? new Date(value.valueOf()) : new Date(value);
    return Number.isNaN(parsed.valueOf())
      ? ""
      : parsed.toISOString().slice(0, 10);
  }

  function inclusivePeriod(startDay, endDay) {
    var start = dateInput(startDay);
    var end = dateInput(endDay);
    if (!start || !end) throw new Error("Укажите корректные даты периода.");
    var startMs = Date.parse(start + "T00:00:00.000Z");
    var endMs = Date.parse(end + "T00:00:00.000Z");
    if (startMs > endMs)
      throw new Error("Дата начала не может быть позже даты окончания.");
    return {
      start: new Date(startMs).toISOString(),
      end: new Date(endMs + DAY).toISOString(),
    };
  }

  function presetDays(period) {
    period = period || {};
    var endMs = Date.parse(period.end);
    return {
      start: dateInput(period.start),
      end: Number.isNaN(endMs) ? "" : dateInput(new Date(endMs - 1)),
    };
  }

  function validBBox(values) {
    return (
      Array.isArray(values) &&
      values.length === 4 &&
      values.every(function (v) {
        return typeof v === "number" && Number.isFinite(v);
      }) &&
      values[0] >= -180 &&
      values[0] < values[2] &&
      values[2] <= 180 &&
      values[1] >= -90 &&
      values[1] < values[3] &&
      values[3] <= 90
    );
  }

  function bboxGeometry(values) {
    if (!validBBox(values))
      throw new Error("Некорректный прямоугольник WGS84.");
    var w = values[0],
      s = values[1],
      e = values[2],
      n = values[3];
    return {
      type: "Polygon",
      coordinates: [
        [
          [w, s],
          [e, s],
          [e, n],
          [w, n],
          [w, s],
        ],
      ],
    };
  }

  function samePoint(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }
  function cross(a, b, c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  }
  function onSegment(a, b, p) {
    return (
      Math.min(a[0], b[0]) <= p[0] &&
      p[0] <= Math.max(a[0], b[0]) &&
      Math.min(a[1], b[1]) <= p[1] &&
      p[1] <= Math.max(a[1], b[1])
    );
  }
  function segmentsIntersect(a, b, c, d) {
    var c1 = cross(a, b, c),
      c2 = cross(a, b, d),
      c3 = cross(c, d, a),
      c4 = cross(c, d, b);
    if (
      ((c1 > 0 && c2 < 0) || (c1 < 0 && c2 > 0)) &&
      ((c3 > 0 && c4 < 0) || (c3 < 0 && c4 > 0))
    )
      return true;
    return (
      (c1 === 0 && onSegment(a, b, c)) ||
      (c2 === 0 && onSegment(a, b, d)) ||
      (c3 === 0 && onSegment(c, d, a)) ||
      (c4 === 0 && onSegment(c, d, b))
    );
  }

  function validateRing(points) {
    if (!Array.isArray(points) || points.length < 3)
      return "Контур должен содержать минимум три вершины.";
    if (
      points.some(function (p) {
        return (
          !Array.isArray(p) ||
          p.length < 2 ||
          !Number.isFinite(p[0]) ||
          !Number.isFinite(p[1]) ||
          p[0] < -180 ||
          p[0] > 180 ||
          p[1] < -90 ||
          p[1] > 90
        );
      })
    )
      return "Координаты контура должны быть конечными значениями WGS84.";
    if (samePoint(points[0], points[points.length - 1]))
      return "Передавайте контур без повторной замыкающей вершины.";
    var unique = [];
    points.forEach(function (p) {
      if (
        !unique.some(function (v) {
          return samePoint(v, p);
        })
      )
        unique.push(p);
    });
    if (unique.length < 3)
      return "Контур должен содержать минимум три разные вершины.";
    for (var a = 0; a < points.length; a += 1)
      for (var b = a + 1; b < points.length; b += 1) {
        if (b === a + 1 || (a === 0 && b === points.length - 1)) continue;
        if (
          segmentsIntersect(
            points[a],
            points[(a + 1) % points.length],
            points[b],
            points[(b + 1) % points.length],
          )
        )
          return "Контур не должен пересекать сам себя.";
      }
    var twiceArea = 0;
    for (var i = 0; i < points.length; i += 1) {
      var next = points[(i + 1) % points.length];
      twiceArea += points[i][0] * next[1] - next[0] * points[i][1];
    }
    if (twiceArea === 0) return "Площадь контура должна быть больше нуля.";
    return null;
  }

  function modelReady(model) {
    return !!model && (model.ready === true || model.status === "ready");
  }

  function taskList(value) {
    var source = Array.isArray(value)
      ? value
      : value && typeof value === "object"
        ? Object.keys(value).filter(function (key) {
            return value[key] !== false && value[key] != null;
          })
        : [];
    return TASKS.filter(function (task) {
      return source.indexOf(task) !== -1;
    });
  }

  function isTraining(dataset) {
    var seen = [];
    function walk(value, key, sourceContext) {
      var explicit =
        sourceContext ||
        /(?:source|dataset|split|provenance|mode|type)/i.test(key || "");
      if (value && typeof value === "object") {
        if (seen.indexOf(value) !== -1) return false;
        seen.push(value);
        return Object.keys(value).some(function (child) {
          return walk(value[child], child, explicit);
        });
      }
      return (
        typeof value === "string" &&
        explicit &&
        /\b(train(?:ing)?|validation|val)\b/i.test(value)
      );
    }
    return !!dataset && walk(dataset, "source");
  }

  function resultStatus(result) {
    var raw = result && (result.data_status || result.status || result.code);
    if (
      raw === "unavailable" ||
      raw === "model_unavailable" ||
      raw === "MODEL_NOT_READY"
    )
      return "model_unavailable";
    return STATUSES.indexOf(raw) >= 0 ? raw : "complete";
  }
  function moduleStatus(module, overall) {
    if (overall === "model_unavailable") return "model_unavailable";
    var raw = module && (module.data_status || module.status);
    return STATUSES.indexOf(raw) >= 0 ? raw : module ? overall : "no_data";
  }
  function nullableMetric(value, status) {
    return status === "no_data" || status === "model_unavailable"
      ? null
      : finiteNumber(value);
  }
  function qualityItem(raw) {
    raw = raw || {};
    return {
      coverage: finiteNumber(
        raw.coverage_fraction !== undefined
          ? raw.coverage_fraction
          : raw.coverage,
      ),
      observedArea: finiteNumber(
        raw.observed_area_ha !== undefined
          ? raw.observed_area_ha
          : raw.observed_area,
      ),
      sceneCount: finiteNumber(
        raw.observation_count !== undefined
          ? raw.observation_count
          : raw.scene_count,
      ),
    };
  }
  function normalizeArtifacts(raw) {
    var entries = Array.isArray(raw)
      ? raw
      : Object.keys(raw || {}).map(function (id) {
          var item = raw[id];
          return typeof item === "string"
            ? { id: id, href: item }
            : Object.assign({ id: id }, item || {});
        });
    return entries
      .filter(function (item) {
        return (
          item && typeof item.id === "string" && typeof item.href === "string"
        );
      })
      .map(function (item) {
        var output = { id: item.id, href: item.href };
        if (typeof item.filename === "string") output.filename = item.filename;
        if (typeof item.media_type === "string")
          output.media_type = item.media_type;
        return output;
      });
  }
  function normalizeResult(input) {
    var result =
      input && input.result && !input.modules ? input.result : input || {};
    var overall = resultStatus(result);
    var modules = result.modules || {};
    var afRaw = modules.af || result.af || null,
      bsRaw = modules.bs || result.bs || null;
    var afStatus = moduleStatus(afRaw, overall),
      bsStatus = moduleStatus(bsRaw, overall);
    var severityRaw =
      (bsRaw && (bsRaw.severity_area_ha || bsRaw.area_by_severity_ha)) || {};
    var quality = result.quality || {};
    var warnings = [];
    [result.warnings, quality.warnings].forEach(function (items) {
      (Array.isArray(items) ? items : []).forEach(function (item) {
        warnings.push(
          typeof item === "string"
            ? item
            : String((item && (item.message || item.code)) || "Предупреждение"),
        );
      });
    });
    return {
      status: overall,
      af: {
        count: nullableMetric(
          afRaw &&
            (afRaw.hotspot_count !== undefined
              ? afRaw.hotspot_count
              : afRaw.count),
          afStatus,
        ),
        status: afStatus,
      },
      bs: {
        area: nullableMetric(
          bsRaw &&
            (bsRaw.burn_area_ha !== undefined
              ? bsRaw.burn_area_ha
              : bsRaw.area_ha),
          bsStatus,
        ),
        severity: {
          1: nullableMetric(severityRaw["1"], bsStatus),
          2: nullableMetric(severityRaw["2"], bsStatus),
          3: nullableMetric(severityRaw["3"], bsStatus),
        },
        status: bsStatus,
      },
      quality: { af: qualityItem(quality.af), bs: qualityItem(quality.bs) },
      warnings: warnings,
      artifacts: normalizeArtifacts(
        result.artifacts || (input && input.artifacts),
      ),
    };
  }

  return {
    dateInput: dateInput,
    inclusivePeriod: inclusivePeriod,
    presetDays: presetDays,
    validBBox: validBBox,
    bboxGeometry: bboxGeometry,
    validateRing: validateRing,
    modelReady: modelReady,
    taskList: taskList,
    isTraining: isTraining,
    normalizeResult: normalizeResult,
  };
});
