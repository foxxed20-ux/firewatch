const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../../web/core.js");

test("inclusive period sends the whole ending UTC day", () => {
  assert.deepEqual(core.inclusivePeriod("2026-09-01", "2026-09-03"), {
    start: "2026-09-01T00:00:00.000Z",
    end: "2026-09-04T00:00:00.000Z",
  });
  assert.throws(
    () => core.inclusivePeriod("2026-09-04", "2026-09-03"),
    /Дата начала/,
  );
  assert.deepEqual(
    core.presetDays({
      start: "2026-09-01T00:00:00Z",
      end: "2026-09-04T00:00:00Z",
    }),
    { start: "2026-09-01", end: "2026-09-03" },
  );
});

test("AOI helpers reject invalid boxes and crossing polygons", () => {
  assert.equal(core.validBBox([30, 50, 31, 51]), true);
  assert.equal(core.validBBox([31, 50, 30, 51]), false);
  assert.deepEqual(
    core.bboxGeometry([30, 50, 31, 51]).coordinates[0][0],
    [30, 50],
  );
  assert.equal(
    core.validateRing([
      [0, 0],
      [2, 2],
      [0, 2],
      [2, 0],
    ]),
    "Контур не должен пересекать сам себя.",
  );
  assert.equal(
    core.validateRing([
      [0, 0],
      [2, 0],
      [0, 2],
    ]),
    null,
  );
});

test("normalizer preserves zero values and makes no-data metrics nullable", () => {
  const complete = core.normalizeResult({
    modules: {
      af: { hotspot_count: 0 },
      bs: { burn_area_ha: 0, severity_area_ha: { 1: 0, 2: 2 } },
    },
    quality: {
      af: { coverage_fraction: 0, observed_area_ha: 0, observation_count: 0 },
    },
  });
  assert.equal(complete.af.count, 0);
  assert.equal(complete.bs.severity["1"], 0);
  assert.equal(complete.quality.af.coverage, 0);
  const empty = core.normalizeResult({
    data_status: "no_data",
    modules: { af: { hotspot_count: 4 }, bs: { burn_area_ha: 8 } },
  });
  assert.equal(empty.af.count, null);
  assert.equal(empty.bs.area, null);
});

test("model readiness and task list accept API forms", () => {
  assert.equal(core.modelReady({ status: "ready" }), true);
  assert.equal(core.modelReady({ status: "available" }), false);
  assert.deepEqual(core.taskList({ af: true, bs: "ready", extra: true }), [
    "af",
    "bs",
  ]);
});
