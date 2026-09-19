const test = require("node:test");
const assert = require("node:assert/strict");
const { searchPayload, validDay, cloudLabel, errorMessage } = require("../../web/imagery.js");
const aoi = { type: "Polygon", coordinates: [[[37, 55], [37.1, 55], [37.1, 55.1], [37, 55.1], [37, 55]]] };

test("scene search preserves polygon and inclusive date contract", () => {
  assert.deepEqual(searchPayload(aoi, "2024-01-01", "2024-12-31", "30"), {
    aoi, date_start: "2024-01-01", date_end: "2024-12-31", cloud_max: 30, limit: 20,
  });
  assert.equal(searchPayload(aoi, "2024-07-13", "2024-07-13", 0).cloud_max, 0);
});
test("invalid dates, absent AOI and excessive periods never reach API", () => {
  assert.equal(validDay("2024-02-30"), false);
  assert.throws(() => searchPayload(null, "2024-07-01", "2024-07-31", 30), /область/);
  assert.throws(() => searchPayload(aoi, "2024-07-02", "2024-07-01", 30), /Дата начала/);
  assert.throws(() => searchPayload(aoi, "2024-01-01", "2025-01-01", 30), /366/);
  assert.throws(() => searchPayload(aoi, "2016-01-01", "2016-01-02", 30), /2017/);
  assert.throws(() => searchPayload(aoi, "2024-01-01", "2024-01-02", ""), /Облачность/);
});
test("cloud estimate absence is distinct from zero", () => {
  assert.equal(cloudLabel(null), "нет оценки");
  assert.equal(cloudLabel(0), "0%");
});
test("configuration, auth, rate limit and timeout errors are actionable", () => {
  assert.match(errorMessage({ body: { error: { code: "GOOGLE_NOT_CONFIGURED" } } }), /не настроен/);
  assert.match(errorMessage({ body: { error: { code: "UNAUTHORIZED" } } }), /токен/);
  assert.match(errorMessage({ status: 429 }), /минуту/);
  assert.match(errorMessage({ name: "AbortError" }), /меньшую область/);
});
