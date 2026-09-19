"""Small deterministic HTTP fixture server for browser QA.

Run from the repository root:
    python tests/frontend/fixture_server.py

Open http://127.0.0.1:8091/?scenario=success.  Available scenarios are:
success, partial, no_data, failed, auth, unavailable, post_retry, missing_job.
The selected scenario is held in the fw-scenario SameSite=Strict cookie.  All
catalogue entries are deliberately labelled ``Тестовая сцена QA``.
"""
from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
REQUEST_LOG = ROOT / "artifacts" / "frontend" / "fixture-requests.jsonl"
HOST, PORT = "127.0.0.1", 8091
SCENARIOS = {"success", "partial", "no_data", "failed", "auth", "unavailable", "post_retry", "missing_job"}
QA_TOKEN = "qa-test-token"
JOB_ID = "qa-job-001"
state = {"retry_key": None, "jobs": {}, "polls": {}}


def quality(fraction: float, observed: float, scenes: int) -> dict:
    return {"coverage_fraction": fraction, "observed_area_ha": observed, "observation_count": scenes}


def result_for(scenario: str, tasks: list[str] | None = None) -> dict:
    tasks = tasks or ["af", "bs"]
    if scenario == "no_data":
        return {
            "job_id": JOB_ID, "model_bundle_id": "official-v1", "tasks": tasks,
            "data_status": "no_data", "modules": {task: None for task in tasks},
            "quality": {task: quality(0, 0, 0) for task in tasks},
            "warnings": ["Для выбранного периода нет пригодных наблюдений."], "artifacts": [],
        }
    partial = scenario == "partial"
    modules = {
        "af": {"data_status": "complete", "hotspot_count": 2},
        "bs": {"data_status": "partial" if partial else "complete", "burn_area_ha": 12.5,
               "severity_area_ha": {"1": 2.5, "2": 4.0, "3": 6.0}},
    }
    qualities = {
        "af": quality(1.0, 18.0, 2),
        "bs": quality(0.5 if partial else 1.0, 9.0 if partial else 18.0, 1),
    }
    return {
        "job_id": JOB_ID, "model_bundle_id": "official-v1", "tasks": tasks,
        "data_status": "partial" if partial else "complete",
        "modules": {task: modules[task] for task in tasks},
        "quality": {task: qualities[task] for task in tasks},
        "warnings": ["Часть площади закрыта облаками."] if partial else [],
        "artifacts": artifacts(),
    }


def artifacts() -> list[dict]:
    base = f"/v1/jobs/{JOB_ID}/artifacts"
    return [
        {"id": "hotspots", "href": f"{base}/hotspots", "media_type": "application/geo+json", "filename": "hotspots.geojson"},
        {"id": "burn_polygons", "href": f"{base}/burn_polygons", "media_type": "application/geo+json", "filename": "burn_polygons.geojson"},
        {"id": "summary", "href": f"{base}/summary", "media_type": "application/json", "filename": "summary.json"},
    ]


def catalog() -> dict:
    return {"datasets": [{
        "dataset_id": "qa-fixture-2024", "label": "Тестовая сцена QA", "mode": "geospatial_demo",
        "bounds": [37.0, 55.0, 37.1, 55.1],
        "date_range": {"start": "2024-07-13T00:00:00Z", "end": "2024-07-28T23:59:59Z"},
        "tasks": ["af", "bs"], "scene_count": 2,
        "source": {"label": "Тестовая сцена QA", "source_dataset": "validation"},
        "presets": [{"preset_id": "qa-all", "label": "Тестовая сцена QA · AF и BS", "tasks": ["af", "bs"],
                     "bounds": [37.0, 55.0, 37.1, 55.1],
                     "period": {"start": "2024-07-13T00:00:00Z", "end": "2024-07-29T00:00:00Z"}}],
    }]}


def hotspot_geojson() -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [37.03, 55.03]}, "properties": {"observed_at": "2024-07-14T12:00:00Z"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [37.07, 55.07]}, "properties": {"observed_at": "2024-07-15T12:00:00Z"}},
    ]}


def polygon_geojson() -> dict:
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"class_id": 3, "area_ha": 6.0},
        "geometry": {"type": "Polygon", "coordinates": [[[37.02, 55.02], [37.08, 55.02], [37.08, 55.08], [37.02, 55.08], [37.02, 55.02]]]},
    }]}


class FixtureHandler(BaseHTTPRequestHandler):
    server_version = "FireWatchFixture/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print("[fixture]" , fmt % args)

    def scenario(self) -> str:
        cookies = self.headers.get("Cookie", "").split(";")
        for cookie in cookies:
            name, _, value = cookie.strip().partition("=")
            if name == "fw-scenario" and value in SCENARIOS:
                return value
        return "success"

    def require_token(self, scenario: str) -> bool:
        if scenario != "auth" or self.headers.get("Authorization") == f"Bearer {QA_TOKEN}":
            return True
        self.send_json(HTTPStatus.UNAUTHORIZED, {"error": {"code": "UNAUTHORIZED", "message": "Введите тестовый токен qa-test-token."}})
        return False

    def send_json(self, status: int, payload: object, extra: dict[str, str] | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items(): self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            selected = parse_qs(parsed.query).get("scenario", [None])[0]
            if selected and selected not in SCENARIOS:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"code": "BAD_SCENARIO", "message": "Неизвестный QA-сценарий."}}); return
            self.serve_file(WEB_ROOT / "index.html", {"Set-Cookie": f"fw-scenario={selected or self.scenario()}; Path=/; SameSite=Strict"}); return
        scenario = self.scenario()
        if (parsed.path.startswith("/v1/") or parsed.path in {"/health/ready", "/health/live"}) and not self.require_token(scenario): return
        if parsed.path == "/v1/catalog": self.send_json(200, catalog()); return
        if parsed.path == "/v1/models":
            status = "unavailable" if scenario == "unavailable" else "ready"
            self.send_json(200, {"models": [{"model_bundle_id": "official-v1", "status": status, "tasks": ["af", "bs"]}]}); return
        if parsed.path in {"/health/ready", "/health/live"}:
            self.send_json(200 if scenario != "unavailable" else 503, {"status": "ready" if scenario != "unavailable" else "unavailable"}); return
        if parsed.path == f"/v1/jobs/{JOB_ID}": self.job_status(scenario); return
        if parsed.path == f"/v1/jobs/{JOB_ID}/result":
            if scenario == "missing_job": self.send_json(404, {"error": {"code": "JOB_NOT_FOUND", "message": "Тестовое задание не найдено."}}); return
            self.send_json(200, result_for(scenario, self.job_tasks())); return
        artifact_prefix = f"/v1/jobs/{JOB_ID}/artifacts/"
        if parsed.path.startswith(artifact_prefix):
            name = parsed.path[len(artifact_prefix):]
            if name == "hotspots": self.send_json(200, hotspot_geojson()); return
            if name == "burn_polygons": self.send_json(200, polygon_geojson()); return
            if name == "summary": self.send_json(200, result_for(scenario, self.job_tasks())); return
            self.send_json(404, {"error": {"code": "ARTIFACT_NOT_FOUND", "message": "Нет такого тестового артефакта."}}); return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        scenario = self.scenario()
        if parsed.path != "/v1/analyses": self.send_json(404, {"error": {"code": "NOT_FOUND", "message": "Маршрут не найден."}}); return
        if not self.require_token(scenario): return
        try:
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": {"code": "BAD_JSON", "message": "Некорректный JSON."}}); return
        key = self.headers.get("Idempotency-Key", "")
        self.log_submission(payload, key, scenario)
        if scenario == "unavailable":
            self.send_json(409, {"error": {"code": "MODEL_NOT_READY", "message": "Тестовая модель недоступна."}}); return
        if scenario == "post_retry":
            if state["retry_key"] is None:
                state["retry_key"] = key; self.send_json(503, {"error": {"code": "RETRY", "message": "Повторите запрос с тем же ключом."}}, {"Retry-After": "1"}); return
            if key != state["retry_key"]:
                self.send_json(409, {"error": {"code": "IDEMPOTENCY_CONFLICT", "message": "Для повтора нужен исходный ключ идемпотентности."}}); return
        tasks = payload.get("tasks") if isinstance(payload, dict) else None
        tasks = [task for task in tasks if task in {"af", "bs"}] if isinstance(tasks, list) else ["af", "bs"]
        state["jobs"][JOB_ID] = {"key": key, "tasks": tasks or ["af", "bs"]}
        state["polls"][JOB_ID] = 0
        self.send_json(202, {"job_id": JOB_ID, "status": "queued", "status_url": f"/v1/jobs/{JOB_ID}", "result_url": f"/v1/jobs/{JOB_ID}/result"}, {"Location": f"/v1/jobs/{JOB_ID}"})

    def job_tasks(self) -> list[str]:
        job = state["jobs"].get(JOB_ID, {})
        return job.get("tasks", ["af", "bs"]) if isinstance(job, dict) else ["af", "bs"]

    def job_status(self, scenario: str) -> None:
        if scenario == "missing_job": self.send_json(404, {"error": {"code": "JOB_NOT_FOUND", "message": "Тестовое задание не найдено."}}); return
        if scenario == "failed": self.send_json(200, {"job_id": JOB_ID, "status": "failed", "stage": "Тестовая ошибка", "progress": 1, "error": {"message": "Тестовый сбой задачи."}}); return
        poll = state["polls"].get(JOB_ID, 0); state["polls"][JOB_ID] = poll + 1
        statuses = [("queued", "Тестовая очередь", 0.05), ("running", "Тестовый анализ", 0.55), ("succeeded", "Тест завершён", 1)]
        status, stage, progress = statuses[min(poll, len(statuses) - 1)]
        self.send_json(200, {"job_id": JOB_ID, "status": status, "stage": stage, "progress": progress, "result_url": f"/v1/jobs/{JOB_ID}/result"}, {"Retry-After": "1"})

    def log_submission(self, payload: object, key: str, scenario: str) -> None:
        REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with REQUEST_LOG.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"scenario": scenario, "idempotency_key": key, "payload": payload}, ensure_ascii=False) + "\n")

    def serve_static(self, path: str) -> None:
        relative = path.lstrip("/") or "index.html"
        candidate = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in candidate.parents and candidate != WEB_ROOT.resolve():
            self.send_json(403, {"error": {"code": "FORBIDDEN", "message": "Недопустимый путь."}}); return
        self.serve_file(candidate)

    def serve_file(self, path: Path, extra: dict[str, str] | None = None) -> None:
        if not path.is_file(): self.send_json(404, {"error": {"code": "NOT_FOUND", "message": "Файл не найден."}}); return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items(): self.send_header(key, value)
        self.end_headers(); self.wfile.write(data)


if __name__ == "__main__":
    print(f"Fixture QA: http://{HOST}:{PORT}/?scenario=success")
    HTTPServer((HOST, PORT), FixtureHandler).serve_forever()
