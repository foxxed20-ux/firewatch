"""Local UI contract fixture; scene metadata is synthetic and marked QA.

python tests/frontend/imagery_fixture_server.py
http://127.0.0.1:8092/?imagery=success
Scenarios: success, empty, unavailable, disabled, auth, slow.
Preview deliberately uses a nonexistent EE map. It tests the tile failure UI,
not real Google imagery. No credentials or paid API calls are involved.
"""
from __future__ import annotations

import json
import time
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import fixture_server as base

base.REQUEST_LOG = base.ROOT / "artifacts/google-imagery/fixture-requests.jsonl"
SCENARIOS = {"success", "empty", "unavailable", "disabled", "auth", "slow"}


class Handler(base.FixtureHandler):
    def imagery_scenario(self):
        for cookie in self.headers.get("Cookie", "").split(";"):
            name, _, value = cookie.strip().partition("=")
            if name == "fw-imagery" and value in SCENARIOS:
                return value
        return "success"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            scenario = parse_qs(parsed.query).get("imagery", ["success"])[0]
            if scenario not in SCENARIOS:
                self.send_json(400, {"error": {"message": "Unknown fixture scenario"}})
                return
            self.serve_file(base.WEB_ROOT / "index.html", {
                "Set-Cookie": f"fw-imagery={scenario}; Path=/; SameSite=Strict",
                "Cache-Control": "no-store",
            })
            return
        if parsed.path == "/v1/client-config":
            self.send_json(200, {
                "maps": {"provider": "osm", "configured": False, "api_key": None},
                "imagery": {"provider": "earth_engine", "configured": self.imagery_scenario() != "disabled", "collection": "COPERNICUS/S2_SR_HARMONIZED"},
            })
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if not path.startswith("/v1/imagery/"):
            super().do_POST()
            return
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        scenario = self.imagery_scenario()
        self.log_submission(payload, path, scenario)
        if scenario == "auth" and self.headers.get("Authorization") != "Bearer qa-test-token":
            self.send_json(401, {"error": {"code": "UNAUTHORIZED"}})
            return
        if scenario in {"disabled", "unavailable"}:
            self.send_json(503, {"error": {"code": "GOOGLE_NOT_CONFIGURED" if scenario == "disabled" else "EARTH_ENGINE_UNAVAILABLE"}})
            return
        if scenario == "slow":
            time.sleep(3)
        if path == "/v1/imagery/search":
            scenes = [] if scenario == "empty" else [{
                "id": "COPERNICUS/S2_SR_HARMONIZED/20240714T084601_20240714T085417_T37UDB_QA",
                "acquired_at": "2024-07-14T08:46:01Z", "cloud_percent": 8.2,
                "sensor": "Тестовая сцена QA", "resolution_m": 10, "bounds": [37.0, 55.0, 37.1, 55.1],
            }, {
                "id": "COPERNICUS/S2_SR_HARMONIZED/20240716T084601_20240716T085417_T37UDB_QA",
                "acquired_at": "2024-07-16T08:46:01Z", "cloud_percent": None,
                "sensor": "Тестовая сцена QA", "resolution_m": 10, "bounds": [37.0, 55.0, 37.1, 55.1],
            }]
            self.send_json(200, {"provider": "qa_fixture", "collection": "COPERNICUS/S2_SR_HARMONIZED", "scenes": scenes, "truncated": False})
            return
        if path == "/v1/imagery/preview":
            self.send_json(200, {
                "scene_id": payload["scene_id"], "tile_url": "https://earthengine.googleapis.com/v1/projects/firewatch-qa/maps/not-a-real-map/tiles/{z}/{x}/{y}",
                "attribution": "QA fixture — no real imagery", "bounds": [37.0, 55.0, 37.1, 55.1],
            })
            return
        self.send_json(404, {"error": {"code": "NOT_FOUND"}})


if __name__ == "__main__":
    print("Synthetic imagery UI fixture: http://127.0.0.1:8092/?imagery=success", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8092), Handler).serve_forever()
