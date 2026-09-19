from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

from firewatch_service.api import create_app
from firewatch_service.config import Settings
from firewatch_service.imagery import ImageryGateway
from firewatch_service.imagery_provider import COLLECTION, ImageryError


AOI = {
    "type": "Polygon",
    "coordinates": [[[30, 50], [30.1, 50], [30.1, 50.1], [30, 50.1], [30, 50]]],
}
SEARCH = {
    "aoi": AOI,
    "date_start": "2026-06-01",
    "date_end": "2026-06-02",
    "cloud_max": 20,
    "limit": 5,
}
SCENE = COLLECTION + "/20260601T100000_20260601T100000_T36UXV"


class FakeProvider:
    def __init__(self):
        self.searches = []
        self.previews = []

    def configuration(self):
        return {
            "provider": "earth_engine",
            "configured": True,
            "collection": COLLECTION,
        }

    def search(self, payload):
        self.searches.append(deepcopy(payload))
        return {
            "provider": "earth_engine",
            "collection": COLLECTION,
            "scenes": [],
            "truncated": False,
        }

    def preview(self, payload):
        self.previews.append(deepcopy(payload))
        return {
            "scene_id": payload["scene_id"],
            "tile_url": "https://earthengine.googleapis.com/v1/projects/test/maps/map/tiles/{z}/{x}/{y}",
            "attribution": "Copernicus Sentinel data",
            "bounds": [30, 50, 30.1, 50.1],
        }


@pytest.fixture
def settings(tmp_path: Path):
    return Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "jobs.sqlite3",
        None,
        public_demo=True,
    )


def client(settings, provider=None):
    return TestClient(create_app(settings, imagery_provider=provider))


def test_public_config_exposes_browser_key_but_not_server_identity(
    settings, monkeypatch
):
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", "/private/service-account.json"
    )
    configured = replace(
        settings,
        public_demo=False,
        token="private-token",
        google_maps_api_key="public-restricted-key",
        google_cloud_project="private-project",
        earth_engine_enabled=True,
    )
    with client(configured) as api:
        response = api.get("/v1/client-config")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "maps": {
                "provider": "google",
                "api_key": "public-restricted-key",
                "configured": True,
            },
            "imagery": {
                "provider": "earth_engine",
                "configured": True,
                "collection": COLLECTION,
            },
        }
        assert "private" not in response.text


def test_default_setup_state_is_stable_without_google_credentials(settings):
    with client(settings) as api:
        config = api.get("/v1/client-config").json()
        assert config["maps"] == {
            "provider": "osm",
            "api_key": None,
            "configured": False,
        }
        assert config["imagery"]["configured"] is False
        for route, payload in [
            ("search", SEARCH),
            ("preview", {"scene_id": SCENE, "aoi": AOI}),
        ]:
            response = api.post("/v1/imagery/" + route, json=payload)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "GOOGLE_NOT_CONFIGURED"


def test_imagery_requires_auth_in_private_mode(settings):
    provider = FakeProvider()
    with client(replace(settings, public_demo=False, token="secret"), provider) as api:
        assert api.post("/v1/imagery/search", json=SEARCH).status_code == 401
        assert (
            api.post(
                "/v1/imagery/preview", json={"scene_id": SCENE, "aoi": AOI}
            ).status_code
            == 401
        )
        assert provider.searches == provider.previews == []
        response = api.post(
            "/v1/imagery/search",
            json=SEARCH,
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200


def test_search_empty_results_preview_and_no_model_jobs(settings):
    provider = FakeProvider()
    with client(settings, provider) as api:
        response = api.post("/v1/imagery/search", json=SEARCH)
        assert response.status_code == 200
        assert response.json()["scenes"] == []
        assert response.json()["truncated"] is False
        assert provider.searches == [SEARCH]
        preview = api.post("/v1/imagery/preview", json={"scene_id": SCENE, "aoi": AOI})
        assert preview.status_code == 200
        assert preview.headers["cache-control"] == "no-store"
        assert "{z}/{x}/{y}" in preview.json()["tile_url"]
        assert "expires_at" not in preview.json()
        assert api.app.state.store.queued() == []
        assert list(settings.artifact_root.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        {"date_start": "2026-06-03"},
        {"date_start": "2025-01-01"},
        {"date_start": "2026-06-01T00:00:00Z"},
        {"date_start": 1780272000},
        {"date_end": "9999-12-31"},
        {"cloud_max": -1},
        {"cloud_max": 101},
        {"limit": 21},
        {"limit": "5"},
        {"unexpected": "field"},
        {"aoi": {"type": "Point", "coordinates": [30, 50]}},
        {
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
            }
        },
        {
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[179, 50], [-179, 50], [-179, 51], [179, 50]]],
            }
        },
        {
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[30, 86], [31, 86], [31, 87], [30, 86]]],
            }
        },
    ],
)
def test_invalid_requests_never_call_google(settings, change):
    provider = FakeProvider()
    with client(settings, provider) as api:
        response = api.post("/v1/imagery/search", json={**SEARCH, **change})
        assert response.status_code == 422, response.text
        assert provider.searches == []


def test_cache_avoids_provider_calls_and_global_budget_caps_misses(settings):
    provider = FakeProvider()
    with client(replace(settings, imagery_rate_limit_per_minute=1), provider) as api:
        first = api.post("/v1/imagery/search", json=SEARCH)
        second = api.post("/v1/imagery/search", json=SEARCH)
        assert first.status_code == second.status_code == 200
        assert len(provider.searches) == 1
        limited = api.post("/v1/imagery/search", json={**SEARCH, "cloud_max": 25})
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "IMAGERY_RATE_LIMITED"
        assert limited.headers["retry-after"] == "60"


def test_long_thin_aoi_cannot_bypass_cost_limit(settings):
    # About 8900 km2, but stretches across half the world: area alone is unsafe.
    provider = FakeProvider()
    thin = {
        "type": "Polygon",
        "coordinates": [[[-90, 0], [90, 0], [90, 0.004], [-90, 0.004], [-90, 0]]],
    }
    with client(settings, provider) as api:
        response = api.post("/v1/imagery/search", json={**SEARCH, "aoi": thin})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "IMAGERY_AOI_TOO_LARGE"
        assert provider.searches == []


def test_provider_failure_never_exposes_credential_details(settings):
    class FailedProvider(FakeProvider):
        def search(self, payload):
            raise RuntimeError("access_token=private-oauth-secret")

    with client(settings, FailedProvider()) as api:
        response = api.post("/v1/imagery/search", json=SEARCH)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "EARTH_ENGINE_UNAVAILABLE"
        assert "private-oauth-secret" not in response.text


def test_timeout_keeps_capacity_until_the_actual_provider_call_finishes(settings):
    started = threading.Event()
    release = threading.Event()

    class SlowProvider(FakeProvider):
        def search(self, payload):
            started.set()
            assert release.wait(5)
            return super().search(payload)

    async def run():
        gateway = ImageryGateway(
            replace(settings, imagery_timeout_seconds=0.05, imagery_max_concurrent=1),
            SlowProvider(),
        )
        try:
            with pytest.raises(ImageryError) as timed_out:
                await gateway.execute("search", SEARCH)
            assert started.is_set()
            assert timed_out.value.status == 504
            assert timed_out.value.code == "EARTH_ENGINE_TIMEOUT"
            with pytest.raises(ImageryError) as busy:
                await gateway.execute("search", {**SEARCH, "cloud_max": 21})
            assert busy.value.code == "IMAGERY_BUSY"
            assert busy.value.status == 429
        finally:
            release.set()
            gateway.close()

    asyncio.run(run())


def test_cache_eviction_and_ttl(settings, monkeypatch):
    import firewatch_service.imagery as module

    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    provider = FakeProvider()

    async def run():
        gateway = ImageryGateway(
            replace(settings, imagery_cache_entries=1, imagery_cache_seconds=5),
            provider,
        )
        try:
            await gateway.execute("search", SEARCH)
            await gateway.execute("search", {**SEARCH, "cloud_max": 21})
            await gateway.execute("search", SEARCH)
            assert len(provider.searches) == 3
            clock[0] += 6
            await gateway.execute("search", SEARCH)
            assert len(provider.searches) == 4
        finally:
            gateway.close()

    asyncio.run(run())


def test_environment_is_opt_in_and_caps_resource_settings(monkeypatch):
    monkeypatch.setenv("FIREWATCH_EARTH_ENGINE_ENABLED", "true")
    monkeypatch.setenv("FIREWATCH_GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.setenv("FIREWATCH_IMAGERY_MAX_CONCURRENT", "200")
    monkeypatch.setenv("FIREWATCH_IMAGERY_CACHE_SECONDS", "999999")
    monkeypatch.setenv("FIREWATCH_IMAGERY_TIMEOUT_SECONDS", "999999")
    settings = Settings.from_env()
    assert settings.earth_engine_enabled
    assert settings.google_cloud_project == "project"
    assert settings.imagery_max_concurrent == 4
    assert settings.imagery_cache_seconds == 300
    assert settings.imagery_timeout_seconds == 60
