from __future__ import annotations

import time
import sys
import json
import os
import socket
import subprocess
import urllib.error
import urllib.request
import threading
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from firewatch_service.api import create_app
from firewatch_service.config import Settings
from firewatch_service.jobs import run_isolated


def _catalog(_: Path) -> dict:
    return {"datasets": [{"id": "demo-v1"}], "limits": {"max_aoi_km2": 1000}}


def _models(_: Path) -> dict:
    return {"models": [{"id": "official-v1", "ready": True, "tasks": ["af", "bs"]}]}


def _analysis(_: dict, output: Path, *__: Path) -> dict:
    (output / "summary.json").write_text('{"ok":true}', encoding="utf-8")
    return {
        "result": {"data_status": "ok", "quality": {"coverage_fraction": 1}},
        "artifacts": {"summary": "summary.json"},
    }


def _prediction(_: dict, output: Path, *__: Path) -> dict:
    (output / "chip_mask.npy").write_bytes(b"test")
    return {"result": {"predictions": []}, "artifacts": {"chip_mask": "chip_mask.npy"}}


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "jobs.sqlite3",
        None,
        public_demo=True,
    )
    return TestClient(
        create_app(
            settings,
            catalog_fn=_catalog,
            models_fn=_models,
            analysis_fn=_analysis,
            prediction_fn=_prediction,
        )
    )


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(40):
        state = client.get(f"/v1/jobs/{job_id}").json()
        if state["status"] in {"succeeded", "failed"}:
            return state
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_analysis_is_idempotent_and_artifacts_are_scoped(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = {
        "dataset_id": "demo-v1",
        "model_bundle_id": "official-v1",
        "aoi": {"bbox": [30, 40, 31, 41]},
        "period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
        "tasks": ["af"],
    }
    first = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "same"}
    )
    second = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "same"}
    )
    assert first.status_code == second.status_code == 202
    job_id = first.json()["job_id"]
    assert job_id == second.json()["job_id"]
    assert _wait(client, job_id)["status"] == "succeeded"
    result = client.get(f"/v1/jobs/{job_id}/result").json()
    assert result["artifacts"][0]["id"] == "summary"
    assert client.get(result["artifacts"][0]["href"]).status_code == 200
    payload["tasks"] = ["bs"]
    assert (
        client.post(
            "/v1/analyses", json=payload, headers={"Idempotency-Key": "same"}
        ).status_code
        == 409
    )


def test_schema_rejects_bad_aoi_and_unknown_bundle(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/v1/analyses",
        json={
            "dataset_id": "demo-v1",
            "model_bundle_id": "nope",
            "aoi": {"bbox": [30, 40, 31, 41]},
            "period": {"start": "2025-01-02T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
            "tasks": ["af"],
        },
        headers={"Idempotency-Key": "bad"},
    )
    assert response.status_code == 422
    response = client.post(
        "/v1/analyses",
        json={
            "dataset_id": "demo-v1",
            "model_bundle_id": "nope",
            "aoi": {"bbox": [30, 40, 31, 41]},
            "period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
            "tasks": ["af"],
        },
        headers={"Idempotency-Key": "unknown"},
    )
    assert response.status_code == 503
    crossing = {
        "type": "Polygon",
        "coordinates": [[[30, 40], [31, 41], [31, 40], [30, 41], [30, 40]]],
    }
    response = client.post(
        "/v1/analyses",
        json={
            "dataset_id": "demo-v1",
            "model_bundle_id": "official-v1",
            "aoi": {"geometry": crossing},
            "period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
            "tasks": ["af"],
        },
        headers={"Idempotency-Key": "crossing"},
    )
    assert response.status_code == 422


def test_isolated_worker_drains_large_result_and_terminates_timeout(
    tmp_path: Path,
) -> None:
    """Queue must be drained before join or a large worker payload can deadlock."""
    fixture = tmp_path / "isolated_fixture.py"
    fixture.write_text(
        "import time\n"
        "def run_analysis(payload, *_):\n"
        "    if payload.get('sleep'): time.sleep(5)\n"
        "    return {'blob': 'x' * 200_000}\n"
        "run_prediction = run_analysis\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        result = run_isolated(
            "analysis",
            {},
            tmp_path,
            tmp_path,
            tmp_path,
            5,
            processor_module="isolated_fixture",
        )
        assert len(result["blob"]) == 200_000
        with pytest.raises(TimeoutError):
            run_isolated(
                "analysis",
                {"sleep": True},
                tmp_path,
                tmp_path,
                tmp_path,
                1,
                processor_module="isolated_fixture",
            )
    finally:
        sys.path.remove(str(tmp_path))


def test_uvicorn_does_not_advertise_unverified_bundle(tmp_path: Path) -> None:
    """A bare manifest cannot make a model public-ready in the deployed ASGI app."""
    data = tmp_path / "data" / "datasets" / "demo-v1"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(
        json.dumps({"dataset_id": "demo-v1", "scenes": []}), encoding="utf-8"
    )
    model = tmp_path / "model" / "af"
    model.mkdir(parents=True)
    (tmp_path / "model" / "manifest.json").write_text(
        json.dumps(
            {"bundle_id": "official-v1", "tasks": {"af": {"path": "af", "sha256": {}}}}
        ),
        encoding="utf-8",
    )
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    environment = {
        **os.environ,
        "FIREWATCH_DATA_ROOT": str(tmp_path / "data"),
        "FIREWATCH_MODEL_ROOT": str(tmp_path / "model"),
        "FIREWATCH_ARTIFACT_ROOT": str(tmp_path / "results"),
        "FIREWATCH_JOB_DB": str(tmp_path / "jobs.sqlite3"),
        "FIREWATCH_PUBLIC_DEMO": "true",
    }
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "firewatch_service.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path.cwd(),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(base + "/health/live", timeout=0.2)
                break
            except urllib.error.URLError:
                time.sleep(0.1)
        else:
            raise AssertionError("uvicorn did not start")
        with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
            model_payload = json.loads(response.read())
        assert (
            model_payload["models"]
            and model_payload["models"][0]["status"] == "unavailable"
        ), model_payload
        payload = json.dumps(
            {
                "dataset_id": "demo-v1",
                "model_bundle_id": "official-v1",
                "aoi": {"bbox": [30, 40, 31, 41]},
                "period": {
                    "start": "2025-01-01T00:00:00Z",
                    "end": "2025-01-02T00:00:00Z",
                },
                "tasks": ["af"],
            }
        ).encode()
        request = urllib.request.Request(
            base + "/v1/analyses",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": "real-worker",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 503
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_queue_rejection_does_not_persist_idempotency_key(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow(_: dict, output: Path, *__: Path) -> dict:
        started.set()
        assert release.wait(5)
        (output / "done.json").write_text("{}", encoding="utf-8")
        return {"result": {"data_status": "ok"}, "artifacts": {"done": "done.json"}}

    settings = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "jobs.sqlite3",
        None,
        max_workers=1,
        max_queued_jobs=1,
        public_demo=True,
    )
    client = TestClient(
        create_app(
            settings,
            catalog_fn=_catalog,
            models_fn=_models,
            analysis_fn=slow,
            prediction_fn=slow,
        )
    )
    payload = {
        "dataset_id": "demo-v1",
        "model_bundle_id": "official-v1",
        "aoi": {"bbox": [30, 40, 31, 41]},
        "period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
        "tasks": ["af"],
    }
    first = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "occupied"}
    )
    assert first.status_code == 202 and started.wait(2)
    # An existing key is replayed even while capacity is unavailable.
    assert (
        client.post(
            "/v1/analyses", json=payload, headers={"Idempotency-Key": "occupied"}
        ).status_code
        == 202
    )
    rejected = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "retry-later"}
    )
    assert rejected.status_code == 429 and rejected.headers["Retry-After"] == "2"
    release.set()
    assert _wait(client, first.json()["job_id"])["status"] == "succeeded"
    retried = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "retry-later"}
    )
    assert retried.status_code == 202


def test_token_auth_and_explicit_public_demo(tmp_path: Path) -> None:
    locked = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "locked-results",
        tmp_path / "locked.sqlite3",
        "secret",
    )
    locked_client = TestClient(
        create_app(
            locked,
            catalog_fn=_catalog,
            models_fn=_models,
            analysis_fn=_analysis,
            prediction_fn=_prediction,
        )
    )
    assert locked_client.get("/v1/catalog").status_code == 401
    assert (
        locked_client.get(
            "/v1/catalog", headers={"Authorization": "Bearer secret"}
        ).status_code
        == 200
    )
    public = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "public-results",
        tmp_path / "public.sqlite3",
        None,
        public_demo=True,
    )
    public_client = TestClient(
        create_app(
            public,
            catalog_fn=_catalog,
            models_fn=_models,
            analysis_fn=_analysis,
            prediction_fn=_prediction,
        )
    )
    assert public_client.get("/v1/catalog").status_code == 200


def test_prediction_result_resolves_published_mask_and_provenance_urls(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "datasets" / "demo-v1"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(
        json.dumps({"dataset_id": "demo-v1", "scenes": [{"chip_id": "AF_test_1"}]}),
        encoding="utf-8",
    )

    def prediction(_: dict, output: Path, *__: Path) -> dict:
        (output / "mask.npy").write_bytes(b"mask-bytes")
        (output / "provenance.json").write_text("{}", encoding="utf-8")
        return {
            "data_status": "ok",
            "items": [{"chip_id": "AF_test_1", "mask_artifact_id": "mask_AF_test_1"}],
            "artifacts": {
                "mask_AF_test_1": "mask.npy",
                "prediction_provenance": "provenance.json",
            },
        }

    settings = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "jobs.sqlite3",
        None,
        public_demo=True,
    )
    client = TestClient(
        create_app(
            settings,
            catalog_fn=_catalog,
            models_fn=_models,
            analysis_fn=_analysis,
            prediction_fn=prediction,
        )
    )
    accepted = client.post(
        "/v1/predictions",
        json={
            "dataset_id": "demo-v1",
            "model_bundle_id": "official-v1",
            "chip_ids": ["AF_test_1"],
        },
        headers={"Idempotency-Key": "prediction-links"},
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    assert _wait(client, job_id)["status"] == "succeeded"
    result = client.get(f"/v1/jobs/{job_id}/result").json()
    assert result["items"][0]["mask_url"].endswith("/artifacts/mask_AF_test_1")
    assert result["provenance_url"].endswith("/artifacts/prediction_provenance")
    downloaded = client.get(result["items"][0]["mask_url"])
    assert downloaded.status_code == 200 and downloaded.content == b"mask-bytes"


def test_model_registry_is_cached_for_public_read_endpoints(tmp_path: Path) -> None:
    calls = 0

    def counted_models(root: Path) -> dict:
        nonlocal calls
        calls += 1
        return _models(root)

    settings = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "jobs.sqlite3",
        None,
        public_demo=True,
        registry_cache_seconds=60,
    )
    client = TestClient(
        create_app(
            settings,
            catalog_fn=_catalog,
            models_fn=counted_models,
            analysis_fn=_analysis,
            prediction_fn=_prediction,
        )
    )
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert calls == 1


def test_accepted_job_persists_private_snapshot_and_model_path(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "data" / "datasets" / "demo-v1"
    folder.mkdir(parents=True)
    scene = {
        "scene_id": "original",
        "chip_id": "chip-a",
        "task": "af",
        "observed_at": "2025-01-01T12:00:00Z",
        "crs": "EPSG:4326",
        "transform": [0.01, 0, 30, 0, -0.01, 41],
        "width": 2,
        "height": 2,
    }
    manifest = {"dataset_id": "demo-v1", "scenes": [scene]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    settings = Settings(
        tmp_path / "data",
        tmp_path / "immutable-model",
        tmp_path / "results",
        tmp_path / "snapshot.sqlite3",
        None,
        public_demo=True,
    )
    app = create_app(settings, models_fn=_models)
    pending = []
    monkeypatch.setattr(
        app.state.executor, "submit", lambda fn, job_id: pending.append((fn, job_id))
    )
    client = TestClient(app)
    payload = {
        "dataset_id": "demo-v1",
        "model_bundle_id": "official-v1",
        "aoi": {"bbox": [30, 40, 31, 41]},
        "period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
        "tasks": ["af"],
    }
    accepted = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "snapshot"}
    )
    assert accepted.status_code == 202
    stored = app.state.store.get(accepted.json()["job_id"])
    assert "_catalog_snapshot" not in stored["payload"]
    assert (
        stored["worker_payload"]["_catalog_snapshot"]["scenes"][0]["scene_id"]
        == "original"
    )
    assert stored["worker_payload"]["_model_root"] == str(settings.model_root.resolve())
    manifest["scenes"][0]["scene_id"] = "modified"
    (folder / "manifest.json").write_text(json.dumps(manifest))
    replay = client.post(
        "/v1/analyses", json=payload, headers={"Idempotency-Key": "snapshot"}
    )
    assert replay.json()["job_id"] == accepted.json()["job_id"]
    unchanged = app.state.store.get(accepted.json()["job_id"])
    assert (
        unchanged["worker_payload"]["_catalog_snapshot"]["scenes"][0]["scene_id"]
        == "original"
    )
    assert len(pending) == 1


def test_html_entrypoint_refreshes_even_with_stale_conditional_headers(
    tmp_path: Path,
) -> None:
    settings = Settings(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "html.sqlite3",
        None,
        public_demo=True,
    )
    client = TestClient(create_app(settings, catalog_fn=_catalog, models_fn=_models))
    for route in ("/", "/index.html"):
        response = client.get(
            route,
            headers={
                "If-Modified-Since": "Wed, 01 Jan 2031 00:00:00 GMT",
                "If-None-Match": '"stale-release"',
            },
        )
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "FireWatch" in response.text
