from __future__ import annotations

import importlib
import hashlib
import logging
import mimetypes
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .jobs import IdempotencyConflict, JobStore, QueueFull, run_isolated
from .schemas import AnalysisRequest, PredictionRequest

SAFE_ARTIFACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _error(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> HTTPException:
    return HTTPException(
        status_code=status, detail={"code": code, "message": message}, headers=headers
    )


def _optional_import(module: str, name: str) -> Callable[..., Any] | None:
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


def _safe_artifacts(raw: Any, output_dir: Path, job_id: str) -> list[dict[str, str]]:
    entries: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, str):
                entries.append((key, value))
            elif isinstance(value, list):
                entries.extend(
                    (f"{key}_{index}", item)
                    for index, item in enumerate(value)
                    if isinstance(item, str)
                )
            elif isinstance(value, dict):
                entries.extend(
                    (f"{key}_{subkey}", item)
                    for subkey, item in value.items()
                    if isinstance(item, str)
                )
    artifacts: list[dict[str, str]] = []
    for artifact_id, relative in entries:
        if (
            not isinstance(artifact_id, str)
            or not SAFE_ARTIFACT.fullmatch(artifact_id)
            or not isinstance(relative, str)
        ):
            continue
        candidate = (output_dir / relative).resolve()
        if output_dir.resolve() not in candidate.parents or not candidate.is_file():
            continue
        artifacts.append(
            {
                "id": artifact_id,
                "href": f"/v1/jobs/{job_id}/artifacts/{artifact_id}",
                "filename": candidate.relative_to(output_dir.resolve()).as_posix(),
            }
        )
    return artifacts


def create_app(
    settings: Settings | None = None,
    *,
    catalog_fn: Callable[[Path], dict[str, Any]] | None = None,
    models_fn: Callable[[Path], dict[str, Any]] | None = None,
    analysis_fn: Callable[..., dict[str, Any]] | None = None,
    prediction_fn: Callable[..., dict[str, Any]] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    store = JobStore(settings.job_db)
    catalog_fn = (
        catalog_fn
        or _optional_import("firewatch_service.catalog", "describe_catalog")
        or (lambda _: {"datasets": [], "limits": {}})
    )
    models_fn = (
        models_fn
        or _optional_import("firewatch_service.catalog", "describe_models")
        or (lambda _: {"models": []})
    )
    injected_processor = analysis_fn is not None or prediction_fn is not None
    analysis_fn = analysis_fn or _optional_import(
        "firewatch_service.processing", "run_analysis"
    )
    prediction_fn = prediction_fn or _optional_import(
        "firewatch_service.processing", "run_prediction"
    )
    executor = ThreadPoolExecutor(
        max_workers=settings.max_workers, thread_name_prefix="firewatch-worker"
    )
    rate_lock = threading.Lock()
    request_times: dict[str, deque[float]] = defaultdict(deque)
    registry_lock = threading.Lock()
    registry_cache: tuple[float, dict[str, Any]] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.recover_running()
        for queued_job in store.queued():
            executor.submit(run_job, queued_job["id"])
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="FireWatch API",
        version="1.0",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.executor = executor

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": "HTTP_ERROR", "message": str(exc.detail)}
        )
        return JSONResponse(
            status_code=exc.status_code, content={"error": detail}, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            str(error.get("msg", "invalid input")) for error in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": "INPUT_SCHEMA_MISMATCH", "message": message[:500]}
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def public_limits(
        request: Request, call_next: Callable[..., Any]
    ) -> Response:
        if request.method == "POST" or request.url.path in {
            "/v1/models",
            "/health/ready",
        }:
            length = request.headers.get("content-length")
            if length is not None and (
                not length.isdigit() or int(length) > settings.max_post_body_bytes
            ):
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "REQUEST_TOO_LARGE",
                            "message": "request body exceeds service limit",
                        }
                    },
                )
            chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > settings.max_post_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "REQUEST_TOO_LARGE",
                                "message": "request body exceeds service limit",
                            }
                        },
                    )
                chunks.append(chunk)
            request._body = b"".join(
                chunks
            )  # Starlette reuses this bounded cache downstream.
            identity = request.client.host if request.client else "unknown"
            now = time.monotonic()
            with rate_lock:
                timestamps = request_times[identity]
                while timestamps and timestamps[0] <= now - 60:
                    timestamps.popleft()
                if len(timestamps) >= settings.rate_limit_per_minute:
                    return JSONResponse(
                        status_code=429,
                        headers={"Retry-After": "60"},
                        content={
                            "error": {
                                "code": "RATE_LIMITED",
                                "message": "too many requests",
                            }
                        },
                    )
                timestamps.append(now)
        return await call_next(request)

    async def authorize(
        request: Request, authorization: str | None = Header(default=None)
    ) -> None:
        if settings.public_demo:
            return
        if not settings.token:
            host = request.client.host if request.client else ""
            if host in {"127.0.0.1", "::1", "localhost"}:
                return
            raise _error(401, "UNAUTHORIZED", "Public access is disabled")
        expected = "Bearer " + settings.token
        if authorization != expected:
            raise _error(401, "UNAUTHORIZED", "Bearer token is required")

    def catalog() -> dict[str, Any]:
        try:
            data = catalog_fn(settings.data_root)
            return data if isinstance(data, dict) else {"datasets": [], "limits": {}}
        except Exception:
            return {"datasets": [], "limits": {}, "status": "catalog_unavailable"}

    def models() -> dict[str, Any]:
        nonlocal registry_cache
        now = time.monotonic()
        with registry_lock:
            if registry_cache is not None and registry_cache[0] > now:
                return registry_cache[1]
            try:
                data = models_fn(settings.model_root)
                result = data if isinstance(data, dict) else {"models": []}
            except Exception:
                result = {"models": [], "status": "model_registry_unavailable"}
            registry_cache = (now + settings.registry_cache_seconds, result)
            return result

    def has_id(document: dict[str, Any], field: str, wanted: str) -> bool:
        candidates = document.get(field, [])
        if isinstance(candidates, dict):
            candidates = candidates.get(field, [])
        return any(
            isinstance(item, dict)
            and (
                item.get("id") == wanted
                or item.get("dataset_id") == wanted
                or item.get("model_bundle_id") == wanted
            )
            for item in candidates
            if isinstance(candidates, list)
        )

    def model_ready(bundle_id: str, tasks: set[str] | None = None) -> bool:
        for item in models().get("models", []):
            if isinstance(item, dict) and (
                item.get("id") == bundle_id or item.get("model_bundle_id") == bundle_id
            ):
                ready = bool(item.get("ready", item.get("status") == "ready"))
                available = set(item.get("tasks", []))
                return ready and (tasks is None or tasks <= available)
        return False

    def registered_chip_ids(dataset_id: str) -> set[str]:
        discover = _optional_import("firewatch_service.catalog", "discover_datasets")
        if discover is None:
            return set()
        try:
            for manifest in discover(settings.data_root):
                if manifest.get("dataset_id") == dataset_id:
                    return {
                        str(scene["chip_id"])
                        for scene in manifest.get("scenes", [])
                        if isinstance(scene, dict)
                        and isinstance(scene.get("chip_id"), str)
                    }
        except Exception:
            pass
        return set()

    def run_job(job_id: str) -> None:
        job = store.get(job_id)
        if not job or job["status"] != "queued":
            return
        created = job["created_at"].replace("Z", "+00:00")
        from datetime import datetime, timezone

        if (
            datetime.now(timezone.utc) - datetime.fromisoformat(created)
        ).total_seconds() > settings.queue_timeout_seconds:
            store.fail(job_id, "QUEUE_TIMEOUT", "Job waited too long in the queue")
            return
        store.running(job_id)
        output_dir = (settings.artifact_root / job_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            fn = analysis_fn if job["kind"] == "analysis" else prediction_fn
            if fn is None:
                raise RuntimeError("processing runtime is not installed")
            worker_payload = job.get("worker_payload") or job["payload"]
            model_root = Path(
                worker_payload.get("_model_root", str(settings.model_root))
            )
            produced = (
                fn(worker_payload, output_dir, settings.data_root, model_root)
                if injected_processor
                else run_isolated(
                    job["kind"],
                    worker_payload,
                    output_dir,
                    settings.data_root,
                    model_root,
                    settings.job_timeout_seconds,
                )
            )
            if not isinstance(produced, dict):
                raise RuntimeError("processing runtime returned invalid result")
            result = produced.get("result", produced)
            if not isinstance(result, dict):
                raise RuntimeError("processing result must be an object")
            result.setdefault("schema_version", "1.0")
            result.setdefault("job_id", job_id)
            result.setdefault("kind", job["kind"])
            result.setdefault("dataset_id", job["payload"]["dataset_id"])
            result.setdefault("model_bundle_id", job["payload"]["model_bundle_id"])
            result.setdefault("warnings", produced.get("warnings", []))
            for key in ("data_status", "tasks", "quality", "provenance", "predictions"):
                if key in produced and key not in result:
                    result[key] = produced[key]
            raw_artifacts = produced.get("artifacts", {})
            result["artifacts"] = _safe_artifacts(raw_artifacts, output_dir, job_id)
            artifact_urls = {
                artifact["id"]: artifact["href"] for artifact in result["artifacts"]
            }
            if isinstance(result.get("items"), list):
                for item in result["items"]:
                    if not isinstance(item, dict):
                        continue
                    artifact_id = item.get("mask_artifact_id")
                    if not isinstance(artifact_id, str):
                        continue
                    mask_url = artifact_urls.get(artifact_id)
                    if mask_url is None:
                        raise RuntimeError(
                            "prediction references an unpublished mask artifact"
                        )
                    item["mask_url"] = mask_url
            for provenance_id in ("prediction_provenance", "provenance"):
                if provenance_id in artifact_urls:
                    result["provenance_url"] = artifact_urls[provenance_id]
                    break
            if isinstance(produced.get("provenance"), dict):
                result.setdefault("provenance", produced["provenance"])
            store.success(job_id, result)
        except TimeoutError:
            store.fail(
                job_id, "JOB_TIMEOUT", "Inference exceeded configured execution limit"
            )
        except Exception:
            logging.getLogger(__name__).exception("Job %s failed", job_id)
            store.fail(job_id, "MODEL_EXECUTION_FAILED", "Model execution failed")

    def submit(
        kind: str, payload: dict[str, Any], key: str | None
    ) -> tuple[dict[str, Any], bool]:
        if not key or len(key) > 256:
            raise _error(
                400, "IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key header is required"
            )
        worker_payload = dict(payload)
        if not injected_processor:
            from .processing import snapshot_request

            worker_payload = snapshot_request(
                kind, payload, settings.data_root, settings.model_root
            )
        try:
            job, created = store.create(
                kind, payload, worker_payload, key, max_active=settings.max_queued_jobs
            )
        except IdempotencyConflict:
            raise _error(
                409,
                "IDEMPOTENCY_CONFLICT",
                "Key was already used with a different request",
            )
        except QueueFull:
            raise _error(
                429, "QUEUE_FULL", "Service queue is full", {"Retry-After": "2"}
            )
        if created:
            executor.submit(run_job, job["id"])
        return job, created

    def accepted(job: dict[str, Any]) -> dict[str, Any]:
        job_id = job["id"]
        return {
            "schema_version": "1.0",
            "job_id": job_id,
            "kind": job["kind"],
            "status": job["status"],
            "status_url": f"/v1/jobs/{job_id}",
            "result_url": f"/v1/jobs/{job_id}/result",
        }

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(_: None = Depends(authorize)) -> Response:
        catalog_payload = catalog()
        datasets = catalog_payload.get("datasets", [])
        required_tasks = {
            task
            for dataset in datasets
            if isinstance(dataset, dict)
            for task in dataset.get("tasks", [])
            if task in {"af", "bs"}
        }
        if not datasets or not required_tasks:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "CATALOG_NOT_READY"},
            )
        registry = models()
        ready_models = [
            m
            for m in registry.get("models", [])
            if isinstance(m, dict)
            and (m.get("ready") or m.get("status") == "ready")
            and required_tasks <= set(m.get("tasks", []))
        ]
        if not ready_models:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "MODEL_NOT_READY"},
            )
        if getattr(executor, "_shutdown", False):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "WORKER_NOT_READY"},
            )
        return JSONResponse(
            content={
                "status": "ready",
                "auth_required": not settings.public_demo and bool(settings.token),
            }
        )

    @app.get("/v1/catalog")
    async def get_catalog(_: None = Depends(authorize)) -> dict[str, Any]:
        return catalog()

    @app.get("/v1/models")
    async def get_models(_: None = Depends(authorize)) -> dict[str, Any]:
        return models()

    @app.post("/v1/analyses", status_code=202)
    async def analyses(
        request: AnalysisRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(authorize),
    ) -> dict[str, Any]:
        if not has_id(catalog(), "datasets", request.dataset_id):
            raise _error(422, "UNKNOWN_DATASET", "dataset_id is not available")
        if not model_ready(request.model_bundle_id, set(request.tasks)):
            raise _error(503, "MODEL_NOT_READY", "model bundle is not ready")
        job, _ = submit("analysis", request.model_dump(mode="json"), idempotency_key)
        response.headers["Location"] = f"/v1/jobs/{job['id']}"
        response.headers["Retry-After"] = "2"
        return accepted(job)

    @app.post("/v1/predictions", status_code=202)
    async def predictions(
        request: PredictionRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(authorize),
    ) -> dict[str, Any]:
        if not has_id(catalog(), "datasets", request.dataset_id):
            raise _error(422, "UNKNOWN_DATASET", "dataset_id is not available")
        if not model_ready(request.model_bundle_id):
            raise _error(503, "MODEL_NOT_READY", "model bundle is not ready")
        if not set(request.chip_ids) <= registered_chip_ids(request.dataset_id):
            raise _error(
                404, "CHIP_NOT_FOUND", "one or more chip_ids are not registered"
            )
        job, _ = submit("prediction", request.model_dump(mode="json"), idempotency_key)
        response.headers["Location"] = f"/v1/jobs/{job['id']}"
        response.headers["Retry-After"] = "2"
        return accepted(job)

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, _: None = Depends(authorize)) -> dict[str, Any]:
        job = store.get(job_id)
        if not job:
            raise _error(404, "JOB_NOT_FOUND", "job does not exist")
        return {
            "schema_version": "1.0",
            "job_id": job_id,
            "kind": job["kind"],
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "error": job["error"],
            "result_url": f"/v1/jobs/{job_id}/result",
        }

    @app.get("/v1/jobs/{job_id}/result")
    async def get_result(job_id: str, _: None = Depends(authorize)) -> dict[str, Any]:
        job = store.get(job_id)
        if not job:
            raise _error(404, "JOB_NOT_FOUND", "job does not exist")
        if job["status"] == "failed":
            raise _error(
                409, "JOB_FAILED", "job failed; inspect its status for the error code"
            )
        if job["status"] != "succeeded":
            raise _error(409, "RESULT_NOT_READY", "job has not completed successfully")
        return job["result"]

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_id}")
    async def artifact(
        job_id: str, artifact_id: str, _: None = Depends(authorize)
    ) -> FileResponse:
        if not SAFE_ARTIFACT.fullmatch(artifact_id):
            raise _error(404, "ARTIFACT_NOT_FOUND", "artifact does not exist")
        job = store.get(job_id)
        if not job or job["status"] != "succeeded":
            raise _error(404, "ARTIFACT_NOT_FOUND", "artifact does not exist")
        entries = {item["id"]: item for item in job["result"].get("artifacts", [])}
        entry = entries.get(artifact_id)
        if not entry:
            raise _error(404, "ARTIFACT_NOT_FOUND", "artifact does not exist")
        # Artifact IDs map only to files emitted under this job directory.
        raw = (settings.artifact_root / job_id).resolve()
        filename = entry.get("filename")
        candidate = (
            (raw / filename).resolve() if isinstance(filename, str) else raw / "invalid"
        )
        if raw not in candidate.parents or not candidate.is_file():
            raise _error(404, "ARTIFACT_NOT_FOUND", "artifact has expired")
        media_type = (
            mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        if candidate.suffix == ".geojson":
            media_type = "application/geo+json"
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        return FileResponse(
            candidate,
            media_type=media_type,
            filename=candidate.name,
            content_disposition_type="attachment",
            headers={"ETag": f'"sha256-{digest}"'},
        )

    if settings.download_root is not None and settings.download_root.is_dir():
        # Explicitly published immutable model archives only; never input rasters.
        app.mount(
            "/downloads",
            StaticFiles(directory=settings.download_root),
            name="downloads",
        )
    web_dir = Path("web")
    if web_dir.is_dir():

        async def index() -> FileResponse:
            # The entry point must refresh after deploy/rollback even when an
            # archive preserves old mtimes. Asset URLs carry content hashes.
            return FileResponse(
                web_dir / "index.html",
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )

        app.add_api_route("/", index, include_in_schema=False)
        app.add_api_route("/index.html", index, include_in_schema=False)
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


app = create_app()
