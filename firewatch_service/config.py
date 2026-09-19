from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    model_root: Path
    artifact_root: Path
    job_db: Path
    token: str | None
    job_timeout_seconds: int = 900
    max_workers: int = 1
    max_queued_jobs: int = 16
    queue_timeout_seconds: int = 300
    max_post_body_bytes: int = 32_768
    rate_limit_per_minute: int = 30
    public_demo: bool = False
    registry_cache_seconds: int = 10
    download_root: Path | None = None
    google_maps_api_key: str | None = None
    google_cloud_project: str | None = None
    earth_engine_enabled: bool = False
    imagery_max_area_km2: int = 10_000
    imagery_max_date_days: int = 366
    imagery_timeout_seconds: int = 20
    imagery_rate_limit_per_minute: int = 10
    imagery_cache_seconds: int = 300
    imagery_cache_entries: int = 128
    imagery_max_concurrent: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        def path(name: str, default: str) -> Path:
            return Path(os.getenv(name, default)).expanduser().resolve()

        def bounded(name: str, default: int, maximum: int) -> int:
            return max(1, min(maximum, int(os.getenv(name, str(default)))))

        return cls(
            google_maps_api_key=os.getenv("FIREWATCH_GOOGLE_MAPS_API_KEY", "").strip()
            or None,
            google_cloud_project=os.getenv("FIREWATCH_GOOGLE_CLOUD_PROJECT", "").strip()
            or None,
            earth_engine_enabled=os.getenv("FIREWATCH_EARTH_ENGINE_ENABLED", "false")
            .strip()
            .lower()
            in {"1", "true", "yes"},
            imagery_max_area_km2=bounded(
                "FIREWATCH_IMAGERY_MAX_AREA_KM2", 10_000, 50_000
            ),
            imagery_max_date_days=bounded("FIREWATCH_IMAGERY_MAX_DATE_DAYS", 366, 366),
            imagery_timeout_seconds=bounded(
                "FIREWATCH_IMAGERY_TIMEOUT_SECONDS", 20, 60
            ),
            imagery_rate_limit_per_minute=bounded(
                "FIREWATCH_IMAGERY_RATE_LIMIT_PER_MINUTE", 10, 60
            ),
            imagery_cache_seconds=bounded("FIREWATCH_IMAGERY_CACHE_SECONDS", 300, 300),
            imagery_cache_entries=bounded("FIREWATCH_IMAGERY_CACHE_ENTRIES", 128, 1024),
            imagery_max_concurrent=bounded("FIREWATCH_IMAGERY_MAX_CONCURRENT", 2, 4),
            download_root=path("FIREWATCH_DOWNLOAD_ROOT", "")
            if os.getenv("FIREWATCH_DOWNLOAD_ROOT")
            else None,
            data_root=path("FIREWATCH_DATA_ROOT", "data/service"),
            model_root=path("FIREWATCH_MODEL_ROOT", "model_bundle"),
            artifact_root=path("FIREWATCH_ARTIFACT_ROOT", "var/results"),
            job_db=path("FIREWATCH_JOB_DB", "var/jobs.sqlite3"),
            token=os.getenv("FIREWATCH_TOKEN") or None,
            job_timeout_seconds=max(
                1, int(os.getenv("FIREWATCH_JOB_TIMEOUT_SECONDS", "900"))
            ),
            max_workers=max(1, int(os.getenv("FIREWATCH_MAX_WORKERS", "1"))),
            max_queued_jobs=max(1, int(os.getenv("FIREWATCH_MAX_QUEUED_JOBS", "16"))),
            queue_timeout_seconds=max(
                1, int(os.getenv("FIREWATCH_QUEUE_TIMEOUT_SECONDS", "300"))
            ),
            max_post_body_bytes=max(
                1_024, int(os.getenv("FIREWATCH_MAX_POST_BODY_BYTES", "32768"))
            ),
            rate_limit_per_minute=max(
                1, int(os.getenv("FIREWATCH_RATE_LIMIT_PER_MINUTE", "30"))
            ),
            public_demo=os.getenv("FIREWATCH_PUBLIC_DEMO", "false").strip().lower()
            in {"1", "true", "yes"},
            registry_cache_seconds=max(
                1, int(os.getenv("FIREWATCH_REGISTRY_CACHE_SECONDS", "10"))
            ),
        )

    def ensure_directories(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.job_db.parent.mkdir(parents=True, exist_ok=True)
