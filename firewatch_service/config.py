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

    @classmethod
    def from_env(cls) -> "Settings":
        def path(name: str, default: str) -> Path:
            return Path(os.getenv(name, default)).expanduser().resolve()

        return cls(
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
