from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
from queue import Empty
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class IdempotencyConflict(Exception):
    pass


class QueueFull(Exception):
    pass


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                worker_payload TEXT,
                idem_key TEXT NOT NULL, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                stage TEXT NOT NULL, result TEXT, error TEXT, created_at TEXT NOT NULL, started_at TEXT,
                finished_at TEXT, UNIQUE(kind, idem_key))""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "worker_payload" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN worker_payload TEXT")
                db.execute(
                    "UPDATE jobs SET worker_payload=payload WHERE worker_payload IS NULL"
                )

    def recover_running(self) -> None:
        """Mark interrupted jobs only when the serving lifespan begins."""
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed', stage='failed', error=? , finished_at=? WHERE status='running'",
                (
                    json.dumps(
                        {
                            "code": "WORKER_LOST",
                            "message": "Service restarted while job was running",
                        }
                    ),
                    utcnow(),
                ),
            )

    def create(
        self,
        kind: str,
        payload: dict[str, Any],
        worker_payload: dict[str, Any],
        idem_key: str,
        *,
        max_active: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        digest = canonical_hash(payload)
        with self.lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM jobs WHERE kind=? AND idem_key=?", (kind, idem_key)
            ).fetchone()
            if existing:
                if existing["payload_hash"] != digest:
                    raise IdempotencyConflict()
                return self._row(existing), False
            if max_active is not None:
                active = int(
                    db.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
                    ).fetchone()[0]
                )
                if active >= max_active:
                    raise QueueFull()
            job_id = "job_" + uuid.uuid4().hex
            now = utcnow()
            db.execute(
                "INSERT INTO jobs (id,kind,payload,worker_payload,payload_hash,idem_key,status,stage,created_at) VALUES (?,?,?,?,?,?,'queued','queued',?)",
                (
                    job_id,
                    kind,
                    json.dumps(payload),
                    json.dumps(worker_payload),
                    digest,
                    idem_key,
                    now,
                ),
            )
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._row(row), True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def queued(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at"
            ).fetchall()
        return [self._row(row) for row in rows]

    def active_count(self) -> int:
        with self._connect() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
                ).fetchone()[0]
            )

    def running(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='running', stage='running', started_at=?, progress=0.05 WHERE id=? AND status='queued'",
                (utcnow(), job_id),
            )

    def success(self, job_id: str, result: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='succeeded', stage='complete', progress=1, result=?, finished_at=? WHERE id=?",
                (json.dumps(result, allow_nan=False), utcnow(), job_id),
            )

    def fail(self, job_id: str, code: str, message: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed', stage='failed', error=?, finished_at=? WHERE id=?",
                (json.dumps({"code": code, "message": message}), utcnow(), job_id),
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("payload", "worker_payload", "result", "error"):
            if item.get(key) is not None:
                item[key] = json.loads(item[key])
        return item


def _processor_child(
    kind: str,
    payload: dict[str, Any],
    output_dir: str,
    data_root: str,
    model_root: str,
    queue: Any,
    processor_module: str,
) -> None:
    """Spawn target: model code is isolated so a timeout can stop it."""
    try:
        processing = importlib.import_module(processor_module)
        fn = (
            processing.run_analysis if kind == "analysis" else processing.run_prediction
        )
        queue.put(
            ("ok", fn(payload, Path(output_dir), Path(data_root), Path(model_root)))
        )
    except Exception as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_isolated(
    kind: str,
    payload: dict[str, Any],
    output_dir: Path,
    data_root: Path,
    model_root: Path,
    timeout_seconds: int,
    *,
    processor_module: str = "firewatch_service.processing",
) -> dict[str, Any]:
    """Run the production processor in its own process and terminate on timeout."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_processor_child,
        args=(
            kind,
            payload,
            str(output_dir),
            str(data_root),
            str(model_root),
            queue,
            processor_module,
        ),
    )
    process.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker exceeded execution limit")
            try:
                state, value = queue.get(timeout=min(0.2, remaining))
                break
            except Empty:
                if not process.is_alive():
                    raise RuntimeError(
                        f"worker exited without a result (exit code {process.exitcode})"
                    ) from None
        process.join(min(5, max(0.1, deadline - time.monotonic())))
        if process.is_alive():
            raise RuntimeError("worker did not exit after returning a result")
        if state != "ok":
            raise RuntimeError(value)
        if not isinstance(value, dict):
            raise RuntimeError("processing runtime returned invalid result")
        return value
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        queue.close()
        queue.join_thread()
