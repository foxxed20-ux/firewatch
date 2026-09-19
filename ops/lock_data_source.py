"""Create a compact, secret-safe provenance lock for competition archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


CHUNK_BYTES = 8 * 1024 * 1024


def _utc(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _safe_url(value: str) -> tuple[str, bool]:
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source-url must be an absolute HTTP(S) URL")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    safe = urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))
    redacted = bool(parsed.username or parsed.password or parsed.query or parsed.fragment)
    return safe, redacted


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
            size += len(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    if size != after.st_size:
        raise RuntimeError(f"hashed size disagrees with filesystem size: {path}")
    return {
        "name": path.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "filesystem_mtime_utc": _utc(after.st_mtime),
    }


def build_lock(args: argparse.Namespace) -> dict[str, object]:
    source_url, url_redacted = _safe_url(args.source_url)
    archives = {"train": _file_record(Path(args.train_archive))}
    if args.test_archive:
        archives["test"] = _file_record(Path(args.test_archive))

    supporting_artifacts = {
        "schema": _file_record(Path(args.schema)),
        "inspection": _file_record(Path(args.inspection)),
    }
    if args.split:
        supporting_artifacts["split"] = _file_record(Path(args.split))

    return {
        "schema_version": 1,
        "kind": "competition_data_source_provenance_lock",
        "observed_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "timestamp_semantics": {
            "observed_at_utc": "time this lock was generated",
            "filesystem_mtime_utc": "filesystem metadata only; not an acquisition timestamp",
            "acquisition_time": "unknown; not asserted",
        },
        "source": {
            "url_without_query_fragment_or_credentials": source_url,
            "sensitive_url_components_removed": url_redacted,
        },
        "archives": archives,
        "supporting_artifacts": supporting_artifacts,
    }


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hash competition archives and local audit artifacts into a compact "
            "source-provenance lock. Archive contents are never inspected."
        )
    )
    parser.add_argument("--train-archive", required=True)
    parser.add_argument("--test-archive")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--schema", default="artifacts/train_schema.json")
    parser.add_argument("--inspection", default="artifacts/data_inspection.json")
    parser.add_argument("--split")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    inputs = [args.train_archive, args.schema, args.inspection]
    if args.test_archive:
        inputs.append(args.test_archive)
    if args.split:
        inputs.append(args.split)
    output = Path(args.output).resolve()
    if output in {Path(value).resolve() for value in inputs}:
        raise ValueError("output must not overwrite an input")

    payload = build_lock(args)
    _write_atomic(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
