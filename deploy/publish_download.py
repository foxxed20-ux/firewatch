"""Publish a SHA-pinned archive into the existing public downloads directory.

Run on the application server as the service user. A temporary file on the same
filesystem is verified before a hard link exposes its final immutable name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_archive(source: Path, directory: Path, name: str, expected: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:zip|tar\.gz)", name):
        raise ValueError("A safe versioned archive filename is required")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Expected SHA256 must have 64 hexadecimal characters")
    expected = expected.lower()
    directory = directory.resolve(strict=True)
    destination = directory / name
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or sha256(destination) != expected:
            raise FileExistsError("Published name already contains another artifact")
        return {
            "status": "already_published",
            "name": name,
            "sha256": expected,
            "bytes": destination.stat().st_size,
        }

    # Stage outside the public directory, but on its parent filesystem. A
    # partially copied archive must never be available through StaticFiles.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".firewatch-download-", dir=directory.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected:
            raise ValueError("Archive SHA256 mismatch; nothing published")
        # link() refuses an existing name, including a concurrent publication.
        os.link(temporary, destination)
        return {
            "status": "published",
            "name": name,
            "sha256": expected,
            "bytes": destination.stat().st_size,
        }
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish_archive(args.archive, args.directory, args.name, args.sha256)
        )
    )


if __name__ == "__main__":
    main()
