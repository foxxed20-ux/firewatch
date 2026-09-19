#!/usr/bin/env python3
"""Fail fast when tracked repository content is unsafe or internally inconsistent."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
DENY_SUFFIXES = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar",
    ".tif", ".tiff", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".onnx",
    ".h5", ".hdf5",
    ".pem", ".key", ".p12", ".pfx",
)
PRIVATE_FILE = re.compile(r"(?:^|/)\.env(?:$|\.)", re.IGNORECASE)
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))[^)]*\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))", re.MULTILINE)
HTML_LINK = re.compile(r"\b(?:src|href)\s*=\s*([\"'])(.*?)\1", re.IGNORECASE)
FENCED_CODE = re.compile(r"(?ms)^ {0,3}(\x60{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$")


def tracked_paths() -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [PurePosixPath(item) for item in result.stdout.decode("utf-8").split("\0") if item]


def is_local_link(destination: str) -> bool:
    value = destination.strip()
    return bool(value) and not (
        value.startswith(("#", "/", "//", "mailto:", "data:"))
        or re.match(r"[a-z][a-z0-9+.-]*:", value, flags=re.IGNORECASE)
    )


def markdown_destinations(text: str) -> list[str]:
    text = FENCED_CODE.sub("", text)
    matches = list(INLINE_LINK.finditer(text)) + list(REFERENCE_LINK.finditer(text))
    return [match.group(1) or match.group(2) for match in matches] + [
        match.group(2) for match in HTML_LINK.finditer(text)
    ]


def local_target_is_tracked(
    markdown: PurePosixPath, destination: str, tracked: set[PurePosixPath]
) -> bool:
    relative = unquote(destination.split("#", 1)[0].split("?", 1)[0]).replace("\\", "/")
    if not relative:
        return True
    candidate = (ROOT / markdown.parent / relative).resolve()
    try:
        candidate_relative = PurePosixPath(candidate.relative_to(ROOT.resolve()).as_posix())
    except ValueError:
        return False
    if candidate_relative in tracked:
        return True
    prefix = f"{candidate_relative.as_posix().rstrip('/')}/"
    return any(path.as_posix().startswith(prefix) for path in tracked)


def forbidden_reason(relative: PurePosixPath) -> str | None:
    path_text = relative.as_posix()
    top_level = relative.parts[0] if relative.parts else ""
    if top_level.startswith((".venv", ".chart-data-")) or top_level in {".codex", "data", "prepared", "runs", "model_bundle"}:
        return "private, local, or generated root"
    if path_text in {"deploy/private", "deploy/tunnel_key"} or path_text.startswith("deploy/private/"):
        return "private deployment material"
    filename = relative.name.lower()
    if filename.startswith(("credentials", "secrets")) and filename.endswith(".json"):
        return "credential material"
    if PRIVATE_FILE.search(path_text) and filename not in {".env.example", "service.env.example"}:
        return "environment file"
    if filename.endswith(DENY_SUFFIXES):
        return "dataset, archive, or model-weight artifact"
    return None


def main() -> int:
    errors: list[str] = []
    paths = tracked_paths()
    tracked = set(paths)
    for relative in paths:
        path_text = relative.as_posix()
        full_path = ROOT / relative
        if reason := forbidden_reason(relative):
            errors.append(f"{reason} is tracked: {path_text}")
        if full_path.is_file() and full_path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"tracked file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB: {path_text}")
        if relative.suffix.lower() not in {".md", ".markdown"} or not full_path.is_file():
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Markdown file is not UTF-8: {path_text}")
            continue
        for destination in markdown_destinations(text):
            if is_local_link(destination) and not local_target_is_tracked(relative, destination, tracked):
                errors.append(f"broken local Markdown link in {path_text}: {destination}")

    if errors:
        print("Repository check failed:", file=sys.stderr)
        print(*[f"- {error}" for error in errors], sep="\n", file=sys.stderr)
        return 1
    print("Repository check passed: tracked paths, file sizes, and local Markdown links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
