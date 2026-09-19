"""Verify an immutable release, optionally extracting a SHA-pinned archive first."""

from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import tarfile


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    if args.archive:
        archive = Path(args.archive)
        if not args.sha256 or digest(archive) != args.sha256.lower():
            raise ValueError("Release archive SHA256 mismatch")
        if root.exists():
            raise ValueError(
                "Release destination already exists; use a new immutable name"
            )
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            names = set()
            for member in members:
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or pure.as_posix() != member.name
                    or ".." in pure.parts
                    or "\\" in member.name
                    or not member.isfile()
                    or member.name in names
                ):
                    raise ValueError("Unsafe or duplicate archive member")
                names.add(member.name)
                if not (root / member.name).resolve().is_relative_to(root):
                    raise ValueError("Escaping archive member")
            root.mkdir(parents=True)
            tar.extractall(root)
    manifest_path = root / "release-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Release manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if actual != set(expected) | {"release-manifest.json"}:
        raise ValueError("Release file inventory differs from manifest")
    for relative, record in expected.items():
        raw_path = root / relative
        path = raw_path.resolve()
        if not path.is_relative_to(root) or raw_path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe release path")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Release file checksum mismatch: {relative}")
    print(
        json.dumps(
            {
                "status": "verified",
                "release": manifest["release"],
                "files": len(expected),
                "manifest_sha256": digest(root / "release-manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
