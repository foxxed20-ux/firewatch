"""Create an immutable code/model release archive, excluding datasets and secrets."""

from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--model-root",
        help="Verified complete model bundle to freeze inside the release",
    )
    parser.add_argument("--bootstrap-without-model", action="store_true")
    args = parser.parse_args()
    if (
        not args.name
        or args.name in {".", ".."}
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in args.name
        )
    ):
        parser.error("unsafe release name")
    if bool(args.model_root) == args.bootstrap_without_model:
        parser.error(
            "choose --model-root for a complete release or --bootstrap-without-model"
        )
    root = Path(args.root).resolve()
    output = root / "artifacts/service/releases"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (args.name + ".tar.gz")
    if archive.exists():
        parser.error("release already exists")
    paths = []
    for folder in ("firewatch_service", "competition", "web"):
        for path in (root / folder).rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                paths.append((path, path.relative_to(root).as_posix()))
    for name in ("requirements-service.txt", "MODEL_API.md", "README_SERVICE.md"):
        if (root / name).is_file():
            paths.append((root / name, name))
    if args.model_root:
        model_root = Path(args.model_root).resolve()
        import sys

        sys.path.insert(0, str(root))
        from competition.service_bridge import describe_models

        registry = describe_models(model_root)
        if not registry.get("available"):
            parser.error(
                "bundle integrity check failed: " + str(registry.get("reason"))
            )
        for path in model_root.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
            ):
                paths.append(
                    (path, "model_bundle/" + path.relative_to(model_root).as_posix())
                )
    files = {}
    with tarfile.open(archive, "x:gz") as tar:
        for path, name in sorted(paths, key=lambda item: item[1]):
            data = path.read_bytes()
            files[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        if args.model_root:
            data = f"FIREWATCH_MODEL_ROOT=/home/red/firewatch/releases/{args.name}/model_bundle\n".encode()
            files["release.env"] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            info = tarfile.TarInfo("release.env")
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
        manifest = json.dumps(
            {"release": args.name, "files": files}, sort_keys=True, indent=2
        ).encode()
        info = tarfile.TarInfo("release-manifest.json")
        info.size = len(manifest)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(manifest))
    receipt = {
        "release": args.name,
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": files,
    }
    (output / (args.name + ".json")).write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    main()
