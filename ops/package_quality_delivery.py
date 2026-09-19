"""Create a minimal, integrity-checked quality-model delivery archive.

The CLI deliberately uses explicit allowlists.  It packages no raw rasters,
probability caches, presentation material, or automatically authored claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any


OPS_ALLOWLIST = (
    "package_quality_delivery.py",
    "train_quality_cnn.py",
    "train_quality_tree.py",
    "cache_cnn_validation.py",
    "cache_quality_cnn.py",
    "prepare_quality_validation.py",
    "compare_quality_candidates.py",
    "compare_quality_balanced_tree.py",
    "blend_quality_tree_cache.py",
    "compare_quality_af.py",
    "compare_quality_spatial.py",
    "bootstrap_bs_comparison.py",
    "select_bundle.py",
    "tune_postprocess.py",
    "validate_submission.py",
    "replay_validation.py",
    "subgroup_diagnostics.py",
)
TEST_ALLOWLIST = (
    "test_competition_contract.py",
    "test_model_runtime_contract.py",
    "test_training_shapes.py",
    "test_tree_contract.py",
    "test_postprocess_tuning.py",
    "test_quality_candidates.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    return path.resolve()


def _safe_add(files: dict[str, Path], archive_name: str, path: Path) -> None:
    normalized = Path(archive_name)
    if normalized.is_absolute() or ".." in normalized.parts or archive_name in files:
        raise ValueError(f"unsafe or duplicate archive path: {archive_name!r}")
    files[normalized.as_posix()] = _require_file(path, label=archive_name)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_require_file(path, label=label).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _declared_model_hashes(manifest: dict[str, Any], bundle: Path) -> dict[Path, str]:
    """Read all component file hashes declared by select_bundle's manifest."""
    expected: dict[Path, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            relative = value.get("path")
            hashes = value.get("sha256")
            if isinstance(relative, str) and isinstance(hashes, dict):
                base = (bundle / relative).resolve()
                if not _inside(base, bundle):
                    raise ValueError(
                        f"manifest model path escapes bundle: {relative!r}"
                    )
                for name, digest in hashes.items():
                    if not isinstance(name, str) or not isinstance(digest, str):
                        raise ValueError("manifest contains a malformed model hash")
                    target = (base / name).resolve()
                    if not _inside(target, bundle):
                        raise ValueError("manifest hash path escapes bundle")
                    if target in expected and expected[target] != digest:
                        raise ValueError(f"conflicting manifest hash for {target}")
                    expected[target] = digest.lower()
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    if not expected:
        raise ValueError("model manifest contains no declared component hashes")
    return expected


def _verify_model_bundle(bundle: Path, manifest: dict[str, Any], root: Path) -> None:
    if not _inside(bundle, root):
        raise ValueError("--model-dir must stay inside the repository")
    declared = _declared_model_hashes(manifest, bundle)
    actual = {path.resolve() for path in bundle.rglob("*") if path.is_file()}
    manifest_path = (bundle / "manifest.json").resolve()
    if manifest_path not in actual:
        raise FileNotFoundError("model bundle lacks manifest.json")
    undeclared = actual - {manifest_path} - set(declared)
    if undeclared:
        raise ValueError(
            f"model files missing manifest hashes: {sorted(map(str, undeclared))}"
        )
    missing = set(declared) - actual
    if missing:
        raise FileNotFoundError(
            f"manifest hashes missing files: {sorted(map(str, missing))}"
        )
    for path, expected in declared.items():
        actual_hash = _sha256(path)
        if actual_hash.lower() != expected:
            raise ValueError(f"model hash mismatch: {path.relative_to(bundle)}")

    # This is the runtime availability gate, independent from manifest parsing.
    from competition.service_bridge import describe_models

    described = describe_models(bundle)
    if not isinstance(described, dict) or described.get("available") is not True:
        raise ValueError(f"model runtime unavailable: {described}")


def _validate_submission(
    submission: Path, evidence: Path, manifest: dict[str, Any]
) -> Path:
    validation = _read_json(
        evidence / "submission_validation.json", label="submission validation"
    )
    if validation.get("status") != "pass":
        raise ValueError("submission_validation.json must have status='pass'")
    if validation.get("submission_sha256") != _sha256(submission):
        raise ValueError("submission bytes differ from independently validated SHA256")
    audit = _read_json(submission.with_suffix(".audit.json"), label="submission audit")
    if audit.get("bundle_id") != manifest.get("bundle_id") or audit.get("rows") != 447:
        raise ValueError("submission audit must match model bundle_id and 447 rows")
    return submission.with_suffix(".audit.json")


def _collect(
    root: Path, bundle: Path, submission: Path, evidence: Path, report: Path
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in (
        "inference.py",
        "train.py",
        "requirements-model.txt",
        "LICENSE",
        "NOTICE",
    ):
        _safe_add(files, name, root / name)
    for path in sorted((root / "competition").glob("*.py")):
        _safe_add(files, path.relative_to(root).as_posix(), path)
    for name in OPS_ALLOWLIST:
        _safe_add(files, f"ops/{name}", root / "ops" / name)
    for name in TEST_ALLOWLIST:
        _safe_add(files, f"tests/{name}", root / "tests" / name)
    _safe_add(files, "artifacts/split.json", root / "artifacts" / "split.json")
    for name in ("best.pt", "config.json"):
        _safe_add(
            files, f"initial/cnn/bs/{name}", root / "artifacts" / "cnn-v1" / "bs" / name
        )
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            _safe_add(
                files, f"model_bundle/{path.relative_to(bundle).as_posix()}", path
            )
    for path in sorted(evidence.glob("*.json")):
        _safe_add(files, f"evidence/{path.name}", path)
    _safe_add(files, f"report/{report.name}", report)
    _safe_add(files, "submission.csv", submission)
    _safe_add(files, "submission.audit.json", submission.with_suffix(".audit.json"))
    return files


def _inventory(files: dict[str, Path], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": manifest["bundle_id"],
        "scope": "Minimal quality-model delivery: runtime code, selected bundle, continuation seed, validation evidence, supplied report, and submission. No raw rasters, probability caches, jury PDFs, or presentation files.",
        "files": {
            name: {"sha256": _sha256(path), "size": path.stat().st_size}
            for name, path in sorted(files.items())
        },
    }


def _write_verified_zip(
    output: Path, files: dict[str, Path], inventory: dict[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, name)
            archive.writestr("DELIVERY_MANIFEST.json", json.dumps(inventory, indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("archive CRC verification failed")
            for name, record in inventory["files"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != record["sha256"]:
                    raise RuntimeError(f"archive content hash mismatch: {name}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_delivery(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    bundle = Path(args.model_dir).resolve()
    submission = _require_file(Path(args.submission), label="submission")
    evidence = Path(args.evidence_dir).resolve()
    report = _require_file(Path(args.report), label="report")
    if (
        not _inside(evidence, root)
        or not _inside(report, root)
        or not _inside(submission, root)
    ):
        raise ValueError(
            "submission, evidence, and report must stay inside the repository"
        )
    if report.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("--report must be a Markdown file supplied by the caller")
    manifest = _read_json(bundle / "manifest.json", label="model manifest")
    if not isinstance(manifest.get("bundle_id"), str) or not manifest["bundle_id"]:
        raise ValueError("model manifest lacks bundle_id")
    _verify_model_bundle(bundle, manifest, root)
    audit = _validate_submission(submission, evidence, manifest)
    files = _collect(root, bundle, submission, evidence, report)
    if files["submission.audit.json"] != audit:
        raise AssertionError("submission audit collection mismatch")
    inventory = _inventory(files, manifest)
    output = Path(args.output).resolve()
    if not _inside(output, root):
        raise ValueError("--output must stay inside the repository")
    _write_verified_zip(output, files, inventory)
    return {
        "bundle_id": manifest["bundle_id"],
        "path": str(output),
        "files": len(files) + 1,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "crc_and_content_hashes_verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_delivery(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
