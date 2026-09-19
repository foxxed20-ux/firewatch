"""Build a self-contained model delivery from an explicit file allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    bundle = Path(args.model_dir).resolve()
    submission = Path(args.submission).resolve()
    output = Path(args.output).resolve()
    evidence = Path(args.evidence_dir).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    files = {}
    for name in (
        "inference.py",
        "train.py",
        "requirements-model.txt",
        "README_MODEL.md",
        "docs/MODEL_REPORT.md",
        "artifacts/split.json",
        "artifacts/data_inspection.json",
        "artifacts/train_schema.json",
        "artifacts/source-training-v1.zip",
        "artifacts/data_source_lock.json",
        "artifacts/split_overlap_audit.json",
    ):
        files[name] = root / name
    for folder, pattern in (
        ("competition", "*.py"),
        ("docs/environment", "*"),
        ("artifacts/training", "*.json"),
    ):
        for path in sorted((root / folder).glob(pattern)):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    for name in (
        "README.md",
        "README_SERVICE.md",
        "MODEL_API.md",
        "THIRD_PARTY.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "docs/MODEL_DELIVERY.md",
        "docs/assets/firewatch-banner.svg",
        "docs/assets/firewatch-bs-demo.png",
        "docs/presentation/presentation.md",
        "docs/presentation/metrics.json",
        "docs/presentation/FireWatch_КосмоХакатон_2026.pptx",
        "docs/evidence/service-release-summary.json",
        "docs/evidence/service-v3-release-summary.json",
        "output/pdf/FireWatch_Project_Passport.pdf",
        "output/pdf/FireWatch_Domain_Research.pdf",
    ):
        if (root / name).is_file():
            files[name] = root / name
    for path in sorted((root / "docs").glob("*.md")):
        if path.name != "EXTERNAL_AGENTS.md":
            files[path.relative_to(root).as_posix()] = path
    for name in (
        "tests/test_competition_contract.py",
        "tests/test_training_shapes.py",
        "tests/test_tree_contract.py",
        "tests/test_model_runtime_contract.py",
        "tests/test_postprocess_tuning.py",
        "artifacts/tree-v1/validation_report.json",
        "artifacts/tree-v1/selected_recipes.json",
        "artifacts/tree-v1/tuned_recipes.json",
        "artifacts/tree-v1/demo_smoke/report.json",
        "artifacts/cnn-v1/af/validation_report.json",
        "artifacts/ensemble-v2/selected_recipes.json",
        "artifacts/ensemble-v2/subgroup_diagnostics.json",
        "artifacts/ensemble-v2/demo_smoke/report.json",
        "artifacts/xgb-v1/bs/comparison.json",
        "artifacts/xgb-v1/bs/bootstrap_comparison.json",
        "artifacts/service/public-mask-parity.json",
        "artifacts/service/public-v3-mask-parity.json",
        "artifacts/service/public-v3-release-summary.json",
        "artifacts/service/server-smoke-ensemble-v3.json",
        "artifacts/service/public-model-download-v3.json",
        "artifacts/frontend/v3-browser-qa.json",
        "artifacts/ensemble-v3/selected_recipes.json",
        "artifacts/ensemble-v3/subgroup_diagnostics.json",
        "artifacts/ensemble-v3/demo_smoke/report.json",
        "artifacts/cnn-v1/bs/comparison.json",
        "artifacts/cnn-v1/bs/bootstrap_comparison.json",
        "artifacts/cnn-v1/bs/validation_cache_provenance.json",
        "artifacts/cnn-v1/bs/cache_download_verification.json",
    ):
        if (root / name).is_file():
            files[name] = root / name
    for name in (
        "select_bundle.py",
        "tune_postprocess.py",
        "verify_bundle_demo.py",
        "validate_submission.py",
        "replay_validation.py",
        "package_delivery.py",
        "compare_tree_candidates.py",
        "bootstrap_bs_comparison.py",
        "subgroup_diagnostics.py",
        "compress_feature_cache.py",
        "cache_cnn_validation.py",
        "verify_training_source.py",
        "lock_data_source.py",
        "build_model_release_evidence.py",
        "audit_split_overlap.py",
    ):
        files["ops/" + name] = root / "ops" / name
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            files["model_bundle/" + path.relative_to(bundle).as_posix()] = path
    for path in sorted(evidence.glob("*.json")):
        files["evidence/" + path.name] = path
        if path.is_relative_to(root):
            # Preserve repository links from the technical report as well.
            files[path.relative_to(root).as_posix()] = path
    files["submission.csv"] = submission
    required_evidence = evidence / "submission_validation.json"
    if (
        not required_evidence.is_file()
        or json.loads(required_evidence.read_text()).get("status") != "pass"
    ):
        raise ValueError("An independently verified written submission is required")
    checked_submission = json.loads(required_evidence.read_text())
    if checked_submission.get("submission_sha256") != sha256(submission):
        raise ValueError(
            "Submission bytes differ from the independently checked artifact"
        )
    audit_path = submission.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if audit.get("bundle_id") != manifest["bundle_id"] or audit.get("rows") != 447:
        raise ValueError("Submission audit belongs to a different bundle or row count")
    files["submission.audit.json"] = audit_path
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing delivery item {name}: {path}")
    inventory = {
        "bundle_id": manifest["bundle_id"],
        "scope": "Competition model code, weights, predictions, evidence and jury documents. Service source is in the linked project repository. No source raster data.",
        "files": {
            name: {"sha256": sha256(path), "size": path.stat().st_size}
            for name, path in files.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("DELIVERY_MANIFEST.json", json.dumps(inventory, indent=2))
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for name, record in inventory["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Archive content hash mismatch: {name}")
    report = {
        "bundle_id": manifest["bundle_id"],
        "path": str(output),
        "files": len(files) + 1,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "crc_and_content_hashes_verified": True,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
