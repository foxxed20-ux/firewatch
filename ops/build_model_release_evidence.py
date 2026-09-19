"""Build a generic release-gate manifest from verified FireWatch receipts.

This command does not run inference, retrain models, or execute the release
audit.  It validates already-produced artifacts and writes a self-contained
input manifest for the release-gate tool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY_MANIFEST = "DELIVERY_MANIFEST.json"
TARGET_WINDOWS = "Windows Python 3.13 CPU"
TARGET_COLAB = "Google Colab Python 3.13 CPU replay"
TARGET_UBUNTU = "Ubuntu Python 3.10 CPU smoke"


class EvidenceError(RuntimeError):
    """Raised when a claimed receipt cannot be verified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON receipt: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON receipt {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON receipt must be an object: {path}")
    return value


def json_from_zip(archive: ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name).decode("utf-8-sig"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid or missing {name} in ZIP") from exc
    require(isinstance(value, dict), f"{name} in ZIP must be an object")
    return value


def json_from_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid packaged JSON {label}: {exc}") from exc
    require(isinstance(value, dict), f"packaged JSON {label} must be an object")
    return value


def canonical_payload_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def unwrap_signed_receipt(value: dict[str, Any], label: str) -> dict[str, Any]:
    if "payload" not in value:
        return value
    payload = value.get("payload")
    claimed = value.get("sha256")
    require(isinstance(payload, dict), f"{label}: payload must be an object")
    require(
        isinstance(claimed, str) and claimed == canonical_payload_sha(payload),
        f"{label}: signed payload SHA-256 mismatch",
    )
    return payload


def safe_zip_names(archive: ZipFile, label: str) -> list[str]:
    infos = [item for item in archive.infolist() if not item.is_dir()]
    names = [item.filename for item in infos]
    require(len(names) == len(set(names)), f"{label}: duplicate ZIP entry")
    for name in names:
        posix = PurePosixPath(name)
        require(
            name == posix.as_posix()
            and not name.startswith("/")
            and "\\" not in name
            and ".." not in posix.parts,
            f"{label}: unsafe or non-portable ZIP entry {name!r}",
        )
    bad = archive.testzip()
    require(bad is None, f"{label}: CRC failure in {bad!r}")
    return names


def component_artifacts(spec: dict[str, Any]) -> list[tuple[str, str]]:
    backend = spec.get("backend")
    if backend == "ensemble":
        components = spec.get("components")
        require(
            isinstance(components, dict) and set(components) == {"cnn", "tree"},
            "bundle ensemble must contain cnn and tree components",
        )
        weight = spec.get("tree_weight")
        require(
            isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and math.isfinite(float(weight))
            and 0.0 <= float(weight) <= 1.0,
            "bundle ensemble tree_weight is invalid",
        )
        result: list[tuple[str, str]] = []
        for component in components.values():
            require(isinstance(component, dict), "bundle component must be an object")
            result.extend(component_artifacts(component))
        return result

    require(backend in {"torch", "lightgbm"}, f"unsupported bundle backend {backend!r}")
    raw_path = spec.get("path")
    hashes = spec.get("sha256")
    require(isinstance(raw_path, str) and raw_path, "bundle component path is missing")
    path = PurePosixPath(raw_path)
    require(
        raw_path == path.as_posix()
        and not raw_path.startswith("/")
        and "\\" not in raw_path
        and ".." not in path.parts,
        f"bundle component path is unsafe or non-portable: {raw_path!r}",
    )
    require(isinstance(hashes, dict) and hashes, f"bundle hashes missing for {raw_path}")
    required = {"best.pt"} if backend == "torch" else {"model.txt", "metadata.json"}
    require(required.issubset(hashes), f"bundle component is incomplete: {raw_path}")
    result = []
    for name, digest in hashes.items():
        require(
            isinstance(name, str)
            and PurePosixPath(name).name == name
            and isinstance(digest, str)
            and len(digest) == 64,
            f"invalid component hash entry in {raw_path}",
        )
        result.append(((path / name).as_posix(), digest.lower()))
    return result


def validate_bundle_zip(path: Path) -> tuple[str, dict[str, bytes]]:
    require(path.is_file(), f"missing bundle ZIP: {path}")
    try:
        with ZipFile(path) as archive:
            names = safe_zip_names(archive, "bundle ZIP")
            require("manifest.json" in names, "bundle ZIP lacks manifest.json")
            manifest = json_from_zip(archive, "manifest.json")
            bundle_id = manifest.get("bundle_id")
            require(
                isinstance(bundle_id, str)
                and re.fullmatch(r"[A-Za-z0-9._-]+", bundle_id) is not None,
                f"invalid bundle_id: {bundle_id!r}",
            )
            require(
                manifest.get("feature_version") == "official-named-v1",
                "unexpected bundle feature_version",
            )
            tasks = manifest.get("tasks")
            require(isinstance(tasks, dict) and set(tasks) == {"af", "bs"}, "bundle tasks must be af and bs")
            artifacts: list[tuple[str, str]] = []
            for spec in tasks.values():
                require(isinstance(spec, dict), "bundle task spec must be an object")
                artifacts.extend(component_artifacts(spec))
            for name, expected in artifacts:
                require(name in names, f"bundle component is absent: {name}")
                require(sha256_bytes(archive.read(name)) == expected, f"bundle component hash mismatch: {name}")
            return bundle_id, {name: archive.read(name) for name in names}
    except BadZipFile as exc:
        raise EvidenceError(f"invalid bundle ZIP {path}: {exc}") from exc


def validate_delivery_zip(
    path: Path, sidecar_path: Path, bundle_entries: dict[str, bytes], bundle_id: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(path.is_file(), f"missing delivery ZIP: {path}")
    sidecar = read_json(sidecar_path)
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    require(sidecar.get("bytes") == actual_size, "delivery sidecar byte size mismatch")
    require(sidecar.get("sha256") == actual_sha, "delivery sidecar SHA-256 mismatch")
    require(sidecar.get("bundle_id") == bundle_id, "delivery sidecar bundle_id mismatch")
    require(sidecar.get("crc_and_content_hashes_verified") is True, "delivery sidecar lacks verification flag")

    try:
        with ZipFile(path) as archive:
            names = safe_zip_names(archive, "delivery ZIP")
            require(sidecar.get("files") == len(names), "delivery sidecar file count mismatch")
            require(DELIVERY_MANIFEST in names, "delivery ZIP lacks DELIVERY_MANIFEST.json")
            manifest = json_from_zip(archive, DELIVERY_MANIFEST)
            require(manifest.get("bundle_id") == bundle_id, "DELIVERY_MANIFEST bundle_id mismatch")
            files = manifest.get("files")
            require(isinstance(files, dict), "DELIVERY_MANIFEST files must be an object")
            require(set(files) == set(names) - {DELIVERY_MANIFEST}, "DELIVERY_MANIFEST file inventory mismatch")
            payloads: dict[str, bytes] = {}
            for name in names:
                data = archive.read(name)
                payloads[name] = data
                if name == DELIVERY_MANIFEST:
                    continue
                receipt = files[name]
                require(isinstance(receipt, dict), f"invalid DELIVERY_MANIFEST entry: {name}")
                require(receipt.get("size") == len(data), f"delivery member size mismatch: {name}")
                require(receipt.get("sha256") == sha256_bytes(data), f"delivery member hash mismatch: {name}")

            embedded_names = {name.removeprefix("model_bundle/") for name in names if name.startswith("model_bundle/")}
            require(embedded_names == set(bundle_entries), "embedded model bundle inventory differs from bundle ZIP")
            for name, data in bundle_entries.items():
                require(payloads[f"model_bundle/{name}"] == data, f"embedded bundle member differs: {name}")
            return manifest, payloads
    except BadZipFile as exc:
        raise EvidenceError(f"invalid delivery ZIP {path}: {exc}") from exc


def validate_submission(
    receipt: dict[str, Any],
    csv_bytes: bytes,
    bundle_receipt: dict[str, Any],
    bundle_id: str,
) -> dict[str, Any]:
    require(receipt == bundle_receipt, "external and packaged submission receipts differ")
    if "bundle_id" in receipt:
        require(receipt.get("bundle_id") == bundle_id, "submission receipt bundle_id mismatch")
    required_true = ("header", "template_order_exact", "canonical_rle", "bs_classes_exclusive")
    require(receipt.get("status") == "pass", "submission receipt did not pass")
    require(receipt.get("data_rows") == 447, "submission receipt must report 447 data rows")
    require(receipt.get("af_chips") == 180 and receipt.get("bs_chips") == 89, "submission chip counts are wrong")
    require(all(receipt.get(key) is True for key in required_true), "submission contract flags are incomplete")
    require(receipt.get("submission_sha256") == sha256_bytes(csv_bytes), "submission CSV SHA-256 mismatch")

    try:
        csv.field_size_limit(max(csv.field_size_limit(), len(csv_bytes)))
        rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise EvidenceError(f"submission.csv is not valid UTF-8 CSV: {exc}") from exc
    require(rows and rows[0] == ["chip_id", "class_id", "rle"], "submission.csv header mismatch")
    require(len(rows) == 448, "submission.csv must contain a header and 447 data rows")
    keys: list[tuple[str, int]] = []
    for row in rows[1:]:
        require(len(row) == 3 and row[0], "submission.csv contains a malformed row")
        try:
            key = (row[0], int(row[1]))
        except ValueError as exc:
            raise EvidenceError("submission.csv class_id is not an integer") from exc
        require(
            (key[0].startswith("AF_") and key[1] == 1)
            or (key[0].startswith("BS_") and key[1] in {1, 2, 3}),
            f"submission.csv contains an unexpected key: {key}",
        )
        keys.append(key)
    require(len(keys) == len(set(keys)), "submission.csv contains duplicate chip/class rows")
    return receipt


def validate_cpu_replay(
    receipt: dict[str, Any],
    bundle_receipt: dict[str, Any],
    bundle_id: str,
    selected_recipes: dict[str, Any],
    bundle_manifest: dict[str, Any],
) -> dict[str, Any]:
    require(receipt == bundle_receipt, "external and packaged CPU replay receipts differ")
    require(receipt.get("bundle_id") == bundle_id, "CPU replay bundle_id mismatch")
    require(receipt.get("device") == "cpu", "CPU replay did not use CPU")
    require(receipt.get("network_blocked") is True, "CPU replay did not record blocked network")
    require(receipt.get("within_tolerance") is True, "CPU replay is outside tolerance")
    tolerance = receipt.get("tolerance")
    require(
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and 0.0 <= float(tolerance) <= 0.005,
        "CPU replay tolerance is missing or exceeds 0.005",
    )
    tolerance = float(tolerance)
    deltas = receipt.get("absolute_delta")
    metrics = receipt.get("metrics")
    require(isinstance(metrics, dict), "CPU replay metrics missing")
    expected = receipt.get("expected_validation")
    metric_keys = {"F1_af", "IoU_burn", "mIoU_sev"}
    require(isinstance(expected, dict) and set(expected) == metric_keys, "CPU replay expected_validation keys are invalid")
    require(isinstance(deltas, dict) and set(deltas) == metric_keys, "CPU replay absolute_delta keys are invalid")
    require(isinstance(selected_recipes, dict), "packaged selected recipes must be an object")
    tasks = bundle_manifest.get("tasks")
    require(isinstance(tasks, dict) and set(tasks) == {"af", "bs"}, "bundle manifest task selections are invalid")
    for task in ("af", "bs"):
        recipe = selected_recipes.get(task)
        spec = tasks.get(task)
        selection = spec.get("selection") if isinstance(spec, dict) else None
        require(isinstance(recipe, dict), f"selected recipe is missing for {task}")
        require(selection == recipe, f"packaged {task} recipe differs from bundle manifest selection")
    expected_from_recipe = {
        "F1_af": selected_recipes["af"].get("F1_af"),
        "IoU_burn": selected_recipes["bs"].get("IoU_burn"),
        "mIoU_sev": selected_recipes["bs"].get("mIoU_sev"),
    }
    require(
        expected == expected_from_recipe,
        "CPU replay expected_validation differs from packaged selected recipes",
    )
    computed: dict[str, float] = {}
    for key in sorted(metric_keys):
        actual_value, expected_value, reported_delta = (
            metrics.get(key),
            expected.get(key),
            deltas.get(key),
        )
        require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (actual_value, expected_value, reported_delta)
            ),
            f"CPU replay contains a non-finite {key} value",
        )
        require(
            0.0 <= float(actual_value) <= 1.0
            and 0.0 <= float(expected_value) <= 1.0,
            f"CPU replay {key} metric is outside [0, 1]",
        )
        delta = abs(float(actual_value) - float(expected_value))
        reported = float(reported_delta)
        require(0.0 <= reported <= tolerance, f"CPU replay {key} delta exceeds tolerance")
        require(
            math.isclose(reported, delta, rel_tol=0.0, abs_tol=1e-12),
            f"CPU replay reported {key} delta does not match actual metrics",
        )
        computed[key] = delta
    require(all(value <= tolerance for value in computed.values()), "CPU replay recomputed deltas exceed tolerance")
    score = metrics.get("Score")
    require(isinstance(score, (int, float)) and math.isfinite(float(score)) and 0 <= float(score) <= 1, "CPU replay Score is invalid")
    return receipt


def validate_native_run(
    path: Path, delivery_entries: dict[str, bytes]
) -> tuple[dict[str, Any], list[Path], int]:
    require(path.is_dir(), f"native run directory is missing: {path}")
    plan_path, packet_path, state_path = path / "plan.json", path / "packet.json", path / "state.json"
    plan = read_json(plan_path)
    packet = read_json(packet_path)
    state_document = read_json(state_path)
    state = unwrap_signed_receipt(state_document, "native state")
    require(state.get("active") is None and state.get("finished_at"), "native run is unfinished")
    require(state.get("plan_sha256") == sha256_file(plan_path), "native plan SHA-256 mismatch")
    require(state.get("packet_sha256") == sha256_file(packet_path), "native packet SHA-256 mismatch")
    mandatory = packet.get("mandatory")
    snapshot = mandatory.get("snapshot") if isinstance(mandatory, dict) else None
    snapshot_files = snapshot.get("files") if isinstance(snapshot, dict) else None
    require(
        isinstance(snapshot_files, list) and snapshot_files,
        "native packet has no mandatory snapshot files",
    )
    seen_snapshot_paths: set[str] = set()
    for item in snapshot_files:
        require(isinstance(item, dict), "native snapshot file entry must be an object")
        name, size, expected = item.get("path"), item.get("bytes"), item.get("sha256")
        require(isinstance(name, str) and name, "native snapshot file path is missing")
        posix = PurePosixPath(name)
        require(
            name == posix.as_posix()
            and not name.startswith("/")
            and "\\" not in name
            and ".." not in posix.parts,
            f"native snapshot path is unsafe or non-portable: {name!r}",
        )
        require(name not in seen_snapshot_paths, f"duplicate native snapshot path: {name}")
        seen_snapshot_paths.add(name)
        require(name in delivery_entries, f"delivery ZIP lacks native snapshot file: {name}")
        data = delivery_entries[name]
        require(
            isinstance(size, int)
            and not isinstance(size, bool)
            and size == len(data),
            f"delivery/native snapshot size mismatch: {name}",
        )
        require(
            isinstance(expected, str) and expected == sha256_bytes(data),
            f"delivery/native snapshot SHA-256 mismatch: {name}",
        )
    plan_checks = plan.get("checks")
    checks = state.get("checks")
    require(isinstance(plan_checks, list) and plan_checks, "native plan contains no checks")
    require(isinstance(checks, list) and checks, "native state contains no checks")
    planned_ids = {item.get("id") for item in plan_checks if isinstance(item, dict) and item.get("required") is True}
    required_checks = [item for item in checks if isinstance(item, dict) and item.get("required") is True]
    require(planned_ids and {item.get("id") for item in required_checks} == planned_ids, "native required check inventory mismatch")

    artifacts = [plan_path, packet_path, state_path]
    for check in required_checks:
        check_id = check.get("id")
        require(
            check.get("status") == "passed"
            and check.get("observed_exit") == 0
            and check.get("capture_complete") is True
            and check.get("cleanup_confirmed") is True,
            f"native check did not pass cleanly: {check_id}",
        )
        for stream in ("stdout", "stderr"):
            output = path / f"{check_id}.{stream}"
            require(output.is_file(), f"native check output missing: {output}")
            claimed = check.get(stream)
            require(isinstance(claimed, dict), f"native {stream} receipt missing for {check_id}")
            require(claimed.get("bytes") == output.stat().st_size, f"native {stream} size mismatch for {check_id}")
            require(claimed.get("sha256") == sha256_file(output), f"native {stream} hash mismatch for {check_id}")
            artifacts.append(output)
    return state, artifacts, len(snapshot_files)


def validate_smoke(receipt: dict[str, Any], bundle_id: str, label: str) -> dict[str, str]:
    require(receipt.get("bundle_id") == bundle_id, f"{label} bundle_id mismatch")
    require(receipt.get("device") == "cpu", f"{label} did not use CPU")
    require(receipt.get("network_connections_blocked") is True, f"{label} did not record blocked network")
    masks = receipt.get("masks")
    require(isinstance(masks, list) and len(masks) == 4, f"{label} must contain four real-scene masks")
    result: dict[str, str] = {}
    task_counts = {"AF": 0, "BS": 0}
    for mask in masks:
        require(isinstance(mask, dict), f"{label} mask receipt is malformed")
        chip_id, digest = mask.get("chip_id"), mask.get("sha256_c_order_uint8")
        require(isinstance(chip_id, str) and isinstance(digest, str) and len(digest) == 64, f"{label} mask digest is invalid")
        if chip_id.startswith("AF_"):
            task, expected_shape, allowed = "AF", [256, 256], {0, 1}
        elif chip_id.startswith("BS_"):
            task, expected_shape, allowed = "BS", [512, 512], {0, 1, 2, 3}
        else:
            raise EvidenceError(f"{label} has an unexpected chip ID: {chip_id}")
        task_counts[task] += 1
        require(mask.get("shape") == expected_shape, f"{label} has wrong shape for {chip_id}")
        require(mask.get("dtype") == "uint8", f"{label} has wrong dtype for {chip_id}")
        counts = mask.get("counts")
        require(isinstance(counts, dict) and counts, f"{label} counts are malformed for {chip_id}")
        parsed_counts: dict[int, int] = {}
        for raw_class, raw_count in counts.items():
            try:
                class_id = int(raw_class)
            except (TypeError, ValueError) as exc:
                raise EvidenceError(f"{label} count class is invalid for {chip_id}") from exc
            require(
                class_id in allowed
                and isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count >= 0,
                f"{label} count is invalid for {chip_id}",
            )
            parsed_counts[class_id] = raw_count
        require(
            sum(parsed_counts.values()) == expected_shape[0] * expected_shape[1],
            f"{label} counts do not cover the full mask for {chip_id}",
        )
        result[chip_id] = digest
    require(len(result) == 4, f"{label} mask chip IDs are not unique")
    require(task_counts == {"AF": 2, "BS": 2}, f"{label} must contain two AF and two BS masks")
    return result


def validate_platform_parity(
    receipt: dict[str, Any], service_summary: dict[str, Any], server_masks: dict[str, str], bundle_id: str
) -> int:
    require(receipt.get("status") == "pass", "public mask parity did not pass")
    require(receipt.get("bundle_id") == bundle_id, "public mask parity bundle_id mismatch")
    checks = receipt.get("checks")
    require(isinstance(checks, list) and len(checks) == 4, "public mask parity must contain four checks")
    parity_masks: dict[str, str] = {}
    for check in checks:
        require(isinstance(check, dict) and check.get("match") is True, "public mask parity contains a mismatch")
        chip_id, digest = check.get("chip_id"), check.get("sha256_c_order_uint8")
        require(isinstance(chip_id, str) and isinstance(digest, str), "public mask parity check is malformed")
        parity_masks[chip_id] = digest
    require(parity_masks == server_masks, "public parity and Ubuntu server smoke digests differ")
    require(service_summary.get("profile") == "web-service" and service_summary.get("verdict") == "PASS", "service release summary did not pass")
    require(bundle_id in str(service_summary.get("release", "")), "service summary release does not identify this bundle")
    targets = service_summary.get("target_environments")
    require(isinstance(targets, list) and any("Python 3.10" in str(value) and "Ubuntu" in str(value) for value in targets), "service summary lacks Ubuntu Python 3.10 evidence")
    gates = service_summary.get("gates")
    require(
        isinstance(gates, list)
        and any(isinstance(gate, dict) and gate.get("id") == "model-parity" and gate.get("status") == "pass" for gate in gates),
        "service summary lacks a passing model-parity gate",
    )
    return len(checks)


def validate_environment_receipts(windows: dict[str, Any], colab: dict[str, Any]) -> None:
    require(str(windows.get("python", "")).startswith("3.13."), "Windows receipt is not Python 3.13")
    require(str(windows.get("platform", "")).startswith("Windows"), "Windows environment receipt has wrong platform")
    require(windows.get("device") == "cpu", "Windows environment receipt is not CPU")
    require(str(colab.get("python", "")).startswith("3.13."), "Colab receipt is not Python 3.13")
    require("Linux" in str(colab.get("platform", "")), "Colab environment receipt has wrong platform")


def artifact(path: Path, *, required: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"artifact is missing: {resolved}")
    try:
        relative = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"artifact is outside project root: {resolved}") from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size": resolved.stat().st_size,
        "required": required,
    }


def evidence(
    *, method: str, procedure: str, environment: str, target: str, observed: str, source: Path
) -> dict[str, str]:
    return {
        "method": method,
        "procedure": procedure,
        "environment": environment,
        "target_environment": target,
        "observed": observed,
        "source": source.resolve().relative_to(PROJECT_ROOT).as_posix(),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    delivery_zip = Path(args.delivery_zip).resolve()
    bundle_zip = Path(args.bundle_zip).resolve()
    evidence_dir = Path(args.evidence_dir).resolve()
    native_dir = Path(args.native_run_dir).resolve()
    sidecar = delivery_zip.with_suffix(".json")

    bundle_id, bundle_entries = validate_bundle_zip(bundle_zip)
    _, delivery_entries = validate_delivery_zip(
        delivery_zip, sidecar, bundle_entries, bundle_id
    )
    recipe_member = (
        f"artifacts/{bundle_id.removeprefix('official-')}/selected_recipes.json"
    )
    required_delivery_evidence = {
        "submission.csv",
        "evidence/submission_validation.json",
        "evidence/cpu_validation_replay.json",
        recipe_member,
    }
    require(
        required_delivery_evidence.issubset(delivery_entries),
        "delivery ZIP lacks required submission, replay, or selected-recipe evidence",
    )

    bundle_manifest_bytes = bundle_entries["manifest.json"]
    bundle_manifest = json_from_bytes(bundle_manifest_bytes, "manifest.json")
    packaged_recipe_bytes = delivery_entries[recipe_member]
    selected_recipes = json_from_bytes(packaged_recipe_bytes, recipe_member)
    recipe_sha = sha256_bytes(packaged_recipe_bytes)
    recipe_path = evidence_dir.parent / "selected_recipes.json"
    require(recipe_path.is_file(), f"missing selected recipe receipt: {recipe_path}")
    require(
        recipe_path.read_bytes() == packaged_recipe_bytes,
        "external and packaged selected recipe receipts differ",
    )

    submission_path = evidence_dir / "submission_validation.json"
    replay_path = evidence_dir / "cpu_validation_replay.json"
    submission = validate_submission(
        read_json(submission_path),
        delivery_entries["submission.csv"],
        json_from_bytes(
            delivery_entries["evidence/submission_validation.json"],
            "evidence/submission_validation.json",
        ),
        bundle_id,
    )
    replay = validate_cpu_replay(
        read_json(replay_path),
        json_from_bytes(
            delivery_entries["evidence/cpu_validation_replay.json"],
            "evidence/cpu_validation_replay.json",
        ),
        bundle_id,
        selected_recipes,
        bundle_manifest,
    )
    native_state, native_artifacts, native_snapshot_files = validate_native_run(
        native_dir, delivery_entries
    )

    demo_path = evidence_dir.parent / "demo_smoke" / "report.json"
    parity_path = (
        Path(args.public_parity).resolve()
        if args.public_parity
        else PROJECT_ROOT / "artifacts/service/public-mask-parity.json"
    )
    server_smoke_name = f"server-smoke-{bundle_id.removeprefix('official-')}.json"
    server_smoke_path = (
        Path(args.server_smoke).resolve()
        if args.server_smoke
        else PROJECT_ROOT / "artifacts/service" / server_smoke_name
    )
    service_summary_path = (
        Path(args.service_summary).resolve()
        if args.service_summary
        else PROJECT_ROOT / "docs/evidence/service-release-summary.json"
    )
    windows_env_path = PROJECT_ROOT / "docs/environment/windows-cpu-inference.json"
    colab_env_path = PROJECT_ROOT / "docs/environment/colab-training.json"
    demo_masks = validate_smoke(read_json(demo_path), bundle_id, "Windows offline smoke")
    server_masks = validate_smoke(read_json(server_smoke_path), bundle_id, "Ubuntu server smoke")
    parity_checks = validate_platform_parity(
        read_json(parity_path), read_json(service_summary_path), server_masks, bundle_id
    )
    require(demo_masks == server_masks, "Windows and Ubuntu smoke mask digests differ")
    validate_environment_receipts(read_json(windows_env_path), read_json(colab_env_path))

    artifact_paths = [
        delivery_zip,
        sidecar,
        bundle_zip,
        recipe_path,
        submission_path,
        replay_path,
        demo_path,
        parity_path,
        server_smoke_path,
        service_summary_path,
        windows_env_path,
        colab_env_path,
        *native_artifacts,
    ]
    artifacts = [artifact(path) for path in dict.fromkeys(artifact_paths)]
    refs = {
        (PROJECT_ROOT / item["path"]).resolve(): item["path"] for item in artifacts
    }

    def ref(path: Path) -> str:
        resolved = path.resolve()
        require(resolved in refs, f"undeclared artifact reference: {resolved}")
        return refs[resolved]

    native_refs = [
        item["path"]
        for item in artifacts
        if Path(item["path"]).parent.as_posix().endswith(native_dir.name)
    ]
    builder_environment = f"{platform.platform()} / Python {platform.python_version()}"
    native_checks = native_state["checks"]
    native_ids = [str(item["id"]) for item in native_checks if item.get("required") is True]

    return {
        "release": bundle_id,
        "scope": (
            "FireWatch competition model CLI delivery: offline inference, submission, "
            "saved-weight validation replay and portable model bundle. Excludes web-service "
            "release, private-test quality and any production-readiness declaration."
        ),
        "profile": "generic",
        "target_environments": [TARGET_WINDOWS, TARGET_COLAB, TARGET_UBUNTU],
        "proof_provenance": {
            "selected_recipes_zip_member": recipe_member,
            "selected_recipes_sha256": recipe_sha,
            "bundle_manifest_sha256": sha256_bytes(bundle_manifest_bytes),
        },
        "artifacts": artifacts,
        "gates": [
            {
                "id": "delivery-integrity",
                "title": "Delivery ZIP sidecar, CRC and per-file hashes",
                "hard": True,
                "status": "pass",
                "evidence": [
                    evidence(
                        method="command",
                        procedure=(
                            "Run ops/build_model_release_evidence.py; compare delivery ZIP size/SHA-256 "
                            "with its sidecar, run ZipFile.testzip(), and recompute every "
                            "DELIVERY_MANIFEST.json member size and SHA-256."
                        ),
                        environment=builder_environment,
                        target=TARGET_WINDOWS,
                        observed=(
                            f"{delivery_zip.name}: {delivery_zip.stat().st_size} bytes, "
                            f"sha256={sha256_file(delivery_zip)}, {len(delivery_entries)} ZIP files; "
                            "sidecar, CRC, inventory and every content hash matched."
                        ),
                        source=sidecar,
                    )
                ],
                "artifact_refs": [ref(delivery_zip), ref(sidecar)],
            },
            {
                "id": "bundle-integrity",
                "title": "Portable model bundle identity and embedded byte parity",
                "hard": True,
                "status": "pass",
                "evidence": [
                    evidence(
                        method="command",
                        procedure=(
                            "Run bundle ZIP CRC, validate manifest bundle_id/tasks/component hashes and "
                            "compare every bundle member byte-for-byte with model_bundle/ in the delivery ZIP."
                        ),
                        environment=builder_environment,
                        target=TARGET_WINDOWS,
                        observed=(
                            f"bundle_id={bundle_id}; {len(bundle_entries)} bundle files; component hashes "
                            "and embedded delivery bytes matched; component paths are POSIX-safe."
                        ),
                        source=bundle_zip,
                    )
                ],
                "artifact_refs": [ref(delivery_zip), ref(bundle_zip)],
            },
            {
                "id": "submission-contract",
                "title": "Official 447-row CSV contract",
                "hard": True,
                "status": "pass",
                "evidence": [
                    evidence(
                        method="external",
                        procedure=(
                            "Validate saved submission against the official sample template, then recompute "
                            "the packaged submission.csv SHA-256 and independently parse its header and rows."
                        ),
                        environment=(
                            "Google Colab Python 3.13 saved CPU inference receipt; archive receipt "
                            "revalidated by the local builder"
                        ),
                        target=TARGET_COLAB,
                        observed=(
                            f"status=pass; rows={submission['data_rows']}; AF chips={submission['af_chips']}; "
                            f"BS chips={submission['bs_chips']}; template order/RLE/exclusivity passed; "
                            f"submission sha256={submission['submission_sha256']}."
                        ),
                        source=submission_path,
                    )
                ],
                "artifact_refs": [ref(delivery_zip), ref(submission_path)],
            },
            {
                "id": "windows-native-smoke",
                "title": "Windows Python 3.13 CPU native contracts and real-scene smoke",
                "hard": True,
                "status": "pass",
                "evidence": [
                    evidence(
                        method="command",
                        procedure=(
                            "Replay the native plan and capture stdout/stderr; run the saved bundle offline "
                            "on two AF and two BS real scenes with network connections blocked."
                        ),
                        environment="Windows 11 / Python 3.13.5 / torch 2.6 CPU",
                        target=TARGET_WINDOWS,
                        observed=(
                            f"Required native checks passed with exit 0 and complete cleanup: {', '.join(native_ids)}; "
                            f"all {native_snapshot_files} checked snapshot files matched delivery ZIP bytes; "
                            "offline smoke produced two AF uint8 256x256 masks and two BS uint8 "
                            f"512x512 masks with full count coverage for bundle_id={bundle_id}."
                        ),
                        source=native_dir / "state.json",
                    )
                ],
                "artifact_refs": [ref(delivery_zip), ref(demo_path), ref(windows_env_path), *native_refs],
            },
            {
                "id": "colab-cpu-validation-replay",
                "title": "Colab Python 3.13 saved-weight CPU replay",
                "hard": True,
                "status": "pass",
                "evidence": [
                    evidence(
                        method="external",
                        procedure=(
                            "Replay saved AF/BS validation chips through the packaged model on CPU with "
                            "network blocked; compare official micro metrics with selected validation receipts."
                        ),
                        environment="Google Colab Python 3.13.15; device=cpu; network blocked",
                        target=TARGET_COLAB,
                        observed=(
                            f"within_tolerance=true; tolerance={replay['tolerance']}; "
                            f"absolute_delta={json.dumps(replay['absolute_delta'], sort_keys=True)}; "
                            f"Score={replay['metrics']['Score']}; AF chips={replay['timing']['af']['chips']}; "
                            f"BS chips={replay['timing']['bs']['chips']}; selected recipe sha256={recipe_sha}. "
                            "Expected metrics matched the packaged recipe and bundle selections exactly. "
                            "This was replay, not retraining."
                        ),
                        source=replay_path,
                    )
                ],
                "artifact_refs": [
                    ref(delivery_zip),
                    ref(bundle_zip),
                    ref(recipe_path),
                    ref(replay_path),
                    ref(colab_env_path),
                ],
            },
            {
                "id": "ubuntu-platform-parity",
                "title": "Ubuntu Python 3.10 CPU public mask parity",
                "hard": True,
                "status": "pass",
                "evidence": [
                    evidence(
                        method="external",
                        procedure=(
                            "Run the bundle on the Ubuntu CPU service, download all four raw uint8 GeoTIFF "
                            "arrays and compare C-order SHA-256 with the independent offline reference masks."
                        ),
                        environment="Ubuntu 22.04 / Python 3.10.12 / torch 2.6 CPU public service",
                        target=TARGET_UBUNTU,
                        observed=(
                            f"status=pass; {parity_checks} of {parity_checks} AF/BS masks matched byte-for-byte; "
                            f"bundle_id={bundle_id}; Windows, local and server smoke digests agree."
                        ),
                        source=parity_path,
                    )
                ],
                "artifact_refs": [
                    ref(delivery_zip),
                    ref(bundle_zip),
                    ref(parity_path),
                    ref(server_smoke_path),
                    ref(service_summary_path),
                ],
            },
            {
                "id": "retraining-within-0.005",
                "title": "Fresh retraining reproduces selected metrics within 0.005",
                "hard": False,
                "status": "unverified",
                "evidence": [],
                "artifact_refs": [ref(bundle_zip), ref(replay_path)],
            },
            {
                "id": "private-test-score",
                "title": "Private competition test score",
                "hard": False,
                "status": "unverified",
                "evidence": [],
                "artifact_refs": [ref(delivery_zip)],
            },
        ],
        "limitations": {
            "retraining_within_0.005": (
                "unverified: saved-weight replay has zero metric delta, but no independent retraining "
                "from initialization was executed"
            ),
            "private_test_score": "unknown: test labels and leaderboard score are unavailable",
            "production_readiness": "not declared; this generic manifest covers the model CLI delivery only",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate existing FireWatch model delivery receipts and build a generic "
            "release-gate evidence manifest. Does not run the final release audit."
        )
    )
    parser.add_argument("--delivery-zip", required=True, help="Final model CLI delivery ZIP")
    parser.add_argument("--bundle-zip", required=True, help="Portable model bundle ZIP")
    parser.add_argument("--evidence-dir", required=True, help="Directory containing submission and replay receipts")
    parser.add_argument("--native-run-dir", required=True, help="Native check run containing plan/packet/state and captured streams")
    parser.add_argument(
        "--server-smoke",
        help=(
            "Optional Ubuntu CPU smoke receipt. Defaults to "
            "artifacts/service/server-smoke-<bundle-id-without-official->.json"
        ),
    )
    parser.add_argument(
        "--public-parity",
        help=(
            "Optional public mask-parity receipt. Defaults to the backward-compatible "
            "artifacts/service/public-mask-parity.json"
        ),
    )
    parser.add_argument(
        "--service-summary",
        help=(
            "Optional compatible web-service release summary. Defaults to the "
            "backward-compatible docs/evidence/service-release-summary.json"
        ),
    )
    parser.add_argument("--output", required=True, help="Output release-gate manifest JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build(args)
        output = Path(args.output).resolve()
        protected = {
            (PROJECT_ROOT / item["path"]).resolve() for item in manifest["artifacts"]
        }
        require(output not in protected, "output must not overwrite an input artifact")
        atomic_json(output, manifest)
    except EvidenceError as exc:
        print(f"evidence build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "built",
                "output": str(output),
                "artifact_root_for_release_gate": str(PROJECT_ROOT),
                "release": manifest["release"],
                "profile": manifest["profile"],
                "artifacts": len(manifest["artifacts"]),
                "gates": len(manifest["gates"]),
                "unverified": [
                    gate["id"] for gate in manifest["gates"] if gate["status"] == "unverified"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
