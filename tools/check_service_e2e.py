"""Exercise a deployed FireWatch candidate with real catalog scenes and models.

Writes observed evidence and downloaded products; never substitutes predictions.
Requires requests and the service's numpy/rasterio/shapely/pyproj dependencies.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin
import uuid

import numpy as np
import requests
from rasterio.io import MemoryFile
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def decode(rle, height, width):
    tokens = [int(value) for value in rle.split()]
    require(len(tokens) % 2 == 0, "Odd RLE length")
    pixels = np.zeros(height * width, dtype=bool)
    previous_end = -1
    for start, length in zip(tokens[::2], tokens[1::2]):
        start -= 1
        end = start + length
        require(
            start >= 0 and length > 0 and end <= pixels.size and start > previous_end,
            "Invalid RLE span",
        )
        pixels[start:end] = True
        previous_end = end
    return pixels.reshape(height, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-timeout", type=int, default=330)
    parser.add_argument("--token-env", default="FIREWATCH_TOKEN")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(
        not output.exists(), "Evidence directory already exists; use a new run name"
    )
    output.mkdir(parents=True)
    session = requests.Session()
    import os

    if os.getenv(args.token_env):
        session.headers["Authorization"] = "Bearer " + os.environ[args.token_env]
    base = args.base_url.rstrip("/") + "/"
    evidence = {
        "base_url": base,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": [],
        "jobs": [],
        "artifacts": [],
    }

    def save(name, data):
        path = output / name
        path.write_text(
            json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )

    def call(method, route, expected=200, **kwargs):
        response = session.request(method, urljoin(base, route), timeout=30, **kwargs)
        evidence["checks"].append(
            {
                "method": method,
                "route": route,
                "status": response.status_code,
                "expected": expected,
            }
        )
        require(
            response.status_code == expected,
            f"{method} {route}: {response.status_code} {response.text[:500]}",
        )
        return response

    def run_job(kind, payload, label, replay=False):
        key = "release-smoke-" + uuid.uuid4().hex
        accepted = call(
            "POST", f"v1/{kind}", 202, json=payload, headers={"Idempotency-Key": key}
        ).json()
        if replay:
            repeated = call(
                "POST",
                f"v1/{kind}",
                202,
                json=payload,
                headers={"Idempotency-Key": key},
            ).json()
            require(
                accepted["job_id"] == repeated["job_id"],
                "Idempotency replay created a second job",
            )
            changed = dict(payload, tasks=["af", "bs"])
            call(
                "POST",
                f"v1/{kind}",
                409,
                json=changed,
                headers={"Idempotency-Key": key},
            )
        started = time.monotonic()
        while True:
            state = call("GET", accepted["status_url"]).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            require(
                time.monotonic() - started < args.job_timeout,
                f"Job deadline exceeded: {accepted['job_id']}",
            )
            time.sleep(1)
        require(state["status"] == "succeeded", f"Failed {label}: {state}")
        result = call("GET", accepted["result_url"]).json()
        save(f"{label}-result.json", result)
        evidence["jobs"].append(
            {
                "label": label,
                "job_id": accepted["job_id"],
                "seconds": round(time.monotonic() - started, 3),
                "status": state["status"],
                "data_status": result.get("data_status"),
                "modules": result.get("modules"),
                "quality": result.get("quality"),
            }
        )
        products = {}
        for artifact in result["artifacts"]:
            response = call("GET", artifact["href"])
            data = response.content
            digest = hashlib.sha256(data).hexdigest()
            require(
                response.headers.get("ETag") == f'"sha256-{digest}"',
                "Artifact hash mismatch",
            )
            filename = f"{label}-{artifact['id']}-{Path(artifact['filename']).name}"
            (output / filename).write_bytes(data)
            evidence["artifacts"].append(
                {
                    "path": filename,
                    "sha256": digest,
                    "bytes": len(data),
                    "url": artifact["href"],
                }
            )
            products[artifact["id"]] = data
        return result, products

    try:
        call("GET", "health/live")
        ready = call("GET", "health/ready").json()
        require(ready["status"] == "ready", "Service not ready")
        catalog = call("GET", "v1/catalog").json()
        registry = call("GET", "v1/models").json()
        save("catalog.json", catalog)
        save("models.json", registry)
        datasets = [item for item in catalog["datasets"] if item.get("presets")]
        require(bool(datasets), "No real prepared scene presets")
        dataset = datasets[0]
        models = [
            item
            for item in registry["models"]
            if item.get("status") == "ready"
            and {"af", "bs"} <= set(item.get("tasks", []))
        ]
        require(bool(models), "No complete AF+BS bundle")
        bundle = models[0]["model_bundle_id"]
        presets = dataset["presets"]
        require(
            {task for preset in presets for task in preset["tasks"]} >= {"af", "bs"},
            "Missing AF/BS real examples",
        )
        to_equal_area = Transformer.from_crs(
            "EPSG:4326", "EPSG:6933", always_xy=True
        ).transform

        for index, preset in enumerate(presets):
            label = preset["preset_id"]
            payload = {
                "dataset_id": dataset["dataset_id"],
                "model_bundle_id": bundle,
                "aoi": {"bbox": preset["bounds"]},
                "period": preset["period"],
                "tasks": preset["tasks"],
            }
            result, products = run_job("analyses", payload, label, replay=index == 0)
            require(
                result["data_status"] in {"complete", "partial"},
                f"Known scene unexpectedly has no observations: {label}",
            )
            require(
                set(result["modules"]) == set(payload["tasks"]),
                "Requested module selection mismatch",
            )
            require(
                "summary" in products and "provenance" in products,
                "Missing downloadable report/provenance",
            )
            summary = json.loads(products["summary"])
            require(
                summary["modules"] == result["modules"]
                and summary["quality"] == result["quality"],
                "Downloadable summary differs from API",
            )
            aoi = box(*preset["bounds"])
            for task in payload["tasks"]:
                quality = result["quality"][task]
                require(
                    quality["observed_area_ha"] > 0
                    and 0 < quality["coverage_fraction"] <= 1,
                    "Invalid coverage",
                )
            if "af" in payload["tasks"]:
                hotspots = json.loads(products["hotspots"])["features"]
                require(
                    result["modules"]["af"]["hotspot_count"] == len(hotspots),
                    "Hotspot count differs from GeoJSON",
                )
                require(
                    all(
                        aoi.buffer(1e-6).covers(shape(f["geometry"])) for f in hotspots
                    ),
                    "Hotspot outside AOI",
                )
                require(
                    len({f["properties"]["hotspot_id"] for f in hotspots})
                    == len(hotspots),
                    "Duplicate hotspot identifiers",
                )
            if "bs" in payload["tasks"]:
                features = json.loads(products["burn_polygons"])["features"]
                areas = {str(k): 0.0 for k in (1, 2, 3)}
                geometries = []
                for feature in features:
                    props, geometry = feature["properties"], shape(feature["geometry"])
                    require(
                        geometry.is_valid and not geometry.is_empty,
                        "Invalid burn geometry",
                    )
                    require(aoi.buffer(1e-6).covers(geometry), "Contour outside AOI")
                    actual = transform(to_equal_area, geometry).area / 10000
                    require(
                        abs(actual - props["area_ha"]) <= 1e-4,
                        "GeoJSON area cannot be reproduced",
                    )
                    areas[str(props["class_id"])] += actual
                    geometries.append(geometry)
                module = result["modules"]["bs"]
                for klass, area in areas.items():
                    require(
                        abs(area - module["severity_area_ha"][klass]) <= 1e-3,
                        "Severity total mismatch",
                    )
                require(
                    abs(sum(areas.values()) - module["burn_area_ha"]) <= 1e-3,
                    "Burn total mismatch",
                )
                union_area = (
                    transform(to_equal_area, unary_union(geometries)).area / 10000
                )
                require(
                    abs(union_area - sum(areas.values())) <= 1e-3,
                    "Burn classes overlap",
                )

        chip_ids = [preset["preset_id"] for preset in presets]
        predictions, products = run_job(
            "predictions",
            {
                "dataset_id": dataset["dataset_id"],
                "model_bundle_id": bundle,
                "chip_ids": chip_ids,
            },
            "raw-predictions",
        )
        require(len(predictions["items"]) == len(chip_ids), "Missing chip prediction")
        for item in predictions["items"]:
            require(
                item["mask_url"] and predictions["provenance_url"],
                "Missing direct product URLs",
            )
            with MemoryFile(products[item["mask_artifact_id"]]) as mem:
                with mem.open() as raster:
                    actual = raster.read(1)
                    require(
                        raster.crs is not None, "Registered geospatial mask lacks CRS"
                    )
            require(
                actual.shape == (item["height"], item["width"])
                and actual.dtype == np.uint8,
                "Mask dimensions/type mismatch",
            )
            reconstructed = np.zeros_like(actual)
            for record in item["rle"]:
                binary = decode(record["rle"], *actual.shape)
                require(
                    not np.any(binary & (reconstructed != 0)), "Overlapping RLE classes"
                )
                reconstructed[binary] = record["class_id"]
            require(
                np.array_equal(actual, reconstructed),
                "RLE does not reproduce downloaded mask",
            )

        outside = {
            "dataset_id": dataset["dataset_id"],
            "model_bundle_id": bundle,
            "aoi": {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[10, 10], [10.1, 10], [10.1, 10.1], [10, 10.1], [10, 10]]
                    ],
                }
            },
            "period": {"start": "2020-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
            "tasks": ["af", "bs"],
        }
        result, products = run_job("analyses", outside, "outside-catalog")
        require(
            result["data_status"] == "no_data", "Outside catalog must yield no_data"
        )
        require(
            "summary" in products and "provenance" in products,
            "no_data report not downloadable",
        )
        for task in outside["tasks"]:
            require(
                result["modules"][task]["data_status"] == "no_data",
                "no_data task missing",
            )
            require(
                result["quality"][task]["coverage_fraction"] == 0
                and result["quality"][task]["unobserved_area_ha"] > 0,
                "no_data coverage incorrect",
            )
        invalid = dict(outside, aoi={"bbox": [50, 50, 40, 40]})
        call(
            "POST",
            "v1/analyses",
            422,
            json=invalid,
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        call(
            "POST",
            "v1/predictions",
            404,
            json={
                "dataset_id": dataset["dataset_id"],
                "model_bundle_id": bundle,
                "chip_ids": ["UNKNOWN_CHIP"],
            },
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        evidence["status"] = "pass"
    except Exception as exc:
        evidence["status"] = "fail"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        save("evidence.json", evidence)
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "jobs": len(evidence["jobs"]),
                    "evidence": str(output / "evidence.json"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
