"""Fetch only the published FireWatch AOI support files and create a local manifest.

The Yandex public directory contains an AOI, not ready-to-serve scenes.  This
tool never downloads the multi-gigabyte renamed train/test archives.  Operators
add independently licensed, georeferenced scene assets to the generated
manifest before the service will expose a dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import stat
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PUBLIC_KEY = "https://disk.yandex.ru/d/-rpmevTflbXZQg"
API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def download(public_path: str, destination: Path) -> None:
    query = urllib.parse.urlencode({"public_key": PUBLIC_KEY, "path": public_path})
    with urllib.request.urlopen(f"{API}?{query}", timeout=30) as response:
        href = json.load(response)["href"]
    with urllib.request.urlopen(href, timeout=60) as response:
        destination.write_bytes(response.read())


def bounds_from_geojson(document: dict) -> list[float] | None:
    """Return WGS84 bbox of the main `aoi` feature without GIS dependencies."""
    feature = next(
        (
            f
            for f in document.get("features", [])
            if f.get("properties", {}).get("role") == "aoi"
        ),
        None,
    )
    if not feature:
        feature = next(iter(document.get("features", [])), None)
    if not isinstance(feature, dict):
        return None
    pairs: list[tuple[float, float]] = []

    def visit(value):
        if (
            isinstance(value, list)
            and len(value) >= 2
            and all(isinstance(x, (int, float)) for x in value[:2])
        ):
            pairs.append((float(value[0]), float(value[1])))
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(feature.get("geometry", {}).get("coordinates", []))
    return (
        [
            min(x for x, _ in pairs),
            min(y for _, y in pairs),
            max(x for x, _ in pairs),
            max(y for _, y in pairs),
        ]
        if pairs
        else None
    )


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        path = Path(info.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not info.filename
            or "\\" in info.filename
        ):
            raise ValueError(f"unsafe ZIP member: {info.filename!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"symlink ZIP member is forbidden: {info.filename!r}")
        if not info.is_dir():
            members.append(info)
    return members


def import_validation_demo(archive_path: Path, data_root: Path) -> Path:
    """Import input-only public-train examples without extracting labels or links."""
    target = data_root / "datasets" / "official-validation-demo"
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive)
        names = {member.filename for member in members}
        if "demo_metadata.json" not in names:
            raise ValueError("demo ZIP has no demo_metadata.json")
        metadata = json.loads(archive.read("demo_metadata.json"))
        if not isinstance(metadata.get("scenes"), list) or len(metadata["scenes"]) != 4:
            raise ValueError("demo_metadata.json must contain exactly four scenes")
        for member in members:
            if member.filename == "demo_metadata.json" or member.filename.endswith(
                ".tif"
            ):
                output = target / member.filename
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output.open("wb") as destination:
                    destination.write(source.read())
    import rasterio
    from rasterio.warp import transform_bounds

    scenes, presets = [], []
    for raw in metadata["scenes"]:
        if (
            not isinstance(raw, dict)
            or raw.get("kind") not in {"af", "bs"}
            or not isinstance(raw.get("chip_id"), str)
        ):
            raise ValueError("invalid scene metadata")
        chip, task = raw["chip_id"], raw["kind"]
        reference = (
            f"af/viirs/{chip}_VIIRS_I1-I5.tif"
            if task == "af"
            else f"bs/sentinel2_post/{chip}_Sentinel-2_post.tif"
        )
        if not (target / reference).is_file():
            raise ValueError(f"missing reference raster for {chip}")
        with rasterio.open(target / reference) as dataset:
            bounds = list(
                transform_bounds(
                    dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21
                )
            )
        scene = {
            "scene_id": chip,
            "chip_id": chip,
            "task": task,
            "reference_raster": reference,
            "training_example": True,
        }
        if task == "af":
            scene["observed_at"] = raw["acq_datetime"]
            instant = datetime.fromisoformat(raw["acq_datetime"].replace("Z", "+00:00"))
            # Query interval includes the observed instant; this is not a new
            # acquisition date and remains labelled as a UI selection preset.
            period = {
                "start": raw["acq_datetime"],
                "end": (instant + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
            }
        else:
            scene["date_pre"] = f"{raw['date_pre']}T00:00:00Z"
            scene["date_post"] = f"{raw['date_post']}T00:00:00Z"
            post = datetime.fromisoformat(scene["date_post"].replace("Z", "+00:00"))
            # Half-open API filtering includes post only before this query end.
            # The date_post field above remains the exact source metadata.
            period = {
                "start": scene["date_pre"],
                "end": (post + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
            }
        scenes.append(scene)
        presets.append(
            {
                "preset_id": chip,
                "label": f"{task.upper()} validation example {chip}",
                "bounds": bounds,
                "period": period,
                "tasks": [task],
                "scene_ids": [chip],
            }
        )
    bounds = [
        min(p["bounds"][0] for p in presets),
        min(p["bounds"][1] for p in presets),
        max(p["bounds"][2] for p in presets),
        max(p["bounds"][3] for p in presets),
    ]
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "official-validation-demo",
        "label": "Official public TRAIN validation demonstration",
        "mode": "geospatial_demo",
        "bounds": bounds,
        "source": {
            "source_dataset": "public train",
            "training_split": "validation",
            "examples_selected_for_display": True,
            "usage": metadata.get("usage"),
            "selection": metadata.get("selection"),
        },
        "source_assets": [
            "input-only georeferenced rasters from model_demo_inputs.zip",
            "demo_metadata.json",
        ],
        "presets": presets,
        "scenes": scenes,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "demo_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="firewatch-public-aoi-v1")
    parser.add_argument("--import-demo-zip", type=Path)
    args = parser.parse_args()
    if args.import_demo_zip:
        print(import_validation_demo(args.import_demo_zip, args.data_root))
        return 0
    target = args.data_root / "datasets" / args.dataset_id
    target.mkdir(parents=True, exist_ok=True)
    download(
        "/fire-aoi/fire_monitoring_aoi.geojson", target / "fire_monitoring_aoi.geojson"
    )
    download("/fire-aoi/README.md", target / "SOURCE_README.md")
    aoi_document = json.loads(
        (target / "fire_monitoring_aoi.geojson").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": args.dataset_id,
        "label": "FireWatch public monitoring AOI (scene assets pending)",
        "mode": "geospatial_demo",
        "bounds": bounds_from_geojson(aoi_document),
        "source": {
            "provider": "Yandex Disk public FireWatch folder",
            "public_key": PUBLIC_KEY,
            "aoi_asset": "fire_monitoring_aoi.geojson",
            "license_status": "verify before adding imagery",
        },
        "scenes": [],
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(target / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
