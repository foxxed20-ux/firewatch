from __future__ import annotations

import sys
import types
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from firewatch_service.imagery_provider import (
    COLLECTION,
    EarthEngineProvider,
    ImageryError,
)


class FakeGeometry:
    def __init__(self, value):
        self.value = value

    def bounds(self):
        return self


class FakeImage:
    def __init__(self, record):
        self.record = record

    def geometry(self):
        return FakeGeometry(self.record["geometry"])

    def id(self):
        return self.record["id"]

    def get(self, name):
        return self.record["properties"].get(name)

    def select(self, bands):
        self.bands = bands
        return self

    def clip(self, aoi):
        self.aoi = aoi
        return self

    def visualize(self, **kwargs):
        self.visualization = kwargs
        return self

    def getMapId(self, _: dict):
        return {
            "tile_fetcher": types.SimpleNamespace(
                url_format=(
                    "https://earthengine.googleapis.com/map/abc/{z}/{x}/{y}?token=map-token"
                )
            )
        }


class FakeList:
    def __init__(self, records):
        self.records = records

    def map(self, fn):
        return [fn(FakeImage(record)) for record in self.records]


class FakeCollection:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def filterBounds(self, value):
        self.calls.append(("bounds", value))
        return self

    def filterDate(self, start, end):
        self.calls.append(("dates", start, end))
        return self

    def filter(self, value):
        self.calls.append(("cloud", value))
        return self

    def sort(self, key, ascending):
        self.calls.append(("sort", key, ascending))
        return self

    def limit(self, value):
        self.calls.append(("limit", value))
        return self

    def toList(self, value):
        self.calls.append(("to_list", value))
        return FakeList(self.records[:value])


class FakeFeatureCollection:
    def __init__(self, features):
        self.features = features

    def getInfo(self):
        return {
            "features": [
                {
                    # EE assigns indexes to mapped features independently of
                    # the source image ID explicitly stored in properties.
                    "id": str(index),
                    "properties": feature["properties"],
                    "geometry": feature["geometry"].value,
                }
                for index, feature in enumerate(self.features)
            ]
        }


def install_fake(monkeypatch, records, tile_url=None):
    collection = FakeCollection(records)
    events = []

    class TestImage(FakeImage):
        def getMapId(self, value):
            if tile_url is not None:
                return {"tile_fetcher": types.SimpleNamespace(url_format=tile_url)}
            return super().getMapId(value)

    def image(value):
        if isinstance(value, FakeImage):
            return value
        if isinstance(value, str):
            return TestImage({"id": value, "properties": {}, "geometry": {}})
        return TestImage(value)

    ee = types.SimpleNamespace(
        ImageCollection=lambda _: collection,
        Geometry=FakeGeometry,
        Image=image,
        Feature=lambda geometry, properties: {
            "geometry": geometry,
            "properties": properties,
        },
        FeatureCollection=FakeFeatureCollection,
        Filter=types.SimpleNamespace(lte=lambda name, value: (name, value)),
        data=types.SimpleNamespace(
            setDeadline=lambda value: events.append(("deadline", value)),
            setMaxRetries=lambda value: events.append(("retries", value)),
        ),
        Initialize=lambda **kwargs: events.append(("initialize", kwargs["project"])),
    )
    google_auth = types.SimpleNamespace(
        default=lambda **kwargs: (
            events.append(("credentials", kwargs["scopes"])) or object(),
            "ignored",
        )
    )
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setitem(sys.modules, "google.auth", google_auth)
    return collection, events


@pytest.fixture
def payload():
    return {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[20, 50], [21, 50], [21, 51], [20, 50]]],
        },
        "date_start": "2026-06-01",
        "date_end": "2026-06-02",
        "cloud_max": 15,
        "limit": 1,
    }


def record(index: int):
    return {
        "id": f"2015112{index}T002653_2015112{index}T102149_T56MNN",
        "properties": {
            "system:time_start": 1780300800000 + index,
            "CLOUDY_PIXEL_PERCENTAGE": index,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[20, 50], [21, 50], [21, 51], [20, 50]]],
        },
    }


def test_search_materializes_bounded_feature_collection(monkeypatch, payload):
    collection, events = install_fake(monkeypatch, [record(8), record(9)])
    response = EarthEngineProvider("project", enabled=True).search(payload)
    assert response["collection"] == COLLECTION
    assert response["truncated"] is True
    assert len(response["scenes"]) == 1
    assert response["scenes"][0]["id"] == (
        COLLECTION + "/20151128T002653_20151128T102149_T56MNN"
    )
    assert ("dates", "2026-06-01", "2026-06-03") in collection.calls
    assert ("cloud", ("CLOUDY_PIXEL_PERCENTAGE", 15.0)) in collection.calls
    assert ("limit", 2) in collection.calls
    assert ("to_list", 2) in collection.calls
    assert events[:4] == [
        ("retries", 0),
        (
            "credentials",
            [
                "https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        ),
        ("initialize", "project"),
        ("deadline", 20000),
    ]


def test_preview_accepts_real_mgrs_id_and_returns_https_tile(monkeypatch, payload):
    install_fake(monkeypatch, [])
    scene_id = COLLECTION + "/20151128T002653_20151128T102149_T56MNN"
    response = EarthEngineProvider("project", enabled=True).preview(
        {"scene_id": scene_id, "aoi": payload["aoi"]}
    )
    assert response["scene_id"] == scene_id
    assert response["tile_url"].startswith("https://earthengine.googleapis.com/")
    assert response["bounds"] == [20.0, 50.0, 21.0, 51.0]


@pytest.mark.parametrize(
    "tile_url",
    [
        "https://evil.example/maps/{z}/{x}/{y}",
        "http://earthengine.googleapis.com/maps/{z}/{x}/{y}",
        "https://user:password@earthengine.googleapis.com/maps/{z}/{x}/{y}",
        "https://earthengine.googleapis.com/maps/{z}/{x}/{y}?access_token=secret",
    ],
)
def test_preview_rejects_unsafe_tile_urls(monkeypatch, payload, tile_url):
    install_fake(monkeypatch, [], tile_url=tile_url)
    scene_id = COLLECTION + "/20151128T002653_20151128T102149_T56MNN"
    with pytest.raises(ImageryError) as error:
        EarthEngineProvider("project", enabled=True).preview(
            {"scene_id": scene_id, "aoi": payload["aoi"]}
        )
    assert (error.value.status, error.value.code) == (503, "EARTH_ENGINE_UNAVAILABLE")


def test_not_configured_never_imports_or_contacts_google(payload):
    provider = EarthEngineProvider(None, enabled=True)
    assert provider.configuration()["reason"] == "project_not_configured"
    with pytest.raises(ImageryError, match="not configured") as error:
        provider.search(payload)
    assert error.value.status == 503
    assert error.value.code == "GOOGLE_NOT_CONFIGURED"


def test_credentials_failure_is_not_configuration(monkeypatch, payload):
    ee = types.SimpleNamespace(
        data=types.SimpleNamespace(
            setDeadline=lambda _: None, setMaxRetries=lambda _: None
        )
    )
    google_auth = types.SimpleNamespace(
        default=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("private credential detail")
        )
    )
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setitem(sys.modules, "google.auth", google_auth)
    with pytest.raises(ImageryError) as error:
        EarthEngineProvider("project", enabled=True).search(payload)
    assert (error.value.status, error.value.code) == (503, "GOOGLE_NOT_CONFIGURED")
    assert "private credential detail" not in error.value.message


def test_rejects_private_or_non_mgrs_scene_identifiers_before_google_import(payload):
    provider = EarthEngineProvider("project", enabled=True)
    for scene_id in [
        "users/private/image",
        COLLECTION + "/20151128T002653_20151128T102149_T12345",
    ]:
        with pytest.raises(ImageryError) as error:
            provider.preview({"scene_id": scene_id, "aoi": payload["aoi"]})
        assert (error.value.status, error.value.code) == (422, "INVALID_SCENE_ID")


def test_search_rejects_invalid_range_without_google_call(payload):
    payload["date_end"] = "2026-05-31"
    with pytest.raises(ImageryError) as error:
        EarthEngineProvider("project", enabled=True).search(payload)
    assert (error.value.status, error.value.code) == (422, "INVALID_DATE_RANGE")


def test_process_wide_lock_rejects_busy_request_without_delayed_work(
    monkeypatch, payload
):
    collection, _ = install_fake(monkeypatch, [record(8)])
    original_get_info = FakeFeatureCollection.getInfo
    started = Event()
    release = Event()

    def slow_get_info(self):
        started.set()
        assert release.wait(1)
        return original_get_info(self)

    monkeypatch.setattr(FakeFeatureCollection, "getInfo", slow_get_info)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            EarthEngineProvider("project-a", enabled=True).search, payload
        )
        assert started.wait(1)
        with pytest.raises(ImageryError) as error:
            EarthEngineProvider("project-b", enabled=True).search(payload)
        assert (error.value.status, error.value.code) == (429, "IMAGERY_BUSY")
        assert collection.calls.count(("to_list", 2)) == 1
        release.set()
        future.result()
    assert collection.calls.count(("to_list", 2)) == 1


def test_search_rejects_malformed_earth_engine_response(monkeypatch, payload):
    install_fake(monkeypatch, [record(8)])
    monkeypatch.setattr(FakeFeatureCollection, "getInfo", lambda _: None)
    with pytest.raises(ImageryError) as error:
        EarthEngineProvider("project", enabled=True).search(payload)
    assert (error.value.status, error.value.code) == (503, "EARTH_ENGINE_UNAVAILABLE")
