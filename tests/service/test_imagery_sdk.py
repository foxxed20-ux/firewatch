from __future__ import annotations

import importlib
import json
import types
from unittest import mock

import ee
import httplib2
from ee.apitestcase import ApiTestCase, GetAlgorithms
from google.auth.credentials import AnonymousCredentials

from firewatch_service.imagery_provider import COLLECTION, EarthEngineProvider


PAYLOAD = {
    "aoi": {
        "type": "Polygon",
        "coordinates": [[[20, 50], [21, 50], [21, 51], [20, 50]]],
    },
    "date_start": "2026-06-01",
    "date_end": "2026-06-02",
    "cloud_max": 15,
    "limit": 1,
}
SCENE_ID = COLLECTION + "/20151128T002653_20151128T102149_T56MNN"


class TestImagerySdkExpression(ApiTestCase):
    """Use the installed EE API test harness; no credentials or HTTP are used."""

    def _provider(self) -> EarthEngineProvider:
        provider = EarthEngineProvider("offline-project", enabled=True)
        provider._initialized = True
        return provider

    def test_cold_initialization_builds_sdk_resources_before_deadline(self):
        """Exercise real ``ee.Initialize`` from a fully cold SDK state.

        The discovery-resource and algorithm metadata boundaries are replaced
        with local EE test assets, so this covers the installed SDK's ordering
        without contacting Google or reading local ADC credentials.
        """
        # ApiTestCase replaces this private builder for expression tests. Reload
        # the installed module for this test so the real initialization path is
        # exercised before we replace only its external discovery boundary.
        importlib.reload(ee.data)
        ee.Reset()
        assert ee.data._get_state().requests_session is None

        build_calls = []
        metadata_calls = []
        old_algorithms = ee.data.getAlgorithms
        old_api = ee.ApiFunction._api

        def build_cloud_resource(*args, **kwargs):
            build_calls.append((args, kwargs))
            return object()

        def get_algorithms():
            metadata_calls.append(True)
            return GetAlgorithms()

        ee.ApiFunction._api = {}
        ee.data.getAlgorithms = get_algorithms
        try:
            with mock.patch(
                "ee.data._cloud_api_utils.build_cloud_resource",
                side_effect=build_cloud_resource,
            ), mock.patch(
                "google.auth.default",
                return_value=(AnonymousCredentials(), "offline-project"),
            ):
                provider = EarthEngineProvider(
                    "offline-project", enabled=True, timeout_seconds=17
                )
                initialized = provider._ensure_initialized()
        finally:
            ee.data.getAlgorithms = old_algorithms
            ee.ApiFunction._api = old_api

        state = ee.data._get_state()
        assert initialized is ee
        assert provider._initialized is True
        assert state.initialized is True
        assert state.requests_session is not None
        assert metadata_calls == [True]
        assert len(build_calls) == 4

        first_args, first_kwargs = build_calls[0]
        assert first_args[0] == "https://earthengine.googleapis.com"
        assert first_kwargs["num_retries"] == 0
        assert isinstance(first_kwargs["http_transport"], httplib2.Http)
        assert first_kwargs["http_transport"].timeout == 17
        assert first_kwargs["timeout"] is None

        # ``setDeadline`` must run only after ``Initialize`` has created the
        # requests session and first discovery resources. It rebuilds both raw
        # and normal Cloud resources with the request deadline applied.
        for _, kwargs in build_calls[2:]:
            assert kwargs["num_retries"] == 0
            assert kwargs["timeout"] == 17
            assert kwargs["http_transport"].timeout == 17

    def test_search_builds_bounded_real_sdk_expression(self):
        expressions = []

        def compute_value(value):
            expressions.append(ee.serializer.encode(value, for_cloud_api=True))
            return {
                "features": [
                    {
                        "id": "0",
                        "properties": {
                            "id": "20151128T002653_20151128T102149_T56MNN",
                            "system:time_start": 1448670413000,
                            "CLOUDY_PIXEL_PERCENTAGE": 12.5,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[20, 50], [21, 50], [21, 51], [20, 50]]],
                        },
                    }
                ]
            }

        ee.data.computeValue = compute_value
        response = self._provider().search(PAYLOAD)

        serialized = json.dumps(expressions, sort_keys=True)
        assert len(expressions) == 1
        assert "Collection.toList" in serialized
        assert "List.map" in serialized
        assert "Image.geometry" in serialized
        assert "Image.id" in serialized
        assert response["scenes"][0]["id"] == SCENE_ID

    def test_preview_builds_real_sdk_expression_and_uses_map_boundary(self):
        map_requests = []

        def get_map_id(params):
            map_requests.append(
                ee.serializer.encode(params["image"], for_cloud_api=True)
            )
            return {
                "tile_fetcher": types.SimpleNamespace(
                    url_format=(
                        "https://earthengine-highvolume.googleapis.com/map/abc/"
                        "{z}/{x}/{y}?token=map-token"
                    )
                )
            }

        ee.data.getMapId = get_map_id
        response = self._provider().preview(
            {"scene_id": SCENE_ID, "aoi": PAYLOAD["aoi"]}
        )

        serialized = json.dumps(map_requests, sort_keys=True)
        assert len(map_requests) == 1
        assert "Image.visualize" in serialized
        assert "Image.clip" in serialized
        assert "Image.select" in serialized
        assert response["tile_url"].startswith(
            "https://earthengine-highvolume.googleapis.com/"
        )
