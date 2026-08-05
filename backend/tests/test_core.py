from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

import app.main as main_module
from app.core.config import Settings
from app.schemas import DetectionReviewRequest
from app.services.copernicus_auth import CopernicusAuth
from app.services.inference import OilUnetInference
from app.services.ollama_explainer import OllamaExplainer
from app.services.risk_calculator import calculate_risk
from app.services.scene_processor import normalize_vv_vh
from app.services.sentinel_catalog import SceneCatalog
from app.services.vectorizer import polygon_area_km2, raster_mask_to_geojson

client = TestClient(main_module.app)


def test_health_is_live_only_and_fail_closed(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "readiness",
        lambda: {
            "live_ready": False,
            "satellite_ready": False,
            "components": {
                "satellite_catalog": False,
                "segmentation_model": False,
                "postgis": False,
                "redis": False,
                "object_storage": False,
            },
        },
    )
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["data_mode"] == "live"
    assert response.json()["status"] == "configuration_required"


@pytest.mark.asyncio
async def test_catalog_never_falls_back_to_synthetic_data():
    auth = CopernicusAuth("", "", "https://example.invalid/token")
    catalog = SceneCatalog(auth, "https://example.invalid", public_stac_url="")
    with pytest.raises(RuntimeError, match="catalog"):
        await catalog.latest([51.85, 42.4, 52.4, 43.1], 14)


def test_public_catalog_converts_s3_assets_to_anonymous_https():
    href = SceneCatalog._public_asset_url("s3://sentinel-s1-l1c/path/to/VV.tif")
    assert href == "https://sentinel-s1-l1c.s3.amazonaws.com/path/to/VV.tif"


def test_inference_without_checkpoint_fails_closed(tmp_path: Path):
    inference = OilUnetInference(str(tmp_path / "missing.pt"))
    assert inference.ready is False
    with pytest.raises(RuntimeError, match="checkpoint"):
        inference.predict(np.ones((2, 8, 8), dtype=np.float32))


def test_bbox_validation_rejects_invalid_latitude():
    response = client.post("/api/scenes/search", json={"bbox": [51.85, -100, 52.4, 43.1], "days_back": 14})
    assert response.status_code == 422


def test_vv_vh_normalization_shape_and_range():
    prepared = normalize_vv_vh(np.ones((4, 4)), np.full((4, 4), 0.2))
    assert prepared.tensor.shape == (2, 4, 4)
    assert float(prepared.tensor.min()) >= 0
    assert float(prepared.tensor.max()) <= 1


def test_mask_to_polygon_and_area():
    mask = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)
    geojson = raster_mask_to_geojson(mask, from_bounds(51, 42, 52, 43, 3, 3), min_pixels=0)
    geometry = geojson["features"][0]["geometry"]
    assert polygon_area_km2(geometry) > 0


def test_risk_logic_is_configurable():
    settings = Settings()
    assert calculate_risk(0.80, 0.90, 0.25, settings).level.value == "HIGH"
    assert calculate_risk(0.30, 0.40, 0.01, settings).level.value == "LOW"


@pytest.mark.asyncio
async def test_ollama_disabled_uses_responsible_fallback():
    text = await OllamaExplainer(False, "http://localhost:11434", "qwen").explain(
        {"area_km2": 1, "mean_confidence": 0.7, "max_confidence": 0.8, "model_version": "test"}
    )
    assert "screening signal" in text
    assert "провер" in text


def test_review_actions_are_restricted():
    assert DetectionReviewRequest(action="CONFIRM", actor="operator").action.value == "CONFIRM"
    with pytest.raises(ValueError):
        DetectionReviewRequest(action="PUBLISH", actor="operator")
