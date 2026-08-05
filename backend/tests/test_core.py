import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

import app.main as main_module
from app.core.config import Settings
from app.schemas import DetectionReviewRequest, LabelReviewRequest
from app.services.copernicus_auth import CopernicusAuth
from app.services.inference import OilUnetInference, load_validated_model_card
from app.services.label_pack import LabelPack
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


def test_candidate_model_card_is_not_treated_as_validated(tmp_path: Path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    checkpoint.with_suffix(".json").write_text(
        '{"schema_version":1,"validation_status":"candidate"}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="promoted"):
        load_validated_model_card(checkpoint)


def test_bbox_validation_rejects_invalid_latitude():
    response = client.post("/api/scenes/search", json={"bbox": [51.85, -100, 52.4, 43.1], "days_back": 14})
    assert response.status_code == 422


def test_vv_vh_normalization_shape_and_range():
    prepared = normalize_vv_vh(np.ones((4, 4)), np.full((4, 4), 0.2))
    assert prepared.tensor.shape == (2, 4, 4)
    assert float(prepared.tensor.min()) >= 0
    assert float(prepared.tensor.max()) <= 1


def test_public_digital_amplitudes_keep_contrast_after_normalization():
    vv = np.array([[20, 40], [80, 160]], dtype=np.float32)
    vh = np.array([[10, 20], [40, 80]], dtype=np.float32)
    prepared = normalize_vv_vh(vv, vh)
    assert float(prepared.tensor[0].min()) == pytest.approx(0)
    assert float(prepared.tensor[0].max()) == pytest.approx(1)
    assert np.unique(prepared.tensor[0]).size > 2


def test_experimental_baseline_is_versioned_and_masks_invalid_pixels(tmp_path: Path):
    inference = OilUnetInference(str(tmp_path / "missing.pt"), experimental_baseline_enabled=True)
    image = np.full((2, 64, 64), 0.7, dtype=np.float32)
    image[:, 24:40, 24:40] = 0.1
    valid = np.ones((64, 64), dtype=bool)
    valid[0, 0] = False
    probability, mask = inference.predict(image, valid)
    assert inference.ready is True
    assert inference.validated is False
    assert inference.model_version.endswith(":experimental")
    assert probability[0, 0] == 0
    assert mask[0, 0] == 0
    assert probability[32, 32] > probability[4, 4]


def test_mask_to_polygon_and_area():
    mask = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)
    geojson = raster_mask_to_geojson(mask, from_bounds(51, 42, 52, 43, 3, 3), min_pixels=0)
    geometry = geojson["features"][0]["geometry"]
    assert polygon_area_km2(geometry) > 0


def test_vectorizer_minimum_is_measured_in_pixels():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    transform = from_bounds(51, 42, 52, 43, 10, 10)
    assert len(raster_mask_to_geojson(mask, transform, min_pixels=8)["features"]) == 1
    assert len(raster_mask_to_geojson(mask, transform, min_pixels=10)["features"]) == 0


def test_risk_logic_is_configurable():
    settings = Settings()
    assert calculate_risk(0.80, 0.90, 0.25, settings).level.value == "HIGH"
    assert calculate_risk(0.30, 0.40, 0.01, settings).level.value == "LOW"
    assert calculate_risk(0.80, 0.90, 0.25, settings, model_validated=False).level.value == "MEDIUM"


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


def make_label_pack(root: Path) -> LabelPack:
    (root / "annotations").mkdir(parents=True)
    (root / "previews").mkdir()
    (root / "previews" / "aktau-sample.png").write_bytes(b"preview")
    annotation = {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "review_status": "unreviewed",
            "reviewed_by": None,
            "sample_id": "aktau-sample",
        },
    }
    (root / "annotations" / "aktau-sample.geojson").write_text(json.dumps(annotation), encoding="utf-8")
    record = {
        "sample_id": "aktau-sample",
        "scene_id": "S1_TEST",
        "acquisition_time": "2026-08-01T00:00:00Z",
        "region": "aktau",
        "region_name": "Побережье Актау",
        "bbox": [51.0, 43.0, 52.0, 44.0],
        "preview": "previews/aktau-sample.png",
        "annotation": "annotations/aktau-sample.geojson",
    }
    (root / "pack.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return LabelPack(root)


def test_label_pack_saves_only_explicit_expert_positive_review(tmp_path: Path):
    pack = make_label_pack(tmp_path)
    payload = LabelReviewRequest(
        status="reviewed_positive",
        reviewed_by="expert",
        polygons=[[[51.1, 43.1], [51.3, 43.1], [51.2, 43.3]]],
    )
    result = pack.review("aktau-sample", payload)
    assert result["review_status"] == "reviewed_positive"
    assert result["polygon_count"] == 1
    assert result["polygons"][0][0] == result["polygons"][0][-1]


def test_label_pack_rejects_polygon_outside_scene(tmp_path: Path):
    pack = make_label_pack(tmp_path)
    payload = LabelReviewRequest(
        status="reviewed_positive",
        reviewed_by="expert",
        polygons=[[[51.1, 43.1], [55.0, 43.1], [51.2, 43.3]]],
    )
    with pytest.raises(ValueError, match="outside"):
        pack.review("aktau-sample", payload)


def test_negative_label_requires_explicit_reviewer_and_no_polygons(tmp_path: Path):
    pack = make_label_pack(tmp_path)
    result = pack.review(
        "aktau-sample",
        LabelReviewRequest(status="reviewed_negative", reviewed_by="expert", polygons=[]),
    )
    assert result["review_status"] == "reviewed_negative"
    assert result["polygon_count"] == 0
    with pytest.raises(ValueError, match="reviewer name"):
        LabelReviewRequest(status="reviewed_negative", reviewed_by="   ", polygons=[])
