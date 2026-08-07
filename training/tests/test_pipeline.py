import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from training.datasets import (
    SarPatchDataset,
    load_manifest,
    split_by_scene,
    split_summary,
)
from training.export import export_candidate_metadata
from training.promote import promote
from training.prepare_label_pack import compatible_items
from training.rasterize_labels import build_manifest, rasterize_record
from training.temporal import refresh_temporal_context
from training.weather import nearest_wind


def create_manifest(root: Path, scenes: int = 6) -> Path:
    records = []
    for index in range(scenes):
        image_path = root / f"image-{index}.npy"
        mask_path = root / f"mask-{index}.npy"
        np.save(
            image_path, np.full((2, 32, 32), index / max(scenes, 1), dtype="float32")
        )
        mask = np.zeros((32, 32), dtype="float32")
        mask[8:16, 8:16] = 1
        np.save(mask_path, mask)
        records.append(
            {
                "sample_id": f"sample-{index}",
                "image": image_path.name,
                "mask": mask_path.name,
                "scene_id": f"scene-{index}",
                "acquisition_time": f"2026-0{index % 6 + 1}-01T00:00:00Z",
                "region": f"region-{index % 3}",
                "bbox": [51, 42, 51.1, 42.1],
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    return manifest


def test_scene_grouped_split_has_no_leakage(tmp_path: Path):
    manifest = create_manifest(tmp_path)
    splits = split_by_scene(load_manifest(manifest), seed=7)
    scene_sets = [
        {record.scene_id for record in records} for records in splits.values()
    ]
    assert scene_sets[0].isdisjoint(scene_sets[1])
    assert scene_sets[0].isdisjoint(scene_sets[2])
    assert scene_sets[1].isdisjoint(scene_sets[2])
    assert sum(len(records) for records in splits.values()) == 6


def test_patch_dataset_validates_and_loads_two_channels(tmp_path: Path):
    manifest = create_manifest(tmp_path, scenes=3)
    records = load_manifest(manifest)
    image, mask, sample_id = SarPatchDataset(manifest, records)[0]
    assert tuple(image.shape) == (2, 32, 32)
    assert tuple(mask.shape) == (1, 32, 32)
    assert sample_id == "sample-0"


def test_candidate_requires_explicit_promotion(tmp_path: Path):
    manifest = create_manifest(tmp_path)
    records = load_manifest(manifest)
    splits = split_by_scene(records)
    checkpoint = tmp_path / "oil_unet.pt"
    checkpoint.write_bytes(b"safe-state-dict-placeholder")
    metadata = tmp_path / "oil_unet.json"
    export_candidate_metadata(
        metadata,
        checkpoint,
        "dataset-sha",
        split_summary(splits),
        {
            "validation": {"iou": 0.8, "precision": 0.8, "recall": 0.8},
            "test": {"iou": 0.8, "precision": 0.8, "recall": 0.8},
        },
        0.55,
    )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "candidate"
    args = argparse.Namespace(
        min_iou=0.5,
        min_precision=0.5,
        min_recall=0.5,
        min_scenes=6,
        min_regions=3,
        min_dates=6,
        approved_by="test-reviewer",
        approval_note="Independent test review",
    )
    assert promote(payload, checkpoint, args)["validation_status"] == "validated"


def label_record(
    root: Path, status: str, features: list, reviewed_by: str | None = "reviewer"
) -> dict:
    from rasterio.transform import from_bounds

    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    np.save(root / "images" / "sample.npy", np.zeros((2, 10, 10), dtype="float32"))
    annotation = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"review_status": status, "reviewed_by": reviewed_by},
    }
    (root / "annotations" / "sample.geojson").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    return {
        "sample_id": "sample",
        "image": "images/sample.npy",
        "annotation": "annotations/sample.geojson",
        "scene_id": "scene-1",
        "acquisition_time": "2026-08-02T14:21:15Z",
        "region": "aktau",
        "bbox": [51, 42, 52, 43],
        "shape": [10, 10],
        "transform": list(from_bounds(51, 42, 52, 43, 10, 10))[:6],
    }


def test_unreviewed_annotation_cannot_become_training_mask(tmp_path: Path):
    record = label_record(tmp_path, "unreviewed", [], reviewed_by=None)
    with pytest.raises(ValueError, match="has not been reviewed"):
        rasterize_record(tmp_path, record)


def test_reviewed_polygon_is_rasterized_and_manifested(tmp_path: Path):
    polygon = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[51.2, 42.2], [51.8, 42.2], [51.8, 42.8], [51.2, 42.8], [51.2, 42.2]]
            ],
        },
    }
    pack_root = tmp_path / "pack"
    record = label_record(pack_root, "reviewed_positive", [polygon])
    mask, reviewer = rasterize_record(pack_root, record)
    assert mask.sum() > 0
    assert reviewer == "reviewer"
    pack = pack_root / "pack.jsonl"
    pack.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = build_manifest(pack, tmp_path / "dataset" / "manifest.jsonl")
    loaded = load_manifest(manifest)
    image, saved_mask, _ = SarPatchDataset(manifest, loaded)[0]
    assert tuple(image.shape) == (2, 10, 10)
    assert float(saved_mask.sum()) > 0


def test_reviewed_negative_produces_an_empty_mask(tmp_path: Path):
    record = label_record(tmp_path, "reviewed_negative", [])
    mask, _ = rasterize_record(tmp_path, record)
    assert mask.sum() == 0


def test_label_pack_scene_selection_uses_distinct_dates_and_dual_pol():
    def item(identifier: str, acquired: str, polarizations: list[str]) -> dict:
        return {
            "id": identifier,
            "properties": {
                "datetime": acquired,
                "sar:instrument_mode": "IW",
                "sar:polarizations": polarizations,
            },
            "assets": {
                "vv": {"href": "s3://sentinel-s1-l1c/path/vv.tif"},
                "vh": {"href": "s3://sentinel-s1-l1c/path/vh.tif"},
            },
        }

    selected = compatible_items(
        [
            item("newest", "2026-08-04T10:00:00Z", ["VV", "VH"]),
            item("same-day", "2026-08-04T08:00:00Z", ["VV", "VH"]),
            item("single-pol", "2026-08-03T10:00:00Z", ["VV"]),
            item("older", "2026-08-02T10:00:00Z", ["VV", "VH"]),
        ],
        limit=3,
    )
    assert [item["id"] for item in selected] == ["newest", "older"]
    assert selected[0]["assets"]["vv"]["href"].startswith("https://")


def test_nearest_wind_matches_acquisition_hour():
    from datetime import datetime, timezone

    context = nearest_wind(
        {
            "latitude": 43.4,
            "longitude": 51.2,
            "hourly": {
                "time": ["2026-08-02T13:00", "2026-08-02T14:00", "2026-08-02T15:00"],
                "wind_speed_10m": [2.0, 3.5, 5.0],
                "wind_direction_10m": [100, 120, 140],
                "wind_gusts_10m": [4.0, 6.0, 8.0],
            },
        },
        datetime(2026, 8, 2, 14, 22, tzinfo=timezone.utc),
    )
    assert context["wind_speed_10m_ms"] == 3.5
    assert context["wind_direction_10m_deg"] == 120
    assert context["timestamp"].startswith("2026-08-02T14:00")


def test_temporal_context_compares_adjacent_scenes_only(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    previous = np.full((2, 8, 8), 0.3, dtype="float32")
    current = previous.copy()
    current[:, 2:6, 2:6] = 0.8
    np.save(images / "previous.npy", previous)
    np.save(images / "current.npy", current)
    records = [
        {
            "sample_id": "previous",
            "region": "aktau",
            "image": "images/previous.npy",
            "acquisition_time": "2026-07-01T00:00:00Z",
        },
        {
            "sample_id": "current",
            "region": "aktau",
            "image": "images/current.npy",
            "acquisition_time": "2026-07-05T00:00:00Z",
        },
    ]
    refresh_temporal_context(records, tmp_path)
    assert records[0]["temporal_context"]["status"] == "unavailable"
    context = records[1]["temporal_context"]
    assert context["status"] == "available"
    assert context["previous_sample_id"] == "previous"
    assert context["days_between"] == 4.0
    assert context["changed_pixel_fraction"] > 0
