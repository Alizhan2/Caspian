import argparse
import json
from pathlib import Path

import numpy as np

from training.datasets import (
    SarPatchDataset,
    load_manifest,
    split_by_scene,
    split_summary,
)
from training.export import export_candidate_metadata
from training.promote import promote


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
