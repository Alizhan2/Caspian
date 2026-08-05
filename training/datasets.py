import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image: str
    mask: str
    scene_id: str
    acquisition_time: str
    region: str
    bbox: tuple[float, float, float, float]

    @property
    def acquisition_date(self) -> str:
        return (
            datetime.fromisoformat(self.acquisition_time.replace("Z", "+00:00"))
            .date()
            .isoformat()
        )


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: str | Path) -> list[SampleRecord]:
    manifest_path = Path(path)
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload: dict[str, Any] = json.loads(line)
            record = SampleRecord(
                sample_id=str(payload["sample_id"]),
                image=str(payload["image"]),
                mask=str(payload["mask"]),
                scene_id=str(payload["scene_id"]),
                acquisition_time=str(payload["acquisition_time"]),
                region=str(payload["region"]),
                bbox=tuple(float(value) for value in payload["bbox"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid manifest record at line {line_number}") from exc
        if record.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {record.sample_id}")
        if (
            len(record.bbox) != 4
            or not record.region.strip()
            or not record.scene_id.strip()
        ):
            raise ValueError(f"Incomplete manifest record: {record.sample_id}")
        datetime.fromisoformat(record.acquisition_time.replace("Z", "+00:00"))
        seen.add(record.sample_id)
        records.append(record)
    if not records:
        raise ValueError("Dataset manifest is empty")
    return records


def split_by_scene(
    records: list[SampleRecord],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[SampleRecord]]:
    if (
        validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation and test fractions must be positive and total less than one"
        )
    groups: dict[str, list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.scene_id, []).append(record)
    if len(groups) < 3:
        raise ValueError(
            "At least three independent scenes are required for train/validation/test"
        )
    scene_ids = sorted(groups)
    random.Random(seed).shuffle(scene_ids)
    validation_count = max(1, round(len(scene_ids) * validation_fraction))
    test_count = max(1, round(len(scene_ids) * test_fraction))
    while validation_count + test_count >= len(scene_ids):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Not enough scenes to create non-empty splits")
    validation_ids = set(scene_ids[:validation_count])
    test_ids = set(scene_ids[validation_count : validation_count + test_count])
    train_ids = set(scene_ids) - validation_ids - test_ids
    return {
        "train": [record for record in records if record.scene_id in train_ids],
        "validation": [
            record for record in records if record.scene_id in validation_ids
        ],
        "test": [record for record in records if record.scene_id in test_ids],
    }


def split_summary(splits: dict[str, list[SampleRecord]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "samples": len(records),
            "scenes": len({record.scene_id for record in records}),
            "regions": sorted({record.region for record in records}),
            "dates": sorted({record.acquisition_date for record in records}),
        }
        for name, records in splits.items()
    }


class SarPatchDataset:
    def __init__(self, manifest_path: str | Path, records: list[SampleRecord]) -> None:
        self.root = Path(manifest_path).resolve().parent
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        image = np.load(self.root / record.image).astype("float32")
        mask = np.load(self.root / record.mask).astype("float32")
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError(f"{record.sample_id}: expected image [2, H, W]")
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2 or mask.shape != image.shape[1:]:
            raise ValueError(f"{record.sample_id}: mask shape does not match image")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{record.sample_id}: mask must be binary")
        return torch.from_numpy(image), torch.from_numpy(mask[None]), record.sample_id


def load_dataset(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility loader for the original aggregate NumPy dataset."""
    root_path = Path(root)
    images, masks = np.load(root_path / "images.npy"), np.load(root_path / "masks.npy")
    if images.ndim != 4 or images.shape[1] != 2 or masks.shape[0] != images.shape[0]:
        raise ValueError("Expected images [N, 2, H, W] and matching masks")
    return images.astype("float32"), masks.astype("float32")
