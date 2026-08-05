"""Conservative adjacent-scene comparison for review prioritisation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

CHANGE_THRESHOLD = 0.18


def _acquired(record: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(record["acquisition_time"].replace("Z", "+00:00"))


def compare_pair(
    current: dict[str, Any], previous: dict[str, Any], root: Path
) -> dict[str, Any]:
    current_image = np.load(root / current["image"])
    previous_image = np.load(root / previous["image"])
    if current_image.shape != previous_image.shape or current_image.ndim != 3:
        return {
            "status": "unavailable",
            "reason": "incompatible_image_shapes",
            "previous_sample_id": previous["sample_id"],
        }
    valid = np.any(current_image > 0, axis=0) & np.any(previous_image > 0, axis=0)
    if not np.any(valid):
        return {
            "status": "unavailable",
            "reason": "no_shared_valid_pixels",
            "previous_sample_id": previous["sample_id"],
        }
    delta = np.mean(np.abs(current_image - previous_image), axis=0)
    values = delta[valid]
    days = (_acquired(current) - _acquired(previous)).total_seconds() / 86400
    return {
        "status": "available",
        "method": "contrast_normalized_vv_vh_absolute_delta",
        "interpretation": "screening_only_not_radiometrically_calibrated",
        "previous_sample_id": previous["sample_id"],
        "previous_acquisition_time": previous["acquisition_time"],
        "days_between": round(days, 2),
        "mean_absolute_delta": round(float(np.mean(values)), 4),
        "changed_pixel_fraction": round(float(np.mean(values >= CHANGE_THRESHOLD)), 4),
        "change_threshold": CHANGE_THRESHOLD,
    }


def refresh_temporal_context(records: list[dict[str, Any]], root: Path) -> None:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_region.setdefault(record["region"], []).append(record)
    for region_records in by_region.values():
        ordered = sorted(region_records, key=_acquired)
        for index, current in enumerate(ordered):
            if index == 0:
                current["temporal_context"] = {
                    "status": "unavailable",
                    "reason": "no_prior_scene_for_region",
                }
                continue
            current["temporal_context"] = compare_pair(
                current, ordered[index - 1], root
            )
