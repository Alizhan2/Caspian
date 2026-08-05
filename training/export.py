import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_CARD_SCHEMA_VERSION = 1


def checkpoint_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_candidate_metadata(
    output: str | Path,
    checkpoint: str | Path,
    dataset_sha256: str,
    split_summary: dict[str, Any],
    metrics: dict[str, Any],
    threshold: float,
) -> Path:
    destination = Path(output)
    payload = {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "validation_status": "candidate",
        "architecture": "unet-resnet34",
        "input_channels": ["VV", "VH"],
        "classes": ["potential_oil_like_anomaly"],
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "dataset_manifest_sha256": dataset_sha256,
        "split_summary": split_summary,
        "metrics": metrics,
        "threshold": float(threshold),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Candidate model; not approved for operational conclusions.",
            "SAR dark spots can be caused by low wind and natural films.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def export_metadata(output: str, model_version: str, preprocessing: dict) -> Path:
    """Compatibility helper retained for earlier callers."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"model_version": model_version, "preprocessing": preprocessing}, indent=2
        ),
        encoding="utf-8",
    )
    return destination
