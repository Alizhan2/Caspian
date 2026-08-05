import json
from pathlib import Path


def export_metadata(output: str, model_version: str, preprocessing: dict) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"model_version": model_version, "preprocessing": preprocessing}, indent=2), encoding="utf-8")
    return destination
