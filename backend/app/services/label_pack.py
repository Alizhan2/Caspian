"""Safe access to locally collected Sentinel-1 annotation packs."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas import LabelReviewRequest

SAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,120}$")


class LabelPack:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = self.root / "pack.jsonl"

    def _records(self) -> list[dict[str, Any]]:
        if not self.manifest.exists():
            return []
        return [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _record(self, sample_id: str) -> dict[str, Any]:
        if not SAMPLE_ID.fullmatch(sample_id):
            raise LookupError("Label sample not found")
        for record in self._records():
            if record.get("sample_id") == sample_id:
                return record
        raise LookupError("Label sample not found")

    def _safe_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("Invalid label-pack path")
        return path

    def _annotation(self, record: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(record["annotation"])
        return json.loads(path.read_text(encoding="utf-8"))

    def list_samples(self) -> list[dict[str, Any]]:
        return [self._response(record, self._annotation(record)) for record in self._records()]

    def preview(self, sample_id: str) -> Path:
        record = self._record(sample_id)
        path = self._safe_path(record["preview"])
        if not path.is_file():
            raise LookupError("Label preview not found")
        return path

    def review(self, sample_id: str, payload: LabelReviewRequest) -> dict[str, Any]:
        record = self._record(sample_id)
        west, south, east, north = record["bbox"]
        for polygon in payload.polygons:
            for longitude, latitude in polygon:
                if not west <= longitude <= east or not south <= latitude <= north:
                    raise ValueError("Polygon point is outside the sample bounds")

        annotation = self._annotation(record)
        features = []
        for index, polygon in enumerate(payload.polygons):
            ring = [list(point) for point in polygon]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            features.append(
                {
                    "type": "Feature",
                    "id": f"{sample_id}-{index + 1}",
                    "properties": {"class": "oil_like_anomaly"},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            )
        annotation["features"] = features
        annotation["properties"].update(
            {
                "review_status": payload.status.value,
                "reviewed_by": payload.reviewed_by.strip(),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "review_note": payload.note,
            }
        )
        path = self._safe_path(record["annotation"])
        temporary = path.with_suffix(".geojson.tmp")
        temporary.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return self._response(record, annotation)

    @staticmethod
    def _response(record: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
        properties = annotation.get("properties", {})
        sample_id = record["sample_id"]
        polygons = [
            feature.get("geometry", {}).get("coordinates", [[]])[0]
            for feature in annotation.get("features", [])
            if feature.get("geometry", {}).get("type") == "Polygon"
        ]
        return {
            "sample_id": sample_id,
            "scene_id": record["scene_id"],
            "acquisition_time": record["acquisition_time"],
            "region": record["region"],
            "region_name": record["region_name"],
            "bbox": record["bbox"],
            "review_status": properties.get("review_status", "unreviewed"),
            "reviewed_by": properties.get("reviewed_by"),
            "reviewed_at": properties.get("reviewed_at"),
            "polygon_count": len(annotation.get("features", [])),
            "polygons": polygons,
            "weather_context": record.get("weather_context"),
            "temporal_context": record.get("temporal_context"),
            "preview_url": f"/api/labeling/samples/{sample_id}/preview",
        }
