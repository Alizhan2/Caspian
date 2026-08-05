"""Configurable coastline mask for removing land pixels from SAR candidates."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class CoastlineMask:
    """Rasterize configured land polygons; never invent a coastline when absent."""

    def __init__(self, geojson_path: str = "") -> None:
        self.path = Path(geojson_path) if geojson_path else None

    @property
    def ready(self) -> bool:
        return bool(self.path and self.path.is_file())

    def apply(
        self, candidate: np.ndarray, transform: Any, crs: Any, valid: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        binary = np.asarray(candidate, dtype=bool)
        if not self.ready:
            return binary, {"status": "unavailable", "reason": "coastline_geojson_not_configured"}
        try:
            from rasterio.features import rasterize
            from rasterio.warp import transform_geom

            payload = json.loads(self.path.read_text(encoding="utf-8"))
            features = (
                payload.get("features", []) if payload.get("type") == "FeatureCollection" else [{"geometry": payload}]
            )
            land_shapes = []
            for feature in features:
                geometry = feature.get("geometry")
                if geometry:
                    land_shapes.append((transform_geom("EPSG:4326", str(crs or "EPSG:4326"), geometry), 1))
            if not land_shapes:
                return binary, {"status": "unavailable", "reason": "coastline_geojson_has_no_land_polygons"}
            land = rasterize(land_shapes, out_shape=binary.shape, transform=transform, fill=0, dtype="uint8").astype(
                bool
            )
            land &= np.asarray(valid, dtype=bool)
            filtered = binary & ~land
            return filtered, {
                "status": "applied",
                "source": "configured_land_geojson",
                "candidate_pixels_before": int(binary.sum()),
                "land_pixels_removed": int((binary & land).sum()),
                "candidate_pixels_after": int(filtered.sum()),
            }
        except Exception:
            return binary, {"status": "unavailable", "reason": "coastline_geojson_unreadable"}
