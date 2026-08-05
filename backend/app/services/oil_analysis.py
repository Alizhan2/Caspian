from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from PIL import Image

from app.services.inference import OilUnetInference
from app.services.scene_processor import normalize_vv_vh
from app.services.sentinel_process import SentinelProcess
from app.services.vectorizer import polygon_area_km2, raster_mask_to_geojson


class RealOilAnalysis:
    """Run the real Sentinel-1 VV/VH to mask-to-polygon pipeline."""

    def __init__(self, process: SentinelProcess, inference: OilUnetInference) -> None:
        self.process, self.inference = process, inference

    async def run(self, scene: dict[str, Any], bbox: list[float], resolution: int) -> dict[str, Any]:
        acquisition = scene["acquisition_time"]
        if isinstance(acquisition, str):
            acquisition = datetime.fromisoformat(acquisition.replace("Z", "+00:00"))
        if acquisition.tzinfo is None:
            acquisition = acquisition.replace(tzinfo=timezone.utc)
        raster_path = await self.process.download_vv_vh(
            bbox,
            ((acquisition - timedelta(days=1)).isoformat(), (acquisition + timedelta(days=1)).isoformat()),
            scene["external_scene_id"],
            scene.get("source_metadata"),
        )
        import rasterio

        with rasterio.open(raster_path) as raster:
            if raster.count < 2:
                raise ValueError("Sentinel Process response must contain VV and VH bands")
            prepared = normalize_vv_vh(raster.read(1), raster.read(2), raster.nodata)
            probability, binary = self.inference.predict(prepared.tensor)
            geometry = raster_mask_to_geojson(binary, raster.transform, raster.crs or "EPSG:4326")
        preview_path = raster_path.with_suffix(".preview.png")
        mask_path = raster_path.with_suffix(".mask.png")
        vv_preview = (np.clip(prepared.tensor[0], 0, 1) * 255).astype("uint8")
        Image.fromarray(vv_preview, mode="L").save(preview_path)
        Image.fromarray((binary.astype("uint8") * 255), mode="L").save(mask_path)
        features = geometry["features"]
        if not features:
            raise LookupError("No candidate anomaly pixels passed the model threshold")
        areas = [polygon_area_km2(feature["geometry"]) for feature in features]
        confidence_values = probability[binary]
        first_ring = features[0]["geometry"]["coordinates"][0][0]
        return {
            "geometry": geometry,
            "area_km2": round(sum(areas), 4),
            "mean_confidence": float(confidence_values.mean()),
            "max_confidence": float(confidence_values.max()),
            "coordinates": [round(value, 4) for value in first_ring],
            "model_version": self.inference.model_version,
            "image_url": str(preview_path),
            "mask_url": str(mask_path),
            "raster_path": str(raster_path),
            "warning": "Potential oil-like anomaly requiring field verification.",
            "anomaly_type": "potential_oil_like_anomaly",
            "verification_status": "requires_field_verification",
        }
