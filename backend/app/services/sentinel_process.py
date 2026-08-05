import logging
from pathlib import Path
from typing import Any

import httpx

from app.services.copernicus_auth import CopernicusAuth

logger = logging.getLogger(__name__)


class SentinelProcess:
    """Sentinel Hub Process API client for VV/VH GeoTIFF acquisitions."""

    def __init__(self, auth: CopernicusAuth, base_url: str, storage_path: str) -> None:
        self.auth = auth
        self.process_url = f"{base_url.rstrip('/')}/api/v1/process"
        self.storage_path = Path(storage_path)

    async def download_vv_vh(self, bbox: list[float], time_range: tuple[str, str], scene_id: str) -> Path:
        token = await self.auth.get_token()
        payload: dict[str, Any] = {"input": {"bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}, "data": [{"type": "sentinel-1-grd", "dataFilter": {"timeRange": {"from": time_range[0], "to": time_range[1]}, "polarization": "DV"}, "processing": {"orthorectify": True}}]}, "output": {"width": 512, "height": 512, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]}, "evalscript": "//VERSION=3\nfunction setup(){return {input:[{bands:['VV','VH']}],output:{bands:2,sampleType:'FLOAT32'}}}\nfunction evaluatePixel(s){return [s.VV,s.VH]}"}
        self.storage_path.mkdir(parents=True, exist_ok=True)
        destination = self.storage_path / f"{scene_id}.tif"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(self.process_url, json=payload, headers={"Authorization": f"Bearer {token}"})
            if response.status_code == 401:
                raise RuntimeError("Copernicus token expired or unauthorized")
            if response.status_code == 429:
                raise RuntimeError("Copernicus rate limit reached; retry later")
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("Copernicus returned an empty scene")
            destination.write_bytes(response.content)
            return destination
        except httpx.TimeoutException as exc:
            logger.error("Sentinel Process API timed out")
            raise RuntimeError("Sentinel download timed out") from exc
