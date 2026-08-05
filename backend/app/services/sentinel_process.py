import asyncio
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

    async def download_vv_vh(
        self,
        bbox: list[float],
        time_range: tuple[str, str],
        scene_id: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> Path:
        assets = (source_metadata or {}).get("assets", {})
        vv_url = assets.get("vv", {}).get("href")
        vh_url = assets.get("vh", {}).get("href")
        if vv_url and vh_url:
            return await asyncio.to_thread(
                self._crop_public_cogs,
                bbox,
                scene_id,
                vv_url,
                vh_url,
                source_metadata or {},
            )
        token = await self.auth.get_token()
        payload: dict[str, Any] = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [
                    {
                        "type": "sentinel-1-grd",
                        "dataFilter": {"timeRange": {"from": time_range[0], "to": time_range[1]}, "polarization": "DV"},
                        "processing": {"orthorectify": True},
                    }
                ],
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": (
                "//VERSION=3\n"
                "function setup(){return {input:[{bands:['VV','VH']}],"
                "output:{bands:2,sampleType:'FLOAT32'}}}\n"
                "function evaluatePixel(s){return [s.VV,s.VH]}"
            ),
        }
        self.storage_path.mkdir(parents=True, exist_ok=True)
        destination = self.storage_path / f"{scene_id}.tif"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    self.process_url, json=payload, headers={"Authorization": f"Bearer {token}"}
                )
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

    def _crop_public_cogs(
        self,
        bbox: list[float],
        scene_id: str,
        vv_url: str,
        vh_url: str,
        source_metadata: dict[str, Any],
    ) -> Path:
        import numpy as np
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window, from_bounds

        self.storage_path.mkdir(parents=True, exist_ok=True)
        destination = self.storage_path / f"{scene_id}.tif"
        properties = source_metadata.get("properties", {})
        fallback_crs = properties.get("proj:epsg")
        fallback_transform = properties.get("proj:transform")
        fallback_shape = properties.get("proj:shape")
        if fallback_crs:
            fallback_crs = f"EPSG:{fallback_crs}"
        if fallback_transform and len(fallback_transform) >= 6:
            fallback_transform = Affine(*fallback_transform[:6])

        def read_asset(url: str) -> tuple[np.ndarray, Affine, Any]:
            with rasterio.open(url) as source:
                source_crs = source.crs or fallback_crs
                source_transform = source.transform
                if source_transform.is_identity and fallback_transform:
                    source_transform = fallback_transform
                if source_crs is None:
                    raise RuntimeError("Public Sentinel-1 COG is missing a CRS")
                projected = transform_bounds("EPSG:4326", source_crs, *bbox, densify_pts=21)
                window = from_bounds(*projected, transform=source_transform).round_offsets().round_lengths()
                storage_transposed = bool(
                    fallback_shape
                    and len(fallback_shape) >= 2
                    and source.width == fallback_shape[0]
                    and source.height == fallback_shape[1]
                )
                conceptual_width = fallback_shape[1] if storage_transposed else source.width
                conceptual_height = fallback_shape[0] if storage_transposed else source.height
                try:
                    window = window.intersection(Window(0, 0, conceptual_width, conceptual_height))
                except rasterio.errors.WindowError as exc:
                    raise RuntimeError("Requested area does not intersect the Sentinel-1 scene") from exc
                if window.width <= 0 or window.height <= 0:
                    raise RuntimeError("Requested area does not intersect the Sentinel-1 scene")
                storage_window = (
                    Window(window.row_off, window.col_off, window.height, window.width)
                    if storage_transposed
                    else window
                )
                data = source.read(
                    1,
                    window=storage_window,
                    out_shape=(512, 512),
                    resampling=Resampling.bilinear,
                ).astype("float32")
                if storage_transposed:
                    data = data.T
                transform = source_transform * Affine.translation(window.col_off, window.row_off)
                transform *= Affine.scale(window.width / 512, window.height / 512)
                return data, transform, source_crs

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            GDAL_HTTP_MULTIRANGE="YES",
        ):
            vv, transform, crs = read_asset(vv_url)
            vh, _, _ = read_asset(vh_url)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=512,
            height=512,
            count=2,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=0,
            compress="deflate",
        ) as target:
            target.write(vv, 1)
            target.write(vh, 2)
        return destination
