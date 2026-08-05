import logging
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from app.schemas import BoundingBox
from app.services.copernicus_auth import CopernicusAuth

logger = logging.getLogger(__name__)


class SceneCatalog:
    def __init__(
        self,
        auth: CopernicusAuth,
        base_url: str,
        public_stac_url: str = "https://earth-search.aws.element84.com/v1",
    ) -> None:
        self.auth = auth
        self.catalog_url = f"{base_url.rstrip('/')}/catalog/v1/search"
        self.public_search_url = f"{public_stac_url.rstrip('/')}/search" if public_stac_url else ""
        self.provider = "Earth Search" if self.public_search_url else "Copernicus Data Space"

    @property
    def ready(self) -> bool:
        return bool(self.public_search_url or self.auth.configured)

    @staticmethod
    def _public_asset_url(href: str) -> str:
        if not href.startswith("s3://"):
            return href
        parsed = urlparse(href)
        return f"https://{parsed.netloc}.s3.amazonaws.com{quote(parsed.path, safe='/')}"

    async def latest(self, bbox: list[float], days_back: int) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("A Sentinel-1 satellite catalog is not configured")
        box = BoundingBox.from_list(bbox)
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days_back)
        query = {
            "bbox": box.as_list,
            "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{now.isoformat().replace('+00:00', 'Z')}",
            "collections": ["sentinel-1-grd"],
            "limit": 50,
        }
        try:
            headers: dict[str, str] = {}
            url = self.public_search_url or self.catalog_url
            if not self.public_search_url:
                token = await self.auth.get_token()
                headers["Authorization"] = f"Bearer {token}"
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=query, headers=headers)
            response.raise_for_status()
            items = response.json().get("features", [])
            if not items:
                raise LookupError("No Sentinel-1 scene found for this area and date range")
            compatible = []
            for candidate in items:
                props = candidate.get("properties", {})
                mode = str(props.get("sar:instrument_mode") or props.get("s1:instrument_mode") or "IW").upper()
                pols = (
                    props.get("sar:polarizations")
                    or props.get("s1:polarizations")
                    or props.get("s1:polarization")
                    or []
                )
                if isinstance(pols, str):
                    pols = pols.replace("&", " ").split()
                normalized = {str(value).upper() for value in pols}
                if mode == "IW" and (not normalized or {"VV", "VH"}.issubset(normalized)):
                    compatible.append(candidate)
            if not compatible:
                raise LookupError("No Sentinel-1 IW dual-polarization scene found")
            item = max(compatible, key=lambda candidate: candidate.get("properties", {}).get("datetime", ""))
            item = deepcopy(item)
            if self.public_search_url:
                for asset in item.get("assets", {}).values():
                    href = asset.get("href")
                    if href:
                        asset["href"] = self._public_asset_url(str(href))
                item["caspian:provider"] = self.provider
            properties = item.get("properties", {})
            external_id = str(item.get("id", ""))
            acquisition_time = properties.get("datetime")
            if not external_id or not acquisition_time:
                raise LookupError("Copernicus returned incomplete scene metadata")
            platform = str(properties.get("platform") or "Sentinel-1").replace("sentinel-", "Sentinel-")
            polarizations = (
                properties.get("sar:polarizations")
                or properties.get("s1:polarizations")
                or properties.get("s1:polarization")
                or ["VV", "VH"]
            )
            if isinstance(polarizations, str):
                polarizations = polarizations.replace("&", " ").split()
            orbit_direction = str(properties.get("sat:orbit_state") or "UNKNOWN").upper()
            return {
                "id": uuid.uuid5(uuid.NAMESPACE_URL, external_id),
                "external_scene_id": external_id,
                "satellite": platform,
                "acquisition_time": acquisition_time,
                "orbit_direction": orbit_direction,
                "polarization": [str(value).upper() for value in polarizations],
                "bbox": item.get("bbox") or box.as_list,
                "status": "available",
                "source_metadata": item,
                "preview_url": item.get("assets", {}).get("thumbnail", {}).get("href"),
            }
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else "network-error"
            logger.error("Sentinel catalog request returned %s", status)
            raise RuntimeError("Sentinel catalog request failed") from exc
