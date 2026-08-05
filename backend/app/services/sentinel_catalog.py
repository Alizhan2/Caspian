import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.schemas import BoundingBox
from app.services.copernicus_auth import CopernicusAuth

logger = logging.getLogger(__name__)


class SceneCatalog:
    def __init__(self, auth: CopernicusAuth, base_url: str) -> None:
        self.auth = auth
        self.catalog_url = f"{base_url.rstrip('/')}/catalog/v1/search"

    async def latest(self, bbox: list[float], days_back: int) -> dict[str, Any]:
        if not self.auth.configured:
            raise RuntimeError("Copernicus OAuth credentials are not configured")
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
            token = await self.auth.get_token()
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(self.catalog_url, json=query, headers={"Authorization": f"Bearer {token}"})
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
                "preview_url": None,
            }
        except httpx.HTTPStatusError as exc:
            logger.error("Sentinel catalog request returned %s", exc.response.status_code)
            raise RuntimeError("Sentinel catalog request failed") from exc
