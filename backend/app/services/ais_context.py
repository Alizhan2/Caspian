"""Authorized AIS context adapter. It remains unavailable until an endpoint is configured."""

from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any
from urllib.parse import urlparse

import httpx
from shapely.geometry import shape


def _distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(radians, (*first, *second))
    value = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(value))


class AisContext:
    """Fetch vessel positions from the project's authorized provider contract."""

    def __init__(self, endpoint: str = "", api_key: str = "", timeout_seconds: int = 12) -> None:
        self.endpoint, self.api_key, self.timeout_seconds = endpoint.strip(), api_key.strip(), timeout_seconds

    @property
    def ready(self) -> bool:
        return bool(self.endpoint)

    async def for_detection(self, geometry: dict[str, Any], bbox: list[float], acquired: datetime) -> dict[str, Any]:
        if not self.ready:
            return {"status": "unavailable", "reason": "authorized_ais_endpoint_not_configured"}
        try:
            window_start, window_end = acquired - timedelta(hours=6), acquired + timedelta(hours=6)
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {
                "west": bbox[0],
                "south": bbox[1],
                "east": bbox[2],
                "north": bbox[3],
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(self.endpoint, params=params, headers=headers)
                response.raise_for_status()
            payload = response.json()
            vessels = payload.get("vessels", []) if isinstance(payload, dict) else []
            centre = shape(geometry).centroid
            distances = []
            for vessel in vessels:
                latitude, longitude = vessel.get("latitude"), vessel.get("longitude")
                if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
                    distances.append(_distance_km((centre.x, centre.y), (longitude, latitude)))
            nearest = min(distances) if distances else None
            return {
                "status": "available",
                "provider": urlparse(self.endpoint).netloc,
                "time_window_hours": 6,
                "vessels_returned": len(distances),
                "nearest_vessel_km": round(nearest, 2) if nearest is not None else None,
                "vessels_within_5km": sum(distance <= 5 for distance in distances),
                "note": "AIS context supports review and does not establish a spill source.",
            }
        except Exception:
            return {"status": "unavailable", "reason": "authorized_ais_request_failed"}
