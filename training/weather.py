"""Historical wind context for SAR annotation and false-alarm review."""

from datetime import datetime
from typing import Any

import httpx

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)


def nearest_wind(payload: dict[str, Any], acquired: datetime) -> dict[str, Any]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError("Historical weather response contains no hourly data")
    parsed = [
        datetime.fromisoformat(value).replace(tzinfo=acquired.tzinfo) for value in times
    ]
    index = min(range(len(parsed)), key=lambda item: abs(parsed[item] - acquired))
    values = {
        name: hourly.get(name, [None] * len(times))[index] for name in HOURLY_VARIABLES
    }
    if values["wind_speed_10m"] is None or values["wind_direction_10m"] is None:
        raise ValueError("Historical weather response is missing wind variables")
    return {
        "status": "available",
        "provider": "Open-Meteo Historical Weather API",
        "source_url": ARCHIVE_URL,
        "model": "best_match",
        "timestamp": parsed[index].isoformat(),
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "wind_speed_10m_ms": values["wind_speed_10m"],
        "wind_direction_10m_deg": values["wind_direction_10m"],
        "wind_gusts_10m_ms": values["wind_gusts_10m"],
    }


async def historical_wind(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    acquired: datetime,
) -> dict[str, Any]:
    date = acquired.date().isoformat()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "cell_selection": "sea",
    }
    try:
        response = await client.get(ARCHIVE_URL, params=params)
        response.raise_for_status()
        context = nearest_wind(response.json(), acquired)
        context["requested_latitude"] = latitude
        context["requested_longitude"] = longitude
        return context
    except (httpx.HTTPError, ValueError, KeyError):
        return {
            "status": "unavailable",
            "provider": "Open-Meteo Historical Weather API",
            "source_url": ARCHIVE_URL,
            "requested_latitude": latitude,
            "requested_longitude": longitude,
            "requested_timestamp": acquired.isoformat(),
        }
