from typing import Any


def polygon_area_km2(geometry: dict[str, Any]) -> float:
    """Approximate a small WGS84 polygon area using an equirectangular projection."""
    import math

    coordinates = geometry.get("coordinates", [[]])[0]
    if len(coordinates) < 4:
        return 0.0
    mean_lat = sum(point[1] for point in coordinates) / len(coordinates)
    earth_km = 111.32
    scale_x = earth_km * math.cos(math.radians(mean_lat))
    area = 0.0
    for first, second in zip(coordinates, coordinates[1:], strict=False):
        x1, y1 = first[0] * scale_x, first[1] * earth_km
        x2, y2 = second[0] * scale_x, second[1] * earth_km
        area += x1 * y2 - x2 * y1
    return round(abs(area) / 2, 4)


def raster_mask_to_geojson(mask: Any, transform: Any, crs: Any = "EPSG:4326", min_pixels: int = 8) -> dict[str, Any]:
    """Vectorize a real raster mask while preserving its geographic coordinates."""
    import numpy as np
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape

    valid = np.asarray(mask, dtype="uint8") > 0
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    features: list[dict[str, Any]] = []
    for geometry, value in shapes(valid.astype("uint8"), mask=valid, transform=transform):
        polygon = shape(geometry)
        pixel_count = polygon.area / pixel_area if pixel_area else 0
        if value == 1 and pixel_count >= min_pixels:
            features.append({"type": "Feature", "properties": {"source": "sentinel-1"}, "geometry": mapping(polygon)})
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": str(crs)}},
    }
