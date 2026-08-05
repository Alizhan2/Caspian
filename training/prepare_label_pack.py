"""Collect multi-date Sentinel-1 patches with review-safe wind context."""

import argparse
import asyncio
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.services.copernicus_auth import CopernicusAuth
from app.services.scene_processor import normalize_vv_vh
from app.services.sentinel_catalog import SceneCatalog
from app.services.sentinel_process import SentinelProcess
from training.weather import historical_wind

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--areas", default="training/areas.caspian.json")
    parser.add_argument("--output", default="training/label-packs/caspian-v1")
    parser.add_argument("--stac-url", default=EARTH_SEARCH_URL)
    parser.add_argument("--days-back", type=int, default=180)
    parser.add_argument("--scenes-per-area", type=int, default=4)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def annotation_template(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "review_status": "unreviewed",
            "reviewed_by": None,
            "reviewed_at": None,
            "sample_id": sample["sample_id"],
            "scene_id": sample["scene_id"],
            "region": sample["region"],
            "instructions": (
                "Draw only expert-reviewed potential oil-like polygons. "
                "Set review_status to reviewed_positive or reviewed_negative."
            ),
        },
    }


def save_preview(tensor: np.ndarray, destination: Path) -> None:
    vv = np.clip(tensor[0], 0, 1)
    vh = np.clip(tensor[1], 0, 1)
    ratio = np.clip((vv - vh + 1) / 2, 0, 1)
    rgb = np.stack([vv, vh, ratio], axis=-1)
    Image.fromarray((rgb * 255).astype("uint8"), mode="RGB").save(destination)


def compatible_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    compatible: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    ordered = sorted(
        items,
        key=lambda item: item.get("properties", {}).get("datetime", ""),
        reverse=True,
    )
    for original in ordered:
        item = deepcopy(original)
        properties = item.get("properties", {})
        mode = str(
            properties.get("sar:instrument_mode")
            or properties.get("s1:instrument_mode")
            or "IW"
        ).upper()
        polarizations = properties.get("sar:polarizations") or []
        if isinstance(polarizations, str):
            polarizations = polarizations.replace("&", " ").split()
        normalized = {str(value).upper() for value in polarizations}
        acquired = str(properties.get("datetime") or "")
        acquisition_date = acquired[:10]
        assets = item.get("assets", {})
        if (
            mode != "IW"
            or (normalized and not {"VV", "VH"}.issubset(normalized))
            or not assets.get("vv", {}).get("href")
            or not assets.get("vh", {}).get("href")
            or not acquisition_date
            or acquisition_date in seen_dates
        ):
            continue
        for asset in assets.values():
            href = asset.get("href")
            if href:
                asset["href"] = SceneCatalog._public_asset_url(str(href))
        compatible.append(item)
        seen_dates.add(acquisition_date)
        if len(compatible) >= limit:
            break
    return compatible


async def discover_scenes(
    client: httpx.AsyncClient,
    stac_url: str,
    bbox: list[float],
    days_back: int,
    limit: int,
) -> list[dict[str, Any]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    response = await client.post(
        f"{stac_url.rstrip('/')}/search",
        json={
            "bbox": bbox,
            "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
            "collections": ["sentinel-1-grd"],
            "limit": 100,
        },
    )
    response.raise_for_status()
    return compatible_items(response.json().get("features", []), limit)


def scene_from_item(item: dict[str, Any]) -> dict[str, Any]:
    properties = item.get("properties", {})
    scene_id = str(item.get("id") or "")
    acquired = str(properties.get("datetime") or "")
    if not scene_id or not acquired:
        raise ValueError("STAC item is missing scene identity or acquisition time")
    return {
        "external_scene_id": scene_id,
        "acquisition_time": acquired,
        "source_metadata": item,
    }


async def prepare_scene(
    process: SentinelProcess,
    weather_client: httpx.AsyncClient,
    area: dict[str, Any],
    output: Path,
    scene: dict[str, Any],
) -> dict[str, Any]:
    acquired = datetime.fromisoformat(scene["acquisition_time"].replace("Z", "+00:00"))
    suffix = scene["external_scene_id"][-6:].lower()
    sample_id = safe_name(f"{area['region']}-{acquired:%Y%m%dT%H%M%S}-{suffix}")
    raster_path = await process.download_vv_vh(
        list(area["bbox"]),
        (scene["acquisition_time"], scene["acquisition_time"]),
        sample_id,
        scene["source_metadata"],
    )
    import rasterio

    with rasterio.open(raster_path) as raster:
        prepared = normalize_vv_vh(raster.read(1), raster.read(2), raster.nodata)
        transform = list(raster.transform)[:6]
        crs = str(raster.crs or "EPSG:4326")
        shape = [raster.height, raster.width]
        bounds = list(raster.bounds)
    image_path = output / "images" / f"{sample_id}.npy"
    preview_path = output / "previews" / f"{sample_id}.png"
    annotation_path = output / "annotations" / f"{sample_id}.geojson"
    np.save(image_path, prepared.tensor)
    save_preview(prepared.tensor, preview_path)
    longitude = (area["bbox"][0] + area["bbox"][2]) / 2
    latitude = (area["bbox"][1] + area["bbox"][3]) / 2
    record = {
        "sample_id": sample_id,
        "image": image_path.relative_to(output).as_posix(),
        "preview": preview_path.relative_to(output).as_posix(),
        "annotation": annotation_path.relative_to(output).as_posix(),
        "raster": raster_path.relative_to(output).as_posix(),
        "scene_id": scene["external_scene_id"],
        "acquisition_time": scene["acquisition_time"],
        "region": area["region"],
        "region_name": area.get("name", area["region"]),
        "bbox": bounds,
        "requested_bbox": area["bbox"],
        "crs": crs,
        "transform": transform,
        "shape": shape,
        "weather_context": await historical_wind(
            weather_client, latitude, longitude, acquired
        ),
    }
    if not annotation_path.exists():
        annotation_path.write_text(
            json.dumps(annotation_template(record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return record


def load_records(index: Path) -> list[dict[str, Any]]:
    if not index.exists():
        return []
    return [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def run(args: argparse.Namespace) -> None:
    if args.days_back < 1 or args.scenes_per_area < 1:
        raise ValueError("days-back and scenes-per-area must be positive")
    areas = json.loads(Path(args.areas).read_text(encoding="utf-8"))
    if not isinstance(areas, list) or not areas:
        raise ValueError("Areas configuration must be a non-empty JSON array")
    output = Path(args.output).resolve()
    for folder in ("rasters", "images", "previews", "annotations", "masks"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    process = SentinelProcess(
        CopernicusAuth("", "", ""),
        "https://sh.dataspace.copernicus.eu",
        str(output / "rasters"),
    )
    index = output / "pack.jsonl"
    records = load_records(index)
    by_scene = {(item["region"], item["scene_id"]): item for item in records}
    async with httpx.AsyncClient(timeout=120) as client:
        for area in areas:
            items = await discover_scenes(
                client,
                args.stac_url,
                area["bbox"],
                args.days_back,
                args.scenes_per_area,
            )
            for item in items:
                scene = scene_from_item(item)
                key = (area["region"], scene["external_scene_id"])
                if key in by_scene:
                    existing = by_scene[key]
                    if "weather_context" not in existing:
                        acquired = datetime.fromisoformat(
                            existing["acquisition_time"].replace("Z", "+00:00")
                        )
                        longitude = (area["bbox"][0] + area["bbox"][2]) / 2
                        latitude = (area["bbox"][1] + area["bbox"][3]) / 2
                        existing["weather_context"] = await historical_wind(
                            client, latitude, longitude, acquired
                        )
                        print(json.dumps({"weather_enriched": existing["sample_id"]}))
                    else:
                        print(json.dumps({"skipped": existing["sample_id"]}))
                    continue
                record = await prepare_scene(process, client, area, output, scene)
                by_scene[key] = record
                records.append(record)
                print(
                    json.dumps(
                        {
                            "prepared": record["sample_id"],
                            "scene_id": record["scene_id"],
                        }
                    )
                )
    records.sort(
        key=lambda item: (item["region"], item["acquisition_time"]), reverse=True
    )
    index.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pack": str(index), "samples": len(records)}))


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
