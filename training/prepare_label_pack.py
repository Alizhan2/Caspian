"""Download real Sentinel-1 patches and create review-safe annotation templates."""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.services.copernicus_auth import CopernicusAuth
from app.services.scene_processor import normalize_vv_vh
from app.services.sentinel_process import SentinelProcess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--areas", default="training/areas.caspian.json")
    parser.add_argument("--output", default="training/label-packs/caspian-v1")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--days-back", type=int, default=30)
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


async def prepare_area(
    client: httpx.AsyncClient,
    process: SentinelProcess,
    area: dict[str, Any],
    output: Path,
    days_back: int,
) -> dict[str, Any]:
    response = await client.post(
        "/scenes/search",
        json={"bbox": area["bbox"], "days_back": days_back},
    )
    response.raise_for_status()
    scene = response.json()
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
    }
    if not annotation_path.exists():
        annotation_path.write_text(
            json.dumps(annotation_template(record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return record


async def run(args: argparse.Namespace) -> None:
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
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=args.api_url.rstrip("/"), timeout=90
    ) as client:
        for area in areas:
            record = await prepare_area(client, process, area, output, args.days_back)
            records.append(record)
            print(
                json.dumps(
                    {"prepared": record["sample_id"], "scene_id": record["scene_id"]}
                )
            )
    index = output / "pack.jsonl"
    index.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pack": str(index), "samples": len(records)}))


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
