"""Convert reviewed GeoJSON annotations into binary masks and a training manifest."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", default="training/label-packs/caspian-v1/pack.jsonl")
    parser.add_argument("--manifest", default="training/data/manifest.jsonl")
    return parser.parse_args()


def rasterize_record(root: Path, record: dict[str, Any]) -> tuple[np.ndarray, str]:
    from affine import Affine
    from rasterio.features import rasterize

    annotation_path = root / record["annotation"]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    properties = annotation.get("properties", {})
    status = properties.get("review_status")
    reviewed_by = properties.get("reviewed_by")
    if status not in {"reviewed_positive", "reviewed_negative"} or not reviewed_by:
        raise ValueError(f"{record['sample_id']}: annotation has not been reviewed")
    features = annotation.get("features", [])
    if status == "reviewed_positive" and not features:
        raise ValueError(
            f"{record['sample_id']}: positive review requires at least one polygon"
        )
    if status == "reviewed_negative" and features:
        raise ValueError(
            f"{record['sample_id']}: negative review cannot contain polygons"
        )
    shape = tuple(int(value) for value in record["shape"])
    transform = Affine(*record["transform"])
    geometries = [(feature["geometry"], 1) for feature in features]
    mask = (
        rasterize(
            geometries, out_shape=shape, transform=transform, fill=0, dtype="uint8"
        )
        if geometries
        else np.zeros(shape, dtype="uint8")
    )
    return mask, str(reviewed_by)


def build_manifest(pack_path: str | Path, manifest_path: str | Path) -> Path:
    pack = Path(pack_path).resolve()
    root = pack.parent
    records = [
        json.loads(line)
        for line in pack.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Label pack is empty")
    manifest = Path(manifest_path).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output_records = []
    for record in records:
        mask, reviewed_by = rasterize_record(root, record)
        mask_path = root / "masks" / f"{record['sample_id']}.npy"
        np.save(mask_path, mask.astype("float32"))
        output_records.append(
            {
                "sample_id": record["sample_id"],
                "image": Path(
                    os.path.relpath(
                        Path(record["image"])
                        if Path(record["image"]).is_absolute()
                        else root / record["image"],
                        manifest.parent,
                    )
                ).as_posix(),
                "mask": Path(os.path.relpath(mask_path, manifest.parent)).as_posix(),
                "scene_id": record["scene_id"],
                "acquisition_time": record["acquisition_time"],
                "region": record["region"],
                "bbox": record["bbox"],
                "reviewed_by": reviewed_by,
            }
        )
    manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in output_records)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps({"manifest": str(build_manifest(args.pack, args.manifest))}))


if __name__ == "__main__":
    main()
