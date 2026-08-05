"""Promote a measured model candidate after explicit human approval."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training.export import checkpoint_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="models/oil_unet.json")
    parser.add_argument("--checkpoint", default="models/oil_unet.pt")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--min-scenes", type=int, default=12)
    parser.add_argument("--min-regions", type=int, default=3)
    parser.add_argument("--min-dates", type=int, default=6)
    return parser.parse_args()


def promote(payload: dict, checkpoint: str | Path, args: argparse.Namespace) -> dict:
    if payload.get("validation_status") != "candidate":
        raise ValueError("Only candidate model cards can be promoted")
    if payload.get("checkpoint_sha256") != checkpoint_sha256(checkpoint):
        raise ValueError("Checkpoint SHA-256 does not match the model card")
    metrics = payload.get("metrics", {}).get("test", {})
    minimums = {
        "iou": args.min_iou,
        "precision": args.min_precision,
        "recall": args.min_recall,
    }
    failures = [
        name
        for name, minimum in minimums.items()
        if float(metrics.get(name, -1)) < minimum
    ]
    summary = payload.get("split_summary", {})
    all_splits = list(summary.values())
    scenes = sum(int(split.get("scenes", 0)) for split in all_splits)
    regions = {region for split in all_splits for region in split.get("regions", [])}
    dates = {date for split in all_splits for date in split.get("dates", [])}
    if scenes < args.min_scenes:
        failures.append(f"scenes<{args.min_scenes}")
    if len(regions) < args.min_regions:
        failures.append(f"regions<{args.min_regions}")
    if len(dates) < args.min_dates:
        failures.append(f"dates<{args.min_dates}")
    if failures:
        raise ValueError(f"Promotion gates failed: {', '.join(failures)}")
    payload["validation_status"] = "validated"
    payload["validated_at"] = datetime.now(timezone.utc).isoformat()
    payload["approved_by"] = args.approved_by
    payload["approval_note"] = args.approval_note
    payload["promotion_gates"] = {
        **minimums,
        "min_scenes": args.min_scenes,
        "min_regions": args.min_regions,
        "min_dates": args.min_dates,
    }
    return payload


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    promoted = promote(payload, args.checkpoint, args)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(promoted, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(metadata_path)
    print(
        json.dumps({"metadata": str(metadata_path), "validation_status": "validated"})
    )


if __name__ == "__main__":
    main()
