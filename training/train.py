"""Train a two-channel U-Net and export a non-promoted model candidate."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from training.datasets import (
    SarPatchDataset,
    load_manifest,
    manifest_sha256,
    split_by_scene,
    split_summary,
)
from training.evaluate import aggregate_metrics, select_threshold
from training.export import export_candidate_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training/data/manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default="models/oil_unet.pt")
    parser.add_argument("--metadata", default="models/oil_unet.json")
    return parser.parse_args()


def dice_loss(logits, targets):
    import torch

    probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum(dim=(1, 2, 3))
    total = probability.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + 1) / (total + 1)).mean()


def evaluate_model(
    model, loader, device
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    import torch
    import torch.nn.functional as functional

    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for images, masks, _sample_ids in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = functional.binary_cross_entropy_with_logits(
                logits, masks
            ) + dice_loss(logits, masks)
            losses.append(float(loss.cpu()))
            probabilities.extend(torch.sigmoid(logits)[:, 0].cpu().numpy())
            targets.extend(masks[:, 0].cpu().numpy())
    return probabilities, targets, float(np.mean(losses))


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_manifest(args.manifest)
    splits = split_by_scene(
        records,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    loaders = {
        name: DataLoader(
            SarPatchDataset(args.manifest, split_records),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
        )
        for name, split_records in splits.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=1).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_iou = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, masks, _sample_ids in loaders["train"]:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.binary_cross_entropy_with_logits(
                logits, masks
            ) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_probabilities, validation_targets, validation_loss = evaluate_model(
            model, loaders["validation"], device
        )
        validation_metrics = aggregate_metrics(
            validation_probabilities, validation_targets, 0.5
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "validation_iou_at_0_5": validation_metrics["iou"],
            }
        )
        if validation_metrics["iou"] > best_iou:
            best_iou = validation_metrics["iou"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(history[-1]))
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_probabilities, validation_targets, _ = evaluate_model(
        model, loaders["validation"], device
    )
    threshold, validation_metrics = select_threshold(
        validation_probabilities, validation_targets
    )
    test_probabilities, test_targets, test_loss = evaluate_model(
        model, loaders["test"], device
    )
    test_metrics = aggregate_metrics(test_probabilities, test_targets, threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    metadata = export_candidate_metadata(
        args.metadata,
        output,
        manifest_sha256(args.manifest),
        split_summary(splits),
        {
            "validation": validation_metrics,
            "test": {**test_metrics, "loss": test_loss},
            "history": history,
        },
        threshold,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "metadata": str(metadata),
                "threshold": threshold,
            }
        )
    )


if __name__ == "__main__":
    main()
