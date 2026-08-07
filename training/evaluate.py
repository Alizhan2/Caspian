from collections.abc import Iterable

import numpy as np


def segmentation_metrics(prediction, target) -> dict[str, float]:
    """Compute IoU, Dice, precision, and recall for binary arrays."""
    pred, truth = np.asarray(prediction).astype(bool), np.asarray(target).astype(bool)
    intersection, union = (pred & truth).sum(), (pred | truth).sum()
    total = pred.sum() + truth.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / total) if total else 1.0,
        "precision": float(intersection / pred.sum()) if pred.sum() else 0.0,
        "recall": float(intersection / truth.sum()) if truth.sum() else 0.0,
    }


def aggregate_metrics(
    probabilities: Iterable[np.ndarray], targets: Iterable[np.ndarray], threshold: float
):
    predictions = [np.asarray(item) >= threshold for item in probabilities]
    truth = [np.asarray(item) >= 0.5 for item in targets]
    if not predictions:
        raise ValueError("Cannot evaluate an empty split")
    return segmentation_metrics(
        np.concatenate([item.ravel() for item in predictions]),
        np.concatenate([item.ravel() for item in truth]),
    )


def select_threshold(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    candidates: Iterable[float] | None = None,
) -> tuple[float, dict[str, float]]:
    thresholds = list(candidates or np.arange(0.3, 0.71, 0.05))
    scored = [
        (float(threshold), aggregate_metrics(probabilities, targets, float(threshold)))
        for threshold in thresholds
    ]
    return max(scored, key=lambda item: (item[1]["iou"], item[1]["recall"]))
