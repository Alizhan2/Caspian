def segmentation_metrics(prediction, target) -> dict[str, float]:
    """Compute IoU, Dice, precision, and recall for binary arrays."""
    pred, truth = prediction.astype(bool), target.astype(bool)
    intersection, union = (pred & truth).sum(), (pred | truth).sum()
    return {"iou": float(intersection / union) if union else 1.0, "dice": float(2 * intersection / (pred.sum() + truth.sum())) if (pred.sum() + truth.sum()) else 1.0, "precision": float(intersection / pred.sum()) if pred.sum() else 0.0, "recall": float(intersection / truth.sum()) if truth.sum() else 0.0}
