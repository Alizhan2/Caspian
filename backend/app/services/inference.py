import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_validated_model_card(model_path: str | Path) -> dict[str, Any]:
    checkpoint = Path(model_path)
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        raise RuntimeError("Segmentation checkpoint is missing its model card")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Segmentation model card is invalid") from exc
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported segmentation model card schema")
    if payload.get("validation_status") != "validated":
        raise RuntimeError("Segmentation checkpoint has not been promoted to validated")
    if payload.get("architecture") != "unet-resnet34" or payload.get("input_channels") != ["VV", "VH"]:
        raise RuntimeError("Segmentation model card is incompatible with the runtime")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if payload.get("checkpoint_sha256") != digest:
        raise RuntimeError("Segmentation checkpoint SHA-256 does not match its model card")
    return payload


class OilUnetInference:
    """Run a validated U-Net or an explicitly experimental SAR baseline."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.58,
        experimental_baseline_enabled: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.experimental_baseline_enabled = experimental_baseline_enabled
        self.validation_status = "experimental" if experimental_baseline_enabled else "not-configured"
        self.checkpoint_error: str | None = None
        self.model_version = (
            "adaptive-sar-screening:v1:experimental" if experimental_baseline_enabled else "not-configured"
        )
        self.model = None
        if self.model_path.is_file():
            try:
                self._load_model()
            except RuntimeError as exc:
                if not self.experimental_baseline_enabled:
                    raise
                self.checkpoint_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None or self.experimental_baseline_enabled

    @property
    def validated(self) -> bool:
        return self.model is not None and self.validation_status == "validated"

    @property
    def backend(self) -> str:
        if self.model is not None:
            return "unet-resnet34"
        return "adaptive-sar-baseline" if self.experimental_baseline_enabled else "not-configured"

    def _load_model(self) -> None:
        try:
            import segmentation_models_pytorch as smp
            import torch

            metadata = load_validated_model_card(self.model_path)
            self.model = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=1)
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
            self.threshold = float(metadata["threshold"])
            self.validation_status = "validated"
            self.model_version = f"oil-unet-resnet34:{self.model_path.name}:{digest}:validated"
        except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.model = None
            raise RuntimeError("Unable to load configured segmentation model") from exc

    def predict(
        self,
        image: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError("A validated SAR segmentation checkpoint is not configured")
        if self.model is not None:
            import torch

            with torch.no_grad():
                logits = self.model(torch.from_numpy(image[None]).float())
                probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        else:
            probability = self._adaptive_sar_probability(image)
        binary = probability >= self.threshold
        if valid_mask is not None:
            binary &= valid_mask.astype(bool)
            probability = np.where(valid_mask, probability, 0)
        return probability.astype("float32"), binary

    @staticmethod
    def _adaptive_sar_probability(image: np.ndarray) -> np.ndarray:
        """Estimate dark-spot probability for experimental screening only."""
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError("Expected a normalized [2, H, W] VV/VH tensor")
        vv = image[0].astype("float32", copy=False)
        vh = image[1].astype("float32", copy=False)
        local_mean = OilUnetInference._box_mean(vv, 51)
        local_squared = OilUnetInference._box_mean(vv * vv, 51)
        local_std = np.sqrt(np.maximum(local_squared - local_mean * local_mean, 1e-4))
        local_contrast = (local_mean - vv) / (local_std + 0.04)
        contrast_score = 1.0 / (1.0 + np.exp(-np.clip((local_contrast - 1.15) * 2.4, -30, 30)))
        absolute_darkness = np.clip((0.58 - vv) / 0.38, 0, 1)
        dual_pol_darkness = np.clip((0.52 - vh) / 0.4, 0, 1)
        probability = contrast_score * (0.5 + 0.3 * absolute_darkness + 0.2 * dual_pol_darkness)
        return OilUnetInference._box_mean(probability, 7).astype("float32")

    @staticmethod
    def _box_mean(array: np.ndarray, kernel_size: int) -> np.ndarray:
        """Compute a same-sized reflected box mean without an ML runtime."""
        radius = kernel_size // 2
        padded = np.pad(array, ((radius, radius), (radius, radius)), mode="reflect")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        totals = (
            integral[kernel_size:, kernel_size:]
            - integral[:-kernel_size, kernel_size:]
            - integral[kernel_size:, :-kernel_size]
            + integral[:-kernel_size, :-kernel_size]
        )
        return totals / float(kernel_size * kernel_size)
