import hashlib
from pathlib import Path

import numpy as np


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
        self.model_version = (
            "adaptive-sar-screening:v1:experimental" if experimental_baseline_enabled else "not-configured"
        )
        self.model = None
        if self.model_path.is_file():
            self._load_model()

    @property
    def ready(self) -> bool:
        return self.model is not None or self.experimental_baseline_enabled

    @property
    def validated(self) -> bool:
        return self.model is not None

    @property
    def backend(self) -> str:
        if self.model is not None:
            return "unet-resnet34"
        return "adaptive-sar-baseline" if self.experimental_baseline_enabled else "not-configured"

    def _load_model(self) -> None:
        try:
            import segmentation_models_pytorch as smp
            import torch

            self.model = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=1)
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
            self.model_version = f"oil-unet-resnet34:{self.model_path.name}:{digest}"
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError("Unable to load configured segmentation model") from exc

    def predict(
        self,
        image: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError("A validated SAR segmentation checkpoint is not configured")
        import torch

        if self.model is not None:
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
        import torch
        import torch.nn.functional as functional

        tensor = torch.from_numpy(image[None]).float()
        vv = tensor[:, :1]
        vh = tensor[:, 1:2]
        with torch.no_grad():
            local_mean = functional.avg_pool2d(vv, 51, stride=1, padding=25)
            local_squared = functional.avg_pool2d(vv.square(), 51, stride=1, padding=25)
            local_std = torch.sqrt(torch.clamp(local_squared - local_mean.square(), min=1e-4))
            local_contrast = (local_mean - vv) / (local_std + 0.04)
            contrast_score = torch.sigmoid((local_contrast - 1.15) * 2.4)
            absolute_darkness = torch.clamp((0.58 - vv) / 0.38, 0, 1)
            dual_pol_darkness = torch.clamp((0.52 - vh) / 0.4, 0, 1)
            probability = contrast_score * (0.5 + 0.3 * absolute_darkness + 0.2 * dual_pol_darkness)
            probability = functional.avg_pool2d(probability, 7, stride=1, padding=3)
        return probability[0, 0].numpy()
