import hashlib
from pathlib import Path

import numpy as np


class OilUnetInference:
    """Load a local open-source segmentation checkpoint and fail closed."""

    def __init__(self, model_path: str, threshold: float = 0.58) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model_version = "not-configured"
        self.model = None
        if self.model_path.is_file():
            self._load_model()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        try:
            import segmentation_models_pytorch as smp
            import torch

            self.model = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=1)
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model.eval()
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
            self.model_version = f"oil-unet-resnet34:{self.model_path.name}:{digest}"
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError("Unable to load configured segmentation model") from exc

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("A validated SAR segmentation checkpoint is not configured")
        import torch

        with torch.no_grad():
            logits = self.model(torch.from_numpy(image[None]).float())
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return probability, probability >= self.threshold
