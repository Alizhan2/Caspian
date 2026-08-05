from pathlib import Path
import numpy as np


def load_dataset(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load paired .npy VV/VH tensors and binary masks."""
    root_path = Path(root)
    images, masks = np.load(root_path / "images.npy"), np.load(root_path / "masks.npy")
    if images.ndim != 4 or images.shape[1] != 2 or masks.shape[0] != images.shape[0]:
        raise ValueError("Expected images [N, 2, H, W] and matching masks")
    return images.astype("float32"), masks.astype("float32")
