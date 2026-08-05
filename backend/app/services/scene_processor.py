from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedScene:
    tensor: np.ndarray
    valid_mask: np.ndarray
    tile_size: int


def normalize_vv_vh(vv: np.ndarray, vh: np.ndarray, nodata: float | None = None) -> PreparedScene:
    """Convert linear backscatter to clipped, per-channel normalized dB arrays."""
    if vv.shape != vh.shape or vv.ndim != 2:
        raise ValueError("VV and VH rasters must be matching 2D arrays")
    valid = np.isfinite(vv) & np.isfinite(vh)
    if nodata is not None:
        valid &= (vv != nodata) & (vh != nodata)
    db = np.stack([10 * np.log10(np.maximum(vv, 1e-8)), 10 * np.log10(np.maximum(vh, 1e-8))])
    db = np.clip(db, -35, 5)
    normalized = (db + 35) / 40
    normalized[:, ~valid] = 0
    return PreparedScene(normalized.astype(np.float32), valid, 512)
