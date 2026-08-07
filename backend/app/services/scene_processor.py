from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedScene:
    tensor: np.ndarray
    valid_mask: np.ndarray
    tile_size: int


def normalize_vv_vh(vv: np.ndarray, vh: np.ndarray, nodata: float | None = None) -> PreparedScene:
    """Normalize calibrated backscatter or public Sentinel-1 digital amplitudes."""
    if vv.shape != vh.shape or vv.ndim != 2:
        raise ValueError("VV and VH rasters must be matching 2D arrays")
    valid = np.isfinite(vv) & np.isfinite(vh)
    if nodata is not None:
        valid &= (vv != nodata) & (vh != nodata)
    channels = np.stack([vv, vh]).astype("float32")
    valid_values = channels[:, valid]
    if valid_values.size and float(np.nanmedian(valid_values)) > 2:
        # Earth Search measurement COGs expose raw digital amplitudes. Their
        # relative contrast remains useful for screening even before applying
        # the per-scene XML calibration LUT.
        log_amplitude = 20 * np.log10(np.maximum(channels, 1e-8))
        normalized = np.zeros_like(log_amplitude, dtype="float32")
        for index in range(2):
            low, high = np.nanpercentile(log_amplitude[index][valid], [1, 99])
            scale = max(float(high - low), 1e-6)
            normalized[index] = np.clip((log_amplitude[index] - low) / scale, 0, 1)
    else:
        db = 10 * np.log10(np.maximum(channels, 1e-8))
        normalized = (np.clip(db, -35, 5) + 35) / 40
    normalized[:, ~valid] = 0
    return PreparedScene(normalized.astype(np.float32), valid, 512)
