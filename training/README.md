# Caspian SAR model pipeline

Training data is a JSON Lines manifest. Every row points to a normalized two-channel NumPy patch and a binary mask:

```json
{"sample_id":"aktau-20260802-0001","image":"images/aktau-20260802-0001.npy","mask":"masks/aktau-20260802-0001.npy","scene_id":"S1D_...","acquisition_time":"2026-08-02T14:21:15Z","region":"aktau","bbox":[51.8,42.75,52.2,43.05]}
```

- `image`: `float32 [2,H,W]`, channels VV and VH, normalized with the same preprocessing as the runtime.
- `mask`: binary `float32 [H,W]`; labels must be produced or reviewed by qualified annotators.
- all patches from one `scene_id` stay in one split to prevent spatial leakage.
- the model card records split scenes, regions and acquisition dates.

## Build a real annotation pack

Collect multi-date real patches for Aktau, Kashagan and Atyrau:

```powershell
$env:PYTHONPATH = "backend"
python -m training.prepare_label_pack --days-back 180 --scenes-per-area 4
```

The collector queries the public Earth Search STAC catalogue, keeps one compatible Sentinel-1 IW VV/VH scene per acquisition date and merges new records into the existing pack without overwriting reviewed GeoJSON. Each record also stores the nearest hourly 10 m wind speed, direction and gust from the Open-Meteo Historical Weather API. Missing weather is explicitly stored as `unavailable` rather than estimated. For every scene after the first in a region, it also stores a contrast-normalized VV/VH difference against the previous date; this is an operator screening aid, not radiometrically calibrated change detection or proof of a spill.

The ignored `training/label-packs/caspian-v1` directory contains two-channel arrays, georeferenced rasters, RGB previews and one GeoJSON annotation template per scene. Open the rasters/templates in the built-in bilingual labeling center, QGIS or another geospatial annotation tool.

An annotation is never interpreted as a negative sample merely because it is empty. A reviewer must set:

- `review_status: reviewed_positive`, add at least one polygon and fill `reviewed_by`; or
- `review_status: reviewed_negative`, leave `features` empty and fill `reviewed_by`.

After every sample has been reviewed, build masks and the training manifest:

```powershell
python -m training.rasterize_labels
```

Train a candidate from the repository root:

```powershell
python -m training.train --manifest training/data/manifest.jsonl --epochs 20
```

Training writes `models/oil_unet.pt` and `models/oil_unet.json`. The model card initially has `validation_status: candidate`, so production will continue using the experimental SAR baseline.

After independent review, promote only if the default metric and dataset-coverage gates pass:

```powershell
python -m training.promote --approved-by "Reviewer name" --approval-note "Dataset and false alarms reviewed"
```

The backend verifies the promoted status, architecture, input channels and checkpoint SHA-256 before enabling U-Net inference. A promoted model is still an experimental screening tool, not proof of an oil spill.
