# Caspian SAR model pipeline

Training data is a JSON Lines manifest. Every row points to a normalized two-channel NumPy patch and a binary mask:

```json
{"sample_id":"aktau-20260802-0001","image":"images/aktau-20260802-0001.npy","mask":"masks/aktau-20260802-0001.npy","scene_id":"S1D_...","acquisition_time":"2026-08-02T14:21:15Z","region":"aktau","bbox":[51.8,42.75,52.2,43.05]}
```

- `image`: `float32 [2,H,W]`, channels VV and VH, normalized with the same preprocessing as the runtime.
- `mask`: binary `float32 [H,W]`; labels must be produced or reviewed by qualified annotators.
- all patches from one `scene_id` stay in one split to prevent spatial leakage.
- the model card records split scenes, regions and acquisition dates.

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
