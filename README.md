# Caspian Guardian AI

Live-only platform for Sentinel-1 SAR screening of potential oil-like anomalies in the Caspian Sea. The system never fabricates a scene when production inputs are missing: the UI shows which component must be configured.

> Every detection is an **experimental screening signal**, not a confirmed oil spill. Operator review and field verification remain mandatory.

## What is implemented

- Public Earth Search STAC discovery for the newest Sentinel-1 GRD IW scene with VV/VH polarization; no API key is required.
- Range-based VV/VH crop from public Sentinel-1 Cloud Optimized GeoTIFFs, so the full source products are not downloaded.
- Optional Copernicus Data Space OAuth2 and Process API fallback with token reuse.
- Local open-source U-Net/ResNet34 inference through `segmentation-models-pytorch`; no proprietary inference API.
- Experimental adaptive SAR dark-spot baseline for end-to-end live screening when validated U-Net weights are unavailable.
- Fail-closed model loading with checkpoint SHA-256 included in every model version.
- Configurable confidence and min/max area filters.
- PostGIS persistence for AOIs, subscriptions, scenes, jobs, detections, reports and review history.
- MinIO storage for original GeoTIFFs, previews and masks.
- Redis/Celery worker with retries, late acknowledgements and a Celery Beat discovery schedule.
- Processing states: `DISCOVERED → DOWNLOADING → PROCESSING → AI_ANALYSIS → REVIEW`.
- Operator actions: confirm, mark false positive, or escalate; every action is audited.
- Optional local Ollama explanation. Ollama writes operator-facing text only and never performs SAR segmentation.
- Russian and Kazakh responsive web UI, GeoJSON API and RU/KK/EN PDF reports.

## Architecture

```text
React/Leaflet
      │
   FastAPI ───────── PostGIS
      │                 │
    Redis ─ Celery ─ Celery Beat
      │         │
Earth Search / Copernicus   U-Net/ResNet34
   STAC + COG / Process API
      │
    MinIO ─── optional Ollama explanation
```

## Required production inputs

1. Copy `.env.example` to `.env`. The default Earth Search catalog is public and needs no API key.
2. Optionally add `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` for the Copernicus Process API fallback.
3. For validated ML inference, place a two-channel U-Net checkpoint at `models/oil_unet.pt`. Without it, the explicitly labelled experimental SAR baseline is used.
4. Change all example database and MinIO passwords before internet deployment.
5. Install Docker Desktop, then start the stack.

On this Windows machine, WSL2 first requires an Administrator PowerShell. Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\enable-wsl2-admin.ps1
```

Restart Windows, install Docker Desktop, and verify the remaining inputs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-live-readiness.ps1
```

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open:

- Product UI: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`
- MinIO console: `http://localhost:9001`

Enable the optional Ollama service only after the core pipeline works:

```powershell
$env:OLLAMA_ENABLED = "true"
docker compose --profile local-ai up --build
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

The model pull is intentionally not automatic because it is large. `OLLAMA_MODEL` can point to another locally installed open-source instruct model.

## Readiness contract

`GET /api/health` returns the state of:

- `satellite_catalog`
- `segmentation_model`
- `postgis`
- `redis`
- `object_storage`

`satellite_ready` becomes true when the public catalog and infrastructure are available, allowing real scene search and previews. `live_ready` additionally requires the segmentation model before analysis can run. There is no synthetic fallback.

## AOI monitoring

`POST /api/areas` accepts a name, WGS84 bounding box and `interval_minutes` from 60 to 180. Celery Beat checks due subscriptions, compares the newest external scene ID with the last processed ID, and queues analysis only for a new scene.

Example:

```json
{
  "name": "Aktau port",
  "bbox": [51.05, 43.30, 51.28, 43.47],
  "interval_minutes": 60
}
```

## Model requirements

The validated runtime target is a `segmentation_models_pytorch.Unet` with a ResNet34 encoder, two input channels (`VV`, `VH`) and one output class. A checkpoint must be trained and evaluated on labelled Caspian SAR data split by both date and region. Record precision, recall, IoU, false alarms per scene and the selected threshold. Until that evaluation is complete, the adaptive SAR baseline keeps the live pipeline operational, but all outputs remain experimental screening signals and its scores are not calibrated oil probabilities.

The reproducible training and annotation workflow is documented in [`training/README.md`](training/README.md). `training.prepare_label_pack` collects real georeferenced patches and review-safe GeoJSON templates for Aktau, Kashagan and Atyrau. The bilingual operator UI includes an expert-labeling center that saves only explicit reviewed-positive or reviewed-negative decisions. Training produces a `candidate` model card; a separate promotion command requires metric, scene, region and date coverage gates plus named human approval. At startup, the backend verifies the model-card status, architecture, channels and checkpoint SHA-256 before marking U-Net as validated.

Ollama is not a replacement for U-Net or SegFormer: an LLM cannot provide reliable pixel-wise SAR segmentation. It is restricted to summarising already computed metrics under a prompt that forbids invented wind, AIS, weather or chemical evidence.

## Verification and current scientific limits

Implemented controls include dual-polarization selection, orthorectification, model thresholding, min/max area limits, immutable scene IDs, model-version traceability and human review.

Before operational environmental use, add and validate:

- coastline/water mask;
- wind fields at acquisition time;
- before/after scene comparison;
- polygon shape features;
- authorized AIS context;
- organisation accounts and role-based permissions;
- reviewed-only Telegram/email/webhook delivery;
- a Caspian-labelled dataset and an independently evaluated checkpoint.

These are not silently simulated by the current build.

## Verification performed in this workspace

- Backend, annotation and training pipeline: 23 tests passed, including reviewer-name validation.
- SQLAlchemy mapper configuration: passed.
- Frontend TypeScript and Vite production build: passed.
- Docker Compose infrastructure: PostGIS, Redis and MinIO healthy.
- Live satellite discovery: Earth Search returned Sentinel-1D scene `S1D_IW_GRDH_1SDV_20260802T142102_20260802T142127_003948_00727D` acquired on 2026-08-02.
- Live imagery crop: VV/VH source ranges were read into a georeferenced 512 x 512, two-band GeoTIFF without downloading the complete source products.
- End-to-end live screening: Celery processed a real Sentinel-1 crop through the experimental adaptive SAR baseline, stored seven candidate polygons in PostGIS and uploaded the raster, preview and mask to MinIO.
- Training smoke test: one complete U-Net epoch produced a candidate state dict, validation/test metrics, selected threshold and non-promoted model card.
- Real annotation pack: three 512 x 512 VV/VH patches, GeoTIFFs, previews and `unreviewed` GeoJSON templates were generated for Aktau, Kashagan and Atyrau; no synthetic labels were created.
- Real model inference: not executed because no validated checkpoint is present.

Official integration references: [Earth Search examples](https://element84.com/earth-search/examples/), [Copernicus Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html), [Sentinel-1 GRD Process API](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html), and [SkyTruth Cerulean Cloud](https://github.com/SkyTruth/cerulean-cloud) as an architectural reference for human-reviewed oil-slick monitoring.
