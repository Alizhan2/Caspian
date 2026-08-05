# Caspian Guardian AI

Live-only platform for Sentinel-1 SAR screening of potential oil-like anomalies in the Caspian Sea. The system never fabricates a scene when production inputs are missing: the UI shows which component must be configured.

> Every detection is an **experimental screening signal**, not a confirmed oil spill. Operator review and field verification remain mandatory.

## What is implemented

- Copernicus Data Space OAuth2 client-credentials authentication with token reuse.
- Sentinel Hub Catalog search for the newest Sentinel-1 GRD IW scene with VV/VH polarization.
- Process API download of orthorectified two-band GeoTIFF imagery.
- Local open-source U-Net/ResNet34 inference through `segmentation-models-pytorch`; no proprietary inference API.
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
Copernicus   U-Net/ResNet34
 Catalog + Process API
      │
    MinIO ─── optional Ollama explanation
```

## Required production inputs

1. Create an OAuth client in the Copernicus Data Space Sentinel Hub dashboard.
2. Put `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in a local `.env` copied from `.env.example`.
3. Place a validated two-channel U-Net checkpoint at `models/oil_unet.pt`.
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

- `copernicus`
- `segmentation_model`
- `postgis`
- `redis`
- `object_storage`

`live_ready` becomes true only when all five are ready. Search and analysis endpoints return `503` otherwise. There is no synthetic fallback.

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

The runtime expects a `segmentation_models_pytorch.Unet` with a ResNet34 encoder, two input channels (`VV`, `VH`) and one output class. A checkpoint must be trained and evaluated on labelled Caspian SAR data split by both date and region. Record precision, recall, IoU, false alarms per scene and the selected threshold. Until that evaluation is complete, all outputs remain experimental screening signals.

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

- Backend: 9 tests passed.
- SQLAlchemy mapper configuration: passed.
- Frontend TypeScript and Vite production build: passed.
- Docker Compose runtime: not executed because Docker is not installed in this Windows environment.
- Live Copernicus acquisition: not executed because no user OAuth credentials are present.
- Real model inference: not executed because no validated checkpoint is present.

Official integration references: [Copernicus Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html), [Catalog API examples](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog/Examples.html), [Sentinel-1 GRD Process API](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html), and [SkyTruth Cerulean Cloud](https://github.com/SkyTruth/cerulean-cloud) as an architectural reference for human-reviewed oil-slick monitoring.
