import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from uuid import UUID, uuid4

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, new_request_id, request_id_context
from app.repositories import PostgresRepository
from app.schemas import (
    AnalysisQueued,
    AnalysisRequest,
    AnalysisStatus,
    AreaCreate,
    AreaResponse,
    BoundingBox,
    DetectionResponse,
    DetectionReviewRequest,
    DetectionReviewResponse,
    ReportRequest,
    ReportResponse,
    SceneResponse,
    SceneSearchRequest,
    utc_now,
)
from app.services.copernicus_auth import CopernicusAuth
from app.services.inference import OilUnetInference
from app.services.object_storage import ObjectStorage
from app.services.oil_analysis import RealOilAnalysis
from app.services.ollama_explainer import OllamaExplainer
from app.services.report_service import build_report_content, create_pdf
from app.services.risk_calculator import calculate_risk
from app.services.sentinel_catalog import SceneCatalog
from app.services.sentinel_process import SentinelProcess
from app.worker import celery_app

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)
repo = PostgresRepository(settings.database_url)
auth = CopernicusAuth(settings.cdse_client_id, settings.cdse_client_secret, settings.cdse_token_url)
catalog = SceneCatalog(auth, settings.cdse_base_url)
process = SentinelProcess(auth, settings.cdse_base_url, settings.storage_path)
inference = OilUnetInference(settings.model_path, settings.model_threshold)
real_analysis = RealOilAnalysis(process, inference)
storage = ObjectStorage(
    settings.minio_endpoint,
    settings.minio_access_key,
    settings.minio_secret_key,
    settings.minio_bucket,
    settings.minio_secure,
)
explainer = OllamaExplainer(settings.ollama_enabled, settings.ollama_base_url, settings.ollama_model)

app = FastAPI(
    title="Caspian Guardian AI API",
    version="1.0.0",
    description="Live Sentinel-1 screening for potential oil-like anomalies in the Caspian Sea.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    token = request_id_context.set(new_request_id())
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id_context.get()
        return response
    finally:
        request_id_context.reset(token)


def _redis_ready() -> bool:
    try:
        return bool(redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1).ping())
    except Exception:
        return False


def readiness() -> dict:
    components = {
        "copernicus": auth.configured,
        "segmentation_model": inference.ready,
        "postgis": repo.ping(),
        "redis": _redis_ready(),
        "object_storage": storage.ready(),
    }
    return {"live_ready": all(components.values()), "components": components}


def require_live() -> None:
    state = readiness()
    if not state["live_ready"]:
        missing = [name for name, ready in state["components"].items() if not ready]
        raise HTTPException(status_code=503, detail={"message": "Live pipeline is not configured", "missing": missing})


@app.get("/api/health")
async def health() -> dict:
    state = readiness()
    return {
        "status": "ok" if state["live_ready"] else "configuration_required",
        "service": "caspian-guardian-api",
        "analysis_mode": settings.analysis_mode,
        "data_mode": "live",
        "credentials_configured": auth.configured,
        "model_configured": inference.ready,
        **state,
        "ollama_enabled": settings.ollama_enabled,
        "model_version": inference.model_version,
    }


@app.post("/api/areas", response_model=AreaResponse)
async def create_area(payload: AreaCreate):
    require_live()
    return repo.add_area(payload.name, payload.bbox, payload.interval_minutes)


@app.get("/api/areas", response_model=list[AreaResponse])
async def list_areas():
    if not repo.ping():
        raise HTTPException(status_code=503, detail="PostGIS is unavailable")
    return repo.list_areas()


async def search_scene(payload: SceneSearchRequest) -> dict:
    require_live()
    box = BoundingBox.from_list(payload.bbox)
    if (box.east - box.west) * (box.north - box.south) > settings.max_bbox_area_degrees:
        raise ValueError("requested area is too large for one analysis")
    return repo.add_scene(await catalog.latest(payload.bbox, payload.days_back))


@app.post("/api/scenes/search", response_model=SceneResponse)
async def scenes_search(payload: SceneSearchRequest):
    try:
        return await search_scene(payload)
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/scenes/latest", response_model=SceneResponse)
async def latest_scene():
    require_live()
    scene = repo.latest_scene()
    if not scene:
        raise HTTPException(status_code=404, detail="No live scene has been discovered yet")
    return scene


@app.get("/api/scenes/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: UUID):
    scene = repo.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene


def _asset_url(object_name: str) -> str:
    return f"/api/assets/{object_name}"


async def run_analysis(job_id: UUID) -> None:
    job = repo.get_job(job_id)
    if not job:
        raise LookupError("Analysis job not found")
    try:
        repo.update_job(
            job_id, status="running", progress=10, stage="DOWNLOADING", started_at=datetime.now(timezone.utc)
        )
        scene = repo.get_scene(job["scene_id"])
        if not scene:
            raise LookupError("Satellite scene not found")
        repo.update_job(job_id, progress=40, stage="PROCESSING")
        detection = await real_analysis.run(scene, job["bbox"], job["resolution"])
        if not settings.min_detection_area_km2 <= detection["area_km2"] <= settings.max_detection_area_km2:
            raise LookupError("Candidate rejected by configured area limits")
        repo.update_job(job_id, progress=72, stage="AI_ANALYSIS", model_version=inference.model_version)
        raster_key = f"scenes/{scene['external_scene_id']}.tif"
        image_key = f"previews/{scene['external_scene_id']}.png"
        mask_key = f"masks/{scene['external_scene_id']}-{inference.model_version.split(':')[-1]}.png"
        storage.upload(Path(detection.pop("raster_path")), raster_key, "image/tiff")
        storage.upload(Path(detection["image_url"]), image_key, "image/png")
        storage.upload(Path(detection["mask_url"]), mask_key, "image/png")
        detection["image_url"], detection["mask_url"] = _asset_url(image_key), _asset_url(mask_key)
        detection["explanation"] = await explainer.explain(detection)
        risk = calculate_risk(
            detection["mean_confidence"], detection["max_confidence"], detection["area_km2"], settings
        )
        detection.update(
            id=uuid4(),
            scene_id=scene["id"],
            risk_level=risk.level,
            acquisition_time=scene["acquisition_time"],
            satellite=scene["satellite"],
        )
        saved = repo.add_detection(detection)
        repo.update_job(
            job_id,
            status="review",
            progress=100,
            stage="REVIEW",
            detection_id=saved["id"],
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.exception("Analysis job failed")
        repo.update_job(
            job_id, status="failed", stage="FAILED", error_message=str(exc), completed_at=datetime.now(timezone.utc)
        )
        raise


def enqueue(job_id: UUID) -> None:
    try:
        celery_app.send_task("guardian.run_analysis", args=[str(job_id)])
    except Exception as exc:
        repo.update_job(
            job_id, status="failed", stage="QUEUE_UNAVAILABLE", error_message="Redis/Celery queue is unavailable"
        )
        raise RuntimeError("Redis/Celery queue is unavailable") from exc


@app.post("/api/analysis", response_model=AnalysisQueued)
async def start_analysis(payload: AnalysisRequest):
    scene = await search_scene(payload)
    job = repo.add_job(scene["id"], payload.analysis_mode.value, payload.bbox, payload.resolution)
    if job["created"]:
        try:
            enqueue(job["job_id"])
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job["job_id"], "status": job["status"]}


@app.get("/api/analysis/{job_id}", response_model=AnalysisStatus)
async def analysis_status(job_id: UUID):
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job


@app.get("/api/detections", response_model=list[DetectionResponse])
async def list_detections():
    if not repo.ping():
        raise HTTPException(status_code=503, detail="PostGIS is unavailable")
    return repo.list_detections()


@app.get("/api/detections/{detection_id}", response_model=DetectionResponse)
async def get_detection(detection_id: UUID):
    detection = repo.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection


@app.post("/api/detections/{detection_id}/review", response_model=DetectionReviewResponse)
async def review_detection(detection_id: UUID, payload: DetectionReviewRequest):
    try:
        review = repo.add_review(detection_id, payload.action.value, payload.actor, payload.note)
        return review
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/detections/{detection_id}/reviews", response_model=list[DetectionReviewResponse])
async def review_history(detection_id: UUID):
    return repo.list_reviews(detection_id)


@app.get("/api/detections.geojson")
async def detections_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": str(item["id"]),
                "properties": {
                    "risk_level": item["risk_level"],
                    "area_km2": item["area_km2"],
                    "confidence": item["mean_confidence"],
                },
                "geometry": item["geometry"]["features"][0]["geometry"],
            }
            for item in repo.list_detections()
        ],
    }


@app.get("/api/assets/{object_name:path}")
async def asset(object_name: str):
    if ".." in object_name:
        raise HTTPException(status_code=400, detail="Invalid object name")
    destination = Path(gettempdir()) / "caspian-guardian" / object_name
    try:
        storage.download(object_name, destination)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    media = "image/tiff" if destination.suffix == ".tif" else "image/png"
    return FileResponse(destination, media_type=media)


@app.post("/api/reports/{detection_id}", response_model=ReportResponse)
async def generate_report(detection_id: UUID, payload: ReportRequest):
    detection = repo.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    title, content = build_report_content(detection, payload.language)
    pdf_path = create_pdf(detection, title, content, settings.storage_path)
    report = {
        "detection_id": detection_id,
        "language": payload.language,
        "title": title,
        "content": content,
        "download_url": f"/api/reports/{detection_id}/download",
        "created_at": utc_now(),
        "pdf_path": str(pdf_path),
    }
    repo.save_report(report)
    return report


@app.get("/api/reports/{detection_id}/download")
async def download_report(detection_id: UUID):
    report = repo.latest_report(detection_id)
    if not report:
        raise HTTPException(status_code=404, detail="Generate the report before downloading")
    path = Path(report["pdf_path"]).resolve()
    if Path(settings.storage_path).resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid report path")
    return FileResponse(path, media_type="application/pdf", filename=f"caspian-guardian-{detection_id}.pdf")


async def monitor_due_areas() -> int:
    processed = 0
    for subscription in repo.due_subscriptions():
        scene = repo.add_scene(await catalog.latest(subscription["bbox"], settings.max_days_back))
        if scene["external_scene_id"] == subscription["last_scene_external_id"]:
            continue
        job = repo.add_job(scene["id"], settings.analysis_mode, subscription["bbox"], 10)
        if job["created"]:
            enqueue(job["job_id"])
        repo.mark_subscription_scene(subscription["subscription_id"], scene["external_scene_id"])
        processed += 1
    return processed
