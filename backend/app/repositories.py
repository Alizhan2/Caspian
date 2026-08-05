from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import MultiPolygon, box, mapping, shape
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models import AnalysisJob, Area, AreaSubscription, Detection, DetectionReview, Report, SatelliteScene


def _database_url(url: str) -> str:
    return url.replace("postgresql://", "postgresql+psycopg://", 1) if url.startswith("postgresql://") else url


class PostgresRepository:
    """Durable repository. No in-memory production fallback is permitted."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(_database_url(database_url), pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def add_area(self, name: str, bbox: list[float], interval_minutes: int = 60) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self.sessions.begin() as session:
            area = Area(id=uuid4(), name=name, geometry=from_shape(box(*bbox), srid=4326), created_at=now)
            session.add(area)
            session.flush()
            session.add(
                AreaSubscription(
                    id=uuid4(), area_id=area.id, enabled=True, interval_minutes=interval_minutes, next_check_at=now
                )
            )
            return {
                "id": area.id,
                "name": area.name,
                "bbox": bbox,
                "interval_minutes": interval_minutes,
                "created_at": area.created_at,
            }

    def list_areas(self) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.execute(
                select(Area, AreaSubscription).join(AreaSubscription).order_by(Area.created_at)
            ).all()
            return [
                {
                    "id": area.id,
                    "name": area.name,
                    "bbox": list(to_shape(area.geometry).bounds),
                    "interval_minutes": subscription.interval_minutes,
                    "created_at": area.created_at,
                }
                for area, subscription in rows
            ]

    def add_scene(self, item: dict[str, Any]) -> dict[str, Any]:
        acquisition = item["acquisition_time"]
        if isinstance(acquisition, str):
            acquisition = datetime.fromisoformat(acquisition.replace("Z", "+00:00"))
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(SatelliteScene).where(SatelliteScene.external_scene_id == item["external_scene_id"])
            )
            if existing:
                return self._scene(existing)
            row = SatelliteScene(
                id=item["id"],
                external_scene_id=item["external_scene_id"],
                satellite=item["satellite"],
                acquisition_time=acquisition,
                orbit_direction=item["orbit_direction"],
                polarization=item["polarization"],
                bbox=item["bbox"],
                source_metadata=item["source_metadata"],
                status=item["status"],
                preview_path=item.get("preview_url"),
            )
            session.add(row)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                with self.sessions() as retry:
                    return self._scene(
                        retry.scalar(
                            select(SatelliteScene).where(SatelliteScene.external_scene_id == item["external_scene_id"])
                        )
                    )
            return self._scene(row)

    def latest_scene(self) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.scalar(select(SatelliteScene).order_by(SatelliteScene.acquisition_time.desc()).limit(1))
            return self._scene(row) if row else None

    def get_scene(self, scene_id: UUID) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(SatelliteScene, scene_id)
            return self._scene(row) if row else None

    @staticmethod
    def _scene(row: SatelliteScene) -> dict[str, Any]:
        return {
            "id": row.id,
            "external_scene_id": row.external_scene_id,
            "satellite": row.satellite,
            "acquisition_time": row.acquisition_time,
            "orbit_direction": row.orbit_direction,
            "polarization": row.polarization,
            "bbox": row.bbox,
            "status": row.status,
            "source_metadata": row.source_metadata,
            "preview_url": row.preview_path,
        }

    def add_job(self, scene_id: UUID, mode: str, bbox: list[float], resolution: int) -> dict[str, Any]:
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(AnalysisJob)
                .where(
                    AnalysisJob.scene_id == scene_id,
                    AnalysisJob.analysis_mode == mode,
                    AnalysisJob.status != "failed",
                )
                .order_by(AnalysisJob.started_at.desc().nullslast())
                .limit(1)
            )
            if existing:
                return {**self._job(existing), "created": False}
            row = AnalysisJob(
                id=uuid4(),
                scene_id=scene_id,
                analysis_mode=mode,
                status="queued",
                progress=0,
                stage="DISCOVERED",
                bbox=bbox,
                resolution=resolution,
            )
            session.add(row)
            session.flush()
            return {**self._job(row), "created": True}

    def get_job(self, job_id: UUID) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(AnalysisJob, job_id)
            return self._job(row) if row else None

    def update_job(self, job_id: UUID, **values: Any) -> dict[str, Any]:
        with self.sessions.begin() as session:
            row = session.get(AnalysisJob, job_id)
            if not row:
                raise LookupError("Analysis job not found")
            for key, value in values.items():
                setattr(row, key, value)
            session.flush()
            return self._job(row)

    @staticmethod
    def _job(row: AnalysisJob) -> dict[str, Any]:
        return {
            "job_id": row.id,
            "scene_id": row.scene_id,
            "status": row.status,
            "progress": row.progress,
            "stage": row.stage,
            "detection_id": row.detection_id,
            "error_message": row.error_message,
            "mode": row.analysis_mode,
            "bbox": row.bbox,
            "resolution": row.resolution,
        }

    def add_detection(self, item: dict[str, Any]) -> dict[str, Any]:
        polygons = [shape(feature["geometry"]) for feature in item["geometry"].get("features", [])]
        multi = MultiPolygon([polygon for polygon in polygons if polygon.geom_type == "Polygon"])
        if multi.is_empty:
            raise ValueError("Detection geometry is empty")
        with self.sessions.begin() as session:
            row = Detection(
                id=item["id"],
                scene_id=item["scene_id"],
                geometry=from_shape(multi, srid=4326),
                area_km2=item["area_km2"],
                mean_confidence=item["mean_confidence"],
                max_confidence=item["max_confidence"],
                risk_level=str(getattr(item["risk_level"], "value", item["risk_level"])),
                anomaly_type=item["anomaly_type"],
                verification_status=item["verification_status"],
                acquisition_time=item["acquisition_time"],
                model_version=item["model_version"],
                coordinates=item["coordinates"],
                image_path=item.get("image_url"),
                mask_path=item.get("mask_url"),
                warning=item["warning"],
                explanation=item.get("explanation"),
                evidence_context=item.get("evidence_context", {}),
            )
            session.add(row)
            session.flush()
            return self._detection(row)

    def list_detections(self) -> list[dict[str, Any]]:
        with self.sessions() as session:
            return [
                self._detection(row) for row in session.scalars(select(Detection).order_by(Detection.created_at)).all()
            ]

    def get_detection(self, detection_id: UUID) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(Detection, detection_id)
            return self._detection(row) if row else None

    @staticmethod
    def _detection(row: Detection) -> dict[str, Any]:
        geometry = mapping(to_shape(row.geometry))
        features = [
            {
                "type": "Feature",
                "properties": {"source": "sentinel-1"},
                "geometry": {"type": "Polygon", "coordinates": coordinates},
            }
            for coordinates in geometry["coordinates"]
        ]
        return {
            "id": row.id,
            "scene_id": row.scene_id,
            "geometry": {"type": "FeatureCollection", "features": features},
            "area_km2": row.area_km2,
            "mean_confidence": row.mean_confidence,
            "max_confidence": row.max_confidence,
            "risk_level": row.risk_level,
            "anomaly_type": row.anomaly_type,
            "verification_status": row.verification_status,
            "coordinates": row.coordinates,
            "model_version": row.model_version,
            "acquisition_time": row.acquisition_time,
            "satellite": "Sentinel-1",
            "image_url": row.image_path or "",
            "mask_url": row.mask_path or "",
            "warning": row.warning,
            "explanation": row.explanation,
            "evidence_context": row.evidence_context or {},
        }

    def add_review(self, detection_id: UUID, action: str, actor: str, note: str | None) -> dict[str, Any]:
        status = {"CONFIRM": "operator_confirmed", "FALSE_POSITIVE": "false_positive", "ESCALATE": "escalated"}[action]
        with self.sessions.begin() as session:
            detection = session.get(Detection, detection_id)
            if not detection:
                raise LookupError("Detection not found")
            detection.verification_status = status
            review = DetectionReview(id=uuid4(), detection_id=detection_id, action=action, actor=actor, note=note)
            session.add(review)
            session.flush()
            return {
                "id": review.id,
                "detection_id": detection_id,
                "action": action,
                "actor": actor,
                "note": note,
                "created_at": review.created_at,
            }

    def list_reviews(self, detection_id: UUID) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(DetectionReview)
                .where(DetectionReview.detection_id == detection_id)
                .order_by(DetectionReview.created_at)
            ).all()
            return [
                {
                    "id": row.id,
                    "detection_id": row.detection_id,
                    "action": row.action,
                    "actor": row.actor,
                    "note": row.note,
                    "created_at": row.created_at,
                }
                for row in rows
            ]

    def save_report(self, item: dict[str, Any]) -> None:
        with self.sessions.begin() as session:
            session.add(
                Report(
                    id=uuid4(),
                    detection_id=item["detection_id"],
                    language=item["language"],
                    title=item["title"],
                    content=item["content"],
                    pdf_path=item["pdf_path"],
                )
            )

    def latest_report(self, detection_id: UUID) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.scalar(
                select(Report).where(Report.detection_id == detection_id).order_by(Report.created_at.desc()).limit(1)
            )
            return {"pdf_path": row.pdf_path} if row else None

    def due_subscriptions(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self.sessions.begin() as session:
            rows = session.execute(
                select(AreaSubscription, Area)
                .join(Area)
                .where(AreaSubscription.enabled.is_(True), AreaSubscription.next_check_at <= now)
            ).all()
            result = []
            for sub, area in rows:
                bbox = list(to_shape(area.geometry).bounds)
                result.append(
                    {
                        "subscription_id": sub.id,
                        "area_id": area.id,
                        "bbox": bbox,
                        "last_scene_external_id": sub.last_scene_external_id,
                    }
                )
                sub.last_checked_at = now
                sub.next_check_at = now + timedelta(minutes=sub.interval_minutes)
            return result

    def mark_subscription_scene(self, subscription_id: UUID, external_scene_id: str) -> None:
        with self.sessions.begin() as session:
            row = session.get(AreaSubscription, subscription_id)
            if row:
                row.last_scene_external_id = external_scene_id
