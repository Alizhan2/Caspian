"""SQLAlchemy/PostGIS models used by the live production pipeline."""

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Area(Base):
    __tablename__ = "areas"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120))
    geometry: Mapped[object] = mapped_column(Geometry("POLYGON", srid=4326))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AreaSubscription(Base):
    __tablename__ = "area_subscriptions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    area_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("areas.id"), unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    interval_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_scene_external_id: Mapped[str | None] = mapped_column(String(255))
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_check_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SatelliteScene(Base):
    __tablename__ = "satellite_scenes"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_scene_id: Mapped[str] = mapped_column(String(255), unique=True)
    satellite: Mapped[str] = mapped_column(String(80))
    acquisition_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    orbit_direction: Mapped[str] = mapped_column(String(20))
    polarization: Mapped[list[str]] = mapped_column(JSONB)
    bbox: Mapped[list[float]] = mapped_column(JSONB)
    source_metadata: Mapped[dict] = mapped_column(JSONB)
    image_path: Mapped[str | None] = mapped_column(Text())
    preview_path: Mapped[str | None] = mapped_column(Text())
    status: Mapped[str] = mapped_column(String(40), default="available")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("satellite_scenes.id"))
    analysis_mode: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(40), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(40), default="DISCOVERED")
    bbox: Mapped[list[float]] = mapped_column(JSONB)
    resolution: Mapped[int] = mapped_column(Integer, default=10)
    detection_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("detections.id", use_alter=True)
    )
    error_message: Mapped[str | None] = mapped_column(Text())
    model_version: Mapped[str | None] = mapped_column(String(80))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("satellite_scenes.id"))
    geometry: Mapped[object] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))
    area_km2: Mapped[float] = mapped_column(Float)
    mean_confidence: Mapped[float] = mapped_column(Float)
    max_confidence: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(20))
    anomaly_type: Mapped[str] = mapped_column(String(80))
    verification_status: Mapped[str] = mapped_column(String(80), default="requires_field_verification")
    acquisition_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str] = mapped_column(String(120))
    coordinates: Mapped[list[float]] = mapped_column(JSONB)
    image_path: Mapped[str | None] = mapped_column(Text())
    mask_path: Mapped[str | None] = mapped_column(Text())
    warning: Mapped[str] = mapped_column(Text())
    explanation: Mapped[str | None] = mapped_column(Text())
    evidence_context: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detection_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("detections.id"))
    language: Mapped[str] = mapped_column(String(5))
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text())
    pdf_path: Mapped[str | None] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DetectionReview(Base):
    __tablename__ = "detection_reviews"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detection_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("detections.id"))
    action: Mapped[str] = mapped_column(String(40))
    actor: Mapped[str] = mapped_column(String(120))
    note: Mapped[str | None] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NotificationEndpoint(Base):
    __tablename__ = "notification_endpoints"
    __table_args__ = (UniqueConstraint("organization", "url", name="uq_notification_endpoint"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(20), default="webhook")
    url: Mapped[str] = mapped_column(Text())
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
