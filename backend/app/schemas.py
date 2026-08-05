from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AnalysisMode(str, Enum):
    OIL_SPILL = "oil_spill"
    WATER_QUALITY = "water_quality"
    MARINE_DEBRIS = "marine_debris"
    ILLEGAL_FISHING = "illegal_fishing"
    SHORELINE_POLLUTION = "shoreline_pollution"


class BoundingBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def ordered(self) -> "BoundingBox":
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("bbox must be ordered west < east and south < north")
        return self

    @classmethod
    def from_list(cls, values: list[float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError("bbox must contain west, south, east, north")
        return cls(west=values[0], south=values[1], east=values[2], north=values[3])

    @field_validator("west", "east")
    @classmethod
    def longitude_range(cls, value: float) -> float:
        if not -180 <= value <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return value

    @field_validator("south", "north")
    @classmethod
    def latitude_range(cls, value: float) -> float:
        if not -90 <= value <= 90:
            raise ValueError("latitude must be between -90 and 90")
        return value

    @property
    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


class AreaCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    bbox: list[float]
    interval_minutes: int = Field(default=60, ge=60, le=180)

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value: list[float]) -> list[float]:
        BoundingBox.from_list(value)
        return value


class AreaResponse(AreaCreate):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime


class SceneSearchRequest(BaseModel):
    bbox: list[float]
    days_back: int = Field(default=14, ge=1, le=30)

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value: list[float]) -> list[float]:
        BoundingBox.from_list(value)
        return value


class AnalysisRequest(SceneSearchRequest):
    resolution: int = Field(default=10, ge=10, le=100)
    analysis_mode: AnalysisMode = AnalysisMode.OIL_SPILL


class SceneResponse(BaseModel):
    id: UUID
    external_scene_id: str
    satellite: str
    acquisition_time: datetime
    orbit_direction: str
    polarization: list[str]
    bbox: list[float]
    status: str
    source_metadata: dict[str, Any]
    preview_url: str | None = None


class AnalysisQueued(BaseModel):
    job_id: UUID
    status: str


class AnalysisStatus(BaseModel):
    job_id: UUID
    status: str
    progress: int
    stage: str
    error_message: str | None = None
    detection_id: UUID | None = None


class DetectionResponse(BaseModel):
    id: UUID
    scene_id: UUID
    geometry: dict[str, Any]
    area_km2: float
    mean_confidence: float
    max_confidence: float
    risk_level: RiskLevel
    anomaly_type: str
    verification_status: str
    coordinates: list[float]
    model_version: str
    acquisition_time: datetime
    satellite: str
    image_url: str
    mask_url: str
    warning: str
    explanation: str | None = None


class ReviewAction(str, Enum):
    CONFIRM = "CONFIRM"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ESCALATE = "ESCALATE"


class DetectionReviewRequest(BaseModel):
    action: ReviewAction
    actor: str = Field(min_length=2, max_length=120)
    note: str | None = Field(default=None, max_length=2000)


class DetectionReviewResponse(BaseModel):
    id: UUID
    detection_id: UUID
    action: ReviewAction
    actor: str
    note: str | None
    created_at: datetime


class LabelReviewStatus(str, Enum):
    REVIEWED_POSITIVE = "reviewed_positive"
    REVIEWED_NEGATIVE = "reviewed_negative"


class LabelSampleResponse(BaseModel):
    sample_id: str
    scene_id: str
    acquisition_time: datetime
    region: str
    region_name: str
    bbox: list[float]
    review_status: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    polygon_count: int
    polygons: list[list[list[float]]]
    preview_url: str


class LabelReviewRequest(BaseModel):
    status: LabelReviewStatus
    reviewed_by: str = Field(min_length=2, max_length=120)
    note: str | None = Field(default=None, max_length=2000)
    polygons: list[list[list[float]]] = Field(default_factory=list, max_length=100)

    @field_validator("reviewed_by")
    @classmethod
    def named_reviewer(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("reviewer name is required")
        return value

    @model_validator(mode="after")
    def consistent_review(self) -> "LabelReviewRequest":
        if self.status == LabelReviewStatus.REVIEWED_POSITIVE and not self.polygons:
            raise ValueError("positive review requires at least one polygon")
        if self.status == LabelReviewStatus.REVIEWED_NEGATIVE and self.polygons:
            raise ValueError("negative review cannot contain polygons")
        for polygon in self.polygons:
            if len(polygon) < 3:
                raise ValueError("each polygon requires at least three points")
            for point in polygon:
                if len(point) != 2:
                    raise ValueError("polygon points must contain longitude and latitude")
        return self


class ReportRequest(BaseModel):
    language: str = Field(default="ru", pattern="^(ru|kk|en)$")


class ReportResponse(BaseModel):
    detection_id: UUID
    language: str
    title: str
    content: str
    download_url: str
    created_at: datetime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
