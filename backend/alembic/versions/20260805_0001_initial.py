"""Create the Caspian Guardian PostGIS schema.

Revision ID: 20260805_0001
"""

from collections.abc import Sequence

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260805_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.create_table(
        "areas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("geometry", geoalchemy2.Geometry("POLYGON", srid=4326), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "satellite_scenes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("external_scene_id", sa.String(length=255), nullable=False, unique=True),
        sa.Column("satellite", sa.String(length=80), nullable=False),
        sa.Column("acquisition_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("orbit_direction", sa.String(length=20), nullable=False),
        sa.Column("polarization", postgresql.JSONB(), nullable=False),
        sa.Column("bbox", postgresql.JSONB(), nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("preview_path", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "analysis_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("satellite_scenes.id"), nullable=False),
        sa.Column("analysis_mode", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(length=80), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "detections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("satellite_scenes.id"), nullable=False),
        sa.Column("geometry", geoalchemy2.Geometry("MULTIPOLYGON", srid=4326), nullable=False),
        sa.Column("area_km2", sa.Float(), nullable=False),
        sa.Column("mean_confidence", sa.Float(), nullable=False),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("anomaly_type", sa.String(length=80), nullable=False),
        sa.Column("verification_status", sa.String(length=80), nullable=False),
        sa.Column("acquisition_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("detection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("detections.id"), nullable=False),
        sa.Column("language", sa.String(length=5), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("pdf_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_table("detections")
    op.drop_table("analysis_jobs")
    op.drop_table("satellite_scenes")
    op.drop_table("areas")
