"""Add live monitoring, review, and durable processing fields."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260805_0002"
down_revision: str | None = "20260805_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "area_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("area_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("areas.id"), nullable=False, unique=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("interval_minutes", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("last_scene_external_id", sa.String(255)),
        sa.Column("last_checked_at", sa.DateTime(timezone=True)),
        sa.Column("next_check_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.add_column("analysis_jobs", sa.Column("stage", sa.String(40), nullable=False, server_default="DISCOVERED"))
    op.add_column("analysis_jobs", sa.Column("bbox", postgresql.JSONB(), nullable=False, server_default="[]"))
    op.add_column("analysis_jobs", sa.Column("resolution", sa.Integer(), nullable=False, server_default="10"))
    op.add_column("analysis_jobs", sa.Column("detection_id", postgresql.UUID(as_uuid=True)))
    op.create_foreign_key("fk_analysis_detection", "analysis_jobs", "detections", ["detection_id"], ["id"], use_alter=True)
    op.add_column("detections", sa.Column("coordinates", postgresql.JSONB(), nullable=False, server_default="[]"))
    op.add_column("detections", sa.Column("image_path", sa.Text()))
    op.add_column("detections", sa.Column("mask_path", sa.Text()))
    op.add_column("detections", sa.Column("warning", sa.Text(), nullable=False, server_default="Experimental screening signal requiring field verification."))
    op.add_column("detections", sa.Column("explanation", sa.Text()))
    op.create_table(
        "detection_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("detection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("detections.id"), nullable=False),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("actor", sa.String(120), nullable=False),
        sa.Column("note", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "notification_endpoints",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization", sa.String(120), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False, server_default="webhook"),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("organization", "url", name="uq_notification_endpoint"),
    )


def downgrade() -> None:
    op.drop_table("notification_endpoints")
    op.drop_table("detection_reviews")
    op.drop_constraint("fk_analysis_detection", "analysis_jobs", type_="foreignkey")
    for column in ("explanation", "warning", "mask_path", "image_path", "coordinates"):
        op.drop_column("detections", column)
    for column in ("detection_id", "resolution", "bbox", "stage"):
        op.drop_column("analysis_jobs", column)
    op.drop_table("area_subscriptions")
