"""Vehicle-requirement tables (Part 4).

Registered on ``AIBase``: created/migrated additively by
``core.database.ensure_ai_schema`` and never dropped by ingestion rebuilds.
Requirement IDs are generated once (UUID-based) and stored, so they survive
application restarts, cluster rebuilds and ingestion rebuilds.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from core.ai_models import AIBase

GENERATION_METHODS = ("RULE_BASED", "LLM_ASSISTED", "HUMAN_AUTHORED")
REVIEW_STATUSES = ("DRAFT", "PENDING_REVIEW", "APPROVED", "REJECTED")


class VehicleRequirement(AIBase):
    __tablename__ = "vehicle_requirements"

    requirement_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_cluster_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_insight_id: Mapped[str | None] = mapped_column(String(80))
    need_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    customer_need: Mapped[str] = mapped_column(Text, nullable=False)
    customer_problem: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_requirement: Mapped[str] = mapped_column(Text, nullable=False)
    requirement_type: Mapped[str] = mapped_column(String(32), nullable=False)
    vehicle_model: Mapped[str] = mapped_column(String(64), nullable=False)
    source_vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    target_generation: Mapped[str] = mapped_column(String(128), nullable=False)
    supporting_evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False)  # {"direct": [...], ...}
    evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    assumptions_json: Mapped[str] = mapped_column(Text, nullable=False)
    acceptance_criteria_json: Mapped[str] = mapped_column(Text, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(32), nullable=False)
    template_id: Mapped[str | None] = mapped_column(String(64))
    requires_refinement: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    refinement_notes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    validation_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RequirementEvidenceLink(AIBase):
    """Evidence snapshot at generation time. ``evidence_scope`` makes explicit
    whether an item is a cluster member (direct) or linked only at the
    need-category level (never proof of an individual complaint)."""

    __tablename__ = "requirement_evidence_links"
    __table_args__ = (UniqueConstraint("requirement_id", "tier", "source_table", "source_key"),)

    link_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requirement_id: Mapped[str] = mapped_column(ForeignKey("vehicle_requirements.requirement_id"), nullable=False, index=True)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)  # DIRECT | SUPPORTING | CONTEXTUAL
    source_table: Mapped[str] = mapped_column(String(64), nullable=False)
    source_key: Mapped[str] = mapped_column(Text, nullable=False)  # feedback ID, P001, COMP-R5, study::attr, equipment code
    source_record_id: Mapped[str | None] = mapped_column(String(64))
    evidence_scope: Mapped[str] = mapped_column(String(32), nullable=False)  # cluster_member | need_category | vehicle_catalogue
    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(Text)


class RequirementVersion(AIBase):
    """Immutable snapshots. Version 1 is always the original generated draft."""

    __tablename__ = "requirement_versions"
    __table_args__ = (UniqueConstraint("requirement_id", "version_number"),)

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requirement_id: Mapped[str] = mapped_column(ForeignKey("vehicle_requirements.requirement_id"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    change_type: Mapped[str] = mapped_column(String(32), nullable=False)  # generated | edited
    generation_method: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    changed_fields_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    changed_by: Mapped[str] = mapped_column(String(128), nullable=False)
    change_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RequirementReviewEvent(AIBase):
    __tablename__ = "requirement_review_history"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requirement_id: Mapped[str] = mapped_column(ForeignKey("vehicle_requirements.requirement_id"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)  # create | edit | submit | approve | reject | reopen | comment
    previous_status: Mapped[str | None] = mapped_column(String(32))
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
