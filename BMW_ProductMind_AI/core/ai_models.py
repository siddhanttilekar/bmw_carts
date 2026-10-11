"""Tables for the AI customer-intelligence layer (Part 2).

These live on a separate declarative base (``AIBase``) so that
``scripts.initialize_database`` - which drops and rebuilds the ingestion
tables - never deletes paid-for AI results. They reference ingestion records
by their deterministic ``record_id`` / original feedback ID strings instead of
database foreign keys, so ingestion rebuilds cannot cascade into them;
references are validated in code (``validate_references``) and in tests.

Schema changes are additive only (see ``core.database.ensure_ai_schema``).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class AIBase(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Curated taxonomy and mappings
# ---------------------------------------------------------------------------


class CustomerNeedCategory(AIBase):
    __tablename__ = "customer_need_taxonomy"

    need_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(16), nullable=False)


class VfcNeedMapping(AIBase):
    """Original BMW VFC level-2 name -> normalised need category. The VFC name
    is stored verbatim; source records are never modified."""

    __tablename__ = "vfc_need_mappings"

    vfc_level2_name: Mapped[str] = mapped_column(Text, primary_key=True)  # "<missing>" for NULL
    need_code: Mapped[str] = mapped_column(ForeignKey("customer_need_taxonomy.need_code"), nullable=False, index=True)
    mapping_status: Mapped[str] = mapped_column(String(32), nullable=False)  # direct | review_required
    rationale: Mapped[str | None] = mapped_column(Text)
    taxonomy_version: Mapped[str] = mapped_column(String(16), nullable=False)


# ---------------------------------------------------------------------------
# AI processing: jobs, API calls, cached extractions
# ---------------------------------------------------------------------------


class AIProcessingJob(AIBase):
    __tablename__ = "ai_processing_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)  # extraction | cluster_grouping | insights
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # running | completed | interrupted | failed
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    selection_json: Mapped[str] = mapped_column(Text, nullable=False)  # ordered original feedback IDs
    selection_description: Mapped[str | None] = mapped_column(Text)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    total_items: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    api_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AIApiCall(AIBase):
    """One row per LLM request attempt (usage logging)."""

    __tablename__ = "ai_api_calls"

    call_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("ai_processing_jobs.job_id"), nullable=False, index=True)
    batch_index: Mapped[int | None] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    item_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # succeeded | invalid_output | refused | api_error | rate_limited
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(128))
    stop_reason: Mapped[str | None] = mapped_column(String(32))
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AIFeedbackExtraction(AIBase):
    """Cache entry: one per (feedback ID, text hash, model, prompt version)."""

    __tablename__ = "ai_feedback_extractions"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    feedback_record_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # succeeded | failed
    summary: Mapped[str | None] = mapped_column(Text)  # short factual summary of the comment
    need_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    observation_version_id: Mapped[str | None] = mapped_column(String(40), index=True)  # 2.0 observe-v1 rows only


class ExtractedCustomerNeed(AIBase):
    __tablename__ = "extracted_customer_needs"
    __table_args__ = (UniqueConstraint("cache_key", "need_index"),)

    need_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cache_key: Mapped[str] = mapped_column(ForeignKey("ai_feedback_extractions.cache_key"), nullable=False, index=True)
    feedback_record_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    need_index: Mapped[int] = mapped_column(Integer, nullable=False)
    problem: Mapped[str | None] = mapped_column(Text)
    customer_need: Mapped[str] = mapped_column(Text, nullable=False)
    need_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    polarity: Mapped[str] = mapped_column(String(16), nullable=False)
    vehicle_system: Mapped[str | None] = mapped_column(String(128))
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    supporting_quote: Mapped[str | None] = mapped_column(Text)
    quote_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)  # quote is a verbatim substring
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)


# ---------------------------------------------------------------------------
# Clusters, cross-source evidence, insights
# ---------------------------------------------------------------------------


class CustomerNeedCluster(AIBase):
    __tablename__ = "customer_need_clusters"

    cluster_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    cluster_level: Mapped[str] = mapped_column(String(32), nullable=False)  # need_category | vfc_category | extracted_need
    need_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vfc_level2_name: Mapped[str | None] = mapped_column(Text)
    parent_cluster_id: Mapped[str | None] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(Text, nullable=False)
    customer_need: Mapped[str | None] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    # Counts are always computed with SQL over cluster_feedback_links / source tables.
    unique_feedback_count: Mapped[int] = mapped_column(Integer, nullable=False)
    classification_count: Mapped[int] = mapped_column(Integer, nullable=False)
    extracted_need_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_note: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ClusterFeedbackLink(AIBase):
    __tablename__ = "cluster_feedback_links"
    __table_args__ = (UniqueConstraint("cluster_id", "feedback_record_id", "extracted_need_id"),)

    link_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[str] = mapped_column(ForeignKey("customer_need_clusters.cluster_id"), nullable=False, index=True)
    feedback_record_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False)
    link_basis: Mapped[str] = mapped_column(String(32), nullable=False)  # vfc_classification | extracted_need
    extracted_need_id: Mapped[str | None] = mapped_column(String(64))


class CrossSourceEvidenceLink(AIBase):
    __tablename__ = "cross_source_evidence_links"

    link_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    need_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    evidence_tier: Mapped[str] = mapped_column(String(16), nullable=False)  # direct | supporting | contextual
    source_table: Mapped[str] = mapped_column(String(64), nullable=False)
    source_key: Mapped[str] = mapped_column(Text, nullable=False)  # P003, competitor row, study::attribute, VFC name
    source_record_ids_json: Mapped[str | None] = mapped_column(Text)
    source_label: Mapped[str] = mapped_column(Text, nullable=False)
    data_origin: Mapped[str] = mapped_column(String(32), nullable=False)
    mapping_basis: Mapped[str] = mapped_column(String(64), nullable=False)
    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    mapping_version: Mapped[str] = mapped_column(String(16), nullable=False)


class GeneratedCustomerInsight(AIBase):
    __tablename__ = "generated_customer_insights"

    insight_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    cluster_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    need_code: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    customer_problem: Mapped[str] = mapped_column(Text, nullable=False)
    customer_need: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    unique_feedback_count: Mapped[int] = mapped_column(Integer, nullable=False)
    classification_count: Mapped[int] = mapped_column(Integer, nullable=False)
    markets_json: Mapped[str] = mapped_column(Text, nullable=False)
    internal_evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    public_evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    study_evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    limitations_json: Mapped[str] = mapped_column(Text, nullable=False)
    text_source: Mapped[str] = mapped_column(String(32), nullable=False)  # deterministic_template | llm
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    validated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    validation_errors_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
