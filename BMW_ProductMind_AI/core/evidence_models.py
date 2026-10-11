"""Evidence-first tables (ProductMind AI 2.0).

AI IDENTIFIES EVIDENCE - PYTHON CALCULATES SCORES - HUMANS DECIDE.

* observation_versions          one row per extraction configuration
                                (model, model version, prompt, taxonomy)
* ai_feedback_observations      validated AI observations per G60 comment
                                (separate from original VFC classifications)
* evidence_selection_runs       requirement-specific evidence selection runs
* requirement_evidence_matches  observation -> requirement matches (immutable per run)
* evidence_match_reviews        human decisions; they override AI matches in
                                every later selection run (never replaced)
* scoring_method_versions       registry of deterministic scoring methods
* agent_documents / agent_document_chunks   uploaded agent resources

All on ``AIBase``: created additively by ``ensure_ai_schema`` and never
dropped by ingestion rebuilds. IDs are generated in Python, never by an LLM.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from core.ai_models import AIBase

OBSERVATION_TYPES = ("COMPLAINT", "IMPROVEMENT_REQUEST", "POSITIVE", "NEUTRAL", "UNCLEAR")
MATCH_ROLES = ("COMPLAINT", "IMPROVEMENT_REQUEST", "POSITIVE", "UNCERTAIN")
REVIEW_DECISIONS = ("ACCEPTED", "REJECTED", "UNCERTAIN")


class ObservationVersion(AIBase):
    __tablename__ = "observation_versions"

    version_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AIFeedbackObservation(AIBase):
    __tablename__ = "ai_feedback_observations"
    __table_args__ = (UniqueConstraint("extraction_cache_key", "observation_index"),)

    observation_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    extraction_cache_key: Mapped[str] = mapped_column(ForeignKey("ai_feedback_extractions.cache_key"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("observation_versions.version_id"), nullable=False, index=True)
    feedback_record_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    observation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    customer_need: Mapped[str] = mapped_column(Text, nullable=False)
    vehicle_system: Mapped[str] = mapped_column(String(160), nullable=False)
    observation_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    evidence_excerpt: Mapped[str] = mapped_column(Text, nullable=False)  # verified verbatim substring
    confidence: Mapped[str] = mapped_column(String(8), nullable=False)
    taxonomy_category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EvidenceSelectionRun(AIBase):
    __tablename__ = "evidence_selection_runs"

    selection_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    requirement_version: Mapped[int] = mapped_column(Integer, nullable=False)
    observation_version_id: Mapped[str] = mapped_column(String(40), nullable=False)
    selector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)  # TAXONOMY | TAXONOMY+LLM_SEMANTIC
    based_on_selection_id: Mapped[str | None] = mapped_column(String(40))
    coverage_json: Mapped[str] = mapped_column(Text, nullable=False)  # extracted vs. direct comments
    counts_json: Mapped[str] = mapped_column(Text, nullable=False)  # AI view at creation (reviews applied at read time)
    llm_job_id: Mapped[str | None] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RequirementEvidenceMatch(AIBase):
    __tablename__ = "requirement_evidence_matches"
    __table_args__ = (UniqueConstraint("selection_id", "observation_id"),)

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    selection_id: Mapped[str] = mapped_column(ForeignKey("evidence_selection_runs.selection_id"), nullable=False, index=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    observation_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False)
    observation_type: Mapped[str] = mapped_column(String(24), nullable=False)
    match_role: Mapped[str] = mapped_column(String(24), nullable=False)
    counter_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    match_method: Mapped[str] = mapped_column(String(40), nullable=False)
    ai_status: Mapped[str] = mapped_column(String(24), nullable=False)  # PROPOSED | UNCERTAIN | NOT_RELEVANT
    match_reason: Mapped[str] = mapped_column(Text, nullable=False)


class EvidenceMatchReview(AIBase):
    __tablename__ = "evidence_match_reviews"

    review_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    observation_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    selection_id: Mapped[str | None] = mapped_column(String(40))
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ScoringMethodVersion(AIBase):
    __tablename__ = "scoring_method_versions"

    method_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[str] = mapped_column(String(16), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    execution_methods: Mapped[str] = mapped_column(String(128), nullable=False)
    formula: Mapped[str] = mapped_column(Text, nullable=False)
    normalization_json: Mapped[str] = mapped_column(Text, nullable=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # DOCUMENTED_CUSTOM_METRIC entries ("CUSTOM:<id>") are usable only once an administrator approves them.
    approved_by: Mapped[str | None] = mapped_column(String(128))
    approval_note: Mapped[str | None] = mapped_column(Text)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)


class AgentDocument(AIBase):
    __tablename__ = "agent_documents"
    __table_args__ = (UniqueConstraint("agent_id", "sha256"),)

    document_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(8), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    uploaded_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentDocumentChunk(AIBase):
    __tablename__ = "agent_document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("agent_documents.document_id"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    location: Mapped[str] = mapped_column(String(128), nullable=False)  # page / sheet+row / line range
    text: Mapped[str] = mapped_column(Text, nullable=False)


class AgentEvidenceAnalysis(AIBase):
    """Step 2: one OpenAI-assisted evidence analysis for (agent content,
    requirement version, evidence version, model, prompt). Rows are never
    updated after completion; a changed instruction/resource/evidence input
    produces a new cache key and a new row (version history)."""

    __tablename__ = "agent_evidence_analyses"

    analysis_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    agent_config_version: Mapped[int] = mapped_column(Integer, nullable=False)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    requirement_version: Mapped[int] = mapped_column(Integer, nullable=False)
    base_selection_id: Mapped[str | None] = mapped_column(String(40))
    observation_version_id: Mapped[str] = mapped_column(String(40), nullable=False)
    scoring_method: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    fingerprint_json: Mapped[str] = mapped_column(Text, nullable=False)  # the inputs the cache key covers
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # completed | failed
    result_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[str | None] = mapped_column(String(64))
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
