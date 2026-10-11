"""Agent-based prioritization tables (Part 5).

Registered on ``AIBase``: created additively by ``ensure_ai_schema`` and never
dropped by ingestion rebuilds. Agents and profiles are versioned; runs store a
full snapshot of the configuration they used (agent versions, weights, method
versions), so a historical run stays reproducible after users edit agents.
Weights are integer basis points (10000 = 100%).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from core.ai_models import AIBase

EXECUTION_METHODS = ("DATA_ANALYSIS", "HUMAN_ASSESSMENT", "AI_ANALYSIS")
BP_TOTAL = 10_000  # 100.00 %


class PrioritizationAgent(AIBase):
    __tablename__ = "prioritization_agents"

    agent_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    execution_method: Mapped[str] = mapped_column(String(32), nullable=False)
    scoring_method: Mapped[str] = mapped_column(String(64), nullable=False)
    default_weight_bp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # soft delete
    configuration_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    instructions: Mapped[str | None] = mapped_column(Text)  # "Custom Instructions" (scope evidence selection only)
    context: Mapped[str | None] = mapped_column(Text)  # 2.0: e.g. "Next-generation BMW 5 Series"
    custom_metric_id: Mapped[str | None] = mapped_column(String(64))  # 2.0: approved DOCUMENTED_CUSTOM_METRIC
    evaluation_questions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    limitations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentResourceAssignment(AIBase):
    """Current resource assignment (each version also snapshots its resources)."""

    __tablename__ = "agent_resource_assignments"
    __table_args__ = (UniqueConstraint("agent_id", "resource_id"),)

    assignment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("prioritization_agents.agent_id"), nullable=False, index=True)
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)


class AgentConfigVersion(AIBase):
    __tablename__ = "agent_config_versions"
    __table_args__ = (UniqueConstraint("agent_id", "version_number"),)

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("prioritization_agents.agent_id"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)  # full config incl. resources
    change_note: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PrioritizationProfile(AIBase):
    __tablename__ = "prioritization_profiles"

    profile_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    profile_name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # draft | active | inactive | archived
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_illustrative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ProfileAgentWeight(AIBase):
    """Weights per profile *version*; pins the agent configuration version."""

    __tablename__ = "profile_agent_weights"
    __table_args__ = (UniqueConstraint("profile_id", "profile_version", "agent_id"),)

    weight_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("prioritization_profiles.profile_id"), nullable=False, index=True)
    profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_id: Mapped[str] = mapped_column(ForeignKey("prioritization_agents.agent_id"), nullable=False)
    agent_version: Mapped[int] = mapped_column(Integer, nullable=False)
    weight_bp: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PriorityAssessment(AIBase):
    """Human assessment for a HUMAN_SCORE agent. Never generated automatically."""

    __tablename__ = "priority_assessments"

    assessment_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(ForeignKey("vehicle_requirements.requirement_id"), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("prioritization_agents.agent_id"), nullable=False, index=True)
    agent_version: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    assessor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_refs_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    assessment_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PrioritizationRun(AIBase):
    __tablename__ = "prioritization_runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("prioritization_profiles.profile_id"), nullable=False, index=True)
    profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    profile_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)  # agents, versions, weights, configs
    method_versions_json: Mapped[str] = mapped_column(Text, nullable=False)
    requirement_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    complete_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    incomplete_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # 2.0: frozen evidence per requirement (selection, validated IDs, extraction versions, normalization)
    evidence_snapshot_json: Mapped[str | None] = mapped_column(Text)
    based_on_run_id: Mapped[str | None] = mapped_column(String(40))  # "Reanalyze With New Evidence"


class AgentExecutionResult(AIBase):
    __tablename__ = "agent_execution_results"
    __table_args__ = (UniqueConstraint("run_id", "requirement_id", "agent_id"),)

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("prioritization_runs.run_id"), nullable=False, index=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(40), nullable=False)
    agent_version: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    scoring_method: Mapped[str] = mapped_column(String(64), nullable=False)
    scoring_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # SCORED | UNSCORED
    score: Mapped[float | None] = mapped_column(Float)
    weight_bp: Mapped[int] = mapped_column(Integer, nullable=False)
    weighted_contribution: Mapped[float | None] = mapped_column(Float)
    inputs_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_refs_json: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    limitations_json: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RequirementPriorityScore(AIBase):
    __tablename__ = "requirement_priority_scores"
    __table_args__ = (UniqueConstraint("run_id", "requirement_id"),)

    score_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("prioritization_runs.run_id"), nullable=False, index=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    requirement_title: Mapped[str] = mapped_column(Text, nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False)  # at run time
    requirement_version: Mapped[int | None] = mapped_column(Integer)  # added in Part 6 (NULL for older runs)
    priority_status: Mapped[str] = mapped_column(String(16), nullable=False)  # COMPLETE | INCOMPLETE
    final_score: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)
    missing_agents_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    critical_flags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class CriticalIssueFlag(AIBase):
    """Safety / regulatory / severe reliability flags. Raised and confirmed by
    humans only; never inferred from broad VFC labels."""

    __tablename__ = "critical_issue_flags"

    flag_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(ForeignKey("vehicle_requirements.requirement_id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)  # SAFETY | REGULATORY_COMPLIANCE | SEVERE_RELIABILITY
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # PROPOSED | CONFIRMED | DISMISSED
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_refs_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    raised_by: Mapped[str] = mapped_column(String(128), nullable=False)
    decided_by: Mapped[str | None] = mapped_column(String(128))
    decision_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
