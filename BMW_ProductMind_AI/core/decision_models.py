"""Decision-intelligence tables (Part 6).

Registered on ``AIBase`` (additive migration via ``ensure_ai_schema``; never
dropped by ingestion rebuilds). Conversations, debates and challenges are
always bound to the prioritization run they were generated from. Messages
store evidence IDs, not verbatim customer text.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.ai_models import AIBase


class DecisionConversation(AIBase):
    __tablename__ = "decision_conversations"

    conversation_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), index=True)
    profile_id: Mapped[str | None] = mapped_column(String(40))
    profile_version: Mapped[int | None] = mapped_column(Integer)
    agent_id: Mapped[str | None] = mapped_column(String(40))
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class DecisionMessage(AIBase):
    __tablename__ = "decision_messages"

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("decision_conversations.conversation_id"), nullable=False, index=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(40))
    agent_id: Mapped[str | None] = mapped_column(String(40))
    intent: Mapped[str] = mapped_column(String(32), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)  # structured, typed items
    evidence_refs_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    response_mode: Mapped[str] = mapped_column(String(32), nullable=False)  # OFFLINE_EVIDENCE_BASED | LLM_ASSISTED
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentDebateSession(AIBase):
    __tablename__ = "agent_debate_sessions"

    debate_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    profile_id: Mapped[str] = mapped_column(String(40), nullable=False)
    profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    requirement_a: Mapped[str] = mapped_column(String(40), nullable=False)
    requirement_b: Mapped[str] = mapped_column(String(40), nullable=False)
    positions_json: Mapped[str] = mapped_column(Text, nullable=False)
    differences_json: Mapped[str] = mapped_column(Text, nullable=False)
    synthesis_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PriorityChallenge(AIBase):
    __tablename__ = "priority_challenges"

    challenge_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    requirement_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(40), nullable=False)
    profile_id: Mapped[str] = mapped_column(String(40), nullable=False)
    profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_id: Mapped[str | None] = mapped_column(String(40))
    challenge_type: Mapped[str] = mapped_column(String(32), nullable=False)  # SCORE | WEIGHT | CRITICAL_ISSUE | ASSUMPTION | OTHER
    argument: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    needs_review_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # OPEN | UNDER_REVIEW | RESOLVED | DISMISSED
    resolution_note: Mapped[str | None] = mapped_column(Text)
    linked_assessment_id: Mapped[str | None] = mapped_column(String(40))
    raised_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PriorityChallengeEvent(AIBase):
    """Audit trail for challenges."""

    __tablename__ = "priority_challenge_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    challenge_id: Mapped[str] = mapped_column(ForeignKey("priority_challenges.challenge_id"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String(16))
    new_status: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PrioritizationScenario(AIBase):
    """A saved hypothetical configuration + its simulated results. Never an
    official result; never overwrites profiles, runs or assessments."""

    __tablename__ = "prioritization_scenarios"

    scenario_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    base_run_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    base_profile_id: Mapped[str] = mapped_column(String(40), nullable=False)
    base_profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_json: Mapped[str] = mapped_column(Text, nullable=False)
    results_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ScenarioComparison(AIBase):
    __tablename__ = "scenario_comparisons"

    comparison_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    left_kind: Mapped[str] = mapped_column(String(16), nullable=False)  # run | scenario
    left_id: Mapped[str] = mapped_column(String(40), nullable=False)
    right_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    right_id: Mapped[str] = mapped_column(String(40), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ManagementReportExport(AIBase):
    """Export log (metadata only; report files are not stored in the database)."""

    __tablename__ = "management_report_exports"

    export_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    report_format: Mapped[str] = mapped_column(String(8), nullable=False)  # xlsx | csv | pdf
    run_id: Mapped[str] = mapped_column(String(40), nullable=False)
    options_json: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
