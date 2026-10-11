"""Prioritization agent builder (Part 5).

Agents are configurable evaluation components - not autonomous LLM agents.
An agent's behaviour comes only from its (execution method, scoring method,
resources); its name is a label. Every create/edit writes a new immutable
configuration version; removal is a soft delete, so historical runs keep
their references.

2.0 builder fields: Agent Name, Objective, Custom Instructions, Resources
(including uploads), Context, Weight, Execution Method (with an approved
scoring rule). Custom Instructions and Context are plain text: they are shown
to people and, for AI_ANALYSIS, passed to the LLM as quoted scope notes for
evidence selection. They are never executed and never change a formula.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.prioritization_models import EXECUTION_METHODS, AgentConfigVersion, AgentResourceAssignment, PrioritizationAgent
from services.agent_scoring import check_compatibility
from services.agent_weights import percent_to_bp


class AgentConfigError(ValueError):
    pass


class AgentSpec(BaseModel):
    name: str
    description: str = ""
    objective: str
    execution_method: str
    scoring_method: str
    resources: list[str]
    weight_percent: float | str = 0
    enabled: bool = True
    instructions: str | None = None
    evaluation_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    context: str | None = None
    custom_metric_id: str | None = None


MAX_INSTRUCTIONS = 4000
MAX_CONTEXT = 1000


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _validate(spec: AgentSpec) -> list[str]:
    if not spec.name.strip():
        raise AgentConfigError("Agent name is required.")
    if not spec.objective.strip():
        raise AgentConfigError("Agent objective is required.")
    if spec.execution_method not in EXECUTION_METHODS:
        raise AgentConfigError(f"Execution method must be one of {', '.join(EXECUTION_METHODS)}.")
    if len(set(spec.resources)) != len(spec.resources):
        raise AgentConfigError("A resource is assigned twice.")
    if spec.instructions and len(spec.instructions) > MAX_INSTRUCTIONS:
        raise AgentConfigError(f"Custom instructions are limited to {MAX_INSTRUCTIONS} characters.")
    if spec.context and len(spec.context) > MAX_CONTEXT:
        raise AgentConfigError(f"Context is limited to {MAX_CONTEXT} characters.")
    if spec.scoring_method == "DOCUMENTED_CUSTOM_METRIC":
        from services.custom_metrics import CUSTOM_METRICS

        if spec.custom_metric_id not in CUSTOM_METRICS:
            raise AgentConfigError("Select a documented custom metric.")
    elif spec.custom_metric_id:
        raise AgentConfigError("A custom metric can only be used with the Documented custom metric scoring rule.")
    errors, warnings = check_compatibility(spec.execution_method, spec.scoring_method, spec.resources)
    if errors:
        raise AgentConfigError(" ".join(errors))
    percent_to_bp(spec.weight_percent)  # validates 0..100 with at most 2 decimals
    return warnings


def agent_snapshot(agent: PrioritizationAgent, resources: list[str], session: Session | None = None) -> dict:
    documents = []
    if session is not None and "RES_UPLOADED_DOCUMENTS" in resources:
        from services.agent_documents import agent_documents

        documents = [{"document_id": d["document_id"], "sha256": d["sha256"], "file_name": d["file_name"]}
                     for d in agent_documents(session, agent.agent_id)]
    return {"agent_id": agent.agent_id, "name": agent.name, "description": agent.description,
            "context": agent.context, "custom_metric_id": agent.custom_metric_id, "documents": documents,
            "objective": agent.objective, "execution_method": agent.execution_method,
            "scoring_method": agent.scoring_method, "resources": sorted(resources),
            "default_weight_bp": agent.default_weight_bp, "enabled": agent.enabled, "deleted": agent.deleted,
            "instructions": agent.instructions, "evaluation_questions": json.loads(agent.evaluation_questions_json),
            "limitations": json.loads(agent.limitations_json), "configuration_version": agent.configuration_version}


def _write_version(session: Session, agent: PrioritizationAgent, resources: list[str], actor: str, note: str) -> None:
    session.execute(delete(AgentResourceAssignment).where(AgentResourceAssignment.agent_id == agent.agent_id))
    session.add_all(AgentResourceAssignment(agent_id=agent.agent_id, resource_id=r) for r in sorted(resources))
    session.add(AgentConfigVersion(agent_id=agent.agent_id, version_number=agent.configuration_version,
                                   snapshot_json=json.dumps(agent_snapshot(agent, resources, session), default=str),
                                   change_note=note,
                                   created_by=actor, created_at=_now()))
    session.flush()


def _require_actor(actor: str) -> str:
    if not actor or not actor.strip():
        raise AgentConfigError("A user identifier is required.")
    return actor.strip()


def create_agent(session: Session, spec: AgentSpec, created_by: str) -> tuple[PrioritizationAgent, list[str]]:
    actor = _require_actor(created_by)
    warnings = _validate(spec)
    now = _now()
    agent = PrioritizationAgent(
        agent_id=f"AGT-{uuid.uuid4().hex[:10].upper()}", name=spec.name.strip(), description=spec.description.strip(),
        objective=spec.objective.strip(), execution_method=spec.execution_method, scoring_method=spec.scoring_method,
        default_weight_bp=percent_to_bp(spec.weight_percent), enabled=spec.enabled, deleted=False,
        configuration_version=1, instructions=spec.instructions, context=(spec.context or "").strip() or None,
        custom_metric_id=spec.custom_metric_id, evaluation_questions_json=json.dumps(spec.evaluation_questions),
        limitations_json=json.dumps(spec.limitations), created_by=actor, created_at=now, updated_at=now)
    session.add(agent)
    session.flush()
    _write_version(session, agent, spec.resources, actor, "created")
    return agent, warnings


def get_agent(session: Session, agent_id: str) -> PrioritizationAgent:
    agent = session.get(PrioritizationAgent, agent_id)
    if agent is None:
        raise AgentConfigError(f"Unknown agent {agent_id!r}")
    return agent


def agent_resources(session: Session, agent_id: str) -> list[str]:
    return sorted(session.scalars(select(AgentResourceAssignment.resource_id).where(AgentResourceAssignment.agent_id == agent_id)))


def current_spec(session: Session, agent_id: str) -> AgentSpec:
    a = get_agent(session, agent_id)
    from services.agent_weights import bp_to_percent

    return AgentSpec(name=a.name, description=a.description, objective=a.objective, execution_method=a.execution_method,
                     scoring_method=a.scoring_method, resources=agent_resources(session, agent_id),
                     weight_percent=str(bp_to_percent(a.default_weight_bp)), enabled=a.enabled, instructions=a.instructions,
                     context=a.context, custom_metric_id=a.custom_metric_id,
                     evaluation_questions=json.loads(a.evaluation_questions_json), limitations=json.loads(a.limitations_json))


def edit_agent(session: Session, agent_id: str, spec: AgentSpec, edited_by: str, note: str = "edited"
               ) -> tuple[PrioritizationAgent, list[str]]:
    """Creates a new configuration version; previous versions and the runs that
    used them are untouched."""
    actor = _require_actor(edited_by)
    agent = get_agent(session, agent_id)
    if agent.deleted:
        raise AgentConfigError("A removed agent cannot be edited; duplicate it instead.")
    warnings = _validate(spec)
    agent.name, agent.description, agent.objective = spec.name.strip(), spec.description.strip(), spec.objective.strip()
    agent.execution_method, agent.scoring_method = spec.execution_method, spec.scoring_method
    agent.default_weight_bp, agent.enabled = percent_to_bp(spec.weight_percent), spec.enabled
    agent.instructions = spec.instructions
    agent.context = (spec.context or "").strip() or None
    agent.custom_metric_id = spec.custom_metric_id
    agent.evaluation_questions_json = json.dumps(spec.evaluation_questions)
    agent.limitations_json = json.dumps(spec.limitations)
    agent.configuration_version += 1
    agent.updated_at = _now()
    _write_version(session, agent, spec.resources, actor, note)
    return agent, warnings


def duplicate_agent(session: Session, agent_id: str, created_by: str, new_name: str | None = None) -> PrioritizationAgent:
    spec = current_spec(session, agent_id)
    spec.name = new_name or f"{spec.name} (copy)"
    agent, _ = create_agent(session, spec, created_by)
    return agent


def set_enabled(session: Session, agent_id: str, enabled: bool, actor: str) -> PrioritizationAgent:
    spec = current_spec(session, agent_id)
    spec.enabled = enabled
    agent, _ = edit_agent(session, agent_id, spec, actor, "activated" if enabled else "deactivated")
    return agent


def remove_agent(session: Session, agent_id: str, actor: str) -> PrioritizationAgent:
    """Soft delete: the agent disappears from builders and new profiles, but its
    versions, assessments and historical results remain."""
    actor = _require_actor(actor)
    agent = get_agent(session, agent_id)
    if agent.deleted:
        return agent
    agent.deleted, agent.enabled = True, False
    agent.configuration_version += 1
    agent.updated_at = _now()
    _write_version(session, agent, agent_resources(session, agent_id), actor, "removed (soft delete)")
    return agent


def version_snapshot(session: Session, agent_id: str, version: int) -> dict:
    row = session.scalar(select(AgentConfigVersion).where(AgentConfigVersion.agent_id == agent_id,
                                                          AgentConfigVersion.version_number == version))
    if row is None:
        raise AgentConfigError(f"Agent {agent_id} has no version {version}")
    return json.loads(row.snapshot_json)


def list_agents(session: Session, include_removed: bool = False) -> list[dict]:
    from services.agent_weights import bp_to_percent

    stmt = select(PrioritizationAgent).order_by(PrioritizationAgent.created_at)
    if not include_removed:
        stmt = stmt.where(PrioritizationAgent.deleted.is_(False))
    out = []
    for a in session.scalars(stmt):
        res = agent_resources(session, a.agent_id)
        _, warnings = check_compatibility(a.execution_method, a.scoring_method, res)
        out.append({**agent_snapshot(a, res, session), "default_weight_percent": bp_to_percent(a.default_weight_bp),
                    "status": "REMOVED" if a.deleted else ("ACTIVE" if a.enabled else "INACTIVE"),
                    "warnings": warnings, "created_by": a.created_by, "updated_at": a.updated_at})
    return out


def versions(session: Session, agent_id: str) -> list[dict]:
    return [{"version": v.version_number, "note": v.change_note, "by": v.created_by, "at": v.created_at,
             "snapshot": json.loads(v.snapshot_json)}
            for v in session.scalars(select(AgentConfigVersion).where(AgentConfigVersion.agent_id == agent_id)
                                     .order_by(AgentConfigVersion.version_number))]


# ---------------------------------------------------------------------------
# Optional starter set (configuration only - no scores, no requirements)
# ---------------------------------------------------------------------------

STARTER_AGENTS = [
    AgentSpec(name="Customer Demand Agent", objective="Evaluate how frequently the customer need appears in BMW G60 feedback.",
              description="Distinct G60 comments linked to the requirement, log-normalised.",
              execution_method="DATA_ANALYSIS", scoring_method="LOG_NORMALIZED_FEEDBACK_COUNT",
              resources=["RES_G60_FEEDBACK", "RES_REQUIREMENTS"], weight_percent=30),
    AgentSpec(name="Evidence Coverage Agent", objective="Measure how much traceable evidence supports the requirement.",
              description="Evidence coverage only - not customer impact.", execution_method="DATA_ANALYSIS",
              scoring_method="EVIDENCE_COVERAGE",
              resources=["RES_REQUIREMENTS", "RES_G60_FEEDBACK", "RES_PUBLIC_FEEDBACK", "RES_CUSTOMER_STUDIES"],
              weight_percent=20),
    AgentSpec(name="Customer Impact Agent", objective="Product manager assesses the impact on the customer experience.",
              execution_method="HUMAN_ASSESSMENT", scoring_method="HUMAN_SCORE",
              resources=["RES_HUMAN_ASSESSMENTS", "RES_REQUIREMENTS"], weight_percent=25,
              evaluation_questions=["How strongly does the problem affect the customer experience?"]),
    AgentSpec(name="Technology Feasibility Agent", objective="Engineering assesses technical feasibility.",
              execution_method="HUMAN_ASSESSMENT", scoring_method="HUMAN_SCORE",
              resources=["RES_HUMAN_ASSESSMENTS", "RES_REQUIREMENTS"], weight_percent=25),
    AgentSpec(name="Market Relevance Agent", objective="Market-derived relevance (only with validated market data).",
              execution_method="DATA_ANALYSIS", scoring_method="VALIDATED_MARKET_RELEVANCE",
              resources=["RES_G60_FEEDBACK", "RES_SALES", "RES_REQUIREMENTS"], weight_percent=0, enabled=False,
              limitations=["Returns UNSCORED until market mappings are validated."]),
]


def seed_starter_agents(session: Session, actor: str) -> list[PrioritizationAgent]:
    """Create the illustrative starter agents (idempotent by name)."""
    existing = {a.name for a in session.scalars(select(PrioritizationAgent).where(PrioritizationAgent.deleted.is_(False)))}
    created = []
    for spec in STARTER_AGENTS:
        if spec.name not in existing:
            created.append(create_agent(session, spec.model_copy(deep=True), actor)[0])
    return created
