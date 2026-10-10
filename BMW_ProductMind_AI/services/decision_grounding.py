"""Shared grounding for Part 6: typed response items, run binding, evidence
reference validation and historical-context notices.

Statement kinds:
  FACT        verified from the database or a stored calculation
  ASSESSMENT  entered by a person
  ASSUMPTION  unverified product-management judgment / prototype methodology
  LIMITATION  missing or weak evidence
  SIMULATION  hypothetical result, not a saved official priority
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import CompetitorEvidence, CustomerStudy, Equipment, Feedback, PublicFeedback
from core.prioritization_models import PriorityAssessment, PrioritizationRun, RequirementPriorityScore
from core.requirement_models import RequirementEvidenceLink, VehicleRequirement

KINDS = ("FACT", "ASSESSMENT", "ASSUMPTION", "LIMITATION", "SIMULATION")
OFFLINE_LABEL = "Evidence-based explanation — not AI-generated"


class Item(BaseModel):
    kind: str
    text: str
    refs: list[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    intent: str
    title: str
    mode: str = "OFFLINE_EVIDENCE_BASED"
    label: str = OFFLINE_LABEL
    requirement_id: str | None = None
    run_id: str | None = None
    profile_id: str | None = None
    profile_version: int | None = None
    agent_id: str | None = None
    data_basis: str = "saved run snapshot"
    notices: list[str] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    evidence_refs: list[dict] = Field(default_factory=list)  # {"table", "key"}
    suggestions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False

    def add(self, kind: str, text: str, refs: list[str] | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"Unknown statement kind {kind}")
        self.items.append(Item(kind=kind, text=text, refs=refs or []))


# ---------------------------------------------------------------------------
# Run binding
# ---------------------------------------------------------------------------


class RunBindingError(ValueError):
    pass


def bind_run(session: Session, run_id: str | None, requirement_id: str | None = None) -> PrioritizationRun:
    """Every explanation is bound to one saved run (never mixing profiles)."""
    if not run_id:
        raise RunBindingError("Select a saved prioritization run.")
    run = session.get(PrioritizationRun, run_id)
    if run is None:
        raise RunBindingError(f"Unknown run {run_id!r}")
    if requirement_id and requirement_id not in json.loads(run.requirement_ids_json):
        raise RunBindingError(f"Requirement {requirement_id} was not part of run {run_id}.")
    return run


def run_agents(run: PrioritizationRun) -> list[dict]:
    """Agent configurations exactly as used by the run (from its snapshot)."""
    return json.loads(run.profile_snapshot_json)["agents"]


def historical_notices(session: Session, run: PrioritizationRun, requirement_id: str | None = None) -> list[str]:
    from core.prioritization_models import PrioritizationAgent

    notices = []
    newer = session.scalar(select(PrioritizationRun.run_id).where(
        PrioritizationRun.profile_id == run.profile_id, PrioritizationRun.created_at > run.created_at).limit(1))
    if newer:
        notices.append(f"Historical context: a newer run ({newer}) exists for this profile; this answer describes run "
                       f"{run.run_id} from {run.created_at:%Y-%m-%d %H:%M}.")
    for a in run_agents(run):
        current = session.get(PrioritizationAgent, a["agent_id"])
        if current is not None and current.configuration_version > a["configuration_version"]:
            notices.append(f"Agent '{a['name']}' has been changed since this run (run used v{a['configuration_version']}, "
                           f"current v{current.configuration_version}).")
    if requirement_id:
        row = session.scalar(select(RequirementPriorityScore).where(
            RequirementPriorityScore.run_id == run.run_id, RequirementPriorityScore.requirement_id == requirement_id))
        req = session.get(VehicleRequirement, requirement_id)
        if row is not None and req is not None and row.requirement_version is not None \
                and req.current_version != row.requirement_version:
            notices.append(f"The requirement was edited after this run (run used v{row.requirement_version}, "
                           f"current v{req.current_version}).")
        if row is not None and req is not None and req.review_status != row.review_status:
            notices.append(f"Review status changed since the run ({row.review_status} → {req.review_status}).")
    return notices


# ---------------------------------------------------------------------------
# Evidence references
# ---------------------------------------------------------------------------


def ref_exists(session: Session, table: str, key: str) -> bool:
    if table == "feedback":
        return session.scalar(select(Feedback.vehicle_code).where(Feedback.original_feedback_id == key)) == "G60"
    if table == "public_feedback":
        return session.scalar(select(PublicFeedback.record_id).where(PublicFeedback.original_public_id == key)) is not None
    if table == "competitor_evidence":
        row = key.removeprefix("COMP-R")
        return row.isdigit() and session.scalar(select(CompetitorEvidence.record_id)
                                                .where(CompetitorEvidence.source_row == int(row))) is not None
    if table == "customer_studies":
        study, _, attribute = key.partition("::")
        return session.scalar(select(CustomerStudy.record_id).where(CustomerStudy.study_name == study,
                                                                    CustomerStudy.attribute == attribute).limit(1)) is not None
    if table == "equipment":
        code, _, page = key.partition("@p")
        return session.scalar(select(Equipment.record_id).where(Equipment.option_code == code,
                                                                Equipment.vehicle_code == "G60").limit(1)) is not None
    if table == "priority_assessments":
        return session.get(PriorityAssessment, key) is not None
    if table == "vehicle_requirements":
        return session.get(VehicleRequirement, key) is not None
    return False


def validate_refs(session: Session, refs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (valid, invalid). Only valid references may be shown."""
    valid, invalid, seen = [], [], set()
    for r in refs:
        key = (r.get("table"), r.get("key"))
        if key in seen:
            continue
        seen.add(key)
        (valid if ref_exists(session, *key) else invalid).append({"table": key[0], "key": key[1]})
    return valid, invalid


def requirement_evidence(session: Session, requirement_id: str) -> dict[str, list[dict]]:
    """Evidence links of a requirement grouped by tier (validated)."""
    out: dict[str, list[dict]] = {"DIRECT": [], "SUPPORTING": [], "CONTEXTUAL": []}
    for l in session.scalars(select(RequirementEvidenceLink).where(RequirementEvidenceLink.requirement_id == requirement_id)
                             .order_by(RequirementEvidenceLink.link_id)):
        if ref_exists(session, l.source_table, l.source_key):
            out[l.tier].append({"table": l.source_table, "key": l.source_key, "scope": l.evidence_scope,
                                "requires_review": l.requires_review})
    return out


def finalize(session: Session, response: AssistantResponse) -> AssistantResponse:
    """Quality gate: drop invalid evidence references (reported as a limitation)."""
    valid, invalid = validate_refs(session, response.evidence_refs)
    response.evidence_refs = valid
    if invalid:
        response.add("LIMITATION", f"{len(invalid)} references could not be verified and were not shown.")
    return response
