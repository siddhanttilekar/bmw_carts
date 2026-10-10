"""Human review workflow for vehicle requirements (Part 4).

State machine (every transition is recorded in requirement_review_history
with timestamp, previous/new status, reviewer and reason):

    (create) ──► DRAFT ──submit──► PENDING_REVIEW ──approve──► APPROVED
                   ▲                     │
                   │                     └──reject──► REJECTED
                   └──────── reopen (from REJECTED or APPROVED) ┘
    edit: allowed in any status; a non-DRAFT requirement returns to DRAFT
          with a recorded transition (so an edited requirement must be
          re-submitted and re-approved).

There is no transition from REJECTED to APPROVED; approval requires
PENDING_REVIEW, a reviewer ID different from the submitter (four-eyes), a
reason and a validation result without errors. Nothing is approved automatically.

Version 1 is always the original generated draft and is never modified;
every edit creates a new immutable version.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.requirement_models import (
    RequirementEvidenceLink,
    RequirementReviewEvent,
    RequirementVersion,
    VehicleRequirement,
)
from services.requirement_generator import AcceptanceCriterion, RequirementDraft
from services.requirement_validator import ValidationResult, validate_requirement

EDITABLE_FIELDS = ("title", "proposed_requirement", "customer_need", "customer_problem", "acceptance_criteria", "assumptions")
SNAPSHOT_FIELDS = ("title", "customer_need", "customer_problem", "proposed_requirement", "requirement_type",
                   "evidence_summary", "assumptions", "acceptance_criteria", "vehicle_model", "target_generation",
                   "requires_refinement", "refinement_notes")


class InvalidTransition(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_requirement_id() -> str:
    """Persistent identifier: generated once at creation and stored."""
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def requirement_as_dict(session: Session, req: VehicleRequirement) -> dict:
    links = session.scalars(select(RequirementEvidenceLink).where(RequirementEvidenceLink.requirement_id == req.requirement_id)
                            .order_by(RequirementEvidenceLink.link_id)).all()
    return {
        "requirement_id": req.requirement_id, "title": req.title, "source_cluster_id": req.source_cluster_id,
        "source_insight_id": req.source_insight_id, "need_code": req.need_code, "customer_need": req.customer_need,
        "customer_problem": req.customer_problem, "proposed_requirement": req.proposed_requirement,
        "requirement_type": req.requirement_type, "vehicle_model": req.vehicle_model,
        "source_vehicle_code": req.source_vehicle_code, "target_generation": req.target_generation,
        "supporting_evidence_ids": json.loads(req.supporting_evidence_ids_json),
        "evidence_summary": req.evidence_summary, "assumptions": json.loads(req.assumptions_json),
        "acceptance_criteria": json.loads(req.acceptance_criteria_json), "generation_method": req.generation_method,
        "template_id": req.template_id, "requires_refinement": req.requires_refinement,
        "refinement_notes": json.loads(req.refinement_notes_json), "review_status": req.review_status,
        "validation": json.loads(req.validation_json), "current_version": req.current_version,
        "created_by": req.created_by, "created_at": req.created_at, "updated_at": req.updated_at,
        "evidence": [{"tier": l.tier, "source_table": l.source_table, "source_key": l.source_key,
                      "source_record_id": l.source_record_id, "evidence_scope": l.evidence_scope,
                      "requires_review": l.requires_review, "note": l.note} for l in links],
    }


def _snapshot(data: dict) -> dict:
    return {k: data[k] for k in SNAPSHOT_FIELDS}


def _record(session, req, action, previous, new, reviewer, reason=None):
    session.add(RequirementReviewEvent(requirement_id=req.requirement_id, action=action, previous_status=previous,
                                       new_status=new, reviewer_id=reviewer, reason=reason,
                                       version_number=req.current_version, created_at=_now()))


def _require_actor(actor: str) -> str:
    if not actor or not actor.strip():
        raise InvalidTransition("A reviewer/editor identifier is required.")
    return actor.strip()


def _validate_and_store(session: Session, req: VehicleRequirement) -> ValidationResult:
    session.flush()
    result = validate_requirement(session, requirement_as_dict(session, req))
    req.validation_json = json.dumps([i.model_dump() for i in result.issues])
    return result


# ---------------------------------------------------------------------------
# Create / edit
# ---------------------------------------------------------------------------


def create_requirement(session: Session, draft: RequirementDraft, author: str,
                       edited: dict | None = None, edit_note: str | None = None) -> VehicleRequirement:
    """Persist a draft as a new requirement (status DRAFT). The generated draft
    is stored as version 1; ``edited`` field changes made before the first save
    become version 2, so the original is always preserved."""
    author = _require_actor(author)
    now = _now()
    req = VehicleRequirement(
        requirement_id=new_requirement_id(), title=draft.title, source_cluster_id=draft.source_cluster_id,
        source_insight_id=draft.source_insight_id, need_code=draft.need_code, customer_need=draft.customer_need,
        customer_problem=draft.customer_problem, proposed_requirement=draft.proposed_requirement,
        requirement_type=draft.requirement_type, vehicle_model=draft.vehicle_model,
        source_vehicle_code=draft.source_vehicle_code, target_generation=draft.target_generation,
        supporting_evidence_ids_json=json.dumps(draft.supporting_evidence_ids()),
        evidence_summary=draft.evidence_summary, assumptions_json=json.dumps(draft.assumptions),
        acceptance_criteria_json=json.dumps([c.model_dump() for c in draft.acceptance_criteria]),
        generation_method=draft.generation_method, template_id=draft.template_id,
        requires_refinement=draft.requires_refinement, refinement_notes_json=json.dumps(draft.refinement_notes),
        review_status="DRAFT", validation_json="[]", current_version=1, created_by=author, created_at=now, updated_at=now)
    session.add(req)
    session.flush()
    session.add_all(RequirementEvidenceLink(requirement_id=req.requirement_id, tier=e.tier, source_table=e.source_table,
                                            source_key=e.source_key, source_record_id=e.source_record_id,
                                            evidence_scope=e.evidence_scope, requires_review=e.requires_review, note=e.note)
                    for e in draft.evidence)
    session.flush()
    data = requirement_as_dict(session, req)
    session.add(RequirementVersion(requirement_id=req.requirement_id, version_number=1, change_type="generated",
                                   generation_method=draft.generation_method, snapshot_json=json.dumps(_snapshot(data), default=str),
                                   changed_fields_json="[]", changed_by=author,
                                   change_note=f"Original {draft.generation_method} draft", created_at=now))
    _record(session, req, "create", None, "DRAFT", author, f"Created from {draft.source_cluster_id} ({draft.generation_method})")
    if edited:
        edit_requirement(session, req.requirement_id, edited, author, edit_note or "Edited before first save")
    else:
        _validate_and_store(session, req)
    session.flush()
    return req


def edit_requirement(session: Session, requirement_id: str, changes: dict, editor: str, note: str | None = None
                     ) -> VehicleRequirement:
    editor = _require_actor(editor)
    req = get_requirement(session, requirement_id)
    unknown = set(changes) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"Fields not editable: {sorted(unknown)}")
    before = requirement_as_dict(session, req)
    changed = []
    for field, value in changes.items():
        if field == "acceptance_criteria":
            value = [AcceptanceCriterion.model_validate(c).model_dump() for c in value]
            if value != before["acceptance_criteria"]:
                req.acceptance_criteria_json = json.dumps(value)
                changed.append(field)
        elif field == "assumptions":
            value = [a.strip() for a in value if a and a.strip()]
            if value != before["assumptions"]:
                req.assumptions_json = json.dumps(value)
                changed.append(field)
        elif (value or "").strip() != before[field]:
            setattr(req, field, (value or "").strip())
            changed.append(field)
    if not changed:
        return req
    req.current_version += 1
    req.updated_at = _now()
    session.flush()
    data = requirement_as_dict(session, req)
    session.add(RequirementVersion(requirement_id=req.requirement_id, version_number=req.current_version, change_type="edited",
                                   generation_method="HUMAN_AUTHORED", snapshot_json=json.dumps(_snapshot(data), default=str),
                                   changed_fields_json=json.dumps(changed), changed_by=editor, change_note=note,
                                   created_at=_now()))
    previous = req.review_status
    if previous != "DRAFT":
        req.review_status = "DRAFT"
        _record(session, req, "edit", previous, "DRAFT", editor,
                f"Edited after {previous}; must be re-submitted. {note or ''}".strip())
    else:
        _record(session, req, "edit", "DRAFT", "DRAFT", editor, note or f"Edited: {', '.join(changed)}")
    _validate_and_store(session, req)
    session.flush()
    return req


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def _transition(session, requirement_id, action, allowed_from, new_status, actor, reason, require_reason):
    actor = _require_actor(actor)
    req = get_requirement(session, requirement_id)
    if req.review_status not in allowed_from:
        raise InvalidTransition(f"Cannot {action} a requirement in status {req.review_status} "
                                f"(allowed from: {', '.join(allowed_from)}).")
    if require_reason and not (reason or "").strip():
        raise InvalidTransition(f"A reason is required to {action}.")
    previous = req.review_status
    req.review_status = new_status
    req.updated_at = _now()
    _record(session, req, action, previous, new_status, actor, (reason or "").strip() or None)
    session.flush()
    return req


def submit_for_review(session: Session, requirement_id: str, actor: str, reason: str | None = None) -> VehicleRequirement:
    req = get_requirement(session, requirement_id)
    result = _validate_and_store(session, req)
    if not result.passed:
        raise InvalidTransition("Fix validation errors before submitting: "
                                + "; ".join(i.message for i in result.errors[:3]))
    return _transition(session, requirement_id, "submit", ("DRAFT",), "PENDING_REVIEW", actor, reason, False)


def approve(session: Session, requirement_id: str, reviewer: str, reason: str) -> VehicleRequirement:
    req = get_requirement(session, requirement_id)
    if req.requires_refinement:
        raise InvalidTransition("A problem-oriented draft that requires refinement cannot be approved; refine it first.")
    result = _validate_and_store(session, req)
    if not result.passed:
        raise InvalidTransition("Cannot approve: validation errors remain.")
    submitter = session.scalar(select(RequirementReviewEvent.reviewer_id).where(
        RequirementReviewEvent.requirement_id == requirement_id, RequirementReviewEvent.action == "submit")
        .order_by(RequirementReviewEvent.event_id.desc()).limit(1))
    if submitter and submitter == (reviewer or "").strip():
        raise InvalidTransition("Four-eyes rule: the approver must differ from the person who submitted the requirement.")
    return _transition(session, requirement_id, "approve", ("PENDING_REVIEW",), "APPROVED", reviewer, reason, True)


def reject(session: Session, requirement_id: str, reviewer: str, reason: str) -> VehicleRequirement:
    return _transition(session, requirement_id, "reject", ("PENDING_REVIEW",), "REJECTED", reviewer, reason, True)


def reopen(session: Session, requirement_id: str, actor: str, reason: str) -> VehicleRequirement:
    return _transition(session, requirement_id, "reopen", ("REJECTED", "APPROVED"), "DRAFT", actor, reason, True)


def add_comment(session: Session, requirement_id: str, actor: str, comment: str) -> VehicleRequirement:
    actor = _require_actor(actor)
    if not comment.strip():
        raise ValueError("Comment is empty.")
    req = get_requirement(session, requirement_id)
    _record(session, req, "comment", req.review_status, req.review_status, actor, comment.strip())
    session.flush()
    return req


def mark_refined(session: Session, requirement_id: str, editor: str, note: str) -> VehicleRequirement:
    """A human confirms that a problem-oriented draft has been refined into a
    specific requirement (after editing the statement). Recorded as an edit."""
    editor = _require_actor(editor)
    req = get_requirement(session, requirement_id)
    if not req.requires_refinement:
        return req
    if not note.strip():
        raise InvalidTransition("Explain how the need was refined.")
    req.requires_refinement = False
    req.requirement_type = req.requirement_type.replace("needs_refinement", "refined_by_human")
    _record(session, req, "edit", req.review_status, req.review_status, editor, f"Marked as refined: {note.strip()}")
    _validate_and_store(session, req)
    session.flush()
    return req


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_requirement(session: Session, requirement_id: str) -> VehicleRequirement:
    req = session.get(VehicleRequirement, requirement_id)
    if req is None:
        raise ValueError(f"Unknown requirement {requirement_id!r}")
    return req


class HistoryRow(BaseModel):
    created_at: datetime
    action: str
    previous_status: str | None
    new_status: str
    reviewer_id: str
    reason: str | None
    version_number: int


def history(session: Session, requirement_id: str) -> list[HistoryRow]:
    return [HistoryRow(created_at=e.created_at, action=e.action, previous_status=e.previous_status, new_status=e.new_status,
                       reviewer_id=e.reviewer_id, reason=e.reason, version_number=e.version_number)
            for e in session.scalars(select(RequirementReviewEvent).where(RequirementReviewEvent.requirement_id == requirement_id)
                                     .order_by(RequirementReviewEvent.event_id))]


def versions(session: Session, requirement_id: str) -> list[dict]:
    return [{"version": v.version_number, "change_type": v.change_type, "generation_method": v.generation_method,
             "changed_fields": json.loads(v.changed_fields_json), "changed_by": v.changed_by, "note": v.change_note,
             "created_at": v.created_at, "snapshot": json.loads(v.snapshot_json)}
            for v in session.scalars(select(RequirementVersion).where(RequirementVersion.requirement_id == requirement_id)
                                     .order_by(RequirementVersion.version_number))]


def list_requirements(session: Session, *, cluster_id: str | None = None, need_code: str | None = None) -> list[dict]:
    stmt = select(VehicleRequirement)
    if cluster_id:
        stmt = stmt.where(VehicleRequirement.source_cluster_id == cluster_id)
    if need_code:
        stmt = stmt.where(VehicleRequirement.need_code == need_code)
    return [{"requirement_id": r.requirement_id, "title": r.title, "status": r.review_status,
             "method": r.generation_method, "cluster": r.source_cluster_id, "need": r.need_code,
             "version": r.current_version, "updated_at": r.updated_at}
            for r in session.scalars(stmt.order_by(VehicleRequirement.updated_at.desc()))]


def status_counts(session: Session) -> dict[str, int]:
    return dict(session.execute(select(VehicleRequirement.review_status, func.count())
                                .group_by(VehicleRequirement.review_status)).all())
