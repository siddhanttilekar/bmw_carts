"""Critical-issue flags (Part 5).

Flags are raised and decided by people only. They are never inferred from
broad VFC labels and never change a priority score; a CONFIRMED flag stays
visible regardless of rank. Confirmation needs explicit evidence references
that belong to the requirement, or an explicit human confirmation note.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.prioritization_models import CriticalIssueFlag
from core.requirement_models import RequirementEvidenceLink, VehicleRequirement

CATEGORIES = ("SAFETY", "REGULATORY_COMPLIANCE", "SEVERE_RELIABILITY")


class FlagError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _check_refs(session: Session, requirement_id: str, refs: list[str]) -> None:
    keys = set(session.scalars(select(RequirementEvidenceLink.source_key)
                               .where(RequirementEvidenceLink.requirement_id == requirement_id)))
    unknown = [r for r in refs if r not in keys]
    if unknown:
        raise FlagError(f"Evidence references not linked to this requirement: {', '.join(unknown[:5])}")


def raise_flag(session: Session, requirement_id: str, category: str, rationale: str, raised_by: str,
               evidence_refs: list[str] | None = None) -> CriticalIssueFlag:
    if session.get(VehicleRequirement, requirement_id) is None:
        raise FlagError(f"Unknown requirement {requirement_id!r}")
    if category not in CATEGORIES:
        raise FlagError(f"Category must be one of {', '.join(CATEGORIES)}")
    if not rationale.strip() or not raised_by.strip():
        raise FlagError("Rationale and user identifier are required.")
    refs = evidence_refs or []
    _check_refs(session, requirement_id, refs)
    flag = CriticalIssueFlag(flag_id=f"FLG-{uuid.uuid4().hex[:10].upper()}", requirement_id=requirement_id,
                             category=category, status="PROPOSED", rationale=rationale.strip(),
                             evidence_refs_json=json.dumps(refs), raised_by=raised_by.strip(), created_at=_now(),
                             updated_at=_now())
    session.add(flag)
    session.flush()
    return flag


def decide_flag(session: Session, flag_id: str, decision: str, decided_by: str, note: str) -> CriticalIssueFlag:
    flag = session.get(CriticalIssueFlag, flag_id)
    if flag is None:
        raise FlagError(f"Unknown flag {flag_id!r}")
    if decision not in ("CONFIRMED", "DISMISSED"):
        raise FlagError("Decision must be CONFIRMED or DISMISSED")
    if not decided_by.strip() or not note.strip():
        raise FlagError("A decision needs a user identifier and a note (the explicit human confirmation).")
    flag.status, flag.decided_by, flag.decision_note, flag.updated_at = decision, decided_by.strip(), note.strip(), _now()
    session.flush()
    return flag


def flags_for(session: Session, requirement_id: str, include_dismissed: bool = False) -> list[dict]:
    stmt = select(CriticalIssueFlag).where(CriticalIssueFlag.requirement_id == requirement_id)
    if not include_dismissed:
        stmt = stmt.where(CriticalIssueFlag.status != "DISMISSED")
    return [{"flag_id": f.flag_id, "category": f.category, "status": f.status, "rationale": f.rationale,
             "evidence_refs": json.loads(f.evidence_refs_json), "raised_by": f.raised_by, "decided_by": f.decided_by,
             "decision_note": f.decision_note, "updated_at": f.updated_at.isoformat()}
            for f in session.scalars(stmt.order_by(CriticalIssueFlag.created_at))]
