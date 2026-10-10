"""Challenge the priority (Part 6).

A challenge records who disputes what (requirement, run, profile version,
optionally an agent), the argument, a snapshot of the existing evidence, and
what would need review. Statuses: OPEN -> UNDER_REVIEW -> RESOLVED | DISMISSED
(reviewers can reopen a resolved/dismissed challenge to UNDER_REVIEW). Every
change is written to priority_challenge_events.

Challenges never change scores. A reassessment must be recorded through the
normal Part 5 assessment service and a new run; a resolution may reference
that assessment (validated to exist and to postdate the challenge).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.decision_models import PriorityChallenge, PriorityChallengeEvent
from core.prioritization_models import PriorityAssessment
from services.decision_grounding import bind_run, run_agents
from services.dynamic_prioritization import agent_results

TYPES = ("SCORE", "WEIGHT", "CRITICAL_ISSUE", "ASSUMPTION", "OTHER")
TRANSITIONS = {
    "start_review": ({"OPEN"}, "UNDER_REVIEW"),
    "resolve": ({"OPEN", "UNDER_REVIEW"}, "RESOLVED"),
    "dismiss": ({"OPEN", "UNDER_REVIEW"}, "DISMISSED"),
    "reopen": ({"RESOLVED", "DISMISSED"}, "UNDER_REVIEW"),
}


class ChallengeError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _event(session, c, action, prev, new, actor, note):
    session.add(PriorityChallengeEvent(challenge_id=c.challenge_id, action=action, previous_status=prev, new_status=new,
                                       actor=actor, note=note, created_at=_now()))


def create_challenge(session: Session, requirement_id: str, run_id: str, argument: str, raised_by: str,
                     challenge_type: str = "SCORE", agent_id: str | None = None) -> PriorityChallenge:
    if challenge_type not in TYPES:
        raise ChallengeError(f"Challenge type must be one of {', '.join(TYPES)}")
    if not argument.strip() or not (raised_by or "").strip():
        raise ChallengeError("An argument and a user identifier are required.")
    run = bind_run(session, run_id, requirement_id)
    agents = {a["agent_id"]: a for a in run_agents(run)}
    if agent_id and agent_id not in agents:
        raise ChallengeError("The challenged agent was not part of this run.")
    results = agent_results(session, run_id, requirement_id)
    snapshot, needs = [], []
    for r in results:
        if agent_id and r["agent_id"] != agent_id:
            continue
        snapshot.append({"agent_id": r["agent_id"], "agent": r["agent_name"], "agent_version": r["agent_version"],
                         "status": r["status"], "score": r["score"], "weight_percent": r["weight_percent"],
                         "weighted_contribution": r["weighted_contribution"],
                         "source_refs": [x["key"] for x in r["source_refs"][:50]], "inputs": r["inputs"]})
        aid = r["inputs"].get("assessment_id")
        if aid:
            a = session.get(PriorityAssessment, aid)
            needs.append(f"Human assessment {aid} by {a.assessor_id if a else '?'} ({r['agent_name']}) needs review.")
        elif r["status"] == "UNSCORED":
            needs.append(f"{r['agent_name']} is UNSCORED: {r['rationale']}")
        else:
            needs.append(f"Methodology/assumptions of {r['agent_name']} ({r['scoring_method']} v{r['scoring_version']}) "
                         "need review; data-derived scores change only if evidence or method change.")
    if challenge_type == "WEIGHT":
        needs.append(f"Profile {run.profile_id} v{run.profile_version} weights need review by the profile owner.")
    if challenge_type == "CRITICAL_ISSUE":
        needs.append("A critical-issue flag must be raised and confirmed by a person (Agent Prioritization → Explain my score).")
    now = _now()
    c = PriorityChallenge(challenge_id=f"CHL-{uuid.uuid4().hex[:10].upper()}", requirement_id=requirement_id, run_id=run_id,
                          profile_id=run.profile_id, profile_version=run.profile_version, agent_id=agent_id,
                          challenge_type=challenge_type, argument=argument.strip(),
                          evidence_snapshot_json=json.dumps(snapshot, default=str), needs_review_json=json.dumps(needs),
                          status="OPEN", raised_by=raised_by.strip(), created_at=now, updated_at=now)
    session.add(c)
    session.flush()
    _event(session, c, "create", None, "OPEN", c.raised_by, c.argument)
    session.flush()
    return c


def respond(session: Session, challenge_id: str, action: str, actor: str, note: str,
            linked_assessment_id: str | None = None) -> PriorityChallenge:
    c = session.get(PriorityChallenge, challenge_id)
    if c is None:
        raise ChallengeError(f"Unknown challenge {challenge_id!r}")
    if action == "comment":
        if not note.strip() or not (actor or "").strip():
            raise ChallengeError("A comment needs a note and a user identifier.")
        _event(session, c, "comment", c.status, c.status, actor.strip(), note.strip())
        session.flush()
        return c
    if action not in TRANSITIONS:
        raise ChallengeError(f"Unknown action {action!r}")
    allowed, new = TRANSITIONS[action]
    if c.status not in allowed:
        raise ChallengeError(f"Cannot {action} a challenge in status {c.status}.")
    if not note.strip() or not (actor or "").strip():
        raise ChallengeError("A reviewer identifier and a response note are required.")
    if linked_assessment_id:
        a = session.get(PriorityAssessment, linked_assessment_id)
        if a is None or a.requirement_id != c.requirement_id or a.assessment_timestamp < c.created_at:
            raise ChallengeError("The linked reassessment must exist, belong to this requirement and postdate the challenge.")
        c.linked_assessment_id = linked_assessment_id
    prev = c.status
    c.status, c.updated_at = new, _now()
    if new in ("RESOLVED", "DISMISSED"):
        c.resolution_note = note.strip()
    _event(session, c, action, prev, new, actor.strip(), note.strip())
    session.flush()
    return c


def challenge_view(session: Session, challenge_id: str) -> dict:
    c = session.get(PriorityChallenge, challenge_id)
    if c is None:
        raise ChallengeError(f"Unknown challenge {challenge_id!r}")
    events = [{"action": e.action, "previous_status": e.previous_status, "new_status": e.new_status, "actor": e.actor,
               "note": e.note, "at": e.created_at}
              for e in session.scalars(select(PriorityChallengeEvent).where(PriorityChallengeEvent.challenge_id == challenge_id)
                                       .order_by(PriorityChallengeEvent.event_id))]
    return {"challenge_id": c.challenge_id, "requirement_id": c.requirement_id, "run_id": c.run_id,
            "profile_id": c.profile_id, "profile_version": c.profile_version, "agent_id": c.agent_id,
            "challenge_type": c.challenge_type, "argument": c.argument, "status": c.status,
            "evidence_snapshot": json.loads(c.evidence_snapshot_json), "needs_review": json.loads(c.needs_review_json),
            "resolution_note": c.resolution_note, "linked_assessment_id": c.linked_assessment_id,
            "raised_by": c.raised_by, "created_at": c.created_at, "events": events}


def list_challenges(session: Session, requirement_id: str | None = None, run_id: str | None = None) -> list[dict]:
    stmt = select(PriorityChallenge.challenge_id).order_by(PriorityChallenge.created_at.desc())
    if requirement_id:
        stmt = stmt.where(PriorityChallenge.requirement_id == requirement_id)
    if run_id:
        stmt = stmt.where(PriorityChallenge.run_id == run_id)
    return [challenge_view(session, cid) for cid in session.scalars(stmt)]
