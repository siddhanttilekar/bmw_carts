"""Read-only decision context for the Part 6 Decision Assistant.

Everything returned here comes from stored records (runs, agent results,
assessments, evidence links, flags). The future assistant must use this
context instead of inventing scores or explanations. Nothing here writes to
the database or calls an LLM.

Supported questions (deterministic helpers):
  * why ranked first / which agent contributed most -> get_decision_context
  * which assessments are missing                   -> get_decision_context["missing_assessments"]
  * what if weights changed                          -> what_if_weights (recomputes from stored scores)
  * difference between two requirements              -> compare_requirements
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.prioritization_models import PriorityAssessment, PrioritizationRun, RequirementPriorityScore
from services.agent_explanations import summarize
from services.agent_weights import WeightEntry, percent_to_bp, validate_weights, weighted_contribution
from services.critical_issues import flags_for
from services.dynamic_prioritization import agent_results, latest_run_for
from services.requirement_review import get_requirement, requirement_as_dict


def get_decision_context(session: Session, requirement_id: str, run_id: str | None = None,
                         profile_id: str | None = None) -> dict:
    req = requirement_as_dict(session, get_requirement(session, requirement_id))
    run_id = run_id or latest_run_for(session, requirement_id, profile_id)
    base = {"requirement_id": requirement_id, "requirement_title": req["title"], "review_status": req["review_status"],
            "profile_id": None, "profile_version": None, "run_id": None, "priority_status": None, "final_score": None,
            "rank": None, "agent_contributions": [], "evidence_references": [], "human_assessments": [],
            "missing_assessments": [], "assumptions": list(req["assumptions"]), "limitations": [],
            "critical_flags": flags_for(session, requirement_id),
            "note": "Priority scores do not change approval status."}
    if run_id is None:
        base["limitations"].append("This requirement has not been prioritized yet.")
        return base
    run = session.get(PrioritizationRun, run_id)
    row = session.scalar(select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == run_id,
                                                                RequirementPriorityScore.requirement_id == requirement_id))
    if run is None or row is None:
        raise ValueError(f"Requirement {requirement_id} is not part of run {run_id}")
    results = agent_results(session, run_id, requirement_id)
    explanations = [r["explanation"] for r in results]
    grouped = summarize(explanations)
    contributions = sorted(
        ({"agent_id": r["agent_id"], "agent_name": r["agent_name"], "agent_version": r["agent_version"],
          "scoring_method": r["scoring_method"], "scoring_version": r["scoring_version"], "status": r["status"],
          "score": r["score"], "weight_percent": r["weight_percent"], "weighted_contribution": r["weighted_contribution"],
          "rationale": r["rationale"], "explanation": r["explanation"]} for r in results),
        key=lambda c: -(c["weighted_contribution"] or -1))
    refs = []
    for r in results:
        for ref in r["source_refs"]:
            item = {"agent": r["agent_name"], **ref}
            if item not in refs:
                refs.append(item)
    return {**base, "profile_id": run.profile_id, "profile_version": run.profile_version, "run_id": run_id,
            "run_created_at": run.created_at.isoformat(), "priority_status": row.priority_status,
            "final_score": row.final_score, "rank": row.rank, "agent_contributions": contributions,
            "top_contributor": contributions[0]["agent_name"] if contributions and contributions[0]["weighted_contribution"] else None,
            "top_contributors": [c["agent_name"] for c in contributions
                                 if contributions and c["weighted_contribution"] is not None
                                 and c["weighted_contribution"] == contributions[0]["weighted_contribution"]],
            "evidence_references": refs,
            "human_assessments": _assessments(session, results),
            "missing_assessments": json.loads(row.missing_agents_json),
            "assumptions": base["assumptions"] + grouped["ASSUMPTION"],
            "limitations": grouped["LIMITATION"], "facts": grouped["FACT"],
            "critical_flags_at_run": json.loads(row.critical_flags_json)}


def _assessments(session: Session, results: list[dict]) -> list[dict]:
    """The stored human assessments that the run's HUMAN_SCORE results used."""
    out = []
    for r in results:
        aid = r["inputs"].get("assessment_id")
        a = session.get(PriorityAssessment, aid) if aid else None
        if a is not None:
            out.append({"agent": r["agent_name"], "assessment_id": a.assessment_id, "score": a.score,
                        "rationale": a.rationale, "assessor_id": a.assessor_id,
                        "assessment_timestamp": a.assessment_timestamp.isoformat(),
                        "evidence_references": json.loads(a.evidence_refs_json), "superseded_since": a.superseded})
    return out


def what_if_weights(session: Session, run_id: str, weights: dict[str, object]) -> dict:
    """Recompute final scores of a stored run with different weights
    ({agent_id: percent}). Uses stored agent scores only - nothing is
    re-scored or saved. Weights must cover the run's agents and total 100%."""
    run = session.get(PrioritizationRun, run_id)
    if run is None:
        raise ValueError(f"Unknown run {run_id!r}")
    agents = {a["agent_id"]: a for a in json.loads(run.profile_snapshot_json)["agents"]}
    if set(weights) != set(agents):
        raise ValueError("What-if weights must be given for exactly the run's agents.")
    bp = {aid: percent_to_bp(p) for aid, p in weights.items()}
    check = validate_weights([WeightEntry(agent_id=aid, weight_bp=v) for aid, v in bp.items()])
    if not check.valid:
        raise ValueError(" ".join(check.errors))
    out = []
    for rid in json.loads(run.requirement_ids_json):
        results = agent_results(session, run_id, rid)
        if any(r["status"] != "SCORED" for r in results):
            out.append({"requirement_id": rid, "priority_status": "INCOMPLETE", "final_score": None})
            continue
        total = sum((weighted_contribution(r["score"], bp[r["agent_id"]]) for r in results), Decimal("0"))
        out.append({"requirement_id": rid, "priority_status": "COMPLETE",
                    "final_score": float(total.quantize(Decimal("0.01"), ROUND_HALF_UP))})
    ranked = sorted((o for o in out if o["final_score"] is not None), key=lambda o: (-o["final_score"], o["requirement_id"]))
    previous, rank = None, 0
    for position, o in enumerate(ranked, start=1):
        if o["final_score"] != previous:
            rank, previous = position, o["final_score"]
        o["rank"] = rank
    return {"run_id": run_id, "weights_percent": {k: str(v) for k, v in weights.items()}, "results": out,
            "note": "Hypothetical recalculation from stored agent scores; not saved."}


def compare_requirements(session: Session, run_id: str, requirement_a: str, requirement_b: str) -> dict:
    a, b = (get_decision_context(session, r, run_id=run_id) for r in (requirement_a, requirement_b))
    by_agent = {}
    for side, ctx in (("a", a), ("b", b)):
        for c in ctx["agent_contributions"]:
            by_agent.setdefault(c["agent_name"], {})[side] = c
    rows = []
    for name, sides in by_agent.items():
        ca, cb = sides.get("a", {}), sides.get("b", {})
        diff = None
        if ca.get("weighted_contribution") is not None and cb.get("weighted_contribution") is not None:
            diff = round(ca["weighted_contribution"] - cb["weighted_contribution"], 4)
        rows.append({"agent": name, "score_a": ca.get("score"), "score_b": cb.get("score"),
                     "contribution_a": ca.get("weighted_contribution"), "contribution_b": cb.get("weighted_contribution"),
                     "contribution_difference": diff})
    return {"run_id": run_id, "requirement_a": requirement_a, "requirement_b": requirement_b,
            "final_score_a": a["final_score"], "final_score_b": b["final_score"], "rank_a": a["rank"], "rank_b": b["rank"],
            "agents": sorted(rows, key=lambda r: -abs(r["contribution_difference"] or 0))}
