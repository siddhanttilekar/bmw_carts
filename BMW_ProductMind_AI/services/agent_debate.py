"""Structured, evidence-bound multi-agent debate (Part 6).

Two saved requirements, one saved run. Each agent's "position" is built only
from its stored scores, rationales and contributions:

  ROUND 1  positions (favours A / favours B / no difference / unable to conclude)
  ROUND 2  disagreements between agents
  ROUND 3  criteria contributing most to the score difference
  ROUND 4  synthesis: agreement, disagreement, missing evidence, ranking under
           the run's profile, and sensitivity to single-agent weight changes

No arguments are invented, agents may abstain, and no winner is declared when
either requirement is incomplete.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.decision_models import AgentDebateSession
from core.prioritization_models import PriorityAssessment, RequirementPriorityScore
from services.decision_grounding import bind_run, historical_notices, run_agents
from services.agent_explanations import calculation_lines, evidence_lines
from services.dynamic_prioritization import agent_results

SENSITIVE_PP = 10  # a flip within +/-10 weight points counts as weight-sensitive


class AgentPosition(BaseModel):
    agent_id: str
    agent_name: str
    scoring_method: str
    weight_percent: float
    score_a: float | None
    score_b: float | None
    contribution_a: float | None
    contribution_b: float | None
    position: str  # FAVOURS_A | FAVOURS_B | NO_DIFFERENCE | UNABLE_TO_CONCLUDE
    basis_a: str
    basis_b: str
    evidence_a: int
    evidence_b: int
    limitations: list[str]
    # 2.0: evidence-first detail from the saved snapshot (no new scores are produced in a debate)
    execution_method: str = ""
    resources: list[str] = []
    abstains: bool = False
    abstain_reason: str | None = None
    evidence_counts_a: dict | None = None
    evidence_counts_b: dict | None = None
    counter_evidence_a: int | None = None
    counter_evidence_b: int | None = None
    calculation_a: list[str] = []
    calculation_b: list[str] = []
    evidence_summary_a: list[str] = []
    evidence_summary_b: list[str] = []


class DebateResult(BaseModel):
    debate_id: str | None
    run_id: str
    profile_id: str
    profile_version: int
    requirement_a: dict
    requirement_b: dict
    notices: list[str]
    round1_positions: list[AgentPosition]
    round2_disagreements: dict
    round3_drivers: list[dict]
    round4_synthesis: dict


def _basis(session: Session, result: dict) -> str:
    if result["status"] != "SCORED":
        return f"UNSCORED: {result['rationale']}"
    aid = result["inputs"].get("assessment_id")
    if aid:
        a = session.get(PriorityAssessment, aid)
        return f"Human assessment by {a.assessor_id}: {a.rationale}" if a else "Human assessment (record missing)"
    return result["rationale"]


def _sensitivity(positions: list[AgentPosition], diff: Decimal) -> list[dict]:
    """Weight-point shift to one agent (taken proportionally from the others)
    that would make the two final scores equal. Exact, from stored scores."""
    out = []
    usable = [p for p in positions if p.score_a is not None and p.score_b is not None]
    for p in usable:
        others = [q for q in usable if q.agent_id != p.agent_id]
        w_others = sum(Decimal(str(q.weight_percent)) for q in others)
        if not others or w_others == 0:
            continue
        avg_other = sum(Decimal(str(q.weight_percent)) * Decimal(str(q.score_a - q.score_b)) for q in others) / w_others
        d_i = Decimal(str(p.score_a - p.score_b))
        if d_i == avg_other:
            continue
        delta = -Decimal(100) * diff / (d_i - avg_other)
        new_weight = Decimal(str(p.weight_percent)) + delta
        if 0 <= new_weight <= 100 and delta <= w_others:
            out.append({"agent": p.agent_name, "weight_change_points": float(round(delta, 2)),
                        "weight_at_tie": float(round(new_weight, 2))})
    return sorted(out, key=lambda x: abs(x["weight_change_points"]))


def run_debate(session: Session, run_id: str, requirement_a: str, requirement_b: str, created_by: str | None = None,
               save: bool = False) -> DebateResult:
    if requirement_a == requirement_b:
        raise ValueError("Select two different requirements.")
    run = bind_run(session, run_id, requirement_a)
    bind_run(session, run_id, requirement_b)
    rows = {r.requirement_id: r for r in session.scalars(select(RequirementPriorityScore).where(
        RequirementPriorityScore.run_id == run_id, RequirementPriorityScore.requirement_id.in_([requirement_a, requirement_b])))}
    res_a = {r["agent_id"]: r for r in agent_results(session, run_id, requirement_a)}
    res_b = {r["agent_id"]: r for r in agent_results(session, run_id, requirement_b)}

    positions = []
    for agent in run_agents(run):
        ra, rb = res_a[agent["agent_id"]], res_b[agent["agent_id"]]
        if ra["status"] == "SCORED" and rb["status"] == "SCORED":
            pos = "FAVOURS_A" if ra["score"] > rb["score"] else "FAVOURS_B" if rb["score"] > ra["score"] else "NO_DIFFERENCE"
        else:
            pos = "UNABLE_TO_CONCLUDE"
        positions.append(AgentPosition(
            agent_id=agent["agent_id"], agent_name=agent["name"], scoring_method=agent["scoring_method"],
            weight_percent=agent["weight_bp"] / 100, score_a=ra["score"], score_b=rb["score"],
            contribution_a=ra["weighted_contribution"], contribution_b=rb["weighted_contribution"], position=pos,
            basis_a=_basis(session, ra), basis_b=_basis(session, rb), evidence_a=len(ra["source_refs"]),
            evidence_b=len(rb["source_refs"]), limitations=sorted(set(ra["limitations"]) | set(rb["limitations"])),
            execution_method=agent.get("execution_method", ""), resources=list(agent.get("resources", [])),
            abstains=pos == "UNABLE_TO_CONCLUDE",
            abstain_reason=("Insufficient evidence: " + "; ".join(
                f"{side}: {r['rationale']}" for side, r in (("A", ra), ("B", rb)) if r["status"] != "SCORED"))
            if pos == "UNABLE_TO_CONCLUDE" else None,
            evidence_counts_a=ra["inputs"].get("evidence_counts"), evidence_counts_b=rb["inputs"].get("evidence_counts"),
            counter_evidence_a=len(ra["inputs"]["counter_evidence_ids"]) if "counter_evidence_ids" in ra["inputs"] else None,
            counter_evidence_b=len(rb["inputs"]["counter_evidence_ids"]) if "counter_evidence_ids" in rb["inputs"] else None,
            calculation_a=calculation_lines(ra), calculation_b=calculation_lines(rb),
            evidence_summary_a=evidence_lines(ra), evidence_summary_b=evidence_lines(rb)))

    favours = {k: [p.agent_name for p in positions if p.position == k]
               for k in ("FAVOURS_A", "FAVOURS_B", "NO_DIFFERENCE", "UNABLE_TO_CONCLUDE")}
    disagreements = {**favours, "agents_disagree": bool(favours["FAVOURS_A"] and favours["FAVOURS_B"])}
    drivers = sorted(({"agent": p.agent_name, "contribution_a": p.contribution_a, "contribution_b": p.contribution_b,
                       "difference": round(p.contribution_a - p.contribution_b, 4)}
                      for p in positions if p.contribution_a is not None and p.contribution_b is not None),
                     key=lambda d: -abs(d["difference"]))

    ra_row, rb_row = rows[requirement_a], rows[requirement_b]
    complete = ra_row.priority_status == "COMPLETE" and rb_row.priority_status == "COMPLETE"
    missing = sorted({f"{p.agent_name} ({'A' if p.score_a is None else ''}{'B' if p.score_b is None else ''})"
                      for p in positions if p.position == "UNABLE_TO_CONCLUDE"})
    if complete:
        diff = Decimal(str(ra_row.final_score)) - Decimal(str(rb_row.final_score))
        higher = ("A" if diff > 0 else "B" if diff < 0 else "tie")
        ranking = (f"Under profile {run.profile_id} v{run.profile_version}, requirement "
                   f"{'A ranks higher' if higher == 'A' else 'B ranks higher' if higher == 'B' else 'A and B are tied'} "
                   f"({ra_row.final_score} vs {rb_row.final_score}; ranks {ra_row.rank} vs {rb_row.rank}).")
        flips = _sensitivity(positions, diff)
        sensitivity = {"margin_points": float(diff), "single_agent_shifts_to_tie": flips,
                       "classification": ("weight-sensitive" if flips and abs(flips[0]["weight_change_points"]) <= SENSITIVE_PP
                                          else "robust to single-agent weight changes of up to "
                                               f"{SENSITIVE_PP} points")}
    else:
        ranking = ("No ranking comparison: "
                   + ", ".join(f"requirement {k} is INCOMPLETE" for k, r in (("A", ra_row), ("B", rb_row))
                               if r.priority_status != "COMPLETE") + ". No winner is declared.")
        sensitivity = {"margin_points": None, "single_agent_shifts_to_tie": [], "classification": "not computable"}
    deciding = favours["FAVOURS_A"] + favours["FAVOURS_B"]
    if not deciding:
        agreement = "No agent could favour either requirement (scores missing or equal)."
    elif not favours["FAVOURS_B"]:
        agreement = f"All {len(deciding)} deciding agents favour A: {', '.join(favours['FAVOURS_A'])}."
    elif not favours["FAVOURS_A"]:
        agreement = f"All {len(deciding)} deciding agents favour B: {', '.join(favours['FAVOURS_B'])}."
    else:
        agreement = (f"Agents are split: {', '.join(favours['FAVOURS_A'])} favour A; "
                     f"{', '.join(favours['FAVOURS_B'])} favour B.")
    synthesis = {
        "agreement": agreement,
        "disagree": disagreements["agents_disagree"],
        "missing_evidence": missing,
        "ranking_under_profile": ranking,
        "sensitivity": sensitivity,
        "note": "Synthesis is computed from stored scores; it is not an official decision.",
    }
    req_info = lambda r: {"requirement_id": r.requirement_id, "title": r.requirement_title, "priority_status": r.priority_status,
                          "final_score": r.final_score, "rank": r.rank}
    result = DebateResult(debate_id=None, run_id=run_id, profile_id=run.profile_id, profile_version=run.profile_version,
                          requirement_a=req_info(ra_row), requirement_b=req_info(rb_row),
                          notices=historical_notices(session, run), round1_positions=positions,
                          round2_disagreements=disagreements, round3_drivers=drivers, round4_synthesis=synthesis)
    if save:
        if not (created_by or "").strip():
            raise ValueError("A user identifier is required to save a debate.")
        debate_id = f"DBT-{uuid.uuid4().hex[:10].upper()}"
        session.add(AgentDebateSession(
            debate_id=debate_id, run_id=run_id, profile_id=run.profile_id, profile_version=run.profile_version,
            requirement_a=requirement_a, requirement_b=requirement_b,
            positions_json=json.dumps([p.model_dump() for p in positions]),
            differences_json=json.dumps({"disagreements": disagreements, "drivers": drivers}),
            synthesis_json=json.dumps(synthesis), created_by=created_by.strip(),
            created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        session.flush()
        result.debate_id = debate_id
    return result


def list_debates(session: Session, run_id: str | None = None) -> list[dict]:
    stmt = select(AgentDebateSession).order_by(AgentDebateSession.created_at.desc())
    if run_id:
        stmt = stmt.where(AgentDebateSession.run_id == run_id)
    return [{"debate_id": d.debate_id, "run_id": d.run_id, "profile_version": d.profile_version,
             "requirement_a": d.requirement_a, "requirement_b": d.requirement_b,
             "synthesis": json.loads(d.synthesis_json), "created_by": d.created_by, "created_at": d.created_at}
            for d in session.scalars(stmt)]
